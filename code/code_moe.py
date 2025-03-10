import sys
import os
sys.path.append("/home/zsarwar/Projects/gpt-neox/")

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
# sys.path.append("/home/zsarwar/Projects/gpt-neox/megatron/mpu/")
import megablocks
import torch
import torch.nn.functional as F

import torch.distributed as dist

dist.init_process_group(backend='nccl')
from megatron.mpu import get_expert_token_counts_for_rank
from megatron.mpu import copy_to_expert_model_parallel_region
from megatron.mpu import gather_from_expert_model_parallel_region
from megatron.mpu import initialize_model_parallel
from megablocks import grouped_gemm_util as gg


hidden_size=256
intermediate_size=512
bsz=2
s_l=3
num_experts=8
top_k = 2





initialize_model_parallel(1)

device = "cuda:0"

class ParallelGroupedMLP(torch.nn.Module):
    def __init__(
        self,
        stride=1,
        multiple_of=256,
    ):
        """
        Copied from SparseMLP
        """
        super(ParallelGroupedMLP, self).__init__()


        self.multiple_of = multiple_of

        world_size = 1# get_model_parallel_world_size()
        self.num_experts = 8 #neox_args.moe_num_experts
        self.experts_per_rank = 8 # divide(self.num_experts, world_size)

        self.hidden_size = hidden_size  #neox_args.hidden_size

        self.per_expert_ff_dim = intermediate_size
        # number of rows per rank is the number of experts * ff dimension
        self.num_rows_per_rank = self.experts_per_rank * self.per_expert_ff_dim

        # input
        self.w1 = torch.nn.Parameter(
            torch.empty(
                self.num_rows_per_rank,
                self.hidden_size,
                device=torch.cuda.current_device(),
                dtype=torch.bfloat16,
            )
        )

        # output
        self.w2 = torch.nn.Parameter(
            torch.empty(
                self.num_rows_per_rank,
                self.hidden_size,
                device=torch.cuda.current_device(),
                dtype=torch.bfloat16,
            )
        )


        # TODO: why do we need this? was in original megablocks code
        self.gradient_scale = None
        if world_size > 1:
            self.gradient_scale = 1 / world_size


    def forward(self, x: torch.Tensor, tokens_per_expert: torch.Tensor):
        grouped_gemm_batch_sizes = tokens_per_expert.cpu().to(torch.long)
        w1, w2 = ((self.w1), (self.w2))
        #w1, w2 = [inter_hidden_size, hidden_size], [inter_hidden_size, hidden_size] 
        # Re-shape the weights for the grouped GEMMs
        w1 = w1.view(self.experts_per_rank, -1, self.hidden_size)
        w2 = w2.view(self.experts_per_rank, -1, self.hidden_size)

        # Compute the MLP
        x = gg.ops.gmm(x, w1, grouped_gemm_batch_sizes, trans_b=True)
        x = F.relu(x)
        return gg.ops.gmm(x, w2, grouped_gemm_batch_sizes)








def print_tensor_info(expert_indices, indices, bin_ids, bins, tokens_per_expert):
    print("\n===== Tensor Information =====")
    print(f"expert_indices          | Shape: {expert_indices.shape} | Data: {expert_indices}")
    print(f"indices          | Shape: {indices.shape} | Data: {indices}")
    print(f"bin_ids          | Shape: {bin_ids.shape} | Data: {bin_ids}")
    print(f"bins             | Shape: {bins.shape} | Data: {bins}")
    print(f"tokens_per_expert| Shape: {tokens_per_expert.shape} | Data: {tokens_per_expert}")
    print("================================\n")




def permute_and_compute(
    input_: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    expert_weights: torch.Tensor,
    bins: torch.Tensor,
    top_k: int,
):

    # Route the tokens for MoE computation.
    ## stack (sl, bs, hs) into (sl * bs, hs)
    input_ = input_.view(-1, input_.shape[-1])

    ## repeat each token top_k times and shuffle tokens to group them by their respective experts
    
    input_ = megablocks.ops.gather(input_, indices, bin_ids, bins, top_k)
    # [sl * bsz * topk, emb] where tokens are chunked by expert assingments [e_1,
                                                                        #    e_2,
                                                                        #    e_n  ]    
    # get tokens routed to this rank's experts only
    input_parallel = copy_to_expert_model_parallel_region(input_, tokens_per_expert) # Does nothing
    local_tokens_per_expert = get_expert_token_counts_for_rank(tokens_per_expert)
    
    # Perform the expert computation for this rank's experts
    
    output_parallel = mlp(input_parallel, local_tokens_per_expert)


    output = gather_from_expert_model_parallel_region(
        output_parallel,
        tokens_per_expert,
    ) # Does nothing



    # Un-route the data for the MoE output
    return megablocks.ops.scatter(
        output,
        indices,
        bin_ids,
        expert_weights,
        bins,
        top_k,
    )




def indices_and_bins(top_expert: torch.Tensor):
    # Sort the expert ids to produce the scatter/gather
    # indices for the permutation.
    #
    # TODO(tgale): Is it worth doing this conversion to 32-bit
    # prior? Could we place the `torch.max` operation to return
    # 32-bit expert indices?
    top_expert = top_expert.int()
    bin_ids, indices = megablocks.ops.sort(top_expert)

    # Histogram the expert ids to identify the number of
    # tokens routed to each expert.
    #
    # TODO(tgale): Does the sorted data produce a more favorable
    # data distribution for histogram? Or is the op parallelism
    # worth more?
    tokens_per_expert = megablocks.ops.histogram(top_expert, num_experts)

    # Calculate the bin bounds for the sorted tokens.
    bins = megablocks.ops.inclusive_cumsum(tokens_per_expert, 0)
    bins = bins.view(1) if not len(bins.size()) else bins
    return indices, bin_ids, bins, tokens_per_expert












mlp = ParallelGroupedMLP()

x = torch.randn(s_l,bsz,num_experts).to(device).to(torch.bfloat16) # indexing into [n:] means getting expert scores for the nth token in every batch


x = x.view(-1, x.shape[-1]) # Flattened - Batch sized chunks containing nth element of the s_l in order

in_shape = x.size()

scores  = F.softmax((x), dim=-1)
expert_weights, expert_indices = torch.topk(scores, top_k, dim=-1)


# save shape so we can re-shape the outputs later
in_shape = x.size()

# both are now (sl * bs * top_k)
expert_weights = expert_weights.flatten()
expert_indices = expert_indices.flatten() 
# Chunks of s_len in bsz * topk ordered by the idea that each batch element's topk choices are contiguous
# (sl_0_bsz_0_topk_1, sl_0_bsz_0_topk_2, sl_0_bsz_1_topk_1, sl_0_bsz_1_topk_2)




indices, bin_ids, bins, tokens_per_expert = indices_and_bins(expert_indices)


# Example usage:
print_tensor_info(expert_indices, indices, bin_ids, bins, tokens_per_expert)

"""
===== Tensor Information =====
expert_indices          | Shape: torch.Size([24]) | Data: tensor([3, 1, 3, 0, 3, 2, 1, 0, 3, 2, 3, 4, 4, 1, 1, 2, 4, 1, 1, 4, 4, 2, 4, 1],
       device='cuda:0')

indices          | Shape: torch.Size([24]) | Data: tensor([ 3,  7,  1,  6, 13, 14, 17, 18, 23,  5,  9, 15, 21,  0,  2,  4,  8, 10,
        11, 12, 16, 19, 20, 22], device='cuda:0', dtype=torch.int32)

bin_ids          | Shape: torch.Size([24]) | Data: tensor([0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4],
       device='cuda:0', dtype=torch.int32)

bins             | Shape: torch.Size([8]) | Data: tensor([ 2,  9, 13, 18, 24, 24, 24, 24], device='cuda:0', dtype=torch.int32)

tokens_per_expert| Shape: torch.Size([8]) | Data: tensor([2, 7, 4, 5, 6, 0, 0, 0], device='cuda:0', dtype=torch.int32)
================================


================================
EXPLANATION

indices_and_bins takes in expert_indices and provides a bunch of information. It will sort indices by expert id in ascending order

indices : chunks of indices telling when each token assigned to expert i appears in the expert_indices tensor in order of appearance.
        : Imagine traversing the expert_indices from l-r expert index wise and putting indices for each expert together (experts are scanned in ascending order)
        e.g expert_indices = [3,5,1,2,4,3]
            indices = [(0, 5), (2), (3), (4), (1) ]

bin_ids : Telling us the span of each expert
        e.g expert_indices = [3,5,1,2,4,3]
        indices = [(0, 5), (2), (3), (4), (1) ]
        bin_ids = [0,0,1,2,3,4]]

bins: Telling us the cumulative sum of tokens in each expert, sorted in ascending order. Cumsum of identical ids in bin_ids. Expert demarcations
    bin_ids = [0,0,1,2,3,4]]
    binds = [2,3,4,5,6]

    
tokens_per_expert: Num tokens assigned to each expert, where experts are sorted in ascending order. Cumsum of this results in bins

"""

tokens = torch.randn(s_l * bsz, hidden_size).to(device).to(dtype=torch.bfloat16)

tokens = permute_and_compute(
    tokens,
    tokens_per_expert,
    indices,
    bin_ids,
    expert_weights,
    bins,
    top_k,
)

print("Indices", indices)
# restore input shape
# tokens = tokens.view(in_shape)


dist.destroy_process_group()
