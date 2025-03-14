import sys
import os
sys.path.append("/home/zsarwar/Projects/neox/gpt-neox-smoe")

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
import megablocks
import torch
import torch.nn.functional as F
import random

import torch.distributed as dist
from megatron.mpu.utils import divide
dist.init_process_group(backend='nccl')
from megatron.mpu import initialize_model_parallel
from megatron.model.activations import get_activation, swish
from megatron.mpu.layers import _initialize_affine_weight_gpu
from megatron.mpu.initialize import get_model_parallel_world_size
from megatron.neox_arguments.arguments import NeoXArgs
from megatron.mpu import copy_to_expert_model_parallel_region
from megatron.mpu import get_expert_token_counts_for_rank
from megatron.mpu import gather_from_expert_model_parallel_region
from megatron.model.init_functions import init_method_zeros
from megatron.mpu import get_model_parallel_group, get_model_parallel_rank, get_model_parallel_world_size, get_data_parallel_group
from megatron import mpu
import numpy as np
import torch.nn as nn
import megablocks.ops
from megablocks import grouped_gemm_util as gg



device = "cuda:0"
dtype = torch.bfloat16

def print_tensor_info(expert_indices, indices, bin_ids, bins, tokens_per_lora, tokens_per_expert):
    print("\n===== Tensor Information =====")
    print(f"tokens_per_lora| Shape: {tokens_per_lora.shape} | Data: {tokens_per_lora}")
    print(f"expert_indices          | Shape: {expert_indices.shape} | Data: {expert_indices}")
    print(f"indices          | Shape: {indices.shape} | Data: {indices}")
    print(f"bin_ids          | Shape: {bin_ids.shape} | Data: {bin_ids}")
    print(f"bins             | Shape: {bins.shape} | Data: {bins}")
    print(f"tokens_per_expert| Shape: {tokens_per_expert.shape} | Data: {tokens_per_expert}")
    print("================================\n")


class MoEAuxLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that compute and scales the grad for auxiliary loss.

    """

    main_loss_backward_scale: torch.Tensor = torch.tensor(1.0)

    @staticmethod
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor):
        """Preserve the aux_loss by storing it in the context to avoid garbage collection.
        
        Args:
            output (torch.Tensor): The output tensor.
            aux_loss (torch.Tensor): The auxiliary loss tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute and scale the gradient for auxiliary loss..

        Args:
            grad_output (torch.Tensor): The gradient of the output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled auxiliary loss gradient.
        """
        (aux_loss,) = ctx.saved_tensors
        aux_loss_backward_scale = MoEAuxLossAutoScaler.main_loss_backward_scale
        scaled_aux_loss_grad = torch.ones_like(aux_loss) * aux_loss_backward_scale
        return grad_output, scaled_aux_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """set the scale of the aux loss.
        
        Args:
            scale (torch.Tensor): The scale value to set. Please ensure that the scale passed in matches the scale of the main_loss.
        """
        MoEAuxLossAutoScaler.main_loss_backward_scale = scale


def generate_integer_vector(n, x):
    """
    Generate an n-element list of integers that sum to x.
    """
    if n > x:
        raise ValueError("Impossible to distribute x into n integers (each must be at least 1).")

    # Generate n-1 random breakpoints in range [1, x-1] to split the sum
    breaks = sorted(random.sample(range(1, x), n-1))

    # Compute the differences between breakpoints to form integer parts
    parts = [breaks[0]] + [breaks[i] - breaks[i-1] for i in range(1, len(breaks))] + [x - breaks[-1]]
    
    return parts



initialize_model_parallel(1)
mpu.get_cuda_rng_tracker().add('model-parallel-rng', 1)



class ScaleGradient(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad):
        return grad * ctx.scale, None


scale_gradient = ScaleGradient.apply






class TopKTokenChoiceRouterLoRa(torch.nn.Module):
    # TODO: how do we ensure that all copies of the router get the same
    # initializations and stay in sync over time? Or is this handled by RNG seeding?
    def __init__(
        self,
        init_method=nn.init.xavier_normal_,
    ):
        super().__init__()
        self.jitter_eps = 0
        self.top_k = lora_top_k
        self.num_experts = num_experts
        self.moe_z_loss_coeff = moe_z_loss_coeff
        self.hidden_size = hidden_size
        self.aux_loss_coeff = moe_lora_aux_loss_coeff
        self.expert_parallel_group = get_model_parallel_group()
        self.expert_parallel_rank = get_model_parallel_rank()
        world_size = get_model_parallel_world_size()
        self.num_loras = moe_lora_experts
        self.num_experts = num_experts
        self.experts_per_rank = divide(self.num_experts, world_size)
        self.loras_per_rank = divide(self.num_loras, world_size)
        self.num_rows_per_rank = self.loras_per_rank *  self.experts_per_rank
        self.total_loras = self.loras_per_rank * self.experts_per_rank


        # Learned router parameters.
        #
        # NOTE: This weight matrix is not parallelized with expert tensor
        # parallelism. Each device needs the entire router weight matrix
        # so that it can route its batch of data correctly.
        self.w1 = torch.nn.Parameter( 
                    torch.empty(
                        self.num_rows_per_rank,
                        self.hidden_size,
                        device=torch.cuda.current_device(),
                        dtype=dtype
                        )
        )

        _initialize_affine_weight_gpu(self.w1, init_method, partition_dim=0, stride=1)

        self.gradient_scale = None
        if world_size > 1:
            self.gradient_scale = 1 / world_size

    def jitter(self, x):
        """
        Apply jittering to the input tensor during training.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Jittered input tensor.
        """
        low = 1.0 - self.jitter_eps
        high = 1.0 + self.jitter_eps
        noise = torch.rand(x.size(), dtype=x.dtype, device=x.device)
        return low + noise * (high - low)

    def _top_k(self, scores):
        """
        Select the top-k experts based on input scores.

        Args:
            scores (torch.Tensor): Input scores from the router.
                (sl * bs, num_experts)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing expert weightings and indices of selected experts.


        """
        if self.top_k == 1:
            return scores.max(dim=-1, keepdim=True)
        return torch.topk(scores, self.top_k, dim=-1)
    

    def switch_load_balancing_loss_func(self,
        probs: torch.Tensor, tokens_per_expert: torch.Tensor, topk: int, moe_aux_loss_coeff: float
    ):
        """Calculate the auxiliary loss for better load balacing. 
        Please refer to the Switch Transformer paper (https://arxiv.org/abs/2101.03961) for details.

        Args:
            probs (torch.Tensor): The softmax probs output by the router for each token. [num_tokens, num_experts]
            tokens_per_expert (torch.Tensor): The number of assigned tokens for each expert. [num_experts]

        Returns:
            torch.Tensor: The auxiliary loss for load balancing.
        """
        num_tokens = probs.shape[0] * topk
        num_experts = probs.shape[1]

        probs_mean_per_expert = probs.mean(dim=0)
        aux_loss = torch.sum(probs_mean_per_expert * tokens_per_expert) * (
            num_experts / num_tokens * moe_aux_loss_coeff
        )
        return aux_loss


    def apply_z_loss(self, logits):
        """Encourages the router's logits to remain small to enhance stability.
        Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.

        Args:
            logits (torch.Tensor): The logits of the router.

        Returns:
            torch.Tensor: The logits after applying the z-loss.
        """
        if self.moe_z_loss_coeff is not None:
            z_loss = z_loss_func(logits, self.moe_z_loss_coeff)
            logits = MoEAuxLossAutoScaler.apply(logits, z_loss)
            return logits, z_loss
        else:
            return logits, None


    
    def apply_load_balancing_loss(
            self,
            probs: torch.Tensor,
            num_local_tokens_per_expert: torch.Tensor,
            activation: torch.Tensor,
        ):
            """Applies auxiliary loss to the MoE layer.

            Args:
                probs (torch.Tensor): The probs output by the router for each token. [num_tokens, num_experts]
                num_local_tokens_per_expert (torch.Tensor): The number of tokens per expert. [num_experts]
                activation (torch.Tensor): The activation tensor to attach the gradient function to.

            Returns:
                torch.Tensor: The activation tensor with the attached gradient function.
            """            
            aux_loss = self.switch_load_balancing_loss_func(
                probs, num_local_tokens_per_expert, self.top_k, self.aux_loss_coeff
            )

            # if torch.cuda.current_device() == 0:
            #     print("Aux loss : ", aux_loss)

            activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
            
            return activation


    def scale_grad(self, w: torch.Tensor):
        """
        Copied from SparseMLP
        """
        if self.gradient_scale is None:
            return w
        return scale_gradient(w, self.gradient_scale)


    def generate_offsets(self, tokens_per_expert, num_loras):
        cum_lenghts = torch.cumsum(tokens_per_expert, dim=0)
        indices = torch.arange(cum_lenghts[-1]) + 1
        segments = (torch.searchsorted(cum_lenghts, indices, right=False) * num_loras).to(device=torch.cuda.current_device())

        return segments




    def forward(self, x, grouped_gemm_batch_sizes):
        """
        Forward pass through the Learned Router.

        Args:
            x (torch.Tensor): Input tensor to be routed.
                (sl, bs, hs)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing
                - expert_weights (sl * bs, top_k): Weights assigned to the selected experts
                - expert_indices (sl * bs, top_k): Indices of the selected experts
        """
        if self.training and self.jitter_eps is not None:
            x = x * self.jitter(x)

        # # x.view shape: (sl * bs, hs)...every token as a row
        # # scores (float) shape: (sl * bs, num_experts)...expert rankings for every token````
        # scores = self.layer(x.view(-1, x.shape[-1])).softmax(dim=-1)
        # #logits, z_loss = self.apply_z_loss(logits)
        # #z_loss_temp = z_loss.detach()


        # # expert_weights (float) shape: (sl * bs, top_k)...value(s) from scores corresponding to the top_k experts
        # # expert_indices (int) shape: (sl * bs, top_k)...index(indices) from scores corresponding to the top_k experts
        # expert_weights, expert_indices = self._top_k(scores)

        # with torch.no_grad():
        #     expert_indices_ft = expert_indices.flatten()
        #     tokens_per_expert = megablocks.ops.histogram(expert_indices_ft, self.num_experts)

        # expert_weights = self.apply_load_balancing_loss(scores, tokens_per_expert, activation=expert_weights)
        # # expert_weights probability mass won't add up to 1 because we took
        # # the topk scores from the softmax
        # # TODO: placeholder for moe_normalize_expert_weights if necessary


        if self.expert_parallel_rank == 0:

            # GMM on the router layer
            w1 = self.scale_grad(self.w1)

            w1 = w1.view(self.experts_per_rank, self.hidden_size, self.loras_per_rank)
            # Logits for each router
            scores = gg.ops.gmm(x, w1, grouped_gemm_batch_sizes).softmax(dim=-1) # [num_tokens, lora_distribution]
            expert_weights, expert_indices = self._top_k(scores)

            #offset_vector = self.generate_offsets(grouped_gemm_batch_sizes, self.loras_per_rank) + expert_indices.squeeze(1)
            
            #lora_routing_counts = torch.bincount(offset_vector, minlength=self.total_loras)
            #print(f"lora_routing_counts : {lora_routing_counts}")

            # broadcast the routing result to all ranks
            expert_weights_broadcast = torch.distributed.broadcast(
                expert_weights,
                src=torch.distributed.get_global_rank(self.expert_parallel_group, 0),
                group=self.expert_parallel_group,
                async_op=True,
            )
            expert_indices_broadcast = torch.distributed.broadcast(
                expert_indices,
                src=torch.distributed.get_global_rank(self.expert_parallel_group, 0),
                group=self.expert_parallel_group,
                async_op=True,
            )


        else:
            # sl * bs
            num_rows = x.view(-1, x.shape[-1]).shape[0]
            expert_weights = torch.empty(
                num_rows,
                self.top_k,
                device=torch.cuda.current_device(),
                dtype=self.params_dtype,
            )
            expert_indices = torch.empty(
                num_rows,
                self.top_k,
                device=torch.cuda.current_device(),
                dtype=torch.int64,
            )

            expert_weights_broadcast = torch.distributed.broadcast(
                expert_weights,
                src=torch.distributed.get_global_rank(self.expert_parallel_group, 0),
                group=self.expert_parallel_group,
                async_op=True,
            )
            expert_indices_broadcast = torch.distributed.broadcast(
                expert_indices,
                src=torch.distributed.get_global_rank(self.expert_parallel_group, 0),
                group=self.expert_parallel_group,
                async_op=True,
            )

        # aux loss for loop

        with torch.no_grad():
            #offset_vector = self.generate_offsets(grouped_gemm_batch_sizes, self.loras_per_rank)
            expert_indices_ft = expert_indices.flatten() #+ offset_vector
            #tokens_per_lora = megablocks.ops.histogram(expert_indices_ft, self.num_loras)

        start = 0
        for i in grouped_gemm_batch_sizes:
            if i > 0:
                with torch.no_grad():
                    tokens_per_lora = megablocks.ops.histogram(expert_indices_ft[start:i + start], self.num_loras)
                expert_weights[start:i + start, :] = self.apply_load_balancing_loss(scores[start:i + start, :], tokens_per_lora, activation=expert_weights[start:i + start, :])
            start = i

        # since both are executing asynchronously, it doesn't matter which one
        # we wait for first
        expert_weights_broadcast.wait()
        expert_indices_broadcast.wait()
        #global_lora_routing_counts.wait()

        #self.global_lora_routing_counts = lora_routing_counts

        return expert_weights, expert_indices


class ParallelGroupedLoRas(torch.nn.Module):
    def __init__(
        self,
        init_method=nn.init.xavier_normal_,
        output_layer_init_method=nn.init.xavier_normal_,
        stride=1,
        multiple_of=256,
    ):
        """
        Copied from SparseMLP
        """
        super().__init__()

        self.multiple_of = multiple_of

        world_size = get_model_parallel_world_size()
        self.num_experts = num_experts
        self.experts_per_rank = divide(self.num_experts, world_size)
        self.hidden_size = hidden_size
        self.zero_init_method = init_method_zeros()

        # Allow custom intermediate size
        if intermediate_size is not None:
            per_expert_ff_dim = intermediate_size
        # Otherwise, 4 x hidden size, padded to multiple of 256
        else:
            per_expert_ff_dim = 4 * self.hidden_size
            per_expert_ff_dim = self.multiple_of * (
                (per_expert_ff_dim + multiple_of - 1) // multiple_of
            )
        self.per_expert_ff_dim = per_expert_ff_dim
        # number of rows per rank is the number of experts * ff dimension
        #self.num_rows_per_rank = self.experts_per_rank * per_expert_ff_dim

        self.num_loras = moe_lora_experts
        self.num_experts = num_experts
        self.experts_per_rank = divide(self.num_experts, world_size)
        self.loras_per_rank = divide(self.num_loras, world_size)
        self.num_cols_per_rank = self.loras_per_rank *  self.experts_per_rank
        self.num_rows = self.experts_per_rank * self.loras_per_rank
        self.lora_rank = lora_rank
        self.total_loras = self.loras_per_rank * self.experts_per_rank
        

        self.w1_A = torch.nn.Parameter(
            torch.empty(
                self.num_rows,
                self.hidden_size * self.lora_rank,
                device=torch.cuda.current_device(),
                dtype=dtype,
            )
        )

        _initialize_affine_weight_gpu(self.w1_A, init_method, partition_dim=0, stride=stride)


        self.w1_B = torch.nn.Parameter(
            torch.empty(
                self.num_rows,
                self.per_expert_ff_dim * self.lora_rank,
                device=torch.cuda.current_device(),
                dtype=dtype,
            )
        )

        _initialize_affine_weight_gpu(self.w1_B, init_method, partition_dim=0, stride=stride)

        # TODO: why do we need this? was in original megablocks code
        self.gradient_scale = None
        if world_size > 1:
            self.gradient_scale = 1 / world_size

    def scale_grad(self, w: torch.Tensor):
        """
        Copied from SparseMLP
        """
        if self.gradient_scale is None:
            return w
        return scale_gradient(w, self.gradient_scale)

    def forward(self, x: torch.Tensor, tokens_per_lora: torch.Tensor, layer: int):
        grouped_gemm_batch_sizes = tokens_per_lora.cpu().to(torch.long)
        if layer == 1:
            # reshape and materialize all loras 
            w1_A, w1_B = (self.scale_grad(self.w1_A), self.scale_grad(self.w1_B))
            w1_A = w1_A.view(self.total_loras, self.hidden_size, self.lora_rank)
            w1_B = w1_B.view(self.total_loras, self.lora_rank, self.per_expert_ff_dim)
            #w1_AB = torch.einsum('ijk,ikm->ijm', w1_A, w1_B)
            return gg.ops.gmm(gg.ops.gmm(x, w1_A, grouped_gemm_batch_sizes), w1_B, grouped_gemm_batch_sizes)



            # GG 
            #return gg.ops.gmm(x, w1_AB, grouped_gemm_batch_sizes)
        elif layer == 2:
            w2_A, w2_B = (self.scale_grad(self.w2_A), self.scale_grad(self.w2_B))
            w2_A = w2_A.view(self.total_loras, self.per_expert_ff_dim, self.lora_rank)
            w2_B = w2_B.view(self.total_loras, self.lora_rank, self.hidden_size)
            #w2_AB = torch.einsum('ijk,ikm->ijm', w2_A, w2_B)
            return gg.ops.gmm(gg.ops.gmm(x, w2_A, grouped_gemm_batch_sizes), w2_B, grouped_gemm_batch_sizes)
            #return gg.ops.gmm(x, w2_AB, grouped_gemm_batch_sizes)
        else:
            print(f"No layer {layer} found")
        



class ParallelGroupedMLP(torch.nn.Module):
    def __init__(
        self,
        
        init_method=nn.init.xavier_normal_,
        output_layer_init_method=nn.init.xavier_normal_,
        stride=1,
        multiple_of=256,
    ):
        """
        Copied from SparseMLP
        """
        super(ParallelGroupedMLP, self).__init__()

        self.activation_type = "relu"
        self.activation_func = F.relu
        self.multiple_of = multiple_of

        world_size = get_model_parallel_world_size()
        self.num_experts = num_experts
        self.experts_per_rank = divide(self.num_experts, world_size)

        self.hidden_size = hidden_size
        self.LoRaRouter = TopKTokenChoiceRouterLoRa(init_method)


        self.num_loras = moe_lora_experts
        self.loras_per_rank = divide(self.num_loras, world_size)

        # init loras for each expert as another PGMLP?
        self.loras = ParallelGroupedLoRas(            
                init_method=init_method,
                output_layer_init_method=output_layer_init_method,
            )

        self.total_loras = self.loras_per_rank * self.experts_per_rank        
        self.sort_end_bit = max(int(np.ceil(np.log2(self.total_loras))), 1)

        # Allow custom intermediate size
        if intermediate_size is not None:
            per_expert_ff_dim = intermediate_size
        # Otherwise, 4 x hidden size, padded to multiple of 256
        else:
            per_expert_ff_dim = 4 * self.hidden_size
            per_expert_ff_dim = self.multiple_of * (
                (per_expert_ff_dim + multiple_of - 1) // multiple_of
            )

        self.per_expert_ff_dim = per_expert_ff_dim
        # number of rows per rank is the number of experts * ff dimension
        self.num_rows_per_rank = self.experts_per_rank * per_expert_ff_dim

        # input
        self.w1 = torch.nn.Parameter(
            torch.empty(
                self.num_rows_per_rank,
                self.hidden_size,
                device=torch.cuda.current_device(),
                dtype=dtype,
            )
        )
        # _initialize_affine_weight_gpu(
        #     self.w1, init_method, partition_dim=0, stride=stride
        # )

        # output
        self.w2 = torch.nn.Parameter(
            torch.empty(
                self.num_rows_per_rank,
                self.hidden_size,
                device=torch.cuda.current_device(),
                dtype=dtype,
            )
        )
        # _initialize_affine_weight_gpu(
        #     self.w2, output_layer_init_method, partition_dim=0, stride=stride
        # )


        # TODO: why do we need this? was in original megablocks code
        self.gradient_scale = None
        if world_size > 1:
            self.gradient_scale = 1 / world_size



    def generate_offsets(self, tokens_per_expert, num_loras):
        cum_lenghts = torch.cumsum(tokens_per_expert, dim=0)
        indices = torch.arange(cum_lenghts[-1]) + 1
        segments = (torch.searchsorted(cum_lenghts, indices, right=False) * num_loras).to(device=torch.cuda.current_device())

        return segments


    def indices_and_bins(self, top_expert: torch.Tensor):
        # Sort the expert ids to produce the scatter/gather
        # indices for the permutation.
        #
        # TODO(tgale): Is it worth doing this conversion to 32-bit
        # prior? Could we place the `torch.max` operation to return
        # 32-bit expert indices?
        top_expert = top_expert.int()
        bin_ids, indices = megablocks.ops.sort(top_expert)
        #bin_ids, indices = megablocks.ops.sort(top_expert)

        # Histogram the expert ids to identify the number of
        # tokens routed to each expert.
        #
        # TODO(tgale): Does the sorted data produce a more favorable
        # data distribution for histogram? Or is the op parallelism
        # worth more?
        tokens_per_expert = megablocks.ops.histogram(top_expert, self.total_loras)

        # Calculate the bin bounds for the sorted tokens.
        bins = megablocks.ops.inclusive_cumsum(tokens_per_expert, 0)
        bins = bins.view(1) if not len(bins.size()) else bins
        return indices, bin_ids, bins, tokens_per_expert


    def permute_and_compute(
        self,
        input_: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        expert_weights: torch.Tensor,
        bins: torch.Tensor,
        top_k: int,
        layer: int
    ):
        """
        grouped_permute_and_compute

        torch.distributed.all_reduce(tensor, op=<RedOpType.SUM: 0>, group=None, async_op=False)

        NOTE: Megablocks sets up all MLP tensors as column parallel and uses transposes on some of the grouped_gemm calls for the ops that would be row parallel. This seems to be fine and since we aren't using the underlying NeoX ColumnParallelLinear and RowParallelLinear classes, there doesn't seem to be a reason to change it...because that'd introduce a lot of additional complexity.

        column parallel linear forward

        ```python
        def forward(self, input_):
            if self.use_mup and self.mup_rescale_parameters:
                input_ /= self.width_mult()
            # Set up backprop all-reduce.
            input_parallel = copy_to_model_parallel_region(input_)
            # Matrix multiply.

            bias = self.bias if not self.skip_bias_add else None
            output_parallel = F.linear(input_parallel, self.weight, bias)
            if self.gather_output:
                # All-gather across the partitions.
                output = gather_from_model_parallel_region(output_parallel)
            else:
                output = output_parallel
            output_bias = self.bias if self.skip_bias_add else None
            return output, output_bias
        ```
        """
        # Route the tokens for MoE computation.
        ## stack (sl, bs, hs) into (sl * bs, hs)
        ## repeat each token top_k times and shuffle tokens to group them by their respective experts
        input_x = megablocks.ops.gather(input_, indices, bin_ids, bins, top_k)
        # get tokens routed to this rank's experts only
        input_parallel = copy_to_expert_model_parallel_region(input_x, tokens_per_expert)

        # get tokens_per_expert for this rank's experts only
        # with torch.no_grad():
        local_tokens_per_expert = get_expert_token_counts_for_rank(tokens_per_expert)
        # if torch.cuda.current_device() == 0:
        #     print(f"{torch.cuda.current_device()}: local_tokens_per_expert {local_tokens_per_expert}, global tokens {tokens_per_expert}")

        # Perform the expert computation for this rank's experts
        # print("local_tokens_per_expert", local_tokens_per_expert)
        output_parallel = self.loras(input_parallel, local_tokens_per_expert, layer)

        # all gather masked results from across Tensor parallel ranks here and cat them together
        # this will replicate the calculation of each expert across all ranks
        # NOTE: this combined all_gather and torch.cat operation is performed by gather_from_model_parallel_region(output_parallel)
        # Unlike ColumnParallelLinear, it is nonsensical in the MoE world
        # to optionally return the output_parallel result...we still have to scatter the tokens back to their original positions
        output = gather_from_expert_model_parallel_region(
            output_parallel,
            tokens_per_expert,
        )

        # Un-route the data for the MoE output
        return megablocks.ops.scatter(
            output,
            indices,
            bin_ids,
            expert_weights,
            bins,
            top_k,
        )

    def scale_grad(self, w: torch.Tensor):
        """
        Copied from SparseMLP
        """
        if self.gradient_scale is None:
            return w
        return scale_gradient(w, self.gradient_scale)



    def forward(self, x: torch.Tensor, tokens_per_expert: torch.Tensor):
        grouped_gemm_batch_sizes = tokens_per_expert.cpu().to(torch.long)
        lora_weights, lora_indices = self.LoRaRouter(x, grouped_gemm_batch_sizes)
        # Create an offset vector

        offset_vector = self.generate_offsets(grouped_gemm_batch_sizes, self.loras_per_rank)        
        # print(tokens_per_expert)
        # print(tokens_per_expert.shape)
        # print(offset_vector)
        # print(offset_vector.shape)
        # print(self.loras_per_rank)
        lora_weights = lora_weights.flatten()
        #lora_indices = lora_indices.flatten() + offset_vector.unsqueeze(-1)
        # print(lora_indices)
        lora_indices = lora_indices + offset_vector.unsqueeze(-1)
        # print(lora_indices)
        lora_indices = lora_indices.flatten()
        # print(lora_indices)



        with torch.no_grad():
            indices, lora_ids, lora_bins, tokens_per_lora = self.indices_and_bins(
                lora_indices
            )

        # print_tensor_info(lora_indices, indices, lora_ids, lora_bins, tokens_per_lora, tokens_per_expert)


        x_1_loras = self.permute_and_compute(
            x,
            tokens_per_lora,
            indices,
            lora_ids,
            lora_weights,
            lora_bins,
            self.LoRaRouter.top_k,
            1,
        )
        
        w1, w2 = (self.scale_grad(self.w1), self.scale_grad(self.w2))

        # Re-shape the weights for the grouped GEMMs
        w1 = w1.view(self.experts_per_rank, -1, self.hidden_size)
        w2 = w2.view(self.experts_per_rank, -1, self.hidden_size)

        # Compute the MLP
        x = gg.ops.gmm(x, w1, grouped_gemm_batch_sizes, trans_b=True)
        scaled_x = x + x_1_loras
        x = self.activation_func(scaled_x)

        x = gg.ops.gmm(x, w2, grouped_gemm_batch_sizes)

        return x
    



#====================================================================================================================================



hidden_size=256
intermediate_size=512
bsz=2
s_l=3
num_experts=4
moe_lora_experts=4
lora_rank=64
lora_top_k=3
top_k = 2
moe_z_loss_coeff=0.0
moe_lora_aux_loss_coeff=0.0


mlps = ParallelGroupedMLP()

x = torch.randn(s_l * bsz * top_k, hidden_size).to(device).to(dtype=dtype)
tokens_per_expert = torch.tensor(generate_integer_vector(num_experts, s_l * bsz * top_k)).to(device).to(dtype=dtype)

x = mlps(x, tokens_per_expert)


dist.destroy_process_group()