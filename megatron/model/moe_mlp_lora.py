# Copyright (c) 2024, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2023 MegaBlocks authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from megatron.model.activations import get_activation, swish
from megatron.mpu.layers import _initialize_affine_weight_gpu
from megatron.mpu.initialize import get_model_parallel_world_size
from megatron.mpu.utils import divide
from megatron.model.router import TopKTokenChoiceRouterLoRa
from megatron.neox_arguments.arguments import NeoXArgs
from megatron.mpu import copy_to_expert_model_parallel_region
from megatron.mpu import get_expert_token_counts_for_rank
from megatron.mpu import gather_from_expert_model_parallel_region
from megatron.model.init_functions import init_method_zeros
import numpy as np
import megablocks.ops
from megablocks import grouped_gemm_util as gg


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



class ParallelGroupedLoRas(torch.nn.Module):
    def __init__(
        self,
        neox_args: NeoXArgs,
        init_method,
        output_layer_init_method,
        stride=1,
        multiple_of=256,
    ):
        """
        Copied from SparseMLP
        """
        super().__init__()
        
        self.multiple_of = multiple_of

        world_size = get_model_parallel_world_size()
        self.num_experts = neox_args.moe_num_experts
        self.experts_per_rank = divide(self.num_experts, world_size)
        self.hidden_size = neox_args.hidden_size
        self.zero_init_method = init_method_zeros()

        # Allow custom intermediate size
        if neox_args.intermediate_size is not None:
            per_expert_ff_dim = neox_args.intermediate_size
        # Otherwise, 4 x hidden size, padded to multiple of 256
        else:
            per_expert_ff_dim = 4 * self.hidden_size
            per_expert_ff_dim = self.multiple_of * (
                (per_expert_ff_dim + multiple_of - 1) // multiple_of
            )
        self.per_expert_ff_dim = per_expert_ff_dim
        # number of rows per rank is the number of experts * ff dimension
        #self.num_rows_per_rank = self.experts_per_rank * per_expert_ff_dim

        self.num_loras = neox_args.moe_lora_experts
        self.num_experts = neox_args.moe_num_experts
        self.experts_per_rank = divide(self.num_experts, world_size)
        self.loras_per_rank = divide(self.num_loras, world_size)
        self.num_cols_per_rank = self.loras_per_rank *  self.experts_per_rank
        self.num_rows = self.experts_per_rank * self.loras_per_rank
        self.lora_rank = neox_args.lora_rank
        self.total_loras = self.loras_per_rank * self.experts_per_rank
        

        self.w1_A = torch.nn.Parameter(
            torch.empty(
                self.num_rows,
                self.hidden_size * self.lora_rank,
                device=torch.cuda.current_device(),
                dtype=neox_args.params_dtype,
            )
        )

        _initialize_affine_weight_gpu(self.w1_A, init_method, partition_dim=0, stride=stride)


        self.w1_B = torch.nn.Parameter(
            torch.empty(
                self.num_rows,
                self.per_expert_ff_dim * self.lora_rank,
                device=torch.cuda.current_device(),
                dtype=neox_args.params_dtype,
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
        neox_args: NeoXArgs,
        init_method,
        output_layer_init_method,
        stride=1,
        multiple_of=256,
    ):
        """
        Copied from SparseMLP
        """
        super(ParallelGroupedMLP, self).__init__()
        self.args = neox_args
        self.activation_func = get_activation(neox_args)
        self.activation_type = neox_args.activation

        self.multiple_of = multiple_of

        world_size = get_model_parallel_world_size()
        self.num_experts = neox_args.moe_num_experts
        self.experts_per_rank = divide(self.num_experts, world_size)

        self.hidden_size = neox_args.hidden_size
        self.LoRaRouter = TopKTokenChoiceRouterLoRa(neox_args, init_method)


        self.num_loras = neox_args.moe_lora_experts
        self.loras_per_rank = divide(self.num_loras, world_size)

        # init loras for each expert as another PGMLP?
        self.loras = ParallelGroupedLoRas(            
                neox_args=neox_args,
                init_method=init_method,
                output_layer_init_method=output_layer_init_method,
            )

        self.total_loras = self.loras_per_rank * self.experts_per_rank        
        self.sort_end_bit = max(int(np.ceil(np.log2(self.total_loras))), 1)

        # Allow custom intermediate size
        if neox_args.intermediate_size is not None:
            per_expert_ff_dim = neox_args.intermediate_size
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
                dtype=neox_args.params_dtype,
            )
        )
        _initialize_affine_weight_gpu(
            self.w1, init_method, partition_dim=0, stride=stride
        )

        # output
        self.w2 = torch.nn.Parameter(
            torch.empty(
                self.num_rows_per_rank,
                self.hidden_size,
                device=torch.cuda.current_device(),
                dtype=neox_args.params_dtype,
            )
        )
        _initialize_affine_weight_gpu(
            self.w2, output_layer_init_method, partition_dim=0, stride=stride
        )


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


        lora_weights = lora_weights.flatten()
        #lora_indices = lora_indices.flatten() + offset_vector.unsqueeze(-1)
        lora_indices = lora_indices + offset_vector.unsqueeze(-1)
        lora_indices = lora_indices.flatten()
    
        with torch.no_grad():
            indices, lora_ids, lora_bins, tokens_per_lora = self.indices_and_bins(
                lora_indices
            )

            #print(f"tokens_per_lora : {tokens_per_lora}")


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
        if self.args.lora_interaction_type == 'addition':          
            scaled_x = x + x_1_loras
            x = self.activation_func(scaled_x)
        elif self.args.lora_interaction_type == 'geglu':
            x = x_1_loras * self.activation_func(x)
        else:
            raise("LoRe interaction not defined")

        x = gg.ops.gmm(x, w2, grouped_gemm_batch_sizes)
    
        return x
