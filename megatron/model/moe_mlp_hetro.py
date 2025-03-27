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
import numpy as np
from megatron.model.activations import get_activation

from megatron.mpu.layers import _initialize_affine_weight_gpu
from megatron.mpu.initialize import get_model_parallel_world_size
from megatron.mpu.utils import divide

from megatron.neox_arguments.arguments import NeoXArgs
from .router import TopKTokenChoiceRouterMinion
from megablocks import grouped_gemm_util as gg
from liger_kernel.ops.swiglu import LigerSiLUMulFunction
from liger_kernel.ops.geglu import LigerGELUMulFunction
import megablocks.ops

from megatron import mpu
from megatron.mpu import get_expert_token_counts_for_rank
from megatron.mpu import get_expert_tokens_for_rank
from megatron.mpu import copy_to_expert_model_parallel_region
from megatron.mpu import gather_from_expert_model_parallel_region
from megatron.neox_arguments.arguments import NeoXArgs



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

        self.activation_func = get_activation(neox_args)
        self.activation_type = neox_args.activation

        self.multiple_of = multiple_of

        world_size = get_model_parallel_world_size()
        self.num_minion_experts = neox_args.num_minion_experts
        self.num_experts = neox_args.moe_num_experts - self.num_minion_experts
        self.experts_per_rank = divide(self.num_experts, world_size)
        self.hidden_size = neox_args.hidden_size

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

    def scale_grad(self, w: torch.Tensor):
        """
        Copied from SparseMLP
        """
        if self.gradient_scale is None:
            return w
        return scale_gradient(w, self.gradient_scale)

    def forward(self, x: torch.Tensor, tokens_per_expert: torch.Tensor):
        grouped_gemm_batch_sizes = tokens_per_expert.cpu().to(torch.long)
        w1, w2 = (self.scale_grad(self.w1), self.scale_grad(self.w2))

        # Re-shape the weights for the grouped GEMMs
        w1 = w1.view(self.experts_per_rank, -1, self.hidden_size)
        w2 = w2.view(self.experts_per_rank, -1, self.hidden_size)

        # Compute the MLP
        # print("x.shape, w1.shape, grouped_gemm_batch_sizes", x.shape, w1.shape, grouped_gemm_batch_sizes)
        x = gg.ops.gmm(x, w1, grouped_gemm_batch_sizes, trans_b=True)
        x = self.activation_func(x)
        return gg.ops.gmm(x, w2, grouped_gemm_batch_sizes)





#========================================================================================================================================================




class ParallelGroupedMinions(torch.nn.Module):
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
        super(ParallelGroupedMinions, self).__init__()

        self.activation_func = get_activation(neox_args)
        self.activation_type = neox_args.activation

        self.multiple_of = multiple_of

        world_size = get_model_parallel_world_size()
        self.num_experts = neox_args.num_minion_experts
        self.lora_rank = neox_args.lora_rank
        self.num_loras = neox_args.moe_lora_experts
        self.experts_per_rank = divide(self.num_experts, world_size)
        self.loras_per_rank = divide(self.num_loras, world_size)
        # self.num_rows = self.experts_per_rank * self.loras_per_rank
        self.hidden_size = neox_args.hidden_size

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
        self.num_rows_per_rank = self.loras_per_rank * self.lora_rank


        self.w1_A = torch.nn.Parameter(
            torch.empty(
                self.num_rows_per_rank,
                self.hidden_size,
                device=torch.cuda.current_device(),
                dtype=neox_args.params_dtype,
            )
        )

        self.w1_B = torch.nn.Parameter(
            torch.empty(
                self.num_rows_per_rank,
                self.per_expert_ff_dim,
                device=torch.cuda.current_device(),
                dtype=neox_args.params_dtype,
            )
        )


        self.w2_A = torch.nn.Parameter(
            torch.empty(
                self.num_rows_per_rank,
                self.per_expert_ff_dim,
                device=torch.cuda.current_device(),
                dtype=neox_args.params_dtype,
            )
        )

        self.w2_B = torch.nn.Parameter(
            torch.empty(
                self.num_rows_per_rank,
                self.hidden_size,
                device=torch.cuda.current_device(),
                dtype=neox_args.params_dtype,
            )
        )

        _initialize_affine_weight_gpu(
            self.w1_A, init_method, partition_dim=0, stride=stride
        )

        _initialize_affine_weight_gpu(
            self.w1_B, init_method, partition_dim=0, stride=stride
        )


        _initialize_affine_weight_gpu(
            self.w2_A, output_layer_init_method, partition_dim=0, stride=stride
        )

        _initialize_affine_weight_gpu(
            self.w2_B, output_layer_init_method, partition_dim=0, stride=stride
        )

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

    def forward(self, x: torch.Tensor, tokens_per_expert: torch.Tensor):
        grouped_gemm_batch_sizes = tokens_per_expert.cpu().to(torch.long)
        w1_A, w1_B = (self.scale_grad(self.w1_A), self.scale_grad(self.w1_B))
        w1_A = w1_A.view(self.loras_per_rank, -1, self.hidden_size)
        w1_B = w1_B.view(self.loras_per_rank, -1, self.per_expert_ff_dim) # T
        w2_A, w2_B = (self.scale_grad(self.w2_A), self.scale_grad(self.w2_B))
        w2_A = w2_A.view(self.loras_per_rank, -1, self.per_expert_ff_dim)
        w2_B = w2_B.view(self.loras_per_rank, -1, self.hidden_size) # T
        # print(f"x.shape, w1_A.shape, w1_B.shape, w2_A.shape, w2_B.shape,", x.shape, w1_A.shape, w1_B.shape, w2_A.shape, w2_B.shape)
        # print("grouped_gemm_batch_sizes", grouped_gemm_batch_sizes)
        # exit()
        x = gg.ops.gmm(gg.ops.gmm(x, w1_A, grouped_gemm_batch_sizes,trans_b=True), w1_B, grouped_gemm_batch_sizes)
        x = self.activation_func(x)
        return gg.ops.gmm(gg.ops.gmm(x, w2_A, grouped_gemm_batch_sizes, trans_b=True), w2_B, grouped_gemm_batch_sizes)









class ParallelDroplessMinions(torch.nn.Module):
    """
    This class defines MoE expert computation, using tensor (model) parallel size as the expert parallel size

    The implication of this parallelism decision is that the expert weights can only be sharded within a single node
    """

    def __init__(
        self,
        neox_args: NeoXArgs,
        init_method,
        output_layer_init_method,
    ):
        """

        Bias is currently not supported
        """
        super(ParallelDroplessMinions, self).__init__()

        # Calculate the number of experts to allocate on this rank
        world_size = mpu.get_model_parallel_world_size()
        assert neox_args.moe_num_experts % world_size == 0
        self.num_experts = neox_args.moe_lora_experts
        self.experts_per_rank = self.num_experts // world_size
        self.top_k = neox_args.lora_top_k

        # Calculate the number of bits needed to represent the expert indices
        # so that we can pass it to radix sort
        self.sort_end_bit = max(int(np.ceil(np.log2(self.num_experts))), 1)

        # decide which parallel grouped MLP implementation to use
        if neox_args.mlp_type == "regular":
            self.mlp = ParallelGroupedMinions(
                neox_args=neox_args,
                init_method=init_method,
                output_layer_init_method=output_layer_init_method,
            )
        else:
            raise KeyError(neox_args.mlp_type)



    def indices_and_bins(self, top_expert: torch.Tensor):
        # Sort the expert ids to produce the scatter/gather
        # indices for the permutation.
        #
        # TODO(tgale): Is it worth doing this conversion to 32-bit
        # prior? Could we place the `torch.max` operation to return
        # 32-bit expert indices?
        top_expert = top_expert.int()
        bin_ids, indices = megablocks.ops.sort(top_expert, self.sort_end_bit)

        # Histogram the expert ids to identify the number of
        # tokens routed to each expert.
        #
        # TODO(tgale): Does the sorted data produce a more favorable
        # data distribution for histogram? Or is the op parallelism
        # worth more?
        tokens_per_expert = megablocks.ops.histogram(top_expert, self.num_experts)

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
        input_ = input_.view(-1, input_.shape[-1])

        ## repeat each token top_k times and shuffle tokens to group them by their respective experts
        # print("input_shape", input_.shape)
        # exit()
        input_ = megablocks.ops.gather(input_, indices, bin_ids, bins, top_k)
        # get tokens routed to this rank's experts only
        input_parallel = copy_to_expert_model_parallel_region(input_, tokens_per_expert)
        # print("input_parallel.shape", input_parallel.shape)
    
        # get tokens_per_expert for this rank's experts only
        # with torch.no_grad():
        local_tokens_per_expert = get_expert_token_counts_for_rank(tokens_per_expert)
        # if torch.cuda.current_device() == 0:
        #     print(f"{torch.cuda.current_device()}: local_tokens_per_expert {local_tokens_per_expert}, global tokens {tokens_per_expert}")

        # Perform the expert computation for this rank's experts
        # print("x.shape, w1.shape, grouped_gemm_batch_sizes", x.shape, w1.shape, grouped_gemm_batch_sizes)
        output_parallel = self.mlp(input_parallel, local_tokens_per_expert)
    
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

    def forward(self, x, expert_weights, expert_indices):
        """
        grouped_forward_once

            x: [sl, bs, hs]
            expert_weights: [sl * bs, top-k]
            expert_indices: [sl * bs, top-k]
        """
        # save shape so we can re-shape the outputs later
        in_shape = x.size()

        # both are now (sl * bs * top_k)
        expert_weights = expert_weights.flatten()
        expert_indices = expert_indices.flatten()

        with torch.no_grad():
            indices, bin_ids, bins, tokens_per_expert = self.indices_and_bins(
                expert_indices
            )

        # print("tokens_per_expert", tokens_per_expert, "expert_idnices")
        # exit()
        x = self.permute_and_compute(
            x,
            tokens_per_expert,
            indices,
            bin_ids,
            expert_weights,
            bins,
            self.top_k,
        )

        # restore input shape
        x = x.view(in_shape)
        return x


def cast_if_autocast_enabled(tensor: torch.Tensor):
    if torch.is_autocast_enabled():
        if tensor.device.type == "cuda":
            dtype = torch.get_autocast_gpu_dtype()
        elif tensor.device.type == "cpu":
            dtype = torch.get_autocast_cpu_dtype()
        else:
            raise NotImplementedError()
        return tensor.to(dtype=dtype)
    return tensor





class ParallelDroplessMinion(torch.nn.Module):
    def __init__(
        self,
        neox_args: NeoXArgs,
        init_method,
        output_layer_init_method,
    ):
        super(ParallelDroplessMinion, self).__init__()

        if neox_args.moe_router_type == "topk":
            self.router = TopKTokenChoiceRouterMinion(
                neox_args,
                init_method,
            )
        else:
            raise ValueError(f"Invalid MoE Router type {neox_args.moe_router_type}")

        self.experts = ParallelDroplessMinions( 
            neox_args,
            init_method,
            output_layer_init_method,
        )

    def forward(self, x):
        # we expect inputs as (sl, bs, hs)
        # neox provides inputs as torch.Size([2048, 4, 768])
        # (sl, bs, hs)

        # NOTE: If we're going to cast the activations to lower precision
        # do it before we permute the tokens to save bandwidth
        x = cast_if_autocast_enabled(x)

        # Compute the expert scores and assignments
        expert_weights, expert_indices = self.router(x)

        # return value should be
        return self.experts(x, expert_weights, expert_indices), None

    