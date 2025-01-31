# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2023 MegaBlocks authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from typing import Optional

import megablocks.ops
import numpy as np
import torch

from megatron import mpu
from megatron.mpu import get_expert_token_counts_for_rank
from megatron.mpu import get_expert_tokens_for_rank
from megatron.mpu import copy_to_expert_model_parallel_region
from megatron.mpu import gather_from_expert_model_parallel_region
from megatron.neox_arguments.arguments import NeoXArgs
from megatron.mpu.layers import _initialize_affine_weight_gpu

from .moe_mlp import ParallelGroupedLLaMAMLP, ParallelGroupedMLP, ParallelSharedMLP
from .router import TopKTokenChoiceRouter, SinkhornRouter
from megatron.model.activations import get_activation, sigmoid

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


class ParallelDroplessMLP(torch.nn.Module):
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
        super(ParallelDroplessMLP, self).__init__()

        # Calculate the number of experts to allocate on this rank
        world_size = mpu.get_model_parallel_world_size()
        assert neox_args.moe_num_experts % world_size == 0
        self.num_experts = neox_args.moe_num_experts
        self.experts_per_rank = self.num_experts // world_size
        self.top_k = neox_args.moe_top_k
        self.hidden_size = neox_args.hidden_size
        # Calculate the number of bits needed to represent the expert indices
        # so that we can pass it to radix sort
        self.sort_end_bit = max(int(np.ceil(np.log2(self.num_experts))), 1)

        # decide which parallel grouped MLP implementation to use
        if neox_args.mlp_type == "regular":
            self.mlp = ParallelGroupedMLP(
                neox_args=neox_args,
                init_method=init_method,
                output_layer_init_method=output_layer_init_method,
            )
        elif neox_args.mlp_type == "llama":
            self.mlp = ParallelGroupedLLaMAMLP(
                neox_args=neox_args,
                init_method=init_method,
                output_layer_init_method=output_layer_init_method,
            )
        else:
            raise KeyError(neox_args.mlp_type)

        self.shared_mlp = ParallelSharedMLP(
                neox_args=neox_args,
                init_method=init_method,
                output_layer_init_method=output_layer_init_method,        
            )



        # self.lambda_e_w1 = torch.nn.Parameter(
        #     torch.empty(
        #         self.hidden_size,
        #         self.hidden_size,
        #         device=torch.cuda.current_device(),
        #         dtype=neox_args.params_dtype,
        #     )
        # )

        # _initialize_affine_weight_gpu(
        #     self.lambda_e_w1, init_method, partition_dim=0, stride=1
        # )

        self.gradient_scale = None
        if world_size > 1:
            self.gradient_scale = 1 / world_size


        #self.sigmoid = sigmoid

    def indices_and_bins(self, top_expert: torch.Tensor):

        top_expert = top_expert.int()
        bin_ids, indices = megablocks.ops.sort(top_expert, self.sort_end_bit)

        tokens_per_expert = megablocks.ops.histogram(top_expert, self.num_experts)

        # Calculate the bin bounds for the sorted tokens.
        bins = megablocks.ops.inclusive_cumsum(tokens_per_expert, 0)
        bins = bins.view(1) if not len(bins.size()) else bins
        return indices, bin_ids, bins, tokens_per_expert

    
    # def lambda_balance(self, x: torch.Tensor, lambda_e):
    #     tokens_per_expert = torch.tensor(x.shape[0]).unsqueeze(0)
    #     grouped_gemm_batch_sizes = tokens_per_expert.cpu().to(torch.long)
    #     lambda_e = self.scale_grad(lambda_e)
    #     lambda_e = lambda_e.view(1, self.hidden_size, self.hidden_size)
    #     return gg.ops.gmm(x, lambda_e, grouped_gemm_batch_sizes)

    # def lambda_scale(self, lambda_e: torch.Tensor, y_shared_exp: torch.Tensor, y_exp: torch.Tensor):
    #     lambda_e = self.sigmoid(lambda_e)
    #     retval = torch.einsum("ij,ij->ij", lambda_e, y_shared_exp) + torch.einsum("ij,ij->ij", torch.ones_like(lambda_e) - lambda_e, y_exp)
    #     return retval


    def scale_grad(self, w: torch.Tensor):
        """
        Copied from SparseMLP
        """
        if self.gradient_scale is None:
            return w
        return scale_gradient(w, self.gradient_scale)


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
        input_ = megablocks.ops.gather(input_, indices, bin_ids, bins, top_k)

        # get tokens routed to this rank's experts only
        input_parallel = copy_to_expert_model_parallel_region(input_, tokens_per_expert)

        # get tokens_per_expert for this rank's experts only
        # with torch.no_grad():
        local_tokens_per_expert = get_expert_token_counts_for_rank(tokens_per_expert)
        # if torch.cuda.current_device() == 0:
        #     print(f"{torch.cuda.current_device()}: local_tokens_per_expert {local_tokens_per_expert}, global tokens {tokens_per_expert}")

        #lambda_e = self.lambda_balance(input_parallel, self.lambda_e_w1)

        # Perform the expert computation for this rank's experts
        #shared_output = self.shared_mlp(input_parallel)

        output_parallel = self.mlp(input_parallel, local_tokens_per_expert)
        #print(f"Norm shared_output : {shared_output.norm()}  - Norm exps : {output_parallel.norm()}")
        
        #output_parallel = self.lambda_scale(lambda_e, shared_output, output_parallel)
        #output_parallel+=shared_output
        #print(f"Norm final : {output_parallel.norm()} ")
    
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

        # Shared expert 
        shared_x = self.shared_mlp(x)
        x = self.permute_and_compute(
            x,
            tokens_per_expert,
            indices,
            bin_ids,
            expert_weights,
            bins,
            self.top_k,
        )
    
        x = x.view(in_shape)
        
        x = x + shared_x
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


class ParallelDroplessMoE(torch.nn.Module):
    def __init__(
        self,
        neox_args: NeoXArgs,
        init_method,
        output_layer_init_method,
    ):
        super(ParallelDroplessMoE, self).__init__()

        if neox_args.moe_router_type == "sinkhorn":
            self.router = SinkhornRouter(
                neox_args,
                init_method,
            )
        elif neox_args.moe_router_type == "topk":
            self.router = TopKTokenChoiceRouter(
                neox_args,
                init_method,
            )
        else:
            raise ValueError(f"Invalid MoE Router type {neox_args.moe_router_type}")

        self.experts = ParallelDroplessMLP(
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
