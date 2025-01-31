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

from .moe_mlp import ParallelGroupedLLaMAMLP, ParallelGroupedMLP
from .router import Router

class IndexAddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, buffer, index, source):
        ctx.save_for_backward(index, source)
        ctx.mark_dirty(buffer)
        buffer.index_add_(0, index, source)
        return buffer
    @staticmethod
    def backward(ctx, grad_output):
        index, source = ctx.saved_tensors
        grad_buffer = torch.zeros_like(grad_output)
        grad_buffer.index_add_(0, index, grad_output)
        return grad_buffer, None, None

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

        self.total_approx_count = 0
        # self.register_buffer('buffer', torch.zeros((neox_args.seq_length * neox_args.train_micro_batch_size_per_gpu * self.num_experts, neox_args.hidden_size)).to('cuda', dtype=torch.bfloat16))
        self.buffer = None

    def indices_and_bins(self, top_expert: torch.Tensor):
        # Sort the expert ids to produce the scatter/gather
        # indices for the permutation.
        #
        # TODO(tgale): Is it worth doing this conversion to 32-bit
        # prior? Could we place the `torch.max` operation to return
        # 32-bit expert indices?
        top_expert = top_expert.int()
        # print(f"expert_indices : ", expert_indices.shape)
        bin_ids, indices = megablocks.ops.sort(top_expert, self.sort_end_bit)
        # print(f"bin_ids : {bin_ids}, indices : ")
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
        queries=None,
        keys=None
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
        seq_len, batch_size = input_.shape[0], input_.shape[1]
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

        # Perform the expert computation for this rank's experts
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
        retval = None
        input_indices = indices // top_k
        if self.buffer is None:
            self.buffer = torch.zeros((seq_len * batch_size * self.num_experts, output.size(-1))).to(output.device, dtype=output.dtype)
            # # e.g. indices 0, 1, 2, 3 will all correspond to input 0 if top_k = 4
        self.buffer.zero_()
        # with torch.no_grad():
        self.buffer.index_add_(dim=0, index=self.num_experts * input_indices + bin_ids, source=output.detach())
        # self.buffer = IndexAddFunction.apply(self.buffer, self.num_experts * input_indices + bin_ids, output)
        # Un-route the data for the MoE output
        return megablocks.ops.scatter(
            output,
            indices,
            bin_ids,
            expert_weights,
            bins,
            top_k,
        ), self.buffer.view(seq_len, batch_size, self.num_experts, -1).transpose(0, 1).transpose(1, 2)

    def forward(self, x, expert_weights, expert_indices, queries=None, keys=None):
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

        x, expert_output = self.permute_and_compute(
            x,
            tokens_per_expert,
            indices,
            bin_ids,
            expert_weights,
            bins,
            self.top_k,
            queries=queries,
            keys=keys
        )

        # restore input shape
        x = x.view(in_shape)
        return x, expert_output


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

        self.router = Router(neox_args, init_method)
        # self.router = TopKTokenChoiceRouter(neox_args, init_method)

        self.experts = ParallelDroplessMLP(
            neox_args,
            init_method,
            output_layer_init_method,
        )

        if neox_args.moe_router_type == "dense_approx_efficient":
            from xformers.components.attention import ScaledDotProduct
            self.attention = ScaledDotProduct().cuda()

        self.routed_mask = None

    def forward(self, x, queries=None, keys=None):
        # router_type = self.router.router_type
        # we expect inputs as (sl, bs, hs)
        # neox provides inputs as torch.Size([2048, 4, 768])
        # (sl, bs, hs)

        # NOTE: If we're going to cast the activations to lower precision
        # do it before we permute the tokens to save bandwidth
        x = cast_if_autocast_enabled(x)

        # Compute the expert scores and assignments
        expert_weights, expert_indices, scores = self.router(x)

        # return value should be
        output, expert_output = self.experts(x, expert_weights, expert_indices, queries=queries, keys=keys)
        
        # if router_type == "dense_approx_efficient":
        if self.routed_mask is None:
            self.routed_mask = torch.zeros(expert_indices.size(0), self.experts.num_experts, dtype=torch.bool).to(expert_indices.device)
        with torch.no_grad():
            self.routed_mask.zero_().scatter_(1, expert_indices, 1)
            att_mask = self.routed_mask.view(queries.shape[1], queries.shape[0], -1).transpose(0, 1).transpose(1, 2)
        # sl*bs x nexperts -> bs x nexperts x 1 x sl
        # call xformers attention
        attn_result = self.attention(
            # bs x sl x nheads x head dim -> bs x nexperts x sl x head dim
            queries.mean(dim=2, keepdim=True).expand(-1, -1, self.experts.num_experts, -1).transpose(1, 2),
            # bs x sl x nheads x head dim -> bs x nexperts x sl x head dim
            keys.mean(dim=2, keepdim=True).expand(-1, -1, self.experts.num_experts, -1).transpose(1, 2), 
            # sl*bs x nexperts x hidden dim -> bs x nexperts x sl x hidden dim
            expert_output,
            # select columns of routed tokens
            att_mask=att_mask.unsqueeze(2)
        )
        # -> bs x nexperts x sl x hidden dim
        # remove value rows of routed tokens
        # rarely a whole sequence can be masked out (all tokens routed to that expert) resulting in nans
        # in either case mask out approximations
        attn_result = torch.where(torch.logical_or(att_mask.unsqueeze(3), torch.isnan(attn_result)), 0, attn_result)

        approx_output = (scores.view(*x.shape[:2], -1).unsqueeze(-1) * attn_result.transpose(1, 2).transpose(0, 1)).sum(dim=2)

        return output + approx_output, None
            # return output + (scores.view(*x.shape[:2], -1).unsqueeze(-1) * torch.where(torch.logical_or(self.routed_mask.unsqueeze(3), torch.isnan(attn_result)), 0, attn_result).transpose(1, 2).transpose(0, 1)).sum(dim=2), None
        # return output, None