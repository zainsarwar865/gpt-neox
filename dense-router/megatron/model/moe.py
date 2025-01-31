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
        buffer
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

        self.buffer = buffer
        self.neox_args = neox_args
        # 1 x nexperts x 1
        self.expert_range = torch.arange(self.num_experts, device='cuda').unsqueeze(0).unsqueeze(2)

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
        router_type: str,
        expert_indices=None,
        scores=None,
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
        in_shape = input_.size()
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
        if "approx" in router_type:
            with torch.no_grad():
                self.buffer.zero_()
                input_indices = indices // top_k
                self.buffer = self.buffer.view(
                                    seq_len * batch_size * self.num_experts, 
                                    -1
                                )
                # ith element of output will be added to index corresponding to input index, and associated expert
                # with torch.no_grad():
                self.buffer.index_add_(dim=0, index=self.num_experts * input_indices + bin_ids, source=output.detach())
                self.buffer = self.buffer.view(
                                    seq_len * batch_size, 
                                    self.num_experts, 
                                    -1
                                )
                # ntokens x topk -> ntokens x nexperts x topk
                expert_indices_expanded = expert_indices.unsqueeze(1).expand(-1, self.num_experts, -1)
                # ntokens x nexperts -> ntokens x nexperts x 1
                # scores_expanded = scores.unsqueeze(2)

                
                if "threshold" in router_type:
                    # ntokens x nexperts x topk == 1 x nexperts x 1 broadcasted, any/all -> ntokens x nexperts
                    routed_to_target = (expert_indices_expanded == self.expert_range).any(dim=-1)
                    notrouted_to_cur = (expert_indices_expanded != self.expert_range).all(dim=-1)

                    # ntokens x nexperts x nexperts
                    # for each token, i,j true if routed to j, not routed to i, and above threshold score for i
                    # top_mask[:, cur_expert, target_expert] equivalent to for loop
                    # threshold = np.sqrt(1/((self.top_k+1)*self.neox_args.moe_num_experts))
                    threshold = 0
                    top_mask = ((notrouted_to_cur.unsqueeze(2) & routed_to_target.unsqueeze(1)) + scores.unsqueeze(2)) > 1 + threshold

                    # not routed to i, routed to j, select output it had for j for all i, sum, average
                    # nexperts x nexperts x hidden_dim divided by nexperts x nexperts x 1
                    # note by nature of the mask, diagonal entries are 0
                    # -> approx for expert j, for tokens not routed to j but routed to i
                    approx = (top_mask.unsqueeze(-1) * self.buffer.unsqueeze(1)).sum(dim=0) / top_mask.sum(dim=0).unsqueeze(-1).clamp(min=1)

                    routed_to_cur = ~notrouted_to_cur
                    notrouted_to_target = ~routed_to_target

                    # ntokens x nexperts x nexperts
                    # for each token, i,j true if a token was routed to i and we want approx for j
                    approx_mask = (routed_to_cur.unsqueeze(2) & notrouted_to_target.unsqueeze(1))
                elif "pairwise" in router_type:
                    routed_to = (expert_indices_expanded == self.expert_range).any(dim=-1)
                    top_mask = (routed_to.unsqueeze(2) & routed_to.unsqueeze(1))
                    # if "accurate" in router_type:
                    #     top_mask = top_mask * scores.unsqueeze(1)
                    # elif "viable" in router_type:
                    #     top_mask = top_mask * scores.unsqueeze(2)
                    approx = torch.einsum('nij,njd->ijd', top_mask.to(self.buffer.dtype), self.buffer) / top_mask.sum(dim=0).unsqueeze(-1).clamp(min=1)
                    # print("approx shape", approx.shape)
                    # approx = (top_mask.unsqueeze(-1) * self.buffer.unsqueeze(1)).sum(dim=0) / top_mask.sum(dim=0).unsqueeze(-1).clamp(min=1)
                    # approx = torch.einsum('nij,njd->ijd', top_mask.to(dtype=self.buffer.dtype), self.buffer) / top_mask.sum(dim=0).unsqueeze(-1).clamp(min=1)
                    approx_mask = (routed_to.unsqueeze(2) & ~routed_to.unsqueeze(1))
            approx_output = torch.matmul((scores.unsqueeze(1) * approx_mask).flatten(1, 2), approx.flatten(0, 1) / self.top_k)
            return megablocks.ops.scatter(
                output,
                indices,
                bin_ids,
                expert_weights,
                bins,
                top_k,
            ) + approx_output - approx_output.detach()

        # Un-route the data for the MoE output
        return megablocks.ops.scatter(
            output,
            indices,
            bin_ids,
            expert_weights,
            bins,
            top_k,
        )

    def forward(self, x, expert_weights, expert_indices, scores, router_type):
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

        with torch.no_grad():
            indices, bin_ids, bins, tokens_per_expert = self.indices_and_bins(
                expert_indices.flatten()
            )

        x = self.permute_and_compute(
            x,
            tokens_per_expert,
            indices,
            bin_ids,
            expert_weights,
            bins,
            self.top_k,
            expert_indices=expert_indices,
            router_type=router_type,
            scores=scores
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


class ParallelDroplessMoE(torch.nn.Module):
    def __init__(
        self,
        neox_args: NeoXArgs,
        init_method,
        output_layer_init_method,
        buffer
    ):
        super(ParallelDroplessMoE, self).__init__()

        self.router = Router(neox_args, init_method)

        self.experts = ParallelDroplessMLP(
            neox_args,
            init_method,
            output_layer_init_method,
            buffer=buffer
        )

    def forward(self, x):
        # we expect inputs as (sl, bs, hs)
        # neox provides inputs as torch.Size([2048, 4, 768])
        # (sl, bs, hs)

        # NOTE: If we're going to cast the activations to lower precision
        # do it before we permute the tokens to save bandwidth
        x = cast_if_autocast_enabled(x)

        # Compute the expert scores and assignments
        expert_weights, expert_indices, scores = self.router(x)

        # return value should be
        return self.experts(x, expert_weights, expert_indices, 
                            scores=scores,
                            router_type=self.router.router_type), None
