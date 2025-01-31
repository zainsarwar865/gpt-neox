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

from xformers.components.attention import ScaledDotProduct

import triton
import triton.language as tl

import os
# os.environ["TRITON_INTERPRET"] = '1'

os.environ["TORCH_LOGS"] = "+dynamo"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"

@triton.jit
def lsh_approximation_kernel(
    input, output, bin_ids, bucket_indices,
    bucket_starts, bucket_ends, reshaped_output,
    input_stride, output_stride, bin_ids_stride, bucket_indices_stride,
    num_experts: tl.constexpr, input_dim: tl.constexpr, output_dim: tl.constexpr, bucket_offset: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    bucket = bucket_offset + tl.program_id(0)
    expert = tl.program_id(1)
    
    bucket_start = tl.load(bucket_starts + bucket)
    bucket_end = tl.load(bucket_ends + bucket)
    
    bucket_size = bucket_end - bucket_start

    assert bucket_size <= BLOCK_SIZE
    
    # Load indices for this block
    bucket_locs = tl.arange(0, BLOCK_SIZE)
    bucket_mask = bucket_locs < bucket_size
    this_bucket_indices = tl.load(bucket_indices + bucket_start + bucket_locs, mask=bucket_mask)
    
    # Load input for this block
    input_dims = tl.arange(0, triton.next_power_of_2(input_dim))
    input_locs = this_bucket_indices[:, None] * input_stride + input_dims[None, :]
    input_mask = bucket_mask[:, None] & (input_dims < input_dim)[None, :]
    input_block = tl.load(input + input_locs, mask=input_mask)
    
    # Compute scores
    scale = tl.sqrt(float(input_dim))
    scores = tl.dot(input_block / scale, tl.trans(input_block / scale))
    
    # Apply mask
    mask = tl.arange(0, BLOCK_SIZE)[:, None] == tl.arange(0, BLOCK_SIZE)[None, :]
    threshold = 1 - 2 / num_experts
    scores = tl.where((mask | (scores < threshold)), 0.0, scores)
    
    # Load expert indices for this block
    expert_indices = tl.load(bin_ids + this_bucket_indices * bin_ids_stride, mask=bucket_mask)
    # Compute approximations for each expert
    expert_mask = expert_indices == expert
    not_expert_mask = expert_indices != expert
    expert_scores = tl.where(expert_mask[:, None] | not_expert_mask[None, :], 0.0, scores)
    
    # Load output for this block
    output_dims = tl.arange(0, triton.next_power_of_2(output_dim))
    output_locs = this_bucket_indices[:, None] * output_stride + output_dims[None, :]
    output_mask = bucket_mask[:, None] & (output_dims < output_dim)[None, :]
    output_block = tl.load(output + output_locs, mask=output_mask)
    
    # Compute approximation
    approx = tl.dot(expert_scores, output_block.to(tl.float32))
    score_counts = tl.sum(expert_scores > 0, axis=1)
    approx = tl.where(score_counts[:, None] > 0, approx / score_counts[:, None], approx)
    
    # Store approximation
    store_locs = this_bucket_indices[:, None] * num_experts * output_dim + expert * output_dim + output_dims[None, :]
    store_mask = bucket_mask[:, None] & (output_dims < output_dim)[None, :]
    tl.atomic_add(reshaped_output + store_locs, approx)
    # tl.store(reshaped_output + store_locs, approx, mask=store_mask)

# Launch the kernel
def launch_lsh_approximation_kernel(input_, output, bin_ids, bucket_indices, bucket_starts, bucket_ends, reshaped_output, num_experts, num_buckets):
    # BLOCK_SIZE = 32  # Adjust based on your hardware and input size
    BLOCK_SIZE = triton.next_power_of_2(int((bucket_ends - bucket_starts).max().item()))
    print(f"{BLOCK_SIZE=}")

    BUCKETS_PER_BATCH = 1

    assert num_buckets % BUCKETS_PER_BATCH == 0

    for bucket_offset in range(0, num_buckets, BUCKETS_PER_BATCH):
    
      grid = (BUCKETS_PER_BATCH, num_experts)
      
      lsh_approximation_kernel[grid](
          input_, output, bin_ids, bucket_indices, bucket_starts, bucket_ends, reshaped_output,
          input_.stride(0), output.stride(0), bin_ids.stride(0), bucket_indices.stride(0),
          num_experts, input_.shape[-1], output.shape[-1], bucket_offset,
          BLOCK_SIZE=BLOCK_SIZE
      )
      print('did batch')

print('start compiling')
kernel = torch.compile(launch_lsh_approximation_kernel)
print('done compiling')

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


class ParallelDroplessMLP(torch.nn.Module):
    """
    This class defines MoE expert computation, using tensor (model) parallel size as the expert parallel size

    The implication of this parallelism def is that the expert weights can only be sharded within a single node
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
        self.buffer = None
        self.routed_mask = None
        self.scores = None
        self.approx = None
        self.attention = ScaledDotProduct().cuda()

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

    def forward(self, x, expert_weights, expert_indices, scores=None, queries=None, keys=None, router_type=None):
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
        # expert_indices = expert_indices.flatten()

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
            self.top_k if router_type != "dense" else self.num_experts,
            expert_indices=expert_indices,
            router_type=router_type,
            scores=scores,
            queries=queries,
            keys=keys
        )

        # restore input shape
        x = x.view(in_shape)
        return x

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
        queries=None,
        keys=None
    ):
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
        # e.g. indices 0, 1, 2, 3 will all correspond to input 0 if top_k = 4
        input_indices = indices // top_k
        approx_output = None
        if router_type == "dense_approx_efficient":
            if self.buffer is None:
                self.buffer = torch.zeros((seq_len * batch_size * self.num_experts, output.size(-1)), dtype=output.dtype, device=output.device)
            self.buffer.zero_()
          # ith element of output will be added to index corresponding to input index, and associated expert
            # with torch.no_grad():
            self.buffer.index_add_(dim=0, index=self.num_experts * input_indices + bin_ids, source=output.detach())
            if self.routed_mask is None:
                self.routed_mask = torch.zeros(expert_indices.size(0), self.num_experts, dtype=torch.bool, device=expert_indices.device, requires_grad=False)
            
            self.routed_mask.zero_().scatter_(1, expert_indices, 1)
            
            att_mask = self.routed_mask.view(queries.shape[1], queries.shape[0], -1).transpose(0, 1).transpose(1, 2)

            attn_result = self.attention(
                # bs x sl x nheads x head dim -> bs x nexperts x sl x head dim
                queries.mean(dim=2, keepdim=True).expand(-1, -1, self.num_experts, -1).transpose(1, 2),
                # bs x sl x nheads x head dim -> bs x nexperts x sl x head dim
                keys.mean(dim=2, keepdim=True).expand(-1, -1, self.num_experts, -1).transpose(1, 2), 
                # sl*bs x nexperts x hidden dim -> bs x nexperts x sl x hidden dim
                self.buffer.view(queries.shape[1], queries.shape[0], self.num_experts, -1).transpose(0, 1).transpose(1, 2),
                # select columns of routed tokens
                att_mask=att_mask.unsqueeze(2)
            )
            # -> bs x nexperts x sl x hidden dim
            # remove value rows of routed tokens
            # rarely a whole sequence can be masked out (all tokens routed to that expert) resulting in nans
            # in either case mask out approximations
            attn_result = torch.where(torch.logical_or(att_mask.unsqueeze(3), torch.isnan(attn_result)), 0, attn_result)
            return megablocks.ops.scatter(
                output,
                indices,
                bin_ids,
                expert_weights,
                bins,
                top_k,
            ) + (scores.view(*in_shape[:2], -1).unsqueeze(-1) * attn_result.transpose(1, 2).transpose(0, 1)).sum(dim=2).view(-1, in_shape[-1])
        elif router_type == "dense_approx_lsh":
            
            if self.buffer is None:
                self.buffer = torch.zeros((seq_len * batch_size, self.num_experts, output.size(-1))).to(output.device, dtype=output.dtype)
                self.buffer.requires_grad = False
            self.buffer.zero_()

            if self.scores is None:
                self.scores = torch.zeros(self.num_experts, seq_len * batch_size // 64, seq_len * batch_size // 64).to(input_.device, dtype=input_.dtype)
                self.scores.requires_grad = False
            self.scores.zero_()
            if self.approx is None:
                self.approx = torch.zeros(self.num_experts, seq_len * batch_size // 64, output.size(-1)).to(output.device, dtype=output.dtype)
                self.approx.requires_grad = False
            self.approx.zero_()
            nbits = 10
            # the example here samples uniform [-0.5, 0.5] https://www.pinecone.io/learn/series/faiss/locality-sensitive-hashing-random-projection/
            lsh_vectors = torch.rand((input_.shape[-1], nbits), dtype=input_.dtype, device=input_.device) - 0.5
            hash_vectors = torch.matmul(input_.detach(), lsh_vectors) > 0

            # since nbits is small, let's use tensor operations to turn each unique binary vector into an integer representation
            exponents = 2 ** torch.arange(nbits - 1, -1, -1).to(hash_vectors.device)
            hashes = torch.sum(exponents * hash_vectors, -1)

            # get buckets for all vectors
            bucket_ids, bucket_indices = megablocks.ops.sort(hashes, nbits)
            bucket_counts = megablocks.ops.histogram(hashes, 2**nbits)
            bucket_ends = megablocks.ops.inclusive_cumsum(bucket_counts, 0)

            scale = np.sqrt(input_.shape[-1])
            bucket_starts = bucket_ends.roll(1) % bucket_ends[-1]
            # print((bucket_counts**2).sum(), flush=True)
            expert_range = torch.arange(self.num_experts, device=self.scores.device).view(-1, 1, 1)
            threshold = 1 - 2 / self.num_experts
            # kernel(input_, output, bin_ids, bucket_indices, bucket_starts, bucket_ends, self.buffer, self.num_experts, 2**nbits)
            for i in range(2**nbits):
                bucket_end = bucket_ends[i]
                bucket_size = bucket_counts[i]
                bucket_start = bucket_end - bucket_size
                this_bucket_indices = bucket_indices[bucket_start : bucket_end]

                if bucket_size > self.scores.shape[1]:
                    this_bucket_indices = this_bucket_indices[torch.randperm(bucket_size)[:self.scores.shape[1]]]
                    bucket_size = self.scores.shape[1]

                this_bucket_input = input_[this_bucket_indices].detach()
                this_bucket_expert_indices = bin_ids[this_bucket_indices]
                this_bucket_output = output[this_bucket_indices].detach()

                torch.matmul(this_bucket_input / scale, this_bucket_input.T / scale, out=self.scores[0, :bucket_size, :bucket_size])
                # zero out similarities between same vector, or any below threshold
                
                self.scores[0, :bucket_size, :bucket_size].masked_fill_(torch.eye(bucket_size, dtype=torch.bool, device=self.scores.device).logical_or(self.scores[0, :bucket_size, :bucket_size] < threshold), 0)

                self.scores[1:].copy_(self.scores[0])
                # broadcasting to nxn matrices, of all 0, all 1, ... all num_experts-1. each entry represents the expert
                
                # for element i of this tensor, we want approximations for expert i, for tokens not routed to expert i, using similarities between tokens to expert i
                # so if an entry's row matches the corresponding input's expert index (we don't want its value)
                # or if an entry's column doesn't match the corresponding input's expert index (we don't want to use its score)
                # set it to 0
                # two masked fills with num_experts x n x 1 and num_experts x 1 x n avoids materializing an n^2 mask
                self.scores[:, :bucket_size, :bucket_size].masked_fill_(expert_range.eq(this_bucket_expert_indices.unsqueeze(1)), 0)
                self.scores[:, :bucket_size, :bucket_size].masked_fill_(expert_range.ne(this_bucket_expert_indices.unsqueeze(0)), 0)

                # matmul (which does a weighted sum), then divide by counts (to make it a weighted average)
                score_counts = (self.scores > 0).sum(dim=-1)

                torch.matmul(self.scores[:, :bucket_size, :bucket_size], this_bucket_output, out=self.approx[:, :bucket_size])
                self.approx[score_counts > 0] /= score_counts[score_counts > 0].unsqueeze(1)
                self.total_approx_count += (score_counts > 0).sum()

                self.buffer.index_add_(0, this_bucket_indices, self.approx[:, :bucket_size].transpose(0, 1))
            return megablocks.ops.scatter(
                output,
                indices,
                bin_ids,
                expert_weights,
                bins,
                top_k,
            ) + (scores.unsqueeze(-1) * self.buffer).sum(dim=1)
        # Un-route the data for the MoE output
        # TODO: Somehow combine this with approx_output
        else:
            return megablocks.ops.scatter(
                output,
                indices,
                bin_ids,
                expert_weights,
                bins,
                top_k,
            )

class ParallelDroplessMoE(torch.nn.Module):
    def __init__(
        self,
        neox_args: NeoXArgs,
        init_method,
        output_layer_init_method,
        get_mlp
    ):
        super(ParallelDroplessMoE, self).__init__()

        self.router = Router(neox_args, init_method)
        self.neox_args = neox_args

        # self.experts = ParallelDroplessMLP(
        #     neox_args,
        #     init_method,
        #     output_layer_init_method,
        # )
        import dataclasses
        neox_args_2 = dataclasses.replace(neox_args)

        neox_args_2.hidden_size = neox_args.moe_shared_expert_hidden_size
        neox_args_2.intermediate_size = neox_args.moe_shared_expert_intermediate_size
        
        self.experts = ParallelDroplessMLP(
            neox_args_2,
            init_method,
            output_layer_init_method,
        )

        if neox_args.moe_use_shared_expert:
            self.shared_expert = get_mlp(neox_args.mlp_type, neox_args)

    def forward(self, x, queries=None, keys=None):
        router_type = self.router.router_type
        # we expect inputs as (sl, bs, hs)
        # neox provides inputs as torch.Size([2048, 4, 768])
        # (sl, bs, hs)

        # NOTE: If we're going to cast the activations to lower precision
        # do it before we permute the tokens to save bandwidth
        x = cast_if_autocast_enabled(x)

        # Compute the expert scores and assignments
        expert_weights, expert_indices, scores = self.router(x)

        exp_out = self.experts(x, expert_weights, expert_indices, 
                            scores=scores,
                            queries=queries,
                            keys=keys,
                            router_type=router_type)
        if self.neox_args.moe_use_shared_expert:
            exp_shared,_ = self.shared_expert(x)
            return exp_out + exp_shared, None                    
        return exp_out, None