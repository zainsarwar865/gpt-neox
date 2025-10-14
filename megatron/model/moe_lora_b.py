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
from .moe_mlp_lora_b import  ParallelGroupedMLP
from .router import TopKTokenChoiceRouter, SinkhornRouter, SparseMixerRouter

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True


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
        if neox_args.mlp_type == "regular":
            self.mlp = ParallelGroupedMLP(
                neox_args=neox_args,
                init_method=init_method,
                output_layer_init_method=output_layer_init_method,
            )

    def permute_and_compute(
        self,
        input_: torch.Tensor,
        
    ):
        input_ = input_.view(-1, input_.shape[-1])
        output_parallel = self.mlp(input_)
        return output_parallel

    def forward(self, x):
        # save shape so we can re-shape the outputs later
        in_shape = x.size()

        x = self.permute_and_compute(
            x,
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
    ):
        super(ParallelDroplessMoE, self).__init__()
        self.experts = ParallelDroplessMLP(
            neox_args,
            init_method,
            output_layer_init_method,
        )
        # self.experts = torch.compile(self.experts)

        self.args = neox_args


    def forward(self, x):
        # we expect inputs as (sl, bs, hs)
        # neox provides inputs as torch.Size([2048, 4, 768])
        # (sl, bs, hs)
        # if self.args.iteration > 1500:
        # NOTE: If we're going to cast the activations to lower precision
        # do it before we permute the tokens to save bandwidth
        
        x = cast_if_autocast_enabled(x)

        # Compute the expert scores and assignments
        # expert_weights, expert_indices = self.router(x)

        # return value should be
        #return self.experts(x, expert_weights, expert_indices), None
        return self.experts(x), None
