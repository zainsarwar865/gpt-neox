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
from .fused_lores_gemm import fused_lores_batched_gemm
import torch
from megatron.model.activations import get_activation, swish
from megatron.mpu.layers import _initialize_affine_weight_gpu
from megatron.mpu.initialize import get_model_parallel_world_size
from megatron.mpu.utils import divide
from megatron.model.router import TopKTokenChoiceRouterLoRa, TopKTokenChoiceRouter
from megatron.neox_arguments.arguments import NeoXArgs
from megatron.mpu import copy_to_expert_model_parallel_region
from megatron.mpu import get_expert_token_counts_for_rank
from megatron.mpu import gather_from_expert_model_parallel_region
from megatron.model.init_functions import init_method_zeros
import numpy as np
from megatron import mpu
import torch.nn as nn
import megablocks.ops
from megablocks import grouped_gemm_util as gg
import os
from megatron.model.logger import tokens_per_lora_log, forward_step_counter
import contextlib

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
            # new:
            # Y = fused_lores_batched_gemm(x, w1_A, w1_B)
            # return Y
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
        self.LoRaRouter = TopKTokenChoiceRouter(neox_args, init_method)


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

        self.w1 = mpu.ColumnParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=self.num_rows_per_rank,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
        )

        # Project back to h.
        self.w2 = mpu.RowParallelLinear(
            neox_args=neox_args,
            input_size=self.num_rows_per_rank,
            output_size=neox_args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            parallel_output=False,
            skip_bias_add=True,
        )
        self.gradient_scale = None
        if world_size > 1:
            self.gradient_scale = 1 / world_size

    

    def indices_and_bins(self, top_expert: torch.Tensor):
        top_expert = top_expert.int()
        bin_ids, indices = megablocks.ops.sort(top_expert)
        tokens_per_expert = megablocks.ops.histogram(top_expert, self.total_loras)
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

        input_x = megablocks.ops.gather(input_, indices, bin_ids, bins, top_k)
        #input_parallel = copy_to_expert_model_parallel_region(input_x, tokens_per_expert)


        # local_tokens_per_expert = get_expert_token_counts_for_rank(tokens_per_expert)

        output_parallel = self.loras(input_x, tokens_per_expert, layer)

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


    # def forward(self, x: torch.Tensor):        
    #     # lora_weights, lora_indices = self.LoRaRouter(x)
    #     # lora_weights = lora_weights.flatten()
    #     # lora_indices = lora_indices 
    #     # lora_indices = lora_indices.flatten()
        
    #     # with torch.no_grad():
    #     #     indices, lora_ids, lora_bins, tokens_per_lora = self.indices_and_bins(
    #     #         lora_indices
    #     #     )
    #     # x_1_loras = self.permute_and_compute(
    #     #     x,
    #     #     tokens_per_lora,
    #     #     indices,
    #     #     lora_ids,
    #     #     lora_weights,
    #     #     lora_bins,
    #     #     self.LoRaRouter.top_k,
    #     #     1,
    #     # )
        
    #     x = self.w1(x)[0]
    #     if self.args.lora_interaction_type == 'addition':          
    #         #scaled_x = x + x_1_loras
    #         x = self.activation_func(x)
    #     # elif self.args.lora_interaction_type == 'geglu':
    #     #     x = x_1_loras * self.activation_func(x)
    #     # else:
    #     #     raise("LoRe interaction not defined")

        
    #     x = self.w2(x)[0]
    
    #     return x


    def forward(self, x: torch.Tensor):
        # 1) Router (small) on default stream to get LoRA assignments
        lora_weights, lora_indices = self.LoRaRouter(x)
        lora_weights = lora_weights.flatten()
        lora_indices = lora_indices.flatten()

        # 2) Prepare streams & events
        if x.is_cuda and torch.cuda.device_count() > 0:
            dense_stream = torch.cuda.Stream(device=x.device)
            lora_stream  = torch.cuda.Stream(device=x.device)
            evt_dense = torch.cuda.Event()
            evt_lora  = torch.cuda.Event()
            cur = torch.cuda.current_stream(x.device)
            # x is read by both streams; keep it alive until their work completes
            x.record_stream(lora_stream)
            x.record_stream(dense_stream)

            # Autocast handling (match whatever the caller enabled)
            # ---- LoRA branch on lora_stream: indices/bins + permute_and_compute ----
            with torch.cuda.stream(lora_stream):
                with torch.no_grad():
                    indices, lora_ids, lora_bins, tokens_per_lora = self.indices_and_bins(lora_indices)
                # with amp_ctx():
                x_lora = self.permute_and_compute(
                    x,
                    tokens_per_lora,
                    indices,
                    lora_ids,
                    lora_weights,
                    lora_bins,
                    self.LoRaRouter.top_k,
                    1,  # layer id
                )
                x_lora.record_stream(lora_stream)
                evt_lora.record(lora_stream)

            # ---- Dense up-projection on dense_stream ----
            with torch.cuda.stream(dense_stream):
                # with amp_ctx():
                x_dense, _bias1 = self.w1(x)   # ColumnParallelLinear, skip_bias_add=True
                x_dense.record_stream(dense_stream)
                evt_dense.record(dense_stream)

            # ---- Join on default stream ----
            cur.wait_event(evt_lora)
            cur.wait_event(evt_dense)

            # 3) Combine + activation on default stream (now both ready)
            if self.args.lora_interaction_type == 'addition':
                x_act = self.activation_func(x_dense + x_lora)
            elif self.args.lora_interaction_type == 'geglu':
                x_act = x_lora * self.activation_func(x_dense)
            else:
                raise RuntimeError("LoRe interaction not defined")

            # 4) Down-projection on default stream
            x_out, _bias2 = self.w2(x_act)  # RowParallelLinear, skip_bias_add=True
            return x_out
