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
import torch.nn.functional as F
from megatron.model.activations import get_activation, swish

from megatron.mpu.layers import _initialize_affine_weight_gpu
from megatron.mpu.initialize import get_model_parallel_world_size
from megatron.mpu.utils import divide

from megatron.neox_arguments.arguments import NeoXArgs
from megatron.model.init_functions import init_method_hyper, init_method_zeros
import math
from megablocks import grouped_gemm_util as gg
from contextlib import nullcontext

class HNClipScaleGradient(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad):
        g = grad * ctx.scale
        n = g.norm()
        return g * min(0.1, 0.1/n), None

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
hn_clip_scale_gradient = HNClipScaleGradient.apply

class MemoryOptimizedParallelGroupedMLP(torch.autograd.Function):
    """GroupedMLP with manually scheduled memory reuse."""

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x, w1, w2, batch_sizes, activation_fn):
        # x: [m, k], w1: [n, k], w2: [n, k]
        if not x.is_contiguous() or not w1.is_contiguous() or not w2.is_contiguous():
            raise ValueError("Expected contiguous 'x', 'w1' and 'w2'.")

        # Layer 0: x @ w1.t().
        sdd_out = gg.backend.gmm(x, w1, batch_sizes, trans_b=True)

        # activation_fn
        activation_fn_out = activation_fn(sdd_out)

        # Layer 1: x @ w2.
        dsd_out = gg.backend.gmm(activation_fn_out, w2, batch_sizes)

        # NOTE: Save the input to the layer and the activation_fn input for
        # gradient computation. We'll re-compute the activation_fn forward
        # pass in the backward pass to avoid materializing another
        # intermediate.
        ctx.x_shape = x.shape
        ctx.sdd_out_shape = sdd_out.shape
        ctx.dtype = x.dtype
        ctx.activation_fn = activation_fn
        ctx.save_for_backward(w1, w2, batch_sizes, x, sdd_out)
        return dsd_out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, ddsd_out):
        if (
            not ctx.needs_input_grad[0]
            or not ctx.needs_input_grad[1]
            or not ctx.needs_input_grad[2]
        ):
            raise ValueError("Expected all MLP inputs to need grad.")

        # Unpack saved tensors
        dtype = ctx.dtype
        saved_tensors = ctx.saved_tensors
        w1, w2 = saved_tensors[:2]
        batch_sizes = saved_tensors[2]
        x = saved_tensors[3]
        sdd_out = saved_tensors[4]

        # Rematerialize activation_fn output.
        activation_fn = ctx.activation_fn
        with torch.set_grad_enabled(True):
            sdd_out.requires_grad = True
            activation_fn_out = activation_fn(sdd_out)
            activation_grad_fn = activation_fn_out.backward

        # Compute dw2 with recomputed activation_fn output.
        dw2 = gg.backend.gmm(activation_fn_out, ddsd_out, batch_sizes, trans_a=True)

        # Compute dactivation_fn_out.
        #
        # NOTE: We reuse the activation_fn_out allocation.
        dactivation_fn_out = activation_fn_out
        gg.backend.gmm(ddsd_out, w2, batch_sizes, trans_b=True, c=dactivation_fn_out)

        # Compute dsdd_out.
        #
        # NOTE: This reuses the dactivation_fn_out allocation.
        if activation_fn is DEFAULT_ACTIVATION_FN:
            dsdd_out = gelu.gelu_backward_(dactivation_fn_out, sdd_out)
        else:
            assert activation_grad_fn is not None
            activation_grad_fn(dactivation_fn_out)
            dsdd_out = sdd_out.grad

        # Compute dw1.
        dw1 = gg.backend.gmm(dsdd_out, x, batch_sizes, trans_a=True)

        # Compute dx.
        #
        # NOTE: This reuses the ddsd_out allocation.
        gg.backend.gmm(dsdd_out, w1, batch_sizes, c=ddsd_out)
        dx = ddsd_out
        return dx, dw1, dw2, None, None


memory_optimized_grouped_mlp = MemoryOptimizedParallelGroupedMLP.apply


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

        self.hidden_size_hn = self.hidden_size + neox_args.hypernetwork_bottleneck_dim + neox_args.hypernetwork_bottleneck_dim
        self.per_expert_lora_dim = self.per_expert_ff_dim + self.hidden_size
        self.num_rows_lora_per_rank = self.experts_per_rank * self.per_expert_lora_dim

        self.w1_emb = torch.nn.Parameter(
                torch.empty(neox_args.hidden_size + neox_args.intermediate_size,
                        neox_args.hypernetwork_bottleneck_dim,
                        device=torch.cuda.current_device(),
                        dtype=neox_args.params_dtype,
                    )
        )

        _initialize_affine_weight_gpu(
            self.w1_emb, init_method, partition_dim=0, stride=1)

        self.w2_emb = torch.nn.Parameter(
                torch.empty(neox_args.hidden_size + neox_args.intermediate_size,
                            neox_args.hypernetwork_bottleneck_dim,
                        device=torch.cuda.current_device(),
                        dtype=neox_args.params_dtype,
                    )
        )

        _initialize_affine_weight_gpu(
            self.w2_emb, init_method, partition_dim=0, stride=1)

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

        
        
        # self.lambda_e = torch.nn.Parameter(
        #     torch.empty(
        #         self.hidden_size * self.experts_per_rank,
        #         self.hidden_size,
        #         device=torch.cuda.current_device(),
        #         dtype=neox_args.params_dtype,
        #     )
        # )
        # _initialize_affine_weight_gpu(
        #     self.lambda_e, init_method, partition_dim=0, stride=stride
        # )
        self.lambda_e_w1 = torch.nn.Parameter(
            torch.empty(
                self.num_rows_per_rank,
                self.hidden_size,
                device=torch.cuda.current_device(),
                dtype=neox_args.params_dtype,
            )
        )
        _initialize_affine_weight_gpu(
            # self.lambda_e_w1, init_method, partition_dim=0, stride=stride
            self.lambda_e_w1, init_method_zeros(), partition_dim=0, stride=stride
        )
        self.lambda_e_w2 = torch.nn.Parameter(
            torch.empty(
                self.num_rows_per_rank,
                self.hidden_size,
                device=torch.cuda.current_device(),
                dtype=neox_args.params_dtype,
            )
        )
        _initialize_affine_weight_gpu(
            # self.lambda_e_w2, init_method, partition_dim=0, stride=stride
            self.lambda_e_w2, init_method_zeros(), partition_dim=0, stride=stride
        )
        self.swish = swish
        # # output
        # HN per expert
        self.lora_rank = self.args.hn_lora_rank
        self.do_per_expert_hn = self.args.per_expert_hn
        self.use_svd = self.args.hn_use_svd
        self.use_qr = self.args.hn_use_qr
        init_dim_multiplier = self.lora_rank * self.experts_per_rank if self.do_per_expert_hn else self.lora_rank
        self.hn_w1_A = torch.nn.Parameter(
            torch.empty(
                self.hidden_size * init_dim_multiplier,
                self.hidden_size_hn,
                device=torch.cuda.current_device(),
                dtype=neox_args.params_dtype,
            )
        )
        self.hn_w1_B = torch.nn.Parameter(
            torch.empty(
                self.per_expert_ff_dim * init_dim_multiplier,
                self.hidden_size_hn,
                device=torch.cuda.current_device(),
                dtype=neox_args.params_dtype,
            )
        )
        self.hn_w2_A = torch.nn.Parameter(
            torch.empty(
                init_dim_multiplier * self.per_expert_ff_dim,
                self.hidden_size_hn,
                device=torch.cuda.current_device(),
                dtype=neox_args.params_dtype,
            )
        )
        self.hn_w2_B = torch.nn.Parameter(
            torch.empty(
                init_dim_multiplier * self.hidden_size,
                self.hidden_size_hn,
                device=torch.cuda.current_device(),
                dtype=neox_args.params_dtype,
            )
        )
        std_w1_A = math.sqrt(1 / 
                        ( 
                            (self.hidden_size_hn) 
                            * (self.args.hidden_size)
            )
        )
        std_w2_A = math.sqrt(1 / 
                        ( 
                            (self.hidden_size_hn) 
                            * (self.args.hidden_size) * 4
            )
        )
        _initialize_affine_weight_gpu(
            # self.hn_w1_A, init_method_hyper(std_w1_A), partition_dim=0, stride=stride
            self.hn_w1_A, init_method, partition_dim=0, stride=stride
        )
        _initialize_affine_weight_gpu(
            # self.hn_w1_B, init_method, partition_dim=0, stride=stride
            # self.hn_w1_B, init_method_hyper(std_w2_A), partition_dim=0, stride=stride
            self.hn_w1_B, init_method_zeros(), partition_dim=0, stride=stride
        )
        _initialize_affine_weight_gpu(
            self.hn_w2_A, init_method, partition_dim=0, stride=stride
            # self.hn_w2_A, init_method_hyper(std_w2_A), partition_dim=0, stride=stride
        )
        _initialize_affine_weight_gpu(
            # self.hn_w2_B, init_method, partition_dim=0, stride=stride
            # self.hn_w2_B, init_method_hyper(std_w1_A), partition_dim=0, stride=stride
            self.hn_w2_B, init_method_zeros(), partition_dim=0, stride=stride
        )

        # TODO: why do we need this? was in original megablocks code
        self.gradient_scale = None
        if world_size > 1:
            self.gradient_scale = 1 / world_size
        self.world_size_ = world_size

        #
        self.lowrank_expert_norm_x = 0
        self.lowrank_expert_norm_y = 0

    def scale_grad(self, w: torch.Tensor):
        """
        Copied from SparseMLP
        """
        if self.gradient_scale is None:
            return w
        return scale_gradient(w, self.gradient_scale)

    def hn_scale_grad(self, w: torch.Tensor):
        """
        Copied from SparseMLP
        """
        gradient_scale = 1 / (self.world_size_ * self.args.train_micro_batch_size_per_gpu * self.args.seq_length * self.args.hn_scalar)
        return hn_clip_scale_gradient(w, gradient_scale)

    def get_factorization(self, matrix):
        with torch.no_grad():
            if self.use_svd:
                U, s, Vh = torch.linalg.svd(matrix.float(), full_matrices = False)
                return torch.cat((U[:, :, 1].to(torch.bfloat16), Vh[:, 1, :].to(torch.bfloat16)), dim=1)
            elif self.use_qr:
                Q, R = torch.linalg.qr(matrix.float())
                # print(f"Matrix shape {matrix.shape} Q shape {Q.shape} R shape {R.shape}")
                return Q[:, :, :5].reshape(self.experts_per_rank, self.args.hidden_size + self.args.intermediate_size).to(torch.bfloat16)

    def set_factorized_state(self):
        w1 = self.w1.view(self.experts_per_rank, self.hidden_size, self.per_expert_ff_dim)
        w2 = self.w2.view(self.experts_per_rank, self.hidden_size, self.per_expert_ff_dim)
        self.w1_factorization = self.get_factorization(w1)
        self.w2_factorization = self.get_factorization(w2)
        
    def factorize(self):
        if self.use_svd and self.args.iteration % 50 == 0:
            self.set_factorized_state()
        elif self.use_qr:
            self.set_factorized_state()

    def hn_emb(self, x: torch.Tensor, bin_ids: torch.Tensor):
        condition = False
        # condition = self.args.iteration % 100 >= self.args.hn_skip_grad
        # with torch.no_grad() if condition else nullcontext():
        with torch.no_grad():
            # embed w1
            w1 = self.w1.view(self.experts_per_rank, self.hidden_size, -1)
            if self.use_svd:
                w1_emb = self.w1_factorization
            elif self.use_qr:
                w1_emb = self.w1_factorization
            else:
                w1_rsum = torch.einsum('ijk->ij', w1)
                w1_csum = torch.einsum('ijk->ik', w1)  
                w1_emb = torch.cat((w1_rsum, w1_csum), dim=1)
            
            # embed w2
            w2 = self.w2.view(self.experts_per_rank, -1, self.hidden_size)
            if self.use_svd:
                w2_emb = self.w2_factorization
            elif self.use_qr:
                w2_emb = self.w2_factorization 
            else:
                w2_csum = torch.einsum('ijk->ij', w2)
                w2_rsum = torch.einsum('ijk->ik', w2) 
                w2_emb = torch.cat((w2_rsum, w2_csum), dim=1)
        w1_emb = torch.matmul(w1_emb, self.w1_emb)
        w2_emb = torch.matmul(w2_emb, self.w2_emb)

        # each token needs its own hypernetwork
        w1_emb = w1_emb[bin_ids]
        w2_emb = w2_emb[bin_ids]
        emb = torch.cat((x, w1_emb, w2_emb), dim=1)
        return emb


    def hn_forward_w1(self, x: torch.Tensor, emb: torch.Tensor, grouped_gemm_batch_sizes):
        condition = False
        # condition = self.args.iteration % self.args.hn_skip_cycle > self.args.hn_skip_grad or self.args.iteration < self.args.hn_start_cycle
        with torch.no_grad() if condition else nullcontext():
            # hn_w1_A, hn_w1_B = (self.hn_scale_grad(self.hn_w1_A), self.hn_scale_grad(self.hn_w1_B))
            hn_w1_A, hn_w1_B = (self.scale_grad(self.hn_w1_A), self.scale_grad(self.hn_w1_B))
            if self.do_per_expert_hn:
                hn_w1_A = hn_w1_A.view(self.experts_per_rank, self.hidden_size_hn, self.hidden_size * self.lora_rank)
                hn_w1_B = hn_w1_B.view(self.experts_per_rank, self.hidden_size_hn, self.per_expert_ff_dim * self.lora_rank)

                w1_A = gg.ops.gmm(emb, hn_w1_A, grouped_gemm_batch_sizes).view(-1, self.hidden_size, self.lora_rank)
                w1_B = gg.ops.gmm(emb, hn_w1_B, grouped_gemm_batch_sizes).view(-1, self.lora_rank, self.per_expert_ff_dim)
            else:
                hn_w1_A = hn_w1_A.view(self.hidden_size_hn, self.hidden_size * self.lora_rank)
                hn_w1_B = hn_w1_B.view(self.hidden_size_hn, self.per_expert_ff_dim * self.lora_rank)

                w1_A = torch.einsum("sj,jk->sk", emb, hn_w1_A).view(-1, self.hidden_size, self.lora_rank)
                w1_B = torch.einsum("sj,jk->sk", emb, hn_w1_B).view(-1, self.lora_rank, self.per_expert_ff_dim)
            return torch.einsum("ikj,ik->ij", w1_B, torch.einsum("ijk,ij->ik", w1_A, x))

    def hn_forward_w2(self, x: torch.Tensor, emb: torch.Tensor, grouped_gemm_batch_sizes):
        condition = False
        # condition = self.args.iteration % self.args.hn_skip_cycle > self.args.hn_skip_grad or self.args.iteration < self.args.hn_start_cycle
        with torch.no_grad() if condition else nullcontext():
            # hn_w2_A, hn_w2_B = (self.hn_scale_grad(self.hn_w2_A), self.hn_scale_grad(self.hn_w2_B))
            hn_w2_A, hn_w2_B = (self.scale_grad(self.hn_w2_A), self.scale_grad(self.hn_w2_B))
            if self.do_per_expert_hn:
                hn_w2_A = hn_w2_A.view(self.experts_per_rank, self.hidden_size_hn, self.per_expert_ff_dim * self.lora_rank)
                hn_w2_B = hn_w2_B.view(self.experts_per_rank, self.hidden_size_hn, self.hidden_size * self.lora_rank)

                w2_A = gg.ops.gmm(emb, hn_w2_A, grouped_gemm_batch_sizes).view(-1, self.per_expert_ff_dim, self.lora_rank)
                w2_B = gg.ops.gmm(emb, hn_w2_B, grouped_gemm_batch_sizes).view(-1, self.lora_rank, self.hidden_size)
            else:
                hn_w2_A = hn_w2_A.view(self.hidden_size_hn, self.per_expert_ff_dim * self.lora_rank)
                hn_w2_B = hn_w2_B.view(self.hidden_size_hn, self.hidden_size * self.lora_rank)

                w2_A = torch.einsum("sj,jk->sk", emb, hn_w2_A).view(-1, self.per_expert_ff_dim, self.lora_rank)
                w2_B = torch.einsum("sj,jk->sk", emb, hn_w2_B).view(-1, self.lora_rank, self.hidden_size)
            return torch.einsum("ikj,ik->ij", w2_B, torch.einsum("ijk,ij->ik", w2_A, x))

    def expert_forward_w1(self, x: torch.Tensor, grouped_gemm_batch_sizes):
        condition = False
        # condition = self.args.iteration % self.args.hn_skip_cycle < self.args.hn_skip_grad
        with torch.no_grad() if condition else nullcontext():
            w1 = self.scale_grad(self.w1)
            w1 = w1.view(self.experts_per_rank, -1, self.hidden_size)
            return gg.ops.gmm(x, w1, grouped_gemm_batch_sizes, trans_b=True)

    def expert_forward_w2(self, x: torch.Tensor, grouped_gemm_batch_sizes):
        condition = False
        # condition = self.args.iteration % self.args.hn_skip_cycle < self.args.hn_skip_grad
        with torch.no_grad() if condition else nullcontext():
            w2 = self.scale_grad(self.w2)
            w2 = w2.view(self.experts_per_rank, -1, self.hidden_size)
            return gg.ops.gmm(x, w2, grouped_gemm_batch_sizes)

    def lambda_balance(self, x: torch.Tensor, lambda_e, grouped_gemm_batch_sizes, trans_b):
        lambda_e = self.scale_grad(lambda_e)
        lambda_e = lambda_e.view(self.experts_per_rank, -1, self.hidden_size)
        return gg.ops.gmm(x, lambda_e, grouped_gemm_batch_sizes, trans_b=trans_b)

    def lambda_scale(self, lambda_e: torch.Tensor, y_e: torch.Tensor, y_l: torch.Tensor):
        # return y_e
        # return torch.einsum("ij,ij->ij", torch.ones_like(y_e), y_e)
        lambda_e = self.swish(lambda_e)
        retval = torch.einsum("ij,ij->ij", torch.ones_like(lambda_e) - lambda_e, y_e) + torch.einsum("ij,ij->ij", lambda_e, y_l)
        return retval

    def forward(self, x: torch.Tensor, tokens_per_expert: torch.Tensor, bin_ids: torch.Tensor):
        self.factorize()
        grouped_gemm_batch_sizes = tokens_per_expert.cpu().to(torch.long)
        lambda_e = self.lambda_balance(x, self.lambda_e_w1, grouped_gemm_batch_sizes, trans_b=True)
        emb = self.hn_emb(x, bin_ids)
        x_e = self.expert_forward_w1(x, grouped_gemm_batch_sizes)
        x_hn = self.hn_forward_w1(x, emb, grouped_gemm_batch_sizes)
        scaled_x = self.lambda_scale(lambda_e, x_e, x_hn)
        x = self.activation_func(scaled_x)
        # self.lowrank_expert_norm_x += torch.norm(x_hn / (x_e + torch.ones_like(scaled_x) * 1e-3)).float().numpy(force=True)
        # eps = 1e-3
        # print(f"Contrib of x_e to x_e {torch.norm((x_e / x_e))}")
        # print(f"Contrib of x_e to scaled x {torch.norm((self.activation_func(x_e).abs() / (x.abs() + (torch.ones_like(x) * eps))))}")
        # print(f"Contrib of x_hn to scaled x {torch.norm((self.activation_func(x_hn).abs() / (x.abs() + (torch.ones_like(x) * eps))))}")
        # print("Max hn val", x_hn.max())
        # print("Min x val", x_e.min())
        # print((x_hn / (scaled_x + torch.ones_like(scaled_x))).to(torch.float64).isinf().any())
        # print((x_hn / (x_e + torch.ones_like(x_e))).to(torch.float64).isinf().any())
        # print("Ratio to x_e", torch.norm((x_hn / (x_e + torch.ones_like(scaled_x))).to(torch.float64)))
        
        lambda_e = self.lambda_balance(x, self.lambda_e_w2, grouped_gemm_batch_sizes, trans_b=False)
        y_e = self.expert_forward_w2(x, grouped_gemm_batch_sizes)
        y_hn = self.hn_forward_w2(x, emb, grouped_gemm_batch_sizes)
        scaled_y = self.lambda_scale(lambda_e, y_e, y_hn)
        # self.lowrank_expert_norm_y += torch.norm(y_hn / (y_e + torch.ones_like(y_e) * 1e-3)).float().numpy(force=True)
        return scaled_y
        # if False:
        #     emb = self.hn_emb(x, bin_ids)
        #     x_e = self.expert_forward_w1(x, grouped_gemm_batch_sizes)
        #     x_e = self.activation_func(x_e)
        #     y_e = self.expert_forward_w2(x_e, grouped_gemm_batch_sizes)
        #     x_hn = self.hn_forward_w1(x, emb, grouped_gemm_batch_sizes)
        #     x_hn = self.activation_func(x_hn)
        #     y_hn = self.hn_forward_w2(x_hn, emb, grouped_gemm_batch_sizes)
        #     return y_e + y_hn
        # elif False:
        #     lambda_e = self.lambda_balance(x, self.lambda_e_w1, grouped_gemm_batch_sizes, trans_b=True)
        #     emb = self.hn_emb(x, bin_ids)
        #     x_e = self.expert_forward_w1(x, grouped_gemm_batch_sizes)
        #     x_hn = self.hn_forward_w1(x, emb, grouped_gemm_batch_sizes)
        #     x = self.activation_func(self.lambda_scale(lambda_e, x_e, x_hn))
        #     lambda_e = self.lambda_balance(x, self.lambda_e_w2, grouped_gemm_batch_sizes, trans_b=False)
        #     y_e = self.expert_forward_w2(x, grouped_gemm_batch_sizes)
        #     y_hn = self.hn_forward_w2(x, emb, grouped_gemm_batch_sizes)
        #     return self.lambda_scale(lambda_e, y_e, y_hn)
        # elif False:
        #     emb = self.hn_emb(x, bin_ids)
        #     x_e = self.expert_forward_w1(x, grouped_gemm_batch_sizes)
        #     x_hn = self.hn_forward_w1(x, emb, grouped_gemm_batch_sizes)
        #     x = self.activation_func(x_e + x_hn)
        #     y_e = self.expert_forward_w2(x, grouped_gemm_batch_sizes)
        #     y_hn = self.hn_forward_w2(x, emb, grouped_gemm_batch_sizes)
        #     return y_e + y_hn
        # else:
        #     emb = self.hn_emb(x, bin_ids)
        #     x_e = self.expert_forward_w1(x, grouped_gemm_batch_sizes)
        #     x_e = self.activation_func(x_e)
        #     y_e = self.expert_forward_w2(x_e, grouped_gemm_batch_sizes)
        #     x_hn = self.hn_forward_w1(x, emb, grouped_gemm_batch_sizes)
        #     x_hn = self.activation_func(x_hn)
        #     y_hn = self.hn_forward_w2(x_hn, emb, grouped_gemm_batch_sizes)
        #     lambda_e = self.lambda_balance(x, self.lambda_e, grouped_gemm_batch_sizes, trans_b=True)
        #     # lambda_e = self.lambda_balance(x, self.lambda_e_w1, grouped_gemm_batch_sizes, trans_b=True)
        #     return self.lambda_scale(lambda_e, y_e, y_hn)

class MemoryOptimizedParallelGroupedLLaMAMLP(torch.autograd.Function):
    """GroupedMLP with manually scheduled memory reuse."""

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x, w1, w3, w2, batch_sizes, activation_fn):
        # x: [m, k], w1: [n, k], w3: [n, k], w2: [n, k]
        if (
            not x.is_contiguous()
            or not w1.is_contiguous()
            or not w3.is_contiguous()
            or not w2.is_contiguous()
        ):
            raise ValueError("Expected contiguous 'x', 'w1', 'w3' and 'w2'.")

        # Layer 0: x @ w1.t().
        sdd_out = gg.backend.gmm(x, w1, batch_sizes, trans_b=True)
        w3_out = gg.backend.gmm(x, w3, batch_sizes, trans_b=True)

        # GeLU.
        activation_fn_out = activation_fn(sdd_out) * w3_out

        # Layer 1: x @ w2.
        dsd_out = gg.backend.gmm(activation_fn_out, w2, batch_sizes)

        # NOTE: Save the input to the layer and the activation_fn input for
        # gradient computation. We'll re-compute the activation_fn forward
        # pass in the backward pass to avoid materializing another
        # intermediate.
        ctx.x_shape = x.shape
        ctx.sdd_out_shape = sdd_out.shape
        ctx.dtype = x.dtype
        ctx.activation_fn = activation_fn
        ctx.save_for_backward(w1, w3, w2, batch_sizes, x, sdd_out, w3_out)
        return dsd_out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, ddsd_out):
        if (
            not ctx.needs_input_grad[0]
            or not ctx.needs_input_grad[1]
            or not ctx.needs_input_grad[2]
        ):
            raise ValueError("Expected all MLP inputs to need grad.")

        # Unpack saved tensors
        dtype = ctx.dtype
        saved_tensors = ctx.saved_tensors
        w1, w3, w2 = saved_tensors[:3]
        batch_sizes = saved_tensors[3]
        x = saved_tensors[4]
        sdd_out, w3_out = saved_tensors[5:7]

        # Rematerialize activation_fn output.
        activation_fn = ctx.activation_fn
        with torch.set_grad_enabled(True):
            sdd_out.requires_grad = True
            w3_out.requires_grad = True
            activation_fn_out = activation_fn(sdd_out) * w3_out
            activation_grad_fn = activation_fn_out.backward

        # Compute dw2 with recomputed activation_fn output.
        dw2 = gg.backend.gmm(activation_fn_out, ddsd_out, batch_sizes, trans_a=True)

        # Compute dactivation_fn_out.
        #
        # NOTE: We reuse the activation_fn_out allocation.
        dactivation_fn_out = activation_fn_out
        gg.backend.gmm(ddsd_out, w2, batch_sizes, trans_b=True, c=dactivation_fn_out)

        # Compute dsdd_out.
        #
        # NOTE: This reuses the dactivation_fn_out allocation.
        assert activation_grad_fn is not None
        activation_grad_fn(dactivation_fn_out)
        dsdd_out = sdd_out.grad
        dw3_out = w3_out.grad

        # Compute dw1.
        dw1 = gg.backend.gmm(dsdd_out, x, batch_sizes, trans_a=True)

        # Compute dw3.
        dw3 = gg.backend.gmm(dw3_out, x, batch_sizes, trans_a=True)

        # Compute dx.
        #
        # NOTE: This reuses the ddsd_out allocation.
        dx = ddsd_out
        gg.backend.gmm(dsdd_out, w1, batch_sizes, c=dx)
        dx += gg.backend.gmm(dw3_out, w3, batch_sizes)
        return dx, dw1, dw3, dw2, None, None


memory_optimized_grouped_llama_mlp = MemoryOptimizedParallelGroupedLLaMAMLP.apply


class ParallelGroupedLLaMAMLP(torch.nn.Module):
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
        super(ParallelGroupedLLaMAMLP, self).__init__()

        self.activation_func = get_activation(neox_args)
        self.activation_type = neox_args.activation

        self.multiple_of = multiple_of

        world_size = get_model_parallel_world_size()
        self.num_experts = neox_args.moe_num_experts
        self.experts_per_rank = divide(self.num_experts, world_size)

        self.hidden_size = neox_args.hidden_size

        # Allow custom intermediate size
        if neox_args.intermediate_size is not None:
            per_expert_ff_dim = neox_args.intermediate_size
        # Otherwise, 8/3 x hidden size, padded to multiple of 256
        # TODO: why is this how we formulate it this way?
        else:
            per_expert_ff_dim = int(2 * neox_args.hidden_size * 4 / 3)
            per_expert_ff_dim = self.multiple_of * (
                (per_expert_ff_dim + multiple_of - 1) // multiple_of
            )

        self.per_expert_ff_dim = per_expert_ff_dim
        # number of rows per rank is the number of experts * ff dimension per expert
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

        # gate
        self.w3 = torch.nn.Parameter(
            torch.empty(
                self.num_rows_per_rank,
                self.hidden_size,
                device=torch.cuda.current_device(),
                dtype=neox_args.params_dtype,
            )
        )
        _initialize_affine_weight_gpu(
            self.w3, init_method, partition_dim=0, stride=stride
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
        w1, w3, w2 = (
            self.scale_grad(self.w1),
            self.scale_grad(self.w3),
            self.scale_grad(self.w2),
        )

        w1 = self.w1.view(self.experts_per_rank, -1, self.hidden_size)
        w3 = w3.view(self.experts_per_rank, -1, self.hidden_size)

        w2 = w2.view(self.experts_per_rank, -1, self.hidden_size)

        # return memory_optimized_grouped_llama_mlp(
        #     x,
        #     w1,
        #     w3,
        #     w2,
        #     grouped_gemm_batch_sizes,
        #     self.activation_func
        # )

        llama_x_w1T = gg.ops.gmm(x, w1, grouped_gemm_batch_sizes, trans_b=True)

        llama_x_w3T = gg.ops.gmm(x, w3, grouped_gemm_batch_sizes, trans_b=True)

        llama_act_x_w1T = self.activation_func(llama_x_w1T)

        # self.w2(self.activation_func(w1_out) * w3_out)
        llama_mlp_out = gg.ops.gmm(
            llama_act_x_w1T
            * llama_x_w3T,  # activation results gated (element-wise) with w3
            w2,  # w2
            grouped_gemm_batch_sizes,  # batch_sizes
        )

        return llama_mlp_out
