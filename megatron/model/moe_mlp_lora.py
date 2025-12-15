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
from torch.nn.functional import softmax

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

# dense_lore.py

class FusedLoREUpProj(nn.Module):
    def __init__(self, H, Dff, L, r, init_method, dtype, device):
        super().__init__()
        self.H = H
        self.Dff = Dff
        self.L = L
        self.r = r

        # Joint up-projection (dense W1 + all A blocks)
        self.W_joint = nn.Parameter(
            torch.empty(H, Dff + L * r, dtype=dtype, device=device)
        )
        init_method(self.W_joint)

        # B_stack directly parameterized
        self.B_stack = nn.Parameter(
            torch.empty(L * r, Dff, dtype=dtype, device=device)
        )
        init_method(self.B_stack)

    def forward(self, X, beta):
        """
        X: [T, H]
        beta: [T, L]  (router softmax masked to top-k)
        """
        # GEMM 1: compute base + all A_i contributions
        Z = X @ self.W_joint

        Z_base = Z[:, :self.Dff]
        Z_lore = Z[:, self.Dff:]  # [T, L*r]

        T = X.shape[0]
        S = Z_lore.view(T, self.L, self.r) * beta.unsqueeze(-1)

        # GEMM 2: combine with B
        lore = S.reshape(T, self.L * self.r) @ self.B_stack

        return Z_base + lore



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

        world_size = get_model_parallel_world_size()

        self.hidden_size = neox_args.hidden_size
        self.LoRaRouter = TopKTokenChoiceRouter(neox_args, init_method)
        self.num_loras = neox_args.moe_lora_experts
        self.lora_rank = neox_args.lora_rank
        
    
        self.up_fused = FusedLoREUpProj(
            H=self.hidden_size,
            Dff=neox_args.intermediate_size,
            L=self.num_loras,
            r=self.lora_rank,
            init_method=init_method,
            dtype=neox_args.params_dtype,
            device=torch.cuda.current_device(),
        )



        # Project back to h.
        self.dense_4h_to_h = mpu.RowParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.intermediate_size,
            output_size=neox_args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
        )

        self.gradient_scale = None
        if world_size > 1:
            self.gradient_scale = 1 / world_size


    def build_beta(self, topk_weights, topk_indices, L):
        T, k = topk_weights.shape
        beta = topk_weights.new_zeros(T, L)
        beta.scatter_(1, topk_indices, topk_weights)
        return beta


    def scale_grad(self, w: torch.Tensor):
        """
        Copied from SparseMLP
        """
        if self.gradient_scale is None:
            return w
        return scale_gradient(w, self.gradient_scale)


    def forward(self, x):
        """
        x: [seq, batch, H] or [T, H]
        """
        # Flatten for routing
        x_flat = x.view(-1, x.shape[-1])     # [T, H]

        # Router
        w, idx = self.LoRaRouter(x)          # [T, k], [T, k]
        w = w.view(x_flat.size(0), -1)
        idx = idx.view(x_flat.size(0), -1)

        # Dense beta [T, L]
        beta = self.build_beta(w, idx, self.num_loras)

        # ------- Fused up-projection -------
        up = self.up_fused(x_flat, beta)     # [T, Dff]

        # Activation
        h = self.activation_func(up)

        # Down-projection (TP-aware)
        out, _ = self.dense_4h_to_h(h)       # [T, H]

        return out.view_as(x)