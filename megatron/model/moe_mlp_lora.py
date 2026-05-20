import torch
from megatron.model.activations import get_activation, swish
from megatron.mpu.layers import _initialize_affine_weight_gpu
from megatron.mpu.initialize import get_model_parallel_world_size
from megatron.mpu.utils import divide
from megatron.model.router import TopKTokenChoiceRouter
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

class DenseLoREUpProj(nn.Module):
    """
    Dense execution, sparse semantics:
      S = X @ A_stack       # [T, H] @ [H, L*r] -> [T, L*r]
      S = S.view(T,L,r) * beta[:, :, None]
      Y = (S.view(T,L*r)) @ B_stack   # [T, L*r] @ [L*r, Dffp_local] -> [T, Dffp_local]
    """
    def __init__(self, H, Dffp_local, L, r, init_A, init_B, dtype, device):
        super().__init__()
        self.H, self.Dffp_local, self.L, self.r = H, Dffp_local, L, r

        self.A = nn.Parameter(torch.empty(L, H, r, dtype=dtype, device=device))
        self.B = nn.Parameter(torch.empty(L, r, Dffp_local, dtype=dtype, device=device))
        init_A(self.A)
        init_B(self.B)

        # cache for stacked views (rebuilt if shapes change)
        self._A_stack = None
        self._B_stack = None
        self._cached_shapes = None

    def _stack_weights(self):
        # A_stack: [H, L*r]  (concat LoRE A blocks column-wise)
        A_stack = self.A.permute(1, 0, 2).contiguous().view(self.H, self.L * self.r)
        # B_stack: [L*r, Dffp_local]  (concat LoRE B blocks row-wise)
        B_stack = self.B.contiguous().view(self.L * self.r, self.Dffp_local)
        return A_stack, B_stack

    @torch.no_grad()
    def _build_beta(self, T, topk_weights, topk_indices, L):
        # topk_*: [T, k] from the router (probabilities already softmaxed)
        beta = topk_weights.new_zeros(T, L)
        beta.scatter_(1, topk_indices, topk_weights)
        return beta

    def forward(self, X, topk_weights, topk_indices):
        """
        X: [T, H]
        topk_weights: [T, k]  (router probabilities over top-k)
        topk_indices: [T, k]  (router selected indices in [0..L-1])
        returns: [T, Dffp_local]
        """
        T = X.shape[0]
        if self._cached_shapes != (self.A.shape, self.B.shape):
            self._A_stack, self._B_stack = self._stack_weights()
            self._cached_shapes = (self.A.shape, self.B.shape)
        beta = self._build_beta(T, topk_weights, topk_indices, self.L)  # [T, L]
        # GEMM 1
        S = X @ self._A_stack                          # [T, L*r]
        S = S.view(T, self.L, self.r) * beta.unsqueeze(-1)
        # Y = S @ self._B_stack    
        # GEMM 2
        Y = S.reshape(T, self.L * self.r) @ self._B_stack  # [T, Dffp_local]
        return Y



# class ParallelGroupedMLP(torch.nn.Module):
#     def __init__(
#         self,
#         neox_args: NeoXArgs,
#         init_method,
#         output_layer_init_method,
#         stride=1,
#         multiple_of=256,
#     ):
#         """
#         Copied from SparseMLP
#         """
#         super(ParallelGroupedMLP, self).__init__()
#         self.args = neox_args
#         self.activation_func = get_activation(neox_args)
#         self.activation_type = neox_args.activation

#         world_size = get_model_parallel_world_size()

#         self.hidden_size = neox_args.hidden_size
#         self.LoRaRouter = TopKTokenChoiceRouter(neox_args, init_method)
#         self.num_loras = neox_args.moe_lora_experts
#         self.lora_rank = neox_args.lora_rank
        
    
#         self.dense_lore_up = DenseLoREUpProj(
#             H=self.hidden_size,
#             Dffp_local=neox_args.intermediate_size,
#             L=self.num_loras,
#             r=self.lora_rank,
#             init_A=init_method,                
#             init_B=init_method,                
#             dtype=neox_args.params_dtype,
#             device=torch.cuda.current_device(),
#         )
        
#         self.dense_h_to_4h = nn.Parameter(
#             torch.empty(neox_args.hidden_size, neox_args.intermediate_size, dtype=neox_args.params_dtype, device=torch.cuda.current_device())
#         )
#         init_method(self.dense_h_to_4h)

#         self.dense_4h_to_h = nn.Parameter(
#             torch.empty( neox_args.intermediate_size,neox_args.hidden_size, dtype=neox_args.params_dtype, device=torch.cuda.current_device())
#         )
#         init_method(self.dense_4h_to_h)

#         self.gradient_scale = None
#         if world_size > 1:
#             self.gradient_scale = 1 / world_size


#     def scale_grad(self, w: torch.Tensor):
#         """
#         Copied from SparseMLP
#         """
#         if self.gradient_scale is None:
#             return w
#         return scale_gradient(w, self.gradient_scale)
    

#     def forward(self, x: torch.Tensor):
#         """
#         x: [seq, batch, hidden] or [tokens, hidden] (router flattens internally)
#         """
#         # 1) Router: get top-k weights/indices per token
#         lora_weights, lora_indices = self.LoRaRouter(x)           # [T, k], [T, k]
#         T = lora_weights.numel() // self.LoRaRouter.top_k
#         lora_weights = lora_weights.view(T, self.LoRaRouter.top_k)
#         lora_indices = lora_indices.view(T, self.LoRaRouter.top_k)
#         # 2) Base up-projection (TP-local output)
#         up_local = x @ self.dense_h_to_4h            # [T, Dffp_local]
#         # 3) Dense-LoRE up-projection to the SAME local shape
#         x_flat = x.view(-1, x.shape[-1])                           # [T, H]
#         lore_local = self.dense_lore_up(x_flat, lora_weights, lora_indices)  # [T, Dffp_local]
#         # 4) Fuse before nonlinearity
#         x = self.activation_func(up_local + lore_local)

#         # 5) Down-projection (RowParallel)
        
#         return x @ self.dense_4h_to_h
    
class ParallelGroupedMLP(torch.nn.Module):
    def __init__(
        self,
        neox_args: NeoXArgs,
        init_method,
        output_layer_init_method,
        stride=1,
        multiple_of=256,
    ):
        super(ParallelGroupedMLP, self).__init__()

        self.args = neox_args
        self.activation_func = get_activation(neox_args)  # should be swish for SwiGLU
        self.activation_type = neox_args.activation

        world_size = get_model_parallel_world_size()

        self.hidden_size = neox_args.hidden_size
        self.intermediate_size = neox_args.intermediate_size

        
        self.LoRaRouter = TopKTokenChoiceRouter(neox_args, init_method)
        self.num_loras = neox_args.moe_lora_experts
        self.lora_rank = neox_args.lora_rank

        
        self.dense_lore_up = DenseLoREUpProj(
            H=self.hidden_size,
            Dffp_local=self.intermediate_size,
            L=self.num_loras,
            r=self.lora_rank,
            init_A=init_method,
            init_B=init_method,
            dtype=neox_args.params_dtype,
            device=torch.cuda.current_device(),
        )

        
        self.w_value = nn.Parameter(
            torch.empty(self.hidden_size, self.intermediate_size,
                        dtype=neox_args.params_dtype,
                        device=torch.cuda.current_device())
        )
        init_method(self.w_value)

        self.w_gate = nn.Parameter(
            torch.empty(self.hidden_size, self.intermediate_size,
                        dtype=neox_args.params_dtype,
                        device=torch.cuda.current_device())
        )
        init_method(self.w_gate)

        
        self.dense_4h_to_h = nn.Parameter(
            torch.empty(self.intermediate_size, self.hidden_size,
                        dtype=neox_args.params_dtype,
                        device=torch.cuda.current_device())
        )
        init_method(self.dense_4h_to_h)

        self.gradient_scale = None
        if world_size > 1:
            self.gradient_scale = 1 / world_size

    def scale_grad(self, w: torch.Tensor):
        if self.gradient_scale is None:
            return w
        return scale_gradient(w, self.gradient_scale)

    def forward(self, x: torch.Tensor):
        """
        x: [seq, batch, hidden] OR [tokens, hidden]
        """

        # ---- flatten ----
        x_flat = x.view(-1, x.shape[-1])   # [T, H]

        # ---- router ----
        lora_weights, lora_indices = self.LoRaRouter(x)
        T = lora_weights.numel() // self.LoRaRouter.top_k
        lora_weights = lora_weights.view(T, self.LoRaRouter.top_k)
        lora_indices = lora_indices.view(T, self.LoRaRouter.top_k)

        # ---- base projections ----
        value = x_flat @ self.w_value     # [T, D]
        gate  = x_flat @ self.w_gate      # [T, D]

        # ---- LoRE (ONLY on value) ----
        lore = self.dense_lore_up(x_flat, lora_weights, lora_indices)

        value = value + lore

        # ---- SwiGLU ----
        x_out = value * self.activation_func(gate)

        # ---- down projection ----
        x_out = x_out @ self.dense_4h_to_h

        # ---- reshape back ----
        x_out = x_out.view_as(x)

        return x_out