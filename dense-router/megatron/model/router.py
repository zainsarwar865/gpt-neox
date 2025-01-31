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

from megatron.neox_arguments.arguments import NeoXArgs
from megatron.mpu import get_model_parallel_group, get_model_parallel_rank, get_model_parallel_world_size, get_data_parallel_group
from megatron.mpu.layers import _initialize_affine_weight_gpu
from megatron.mpu.utils import divide
from megablocks import grouped_gemm_util as gg
from megatron import print_rank_0, mpu

import megablocks.ops

class MoEAuxLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that compute and scales the grad for auxiliary loss.

    """

    main_loss_backward_scale: torch.Tensor = torch.tensor(1.0)

    @staticmethod
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor):
        """Preserve the aux_loss by storing it in the context to avoid garbage collection.
        
        Args:
            output (torch.Tensor): The output tensor.
            aux_loss (torch.Tensor): The auxiliary loss tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute and scale the gradient for auxiliary loss..

        Args:
            grad_output (torch.Tensor): The gradient of the output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled auxiliary loss gradient.
        """
        (aux_loss,) = ctx.saved_tensors
        aux_loss_backward_scale = MoEAuxLossAutoScaler.main_loss_backward_scale
        scaled_aux_loss_grad = torch.ones_like(aux_loss) * aux_loss_backward_scale
        return grad_output, scaled_aux_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """set the scale of the aux loss.
        
        Args:
            scale (torch.Tensor): The scale value to set. Please ensure that the scale passed in matches the scale of the main_loss.
        """
        MoEAuxLossAutoScaler.main_loss_backward_scale = scale

class SinkhornRouter(torch.nn.Module):
    # TODO: reduce precision on expert_indices? it looks like it's currently int64
    # TODO: how do we ensure that all copies of the router get the same
    # initializations and stay in sync over time? Or is this handled by RNG seeding?

    ### Sinkhorn

    # - https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/moe_utils.py
    # - https://github.com/fanshiqing/grouped_gemm
    #     - NVIDIA forked original implementation and is using this in Megatron Core now
    # - https://github.com/NVIDIA/Megatron-LM/blob/cafda9529d9956578014d4cb89b69b741702b514/megatron/core/transformer/moe/router.py#L215: this his how megatron actually does its router forward pass

    def __init__(
        self,
        neox_args: NeoXArgs,
        init_method,
    ):
        super().__init__()
        self.top_k = neox_args.moe_top_k
        self.params_dtype = neox_args.params_dtype

        # expert parallel group rank, for purposes of deciding if I should compute the router or wait for the result to be broadcast to me
        self.expert_parallel_group = get_model_parallel_group()
        self.expert_parallel_rank = get_model_parallel_rank()

        # Sinkhorn router parameters.
        #
        # NOTE: This weight matrix is not parallelized with expert tensor
        # parallelism. Each device needs the entire router weight matrix
        # so that it can route its batch of data correctly.
        self.layer = torch.nn.Linear(
            neox_args.hidden_size,
            neox_args.moe_num_experts,
            bias=False,
            dtype=neox_args.params_dtype,
            device=torch.cuda.current_device(),
        )
        init_method(self.layer.weight)

    def sinkhorn(self, cost: torch.Tensor, tol: float = 0.0001):
        """Sinkhorn based MoE routing function"""
        cost = torch.exp(cost)
        d0 = torch.ones(cost.size(0), device=cost.device, dtype=cost.dtype)
        d1 = torch.ones(cost.size(1), device=cost.device, dtype=cost.dtype)

        eps = 0.00000001
        error = 1e9
        d1_old = d1
        while error > tol:
            d0 = (1 / d0.size(0)) * 1 / (torch.sum(d1 * cost, 1) + eps)
            d1 = (1 / d1.size(0)) * 1 / (torch.sum(d0.unsqueeze(1) * cost, 0) + eps)
            error = torch.mean(torch.abs(d1_old - d1))
            d1_old = d1
        return d1 * cost * d0.unsqueeze(1)

    def sinkhorn_load_balancing(self, logits: torch.Tensor):
        """Apply sinkhorn routing to the logits tensor.

        Args:
            logits (torch.Tensor): The logits tensor, as (bs * sl, hidden_size)

        Returns:
            torch.Tensor: The logits tensor after applying sinkhorn routing.
        """

        def _sinkhorn_activation(logits):
            if self.top_k == 1:
                logits = torch.sigmoid(logits)
            else:  # k > 1
                logits = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(
                    logits
                )
            return logits

        # assert self.config.moe_aux_loss_coeff == 0, "Sinkhorn routing does not support aux loss."
        if self.training:
            with torch.no_grad():
                norm_logits = self.sinkhorn(
                    logits.to(dtype=torch.float32)
                )  # explicit fp32 conversion for stability
                _, indices = torch.topk(norm_logits, k=self.top_k, dim=1)
            logits = _sinkhorn_activation(logits)
            scores = torch.gather(logits, 1, indices)
        # at inference, just top_k it...sinkhorn algorithm doesn't support autoregressive generation
        else:
            logits = _sinkhorn_activation(logits)
            scores, indices = torch.topk(logits, k=self.top_k, dim=1)
        return scores, indices

    def forward(self, x):
        """
        Forward pass through the Sinkhorn Router.

        Only compute on rank 0 in the expert parallel group and broadcast to everyone else to avoid weird states where things get out of sync.

        Args:
            x (torch.Tensor): Input tensor to be routed.
                (sl, bs, hs)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing
                - expert_weights (sl * bs, top_k): Weights assigned to the selected experts
                - expert_indices (sl * bs, top_k): Indices of the selected experts
        """
        if self.expert_parallel_rank == 0:
            # x.view shape: (sl * bs, hs)...every token as a row
            # router_logits (float) shape: (sl * bs, num_experts)...expert rankings for every token
            router_logits = self.layer(x.view(-1, x.shape[-1]))

            # expert_weights (float) shape: (sl * bs, top_k)...value(s) from scores corresponding to the top_k experts
            # expert_indices (int) shape: (sl * bs, top_k)...index(indices) from scores corresponding to the top_k experts
            expert_weights, expert_indices = self.sinkhorn_load_balancing(router_logits)

            # broadcast the routing result to all ranks
            expert_weights_broadcast = torch.distributed.broadcast(
                expert_weights,
                src=torch.distributed.get_global_rank(self.expert_parallel_group, 0),
                group=self.expert_parallel_group,
                async_op=True,
            )
            expert_indices_broadcast = torch.distributed.broadcast(
                expert_indices,
                src=torch.distributed.get_global_rank(self.expert_parallel_group, 0),
                group=self.expert_parallel_group,
                async_op=True,
            )
        else:
            # sl * bs
            num_rows = x.view(-1, x.shape[-1]).shape[0]
            expert_weights = torch.empty(
                num_rows,
                self.top_k,
                device=torch.cuda.current_device(),
                dtype=self.params_dtype,
            )
            expert_indices = torch.empty(
                num_rows,
                self.top_k,
                device=torch.cuda.current_device(),
                dtype=torch.int64,
            )

            expert_weights_broadcast = torch.distributed.broadcast(
                expert_weights,
                src=torch.distributed.get_global_rank(self.expert_parallel_group, 0),
                group=self.expert_parallel_group,
                async_op=True,
            )
            expert_indices_broadcast = torch.distributed.broadcast(
                expert_indices,
                src=torch.distributed.get_global_rank(self.expert_parallel_group, 0),
                group=self.expert_parallel_group,
                async_op=True,
            )

        # since both are executing asynchronously, it doesn't matter which one
        # we wait for first
        expert_weights_broadcast.wait()
        expert_indices_broadcast.wait()

        return expert_weights, expert_indices

def z_loss_func(logits, z_loss_coeff):
    """Encourages the router's logits to remain small to enhance stability.
    Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.

    Args:
        logits (torch.Tensor): The logits of the router.

    Returns:
        torch.Tensor: The logits after applying the z-loss.
    """

    z_loss = torch.mean(torch.square(torch.logsumexp(logits, dim=-1))) * z_loss_coeff
    return z_loss

class Router(torch.nn.Module):
    def __init__(
        self,
        neox_args: NeoXArgs,
        init_method,
    ):
        super().__init__()
        self.jitter_eps = neox_args.moe_jitter_eps
        self.top_k = neox_args.moe_top_k

        self.layer = torch.nn.Linear(
            neox_args.hidden_size,
            neox_args.moe_num_experts,
            bias=False,
            dtype=neox_args.params_dtype,
            device=torch.cuda.current_device(),
        )
        self.router_type = neox_args.moe_router_type
        self.neox_args = neox_args
        
        self.moe_aux_loss_coeff = neox_args.moe_aux_loss_coeff / neox_args.gradient_accumulation_steps
        self.moe_z_loss_coeff = neox_args.moe_z_loss_coeff / neox_args.gradient_accumulation_steps


        init_method(self.layer.weight)
        self.num_experts = neox_args.moe_num_experts
        self.tokens_per_batch = neox_args.seq_length * neox_args.train_micro_batch_size_per_gpu
        self.expert_parallel_group = get_model_parallel_group()
        self.expert_parallel_rank = get_model_parallel_rank()
        self.data_parallel_group = get_data_parallel_group()

        self.reset_logging_buffers()

    def sinkhorn(self, cost: torch.Tensor, tol: float = 0.0001):
        """Sinkhorn based MoE routing function"""
        cost = torch.exp(cost)
        d0 = torch.ones(cost.size(0), device=cost.device, dtype=cost.dtype)
        d1 = torch.ones(cost.size(1), device=cost.device, dtype=cost.dtype)

        eps = 0.00000001
        error = 1e9
        d1_old = d1
        while error > tol:
            d0 = (1 / d0.size(0)) * 1 / (torch.sum(d1 * cost, 1) + eps)
            d1 = (1 / d1.size(0)) * 1 / (torch.sum(d0.unsqueeze(1) * cost, 0) + eps)
            error = torch.mean(torch.abs(d1_old - d1))
            d1_old = d1
        return d1 * cost * d0.unsqueeze(1)

    def sinkhorn_load_balancing(self, logits: torch.Tensor):
        """Apply sinkhorn routing to the logits tensor.

        Args:
            logits (torch.Tensor): The logits tensor, as (bs * sl, hidden_size)

        Returns:
            torch.Tensor: The logits tensor after applying sinkhorn routing.
        """

        def _sinkhorn_activation(logits):
            if self.top_k == 1:
                logits = torch.sigmoid(logits)
            else:  # k > 1
                logits = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(
                    logits
                )
            return logits

        # assert self.config.moe_aux_loss_coeff == 0, "Sinkhorn routing does not support aux loss."
        if self.training:
            with torch.no_grad():
                norm_logits = self.sinkhorn(
                    logits.to(dtype=torch.float32)
                )  # explicit fp32 conversion for stability
                _, indices = torch.topk(norm_logits, k=self.top_k, dim=1)
            logits = _sinkhorn_activation(logits)
            scores = torch.gather(logits, 1, indices)
        # at inference, just top_k it...sinkhorn algorithm doesn't support autoregressive generation
        else:
            logits = _sinkhorn_activation(logits)
            scores, indices = torch.topk(logits, k=self.top_k, dim=1)
        return scores, indices

    def jitter(self, x):
        low = 1.0 - self.jitter_eps
        high = 1.0 + self.jitter_eps
        noise = torch.rand(x.size(), dtype=x.dtype, device=x.device)
        return low + noise * (high - low)
    
    def _top_k(self, scores):
        if self.top_k == 1:
            return scores.max(dim=-1, keepdim=True)
        return torch.topk(scores, self.top_k, dim=-1)
    
    def switch_load_balancing_loss_func(self,
        probs: torch.Tensor, tokens_per_expert: torch.Tensor, topk: int, moe_aux_loss_coeff: float
    ):
        num_tokens = probs.shape[0]
        num_experts = probs.shape[1]

        aggregated_probs_per_expert = probs.sum(dim=0)
        aux_loss = torch.sum(aggregated_probs_per_expert * tokens_per_expert) * (
            num_experts * moe_aux_loss_coeff / (num_tokens * num_tokens * topk)
        )
        return aux_loss

    def apply_load_balancing_loss(
            self,
            probs: torch.Tensor,
            num_local_tokens_per_expert: torch.Tensor,
            activation: torch.Tensor,
        ):
            """Applies auxiliary loss to the MoE layer.

            Args:
                probs (torch.Tensor): The probs output by the router for each token. [num_tokens, num_experts]
                num_local_tokens_per_expert (torch.Tensor): The number of tokens per expert. [num_experts]
                activation (torch.Tensor): The activation tensor to attach the gradient function to.

            Returns:
                torch.Tensor: The activation tensor with the attached gradient function.
            """
            aux_loss = self.switch_load_balancing_loss_func(
                probs, num_local_tokens_per_expert, self.top_k, self.moe_aux_loss_coeff
            )
            activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
            return activation, aux_loss
        
    def apply_z_loss(self, logits):
        """Encourages the router's logits to remain small to enhance stability.
        Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.

        Args:
            logits (torch.Tensor): The logits of the router.

        Returns:
            torch.Tensor: The logits after applying the z-loss.
        """
        if self.moe_z_loss_coeff is not None:
            z_loss = z_loss_func(logits, self.moe_z_loss_coeff)
            logits = MoEAuxLossAutoScaler.apply(logits, z_loss)
            return logits, z_loss
        else:
            return logits, None
    
    def forward(self, x, router_type_override=None):
        router_type = router_type_override
        
        if router_type is None:
            router_type = self.router_type
        
        if self.training and self.jitter_eps is not None:
            x = x * self.jitter(x)

        logits = self.layer(x.view(-1, x.shape[-1]))
        # logits, z_loss = self.apply_z_loss(logits)
        # z_loss_temp = z_loss.detach()
        # global_z_loss = torch.distributed.all_reduce(
        #         z_loss_temp,
        #         group=self.data_parallel_group,
        #         op=torch.distributed.ReduceOp.SUM,
        #         async_op=True,
        #     )
        
        # if router_type == "topk" or router_type == "dense_approx" or router_type == "dense_approx_lsh" or router_type == "dense_approx_efficient" or router_type == "expert_prob_approx":
        #     scores = logits.softmax(dim=-1)
        #     expert_weights, expert_indices = self._top_k(scores)
        
        if router_type == "sparsemixer":
            expert_weights, scores, expert_indices = sparsemixerv2_routing(logits, self.top_k, self.jitter_eps, self.training)

        elif router_type == "dense":
            scores = logits.softmax(dim=-1)
            expert_weights, expert_indices = torch.topk(scores, self.num_experts, dim=-1)
        elif "sinkhorn" in router_type:
            scores = logits.softmax(dim=-1)
            expert_weights, expert_indices = self.sinkhorn_load_balancing(logits)
        else:
            # raise ValueError(f"Invalid MoE Router type {router_type}")
            scores = logits.softmax(dim=-1)
            expert_weights, expert_indices = self._top_k(scores)
        
        with torch.no_grad():
            expert_indices_ft = expert_indices.flatten()
            tokens_per_expert = megablocks.ops.histogram(expert_indices_ft, self.num_experts)
            
        expert_weights, lbl_loss = self.apply_load_balancing_loss(scores, tokens_per_expert, activation=expert_weights)
        lbl_loss_temp = lbl_loss.detach()
        global_lbl_loss = torch.distributed.all_reduce(
                lbl_loss_temp,
                group=self.data_parallel_group,
                op=torch.distributed.ReduceOp.SUM,
                async_op=True,
            )

        # expert_weights probability mass won't add up to 1 because we took
        # the topk scores from the softmax
        # TODO: placeholder for moe_normalize_expert_weights if necessary
        routing_counts = torch.bincount(expert_indices.flatten(),minlength=self.num_experts)
        global_routing_counts = torch.distributed.all_reduce(
                routing_counts,
                group=self.data_parallel_group,
                op=torch.distributed.ReduceOp.SUM,
                async_op=True,
            )
        # overlap comms
        # global_z_loss.wait()

        # if self.expert_parallel_rank == 0 and self.training:
        #     self.save_z_loss(
        #         z_loss_temp / ( self.neox_args.global_num_gpus * self.moe_z_loss_coeff )
        #     )
        global_lbl_loss.wait()

        if self.expert_parallel_rank == 0 and self.training:
            self.save_lbl_loss(
                lbl_loss_temp / ( self.neox_args.global_num_gpus * self.moe_z_loss_coeff )
            )
        global_routing_counts.wait()

        if self.expert_parallel_rank == 0 and self.training:
            self.save_routing_counts_train(routing_counts)

        return expert_weights, expert_indices, scores
    
    def reset_logging_buffers(self):
        self.train_routing_count_buffer = None
        self.lbl_loss_buffer = None
        self.z_loss_buffer = None

    def save_routing_counts_train(self, routing_counts):
        if self.train_routing_count_buffer == None:
            self.train_routing_count_buffer = routing_counts.unsqueeze(0)
        else:
            self.train_routing_count_buffer = torch.cat([self.train_routing_count_buffer,
                                                            routing_counts.unsqueeze(0)],
                                                            dim=0)

    def save_lbl_loss(self, new_loss):
        if self.lbl_loss_buffer == None:
            self.lbl_loss_buffer = new_loss.unsqueeze(0)
        else:
            self.lbl_loss_buffer = torch.cat([self.lbl_loss_buffer,
                                                new_loss.unsqueeze(0)],
                                                dim=0)
    
    def save_z_loss(self, new_loss):
        if self.z_loss_buffer == None:
            self.z_loss_buffer = new_loss.unsqueeze(0)
        else:
            self.z_loss_buffer = torch.cat([self.z_loss_buffer,
                                            new_loss.unsqueeze(0)],
                                            dim=0)
