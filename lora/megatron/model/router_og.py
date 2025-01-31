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
import megablocks.ops

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
        self.num_experts = neox_args.moe_num_experts
        self.tokens_per_batch = neox_args.seq_length * neox_args.train_micro_batch_size_per_gpu

        # expert parallel group rank, for purposes of deciding if I should compute the router or wait for the result to be broadcast to me
        self.expert_parallel_group = get_model_parallel_group()
        self.expert_parallel_rank = get_model_parallel_rank()
        self.data_parallel_group = get_data_parallel_group()

        self.global_routing_counts = torch.zeros(neox_args.moe_num_experts)

        # print("SinkhornRouter EP group:{} EP rank:{}".format(self.expert_parallel_group,self.expert_parallel_rank))

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

        # flipped RWT Black mamba
        #d0 \in R^(BS * SL)
        #d1 \in R^E

        #black mamba init
        d0 = (self.tokens_per_batch / self.num_experts) * torch.exp(2 * cost).sum(axis=1)
        # assert d0.shape[0] == cost.size(0)
        # print('d0',d0.shape)
        # print('d1',d1.shape)
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

    def scale_grad(self, w: torch.Tensor):
        """
        Copied from SparseMLP
        """
        if self.gradient_scale is None:
            return w
        return scale_gradient(w, self.gradient_scale)

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
            # it seems that (sl * bs, top_k) here is really just the kth index of each expert
            # to get them all we really just need and all reduce sum
            
            # print("expert_weights",expert_weights[:10])
            # print("expert_indices",expert_indices[:10])
            routing_counts = torch.bincount(expert_indices.squeeze(1),minlength=self.num_experts)
            # print('local_routing_counts',routing_counts)

            # exit(0)


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
            global_routing_counts = torch.distributed.all_reduce(
                routing_counts,
                group=self.data_parallel_group,
                op=torch.distributed.ReduceOp.SUM,
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
        global_routing_counts.wait()

        self.global_routing_counts = routing_counts

        # print("routing_counts after AR",routing_counts)
        # print("global_routing_counts",global_routing_counts)

        # exit(0)

        return expert_weights, expert_indices


class TopKTokenChoiceRouter(torch.nn.Module):
    # TODO: how do we ensure that all copies of the router get the same
    # initializations and stay in sync over time? Or is this handled by RNG seeding?

    def __init__(
        self,
        neox_args: NeoXArgs,
        init_method,
    ):
        super().__init__()
        self.jitter_eps = neox_args.moe_jitter_eps
        self.top_k = neox_args.moe_top_k

        # Learned router parameters.
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

    def jitter(self, x):
        """
        Apply jittering to the input tensor during training.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Jittered input tensor.
        """
        low = 1.0 - self.args.moe_jitter_eps
        high = 1.0 + self.args.moe_jitter_eps
        noise = torch.rand(x.size(), dtype=x.dtype, device=x.device)
        return low + noise * (high - low)

    def _top_k(self, scores):
        """
        Select the top-k experts based on input scores.

        Args:
            scores (torch.Tensor): Input scores from the router.
                (sl * bs, num_experts)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing expert weightings and indices of selected experts.


        """
        if self.top_k == 1:
            return scores.max(dim=-1, keepdim=True)
        return torch.topk(scores, self.top_k, dim=-1)

    def forward(self, x):
        """
        Forward pass through the Learned Router.

        Args:
            x (torch.Tensor): Input tensor to be routed.
                (sl, bs, hs)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing
                - expert_weights (sl * bs, top_k): Weights assigned to the selected experts
                - expert_indices (sl * bs, top_k): Indices of the selected experts
        """
        if self.training and self.jitter_eps is not None:
            x = x * self.jitter(x)

        # x.view shape: (sl * bs, hs)...every token as a row
        # scores (float) shape: (sl * bs, num_experts)...expert rankings for every token
        scores = self.layer(x.view(-1, x.shape[-1])).softmax(dim=-1)

        # expert_weights (float) shape: (sl * bs, top_k)...value(s) from scores corresponding to the top_k experts
        # expert_indices (int) shape: (sl * bs, top_k)...index(indices) from scores corresponding to the top_k experts
        expert_weights, expert_indices = self._top_k(scores)
        # expert_weights probability mass won't add up to 1 because we took
        # the topk scores from the softmax
        # TODO: placeholder for moe_normalize_expert_weights if necessary

        return expert_weights, expert_indices






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




















class TopKTokenChoiceRouterLoRa(torch.nn.Module):
    # TODO: how do we ensure that all copies of the router get the same
    # initializations and stay in sync over time? Or is this handled by RNG seeding?

    def __init__(
        self,
        neox_args: NeoXArgs,
        init_method,
    ):
        super().__init__()
        self.jitter_eps = neox_args.moe_jitter_eps
        self.top_k = neox_args.lora_top_k
        self.params_dtype = neox_args.params_dtype
        self.hidden_size = neox_args.hidden_size
        self.aux_loss_coeff = neox_args.moe_aux_loss_coeff

        # expert parallel group rank, for purposes of deciding if I should compute the router or wait for the result to be broadcast to me
        self.expert_parallel_group = get_model_parallel_group()
        self.expert_parallel_rank = get_model_parallel_rank()

        world_size = get_model_parallel_world_size()
        self.num_loras = neox_args.moe_lora_experts
        self.num_experts = neox_args.moe_num_experts
        self.experts_per_rank = divide(self.num_experts, world_size)
        self.loras_per_rank = divide(self.num_loras, world_size)
        self.num_rows_per_rank = self.loras_per_rank *  self.experts_per_rank
        self.total_loras = self.loras_per_rank * self.experts_per_rank

        self.w1 = torch.nn.Parameter( 
                    torch.empty(
                        self.num_rows_per_rank,
                        self.hidden_size,
                        device=torch.cuda.current_device(),
                        dtype=neox_args.params_dtype
                        )
        )

        _initialize_affine_weight_gpu(self.w1, init_method, partition_dim=0, stride=1)

        self.gradient_scale = None
        if world_size > 1:
            self.gradient_scale = 1 / world_size



    def jitter(self, x):
        """
        Apply jittering to the input tensor during training.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Jittered input tensor.
        """
        low = 1.0 - self.args.moe_jitter_eps
        high = 1.0 + self.args.moe_jitter_eps
        noise = torch.rand(x.size(), dtype=x.dtype, device=x.device)
        return low + noise * (high - low)

    def _top_k(self, scores):
        """
        Select the top-k experts based on input scores.

        Args:
            scores (torch.Tensor): Input scores from the router.
                (sl * bs, num_experts)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing expert weightings and indices of selected experts.


        """
        if self.top_k == 1:
            return scores.max(dim=-1, keepdim=True)
        return torch.topk(scores, self.top_k, dim=-1)


    def scale_grad(self, w: torch.Tensor):
        """
        Copied from SparseMLP
        """
        if self.gradient_scale is None:
            return w
        return scale_gradient(w, self.gradient_scale)


    def generate_offsets(self, tokens_per_expert, num_loras):
        cum_lenghts = torch.cumsum(tokens_per_expert, dim=0)
        indices = torch.arange(cum_lenghts[-1]) + 1
        segments = (torch.searchsorted(cum_lenghts, indices, right=False) * num_loras).to(device=torch.cuda.current_device())

        return segments


    def switch_load_balancing_loss_func(self,
        probs: torch.Tensor, tokens_per_lora: torch.Tensor, topk: int, moe_aux_loss_coeff: float
    ):
        """Calculate the auxiliary loss for better load balacing. 
        Please refer to the Switch Transformer paper (https://arxiv.org/abs/2101.03961) for details.

        Args:
            probs (torch.Tensor): The softmax probs output by the router for each token. [num_tokens, num_experts]
            tokens_per_expert (torch.Tensor): The number of assigned tokens for each expert. [num_experts]

        Returns:
            torch.Tensor: The auxiliary loss for load balancing.
        """
        # Compute batch loss per expert's loras
        num_tokens = probs.shape[0] * topk
        num_experts = probs.shape[1]
        probs_mean_per_expert = probs.mean(dim=0)
        aux_loss = torch.sum(probs_mean_per_expert * tokens_per_lora) * (
            num_experts / num_tokens * moe_aux_loss_coeff
        )
        return aux_loss

    def apply_load_balancing_loss(
            self,
            probs: torch.Tensor,
            num_local_tokens_per_loras: torch.Tensor,
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
            # moe_aux_loss_coeff = (
            #     self.config.moe_aux_loss_coeff / parallel_state.get_tensor_model_parallel_world_size()
            # )
        
            
            
            
            aux_loss = self.switch_load_balancing_loss_func(
                probs, num_local_tokens_per_loras, self.top_k, self.aux_loss_coeff
            )
            #print("Load balancing loss is : ", aux_loss)
            # save_to_aux_losses_tracker(
            #     "load_balancing_loss",
            #     aux_loss / moe_aux_loss_coeff,
            #     self.layer_number,
            #     self.config.num_layers,
            # )
            activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
            
            return activation


    def forward(self, x, grouped_gemm_batch_sizes):
        """
        Forward pass through the Learned Router.

        Args:
            x (torch.Tensor): Input tensor to be routed.
                (sl, bs, hs)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing
                - expert_weights (sl * bs, top_k): Weights assigned to the selected experts
                - expert_indices (sl * bs, top_k): Indices of the selected experts
        """
        if self.training and self.jitter_eps is not None:
            x = x * self.jitter(x)



        if self.expert_parallel_rank == 0:

            # GMM on the router layer
            w1 = self.scale_grad(self.w1)
            w1 = w1.view(self.experts_per_rank, self.hidden_size, self.loras_per_rank)
            
            # Logits for each router
            scores = gg.ops.gmm(x, w1, grouped_gemm_batch_sizes).softmax(dim=-1) # [num_tokens, lora_distribution]
            expert_weights, expert_indices = self._top_k(scores)

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


        # aux loss for loop

        with torch.no_grad():
            #offset_vector = self.generate_offsets(grouped_gemm_batch_sizes, self.loras_per_rank)
            expert_indices_ft = expert_indices.flatten() #+ offset_vector
            #tokens_per_lora = megablocks.ops.histogram(expert_indices_ft, self.num_loras)

        start = 0
        for i in grouped_gemm_batch_sizes:
            if i > 0:
                with torch.no_grad():
                    tokens_per_lora = megablocks.ops.histogram(expert_indices_ft[start:i + start], self.num_loras)
                #print(f"tokens_per_lora : {tokens_per_lora}")
                expert_weights_ = self.apply_load_balancing_loss(scores[start:i + start, :], tokens_per_lora, activation=expert_weights)
            start = i

        # since both are executing asynchronously, it doesn't matter which one
        # we wait for first
        expert_weights_broadcast.wait()
        expert_indices_broadcast.wait()

        return expert_weights, expert_indices





class SinkhornLoRaRouter(torch.nn.Module):
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
        self.top_k = neox_args.lora_top_k
        self.params_dtype = neox_args.params_dtype
        self.hidden_size = neox_args.hidden_size

        # expert parallel group rank, for purposes of deciding if I should compute the router or wait for the result to be broadcast to me
        self.expert_parallel_group = get_model_parallel_group()
        self.expert_parallel_rank = get_model_parallel_rank()

        # Sinkhorn router parameters.
        #
        # NOTE: This weight matrix is not parallelized with expert tensor
        # parallelism. Each device needs the entire router weight matrix
        # so that it can route its batch of data correctly.
        
        
        # Replace this with GroupedGemms
        # self.layer = torch.nn.Linear(
        #     neox_args.hidden_size,
        #     neox_args.moe_num_experts,
        #     bias=False,
        #     dtype=neox_args.params_dtype,
        #     device=torch.cuda.current_device(),
        # )

        # init_method(self.layer.weight)
        world_size = get_model_parallel_world_size()
        self.num_loras = neox_args.lora_num_experts
        self.num_experts = neox_args.moe_num_experts
        self.experts_per_rank = divide(self.num_experts, world_size)
        self.loras_per_rank = divide(self.num_loras, world_size)
        self.num_cols_per_rank = self.loras_per_rank *  self.experts_per_rank

        self.w1 = torch.nn.Parameter( 
                    torch.empty(
                        self.hidden_size,
                        self.num_cols_per_rank,
                        device=torch.cuda.current_device(),
                        dtype=neox_args.params_dtype
                        )
        )

        _initialize_affine_weight_gpu(self.w1, init_method, partition_dim=0, stride=1)

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

    def forward(self, x, tokens_per_expert):
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

            # GMM on the router layer
            w1 = self.scale_grad(self.w1)

            w1 = w1.view(self.experts_per_rank, self.hidden_size, self.loras_per_rank)

            # Logits for each router
            grouped_gemm_batch_sizes = tokens_per_expert.cpu().to(torch.long)
            router_logits = gg.ops.gmm(x, w1, grouped_gemm_batch_sizes) # [num_tokens, lora_distribution]

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