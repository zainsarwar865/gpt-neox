
import torch
from megatron.neox_arguments.arguments import NeoXArgs
from megatron.mpu import get_model_parallel_group, get_model_parallel_rank, get_model_parallel_world_size, get_data_parallel_group, get_data_parallel_world_size
import megablocks.ops
from megatron.mpu.utils import divide
from megatron.mpu.layers import _initialize_affine_weight_gpu
from megablocks import grouped_gemm_util as gg
from megatron import print_rank_0
import torch.distributed as dist

from megatron import mpu


def global_sum(x, group=None):
        if group is None:
            group = mpu.get_data_parallel_group()
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)
        return x


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

class TopKTokenChoiceRouter(torch.nn.Module):
    def __init__(
        self,
        neox_args: NeoXArgs,
        init_method,
    ):
        super().__init__()
        self.jitter_eps = neox_args.moe_jitter_eps
        self.top_k = neox_args.moe_top_k
        self.num_experts = neox_args.moe_lora_experts
        self.aux_loss_coeff = neox_args.moe_lora_aux_loss_coeff
        self.moe_z_loss_coeff = neox_args.moe_z_loss_coeff
        self.args = neox_args
        self.layer = torch.nn.Linear(
            neox_args.hidden_size,
            neox_args.moe_lora_experts,
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
        low = 1.0 - self.jitter_eps
        high = 1.0 + self.jitter_eps
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

    def switch_load_balancing_loss_func(
        self,
        probs: torch.Tensor,                # [T, E], full softmax
        selected_expert_ids: torch.Tensor,  # [T, k], top-k expert IDs (ints)
        topk: int,
        moe_aux_loss_coeff: float,
    ):
        """
        Computes EXACT Switch-Transformer LBL with global synchronization:
        LBL = E * sum_i ( f_i_global * P_i_global )
        """

        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

        T, E = probs.shape

        # ------------------------------------------------------------
        # 1. Compute LOCAL token counts per expert: f_i numerator
        # ------------------------------------------------------------
        # Flatten selected expert IDs: [T*k]
        expert_ids = selected_expert_ids.reshape(-1)

        local_counts = torch.bincount(expert_ids, minlength=E).float()   # [E]

        # ------------------------------------------------------------
        # 2. GLOBAL sum to get global token counts
        # ------------------------------------------------------------
        global_counts = local_counts.clone()
        torch.distributed.all_reduce(global_counts, op=torch.distributed.ReduceOp.SUM)

        # if self.args.iteration % 20 == 0:                    
        #     print_rank_0(global_counts)

        total_tokens_global = global_counts.sum()

        f_i = global_counts / (total_tokens_global + 1e-9)               # [E]
        P_local = probs.sum(dim=0)                                       # [E]
        P_global = P_local.clone()
        torch.distributed.all_reduce(P_global, op=torch.distributed.ReduceOp.SUM)

        # Normalize by GLOBAL token count → true P_i
        P_i = P_global / (total_tokens_global + 1e-9)                    # [E]

        # ------------------------------------------------------------
        # 5. Final LBL per the paper
        # ------------------------------------------------------------
        # LBL = E * sum_i f_i * P_i
        lbl = (E * (f_i * P_i).sum()) * moe_aux_loss_coeff

        return lbl




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


    
    # def apply_load_balancing_loss(
    #         self,
    #         probs: torch.Tensor,
    #         num_local_tokens_per_expert: torch.Tensor,
    #         activation: torch.Tensor,
    #     ):
    #         """Applies auxiliary loss to the MoE layer.

    #         Args:
    #             probs (torch.Tensor): The probs output by the router for each token. [num_tokens, num_experts]
    #             num_local_tokens_per_expert (torch.Tensor): The number of tokens per expert. [num_experts]
    #             activation (torch.Tensor): The activation tensor to attach the gradient function to.

    #         Returns:
    #             torch.Tensor: The activation tensor with the attached gradient function.
    #         """            
    #         aux_loss = self.switch_load_balancing_loss_func(
    #             probs, num_local_tokens_per_expert, self.top_k, self.aux_loss_coeff
    #         )
    #         #print(num_local_tokens_per_expert, aux_loss)
    #         if self.args.iteration % 20 == 0:            
                
    #             print_rank_0(num_local_tokens_per_expert)
    #             print_rank_0(aux_loss.item())
    #         activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
            
    #         return activation




    def apply_load_balancing_loss(
            self,
            probs: torch.Tensor,                 # [T, num_experts]
            local_tokens_per_expert: torch.Tensor, # [num_experts]
            activation: torch.Tensor,
        ):
            """
            GLOBAL load balancing loss:
                f_i = global token fraction for expert i
                p_i = global mean router probability for expert i
                L = num_experts
                aux_loss = L * sum_i f_i * p_i
            """

            world_group = get_data_parallel_group()
            world_size  = get_data_parallel_world_size()

            # -------- 1) GLOBAL TOKEN COUNTS --------
            tokens_global = local_tokens_per_expert.clone().float()
            global_sum(tokens_global, group=world_group)

            # normalize into fractions
            f_i = tokens_global / tokens_global.sum()

            # -------- 2) GLOBAL MEAN PROBABILITIES --------
            # local mean
            p_local = probs.mean(dim=0)  # [num_experts]

            # sum over world
            p_global = p_local.clone()
            global_sum(p_global, group=world_group)

            # global mean
            p_i = p_global / world_size

            # -------- 3) GLOBAL AUX LOSS --------
            L = self.num_experts
            aux_loss = L * (f_i * p_i).sum() * self.aux_loss_coeff

            # attach to activation
            activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)

            return activation



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
        #logits, z_loss = self.apply_z_loss(logits)
        #z_loss_temp = z_loss.detach()


        # expert_weights (float) shape: (sl * bs, top_k)...value(s) from scores corresponding to the top_k experts
        # expert_indices (int) shape: (sl * bs, top_k)...index(indices) from scores corresponding to the top_k experts
        expert_weights, expert_indices = self._top_k(scores)

        with torch.no_grad():
            expert_indices_ft = expert_indices.flatten()
            # tokens_per_expert = megablocks.ops.histogram(expert_indices_ft, self.num_experts)
            tokens_per_expert = torch.bincount(
                expert_indices_ft, minlength=self.num_experts)


        # ---- GLOBAL LOAD BALANCING LOSS (paper exact Eq. 4–6) ----
        aux_loss = self.switch_load_balancing_loss_func(
            probs=scores,                # full softmax distribution [T, E]
            selected_expert_ids=expert_indices,  # top-k expert IDs [T, k]
            topk=self.top_k,
            moe_aux_loss_coeff=self.aux_loss_coeff
        )

        expert_weights = MoEAuxLossAutoScaler.apply(expert_weights, aux_loss)
        # expert_weights probability mass won't add up to 1 because we took
        # the topk scores from the softmax
        # TODO: placeholder for moe_normalize_expert_weights if necessary
        return expert_weights, expert_indices


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

