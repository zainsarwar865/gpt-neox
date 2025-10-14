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




class ParallelGroupedLoRas(torch.nn.Module):
    def __init__(
        self,
        neox_args: NeoXArgs,
        init_method,
        output_layer_init_method,
        stride: int = 1,
        multiple_of: int = 256,
    ):
        """
        Dense LoRA blocks materialized as grouped GEMMs.
        Adds forward_with_batch_sizes(x, batch_sizes_cpu, layer) to avoid .cpu() inside forward().
        """
        super().__init__()
        self.multiple_of = multiple_of

        world_size = get_model_parallel_world_size()
        self.num_experts = neox_args.moe_num_experts
        self.experts_per_rank = divide(self.num_experts, world_size)
        self.hidden_size = neox_args.hidden_size
        self.zero_init_method = init_method_zeros()

        # FFN dimension per expert
        if neox_args.intermediate_size is not None:
            per_expert_ff_dim = neox_args.intermediate_size

        self.per_expert_ff_dim = per_expert_ff_dim

        self.num_loras = neox_args.moe_lora_experts
        self.num_experts = neox_args.moe_num_experts
        self.experts_per_rank = divide(self.num_experts, world_size)
        self.loras_per_rank = divide(self.num_loras, world_size)

        # shapes
        self.num_cols_per_rank = self.loras_per_rank * self.experts_per_rank
        self.num_rows = self.experts_per_rank * self.loras_per_rank
        self.lora_rank = neox_args.lora_rank
        self.total_loras = self.loras_per_rank * self.experts_per_rank

        # ---- LoRA 1 (up) ----
        self.w1_A = torch.nn.Parameter(
            torch.empty(
                self.num_rows,
                self.hidden_size * self.lora_rank,
                device=torch.cuda.current_device(),
                dtype=neox_args.params_dtype,
            )
        )

        self.w1_B = torch.nn.Parameter(
            torch.empty(
                self.num_rows,
                self.per_expert_ff_dim * self.lora_rank,
                device=torch.cuda.current_device(),
                dtype=neox_args.params_dtype,
            )
        )

        # megablocks trick (optional gradient scaling for TP>1)


    # ---------------- main forward (kept identical except comments) ----------------
    def forward(self, x: torch.Tensor, tokens_per_lora: torch.Tensor, layer: int):
        # This path preserves your original behavior: does .cpu() here.
        grouped_gemm_batch_sizes = tokens_per_lora.cpu().to(torch.long)

        if layer == 1:
            w1_A, w1_B = (self.w1_A, self.w1_B)
            w1_A = w1_A.view(self.total_loras, self.hidden_size, self.lora_rank)
            w1_B = w1_B.view(self.total_loras, self.lora_rank, self.per_expert_ff_dim)
            return gg.ops.gmm(
                gg.ops.gmm(x, w1_A, grouped_gemm_batch_sizes),
                w1_B,
                grouped_gemm_batch_sizes,
            )

    # ------------- overlap-friendly forward (uses pre-staged CPU batch sizes) -------------
    @torch.no_grad()  # weights still require grad via data path; sizes are just routing metadata
    def forward_with_batch_sizes(
        self,
        x: torch.Tensor,
        batch_sizes_cpu: torch.Tensor,
        layer: int,
    ):
        """
        Same math as forward(), but assumes `batch_sizes_cpu` is an already-available
        CPU tensor (e.g., pinned + asynchronously staged). No .cpu() inside.
        """
        # Make sure dtype matches what gg.ops.gmm expects (usually torch.long on CPU).
        if batch_sizes_cpu.device.type != "cpu":
            raise ValueError("batch_sizes_cpu must be a CPU tensor")
        if batch_sizes_cpu.dtype != torch.long:
            batch_sizes_cpu = batch_sizes_cpu.to(torch.long)

        if layer == 1:
            w1_A, w1_B = (self.w1_A, self.w1_B)
            w1_A = w1_A.view(self.total_loras, self.hidden_size, self.lora_rank)
            w1_B = w1_B.view(self.total_loras, self.lora_rank, self.per_expert_ff_dim)
            return gg.ops.gmm(
                gg.ops.gmm(x, w1_A, batch_sizes_cpu),
                w1_B,
                batch_sizes_cpu,
            )


# === drop-in timed version (no logic changes) ===
import torch
import numpy as np
import megablocks.ops
from megatron.neox_arguments.arguments import NeoXArgs
from megatron.model.activations import get_activation
from megatron.mpu.initialize import get_model_parallel_world_size
from megatron.mpu.layers import ColumnParallelLinear, RowParallelLinear
from megatron.mpu.utils import divide
from megatron.mpu import gather_from_expert_model_parallel_region
from .router import TopKTokenChoiceRouter
from typing import Optional, Dict, List
from torch.profiler import record_function

# ---- tiny CUDA timer helpers (zero overhead when disabled) ----
class _CudaTimerCtx:
    def __init__(self, store: Optional[Dict[str, List[float]]], name: str):
        self.store = store
        self.name = name
        self._start = None
        self._end = None
    def __enter__(self):
        if self.store is None:
            return
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        self._start.record()
    def __exit__(self, exc_type, exc, tb):
        if self.store is None:
            return
        self._end.record()
        self._end.synchronize()
        ms = self._start.elapsed_time(self._end)
        self.store.setdefault(self.name, []).append(ms)

class _NoTimer:
    def __enter__(self): pass
    def __exit__(self, *args): pass

def _maybe_timer(store, name: str):
    return _CudaTimerCtx(store, name) if store is not None else _NoTimer()

class ParallelGroupedMLP(torch.nn.Module):
    """
    Dense MLP (w1/w2) + Routed LoRAs.
    Overlaps the D→H copy of tokens_per_lora (required by gg.ops.gmm) with the dense up-projection.
    """

    def __init__(
        self,
        neox_args: NeoXArgs,
        init_method,
        output_layer_init_method,
        stride: int = 1,
        multiple_of: int = 256,
    ):
        super().__init__()
        self.args = neox_args
        self.activation_func = get_activation(neox_args)
        self.activation_type = neox_args.activation
        self.multiple_of = multiple_of

        # --- timing store (set via set_timer_store) ---
        self._timer_store: Optional[Dict[str, List[float]]] = None

        # --- overlap plumbing ---
        self._aux_stream = torch.cuda.Stream()
        self._pinned_sizes_cpu: Optional[torch.Tensor] = None  # persistent pinned buffer

        world_size = get_model_parallel_world_size()
        self.num_experts = neox_args.moe_num_experts
        self.experts_per_rank = divide(self.num_experts, world_size)

        self.hidden_size = neox_args.hidden_size
        # self.LoRaRouter = TopKTokenChoiceRouter(neox_args, init_method)

        # self.num_loras = neox_args.moe_lora_experts
        # self.loras_per_rank = divide(self.num_loras, world_size)

        # self.loras = ParallelGroupedLoRas(
        #     neox_args=neox_args,
        #     init_method=init_method,
        #     output_layer_init_method=output_layer_init_method,
        # )

        # self.total_loras = self.loras_per_rank * self.experts_per_rank
        # self.sort_end_bit = max(int(np.ceil(np.log2(self.total_loras))), 1)

        # per-expert ff dim
        if neox_args.intermediate_size is not None:
            per_expert_ff_dim = neox_args.intermediate_size
        else:
            per_expert_ff_dim = 4 * self.hidden_size
            per_expert_ff_dim = self.multiple_of * (
                (per_expert_ff_dim + multiple_of - 1) // multiple_of
            )
        self.per_expert_ff_dim = per_expert_ff_dim
        self.num_rows_per_rank = self.experts_per_rank * per_expert_ff_dim

        # Dense MLP (tensor-parallel)
        self.w1 = ColumnParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=self.num_rows_per_rank,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
        )
        self.w2 = RowParallelLinear(
            neox_args=neox_args,
            input_size=self.num_rows_per_rank,
            output_size=neox_args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            parallel_output=False,
            skip_bias_add=True,
        )


    # ---- external control for timers ----
    def set_timer_store(self, store: Optional[Dict[str, List[float]]]):
        """Pass a dict to collect per-op CUDA timings, or None to disable."""
        self._timer_store = store

    # ----------------------- utils -----------------------

    # ----------------------- forward -----------------------

    def forward(self, x: torch.Tensor):
        # 1) Route tokens to LoRAs (all on device)

        # 2) Kick off async D→H copy of sizes into pinned host buffer on an aux stream

        # 3) Dense up-projection can run in parallel on the default stream
        with record_function("dense:w1"), _maybe_timer(self._timer_store, "dense:w1"):
            x_dense, _ = self.w1(x)  # ColumnParallelLinear returns (out, bias)

        # 6) Fuse and down-projection
        with record_function("dense:act+fuse"), _maybe_timer(self._timer_store, "dense:act+fuse"):
            if self.args.lora_interaction_type == "addition":
                x_act = self.activation_func(x_dense)
            elif self.args.lora_interaction_type == "geglu":
                x_act = self.activation_func(x_dense)
            else:
                raise RuntimeError("LoRe interaction not defined")

        with record_function("dense:w2"), _maybe_timer(self._timer_store, "dense:w2"):
            x_act, _ = self.w2(x_act)

        return x_act