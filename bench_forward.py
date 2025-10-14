#!/usr/bin/env python
# Run exactly like training:
#   ../../deepy.py ../../bench_mlp_like_train.py ../lora.yml
#
# Uses the same PyTorch profiler pattern as your train() loop.
# Env toggles:
#   MLP_PATH=experts|router    # default: experts (MLP-only)
#   SAVE_CHROME=1              # also export chrome trace json
#   PROFILE_DIR=./profile_trace
#
# YAML controls:
#   profile: true
#   profile_step_start: <int>
#   profile_step_stop:  <int>
#   tensorboard_dir: <dir>
#   train_iters: <int>

import os
import time
import torch

from megatron.neox_arguments import NeoXArgs
from megatron.initialize import initialize_megatron
from megatron.utils import print_rank_0
from megatron.model.init_functions import init_method_normal, scaled_init_method_normal, init_method_zeros



# ---- profiler summarizer -----------------------------------------------------
import re
import numpy as np

def _ms(us):  # microseconds -> milliseconds
    return us / 1000.0

# ---- robust profiler summarizer ---------------------------------------------
import re
import numpy as np

def _ms(us):  # microseconds -> milliseconds
    return (us or 0.0) / 1000.0

def _get(e, name, default=0):
    return getattr(e, name, default) or 0

# def summarize_profile(prof, interesting=None, top_k=20, prefer_cuda=True):
#     """
#     Prints:
#       1) Top ops by total time (prefers CUDA if present; else CPU)
#       2) Per-range stats (p50/p90/p95/p99/mean) for selected record_function names.
#     """
#     assert prof is not None

#     events = prof.key_averages(group_by_input_shape=False)

#     # Detect if CUDA timing fields exist & are nonzero
#     has_cuda_field = any(hasattr(e, "cuda_time_total") for e in events)
#     total_cuda_us = sum(_get(e, "cuda_time_total", 0) for e in events) if has_cuda_field else 0
#     use_cuda = prefer_cuda and has_cuda_field and total_cuda_us > 0

#     # Choose fields
#     total_us = total_cuda_us if use_cuda else sum(_get(e, "cpu_time_total", 0) for e in events)
#     time_field = "cuda_time_total" if use_cuda else "cpu_time_total"
#     self_field = "self_cuda_time_total" if use_cuda else "self_cpu_time_total"

#     print(f"\n=== Key averages (sorted by {'CUDA' if use_cuda else 'CPU'} time) ===")
#     hdr = f"{'name':50} {'calls':>7} {'tot_ms':>10} {'%tot':>7} {'self_ms':>10}"
#     print(hdr)
#     print("-"*len(hdr))

#     events_sorted = sorted(
#         events,
#         key=lambda e: _get(e, time_field, 0),
#         reverse=True
#     )[:top_k]

#     for e in events_sorted:
#         tot_us = _get(e, time_field, 0)
#         self_us = _get(e, self_field, 0)
#         pct = (tot_us * 100.0) / (total_us or 1)
#         print(f"{e.key[:50]:50} {e.count:7d} {_ms(tot_us):10.3f} {pct:7.2f} {_ms(self_us):10.3f}")

#     # 2) Percentiles for your custom ranges (record_function names)
#     if not interesting:
#         return

#     # Some versions expose `prof.events()`; keep a guard.
#     try:
#         full = prof.events()
#     except Exception:
#         print("\n(Profiler events() not available on this build/version.)")
#         return

#     name_to_ms = {}
#     for pattern in interesting:
#         rx = re.compile(pattern)
#         durs = []
#         for ev in full:
#             # prefer CUDA if present; else CPU
#             dur_us = None
#             if use_cuda and hasattr(ev, "cuda_time_total") and ev.cuda_time_total:
#                 dur_us = ev.cuda_time_total
#             elif hasattr(ev, "cpu_time_total") and ev.cpu_time_total:
#                 dur_us = ev.cpu_time_total
#             if dur_us is not None and ev.name and rx.fullmatch(ev.name):
#                 durs.append(_ms(dur_us))
#         if durs:
#             name_to_ms[pattern] = np.asarray(durs, dtype=np.float64)

#     if not name_to_ms:
#         print("\n(No matching record_function ranges found for 'interesting')")
#         return

#     print("\n=== Selected ranges (ms) — latency distribution ===")
#     hdr = f"{'range':35} {'N':>7} {'mean':>9} {'p50':>9} {'p90':>9} {'p95':>9} {'p99':>9} {'max':>9}"
#     print(hdr)
#     print("-"*len(hdr))
#     for name, arr in name_to_ms.items():
#         mean = arr.mean()
#         p50  = np.percentile(arr, 50)
#         p90  = np.percentile(arr, 90)
#         p95  = np.percentile(arr, 95)
#         p99  = np.percentile(arr, 99)
#         mx   = arr.max()
#         print(f"{name[:35]:35} {len(arr):7d} {mean:9.3f} {p50:9.3f} {p90:9.3f} {p95:9.3f} {p99:9.3f} {mx:9.3f}")
# -----------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# === import your module ===
# Change to your actual module path if needed (you said imports exist).
from megatron.model.moe_lora_b import ParallelDroplessMoE  # <-- adjust if different

def _set_cuda_from_local_rank():
    if torch.cuda.is_available():
        lr = int(os.environ.get("LOCAL_RANK", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0")))
        torch.cuda.set_device(lr)

@torch.inference_mode()
def main(input_args=None, overwrite_values=None):
    # Parse args exactly like train.py so deepy/deepspeed flags are accepted
    neox_args = NeoXArgs.consume_neox_args(input_args=input_args, overwrite_values=overwrite_values)
    neox_args.configure_distributed_args()
    _set_cuda_from_local_rank()

    # We only need infra init, not data/optimizer/etc.
    neox_args.do_train = False
    neox_args.do_valid = False
    neox_args.do_test  = False
    neox_args.eval_iters = 0
    if getattr(neox_args, "vocab_size", None) is None:
        neox_args.vocab_size = 50257

    initialize_megatron(neox_args=neox_args)

    device = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("cpu")

    # Build your MoE wrapper that hosts the MLP
    init_fn = init_method_zeros()
    out_init_fn = init_method_zeros()
    module = ParallelDroplessMoE(
        neox_args=neox_args,
        init_method=init_fn,
        output_layer_init_method=out_init_fn,
    ).to(device)
    module.eval()

    # Shapes / dtype per config
    S = neox_args.seq_length
    B = neox_args.train_micro_batch_size_per_gpu
    H = neox_args.hidden_size
    dtype = getattr(neox_args, "params_dtype", torch.float16)

    # Synthetic activations
    x = torch.randn(S, B, H, device=device, dtype=dtype)

    # Routing tensors for MLP-only path (your config moe_num_experts=1, top_k=1)
    T = S * B
    top_k = neox_args.moe_top_k
    expert_weights = torch.ones((T, top_k), device=device, dtype=dtype)
    expert_indices = torch.zeros((T, top_k), device=device, dtype=torch.long)

    # Autocast to match your params dtype
    autocast_dtype = torch.bfloat16 if dtype is torch.bfloat16 else torch.float16
    amp_ctx = torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=autocast_dtype)

    # What to profile?
    mlp_path = os.environ.get("MLP_PATH", "router").lower()  # 'experts' or 'router'
    assert mlp_path in ("router"), "MLP_PATH must be 'experts' or 'router'"

    # Profiler setup exactly like your train() loop
    do_profile = bool(getattr(neox_args, "profile", False))
    start_it = int(getattr(neox_args, "profile_step_start", 5))
    stop_it  = int(getattr(neox_args, "profile_step_stop", start_it + 10))
    total_iters = int(getattr(neox_args, "train_iters", stop_it))  # drive the outer loop

    # Optional chrome export
    save_chrome = os.environ.get("SAVE_CHROME", "0") == "1"
    outdir = os.path.abspath(os.environ.get("PROFILE_DIR", "./profile_trace"))
    os.makedirs(outdir, exist_ok=True)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    if do_profile:
        schedule = torch.profiler.schedule(
            wait=start_it,
            warmup=1,
            active=max(1, stop_it - start_it),
        )
        prof = torch.profiler.profile(
            schedule=schedule,
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(neox_args.tensorboard_dir),
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
            with_modules=True,
            with_stack=True,
        )
        prof.start()

    iteration = 0
    if rank == 0:
        print(f"[bench] device={torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu'} | dtype={dtype} | path={mlp_path}")
        print(f"[bench] iterations={total_iters} | profile={do_profile} [{start_it}..{stop_it}] | TB={neox_args.tensorboard_dir}")

    # Lightweight timers
    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()

    while iteration < total_iters:
        if do_profile:
            prof.step()
        if do_profile and iteration == start_it:
            torch.cuda.cudart().cudaProfilerStart()

        # -------- one "train step" but just the MLP --------


        # router + mlp end-to-end
        _y, _ = module(x)

        if do_profile and iteration == stop_it:
            torch.cuda.cudart().cudaProfilerStop()
            prof.stop()
        iteration += 1

    if device.type == "cuda": torch.cuda.synchronize()
    t1 = time.perf_counter()

    # Optional Chrome trace export (last finished profile)
    if save_chrome and do_profile:
        trace_name = f"{'mlp_only' if mlp_path=='experts' else 'router_plus_mlp'}_rank{rank}.json"
        prof.export_chrome_trace(os.path.join(outdir, trace_name))
        if rank == 0:
            print(f"[bench] chrome trace saved: {os.path.join(outdir, trace_name)}")

    # Print simple numbers so you also get a quick sanity check
    wall_ms = (t1 - t0) * 1000.0
    ms_per = wall_ms / max(1, total_iters)
    toks_s = (S * B) / (ms_per / 1000.0)
    if rank == 0:
        print("=== MLP microbench (sanity numbers) ===")
        print(f"S,B,H:                {S},{B},{H}")
        print(f"Iters:                {total_iters}  (profile [{start_it}..{stop_it}] active)")
        print(f"Avg latency:          {ms_per:.3f} ms/iter")
        print(f"Throughput:           {toks_s:,.0f} tokens/sec  (tokens = S*B)")
        print("Trace is in TensorBoard (from on_trace_ready); set SAVE_CHROME=1 for a JSON too.")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

if __name__ == "__main__":
    main()