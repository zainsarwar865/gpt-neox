# import torch
# import torch.nn as nn
# import megablocks.ops
# from megablocks import grouped_gemm_util as gg


# lora_rank = 8
# num_loras = 16
# num_experts = 8
# hidden_size = 768
# per_expert_ff_dim = hidden_size * 4
# token_batch_size = 73728
# total_loras = num_loras * num_experts 

# num_rows = num_experts * num_loras


# w1_A = torch.nn.Parameter(
#     torch.empty(
#         num_rows,
#         hidden_size * lora_rank,
#         device=torch.cuda.current_device(),
#         dtype=torch.bfloat16,
#     )
# )


# w1_B = torch.nn.Parameter(
#     torch.empty(
#         num_rows,
#         per_expert_ff_dim * lora_rank,
#         device=torch.cuda.current_device(),
#         dtype=torch.bfloat16,
#     )
# )


# w1_A = w1_A.view(total_loras, hidden_size, lora_rank)

# # w1_B = w1_B.view(total_loras, lora_rank, per_expert_ff_dim)
# #w1_AB = torch.einsum('ijk,ikm->ijm', w1_A, w1_B)
# # new:
# # Y = fused_lores_batched_gemm(x, w1_A, w1_B)
# # return Y

# x = torch.randn((token_batch_size, hidden_size)).to(torch.cuda.current_device()).to(torch.bfloat16)
# uni_tokens = token_batch_size // total_loras
# grouped_gemm_batch_sizes = torch.tensor([uni_tokens] * total_loras)
# #res = gg.ops.gmm(gg.ops.gmm(x, w1_A, grouped_gemm_batch_sizes), w1_B, grouped_gemm_batch_sizes)
# print(x.shape, w1_A.shape)
# res = gg.ops.gmm(x, w1_A, grouped_gemm_batch_sizes)




import torch
import torch.nn as nn
import megablocks.ops
from megablocks import grouped_gemm_util as gg


lora_rank = 8
num_loras = 16
num_experts = 8
hidden_size = 768
per_expert_ff_dim = hidden_size * 4
token_batch_size = 73728
total_loras = num_loras * num_experts 

num_rows = num_experts * num_loras


w1 = torch.nn.Parameter(
    torch.empty(
        per_expert_ff_dim * num_experts,
        hidden_size,
        device=torch.cuda.current_device(),
        dtype=torch.bfloat16
    )
)

w1 = w1.view(num_experts, hidden_size, per_expert_ff_dim)


x = torch.randn((token_batch_size, hidden_size)).to(torch.cuda.current_device()).to(torch.bfloat16)
uni_tokens = token_batch_size // num_experts
grouped_gemm_batch_sizes = torch.tensor([uni_tokens] * num_experts)
res  = gg.ops.gmm(x, w1, grouped_gemm_batch_sizes)


# import torch
# import megablocks.ops
# from megablocks import grouped_gemm_util as gg

# # Constants
# num_loras = 16
# num_experts = 8
# hidden_size = 768
# lora_rank = 8
# per_expert_ff_dim = hidden_size * 4
# token_batch_size = 73728
# total_experts = num_loras * num_experts  # 128
# tokens_per_expert = token_batch_size // total_experts

# # Inputs
# x = torch.randn((token_batch_size, hidden_size), dtype=torch.bfloat16, device='cuda')
# w1_A = torch.randn((total_experts, hidden_size, lora_rank), dtype=torch.bfloat16, device='cuda')
# grouped_batch_sizes = torch.tensor([tokens_per_expert] * total_experts)

# # Parallelism setup
# num_streams = 8
# experts_per_stream = total_experts // num_streams

# streams = [torch.cuda.Stream() for _ in range(num_streams)]
# results = [None for _ in range(num_streams)]

# torch.cuda.synchronize()

# for i in range(num_streams):
#     start_expert = i * experts_per_stream
#     end_expert = (i + 1) * experts_per_stream

#     x_start = start_expert * tokens_per_expert
#     x_end = end_expert * tokens_per_expert

#     x_chunk = x[x_start:x_end]                       # (batch_subset, hidden_size)
#     w_chunk = w1_A[start_expert:end_expert]          # (experts_subset, hidden_size, r)
#     bs_chunk = grouped_batch_sizes[start_expert:end_expert]  # (experts_subset,)

#     with torch.cuda.stream(streams[i]):
#         results[i] = gg.ops.gmm(x_chunk, w_chunk, bs_chunk)

# # Wait for all streams to finish
# for s in streams:
#     s.synchronize()

# # Merge the results
# res = torch.cat(results, dim=0)
# print("Final shape:", res.shape)