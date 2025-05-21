# fused_lores_gemm.py
import torch
import triton
import triton.language as tl

@triton.jit
def fused_lores_gemm(
    # pointers
    X_ptr, W1_ptr, W2_ptr, Y_ptr,
    # sizes
    B, D, K, P,
    # strides
    stride_Xb, stride_Xd,
    stride_W1n, stride_W1d, stride_W1k,
    stride_W2n, stride_W2k, stride_W2p,
    stride_Yn, stride_Yb, stride_Yp,
    # compile-time constants
    BLOCK_B: tl.constexpr, BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_P: tl.constexpr
):
    n_idx = tl.program_id(0)
    b_idx = tl.program_id(1)
    p_idx = tl.program_id(2)

    # compute pointers to this block
    X_block = X_ptr + b_idx * BLOCK_B * stride_Xb
    W1_base = W1_ptr + n_idx * stride_W1n
    W2_base = W2_ptr + n_idx * stride_W2n
    Y_block = Y_ptr + n_idx * stride_Yn + b_idx * BLOCK_B * stride_Yb + p_idx * BLOCK_P * stride_Yp

    # accumulator
    acc = tl.zeros((BLOCK_B, BLOCK_P), dtype=tl.float32)

    for k_off in range(0, K, BLOCK_K):
        # load X slice: [BLOCK_B, BLOCK_D]
        x = tl.load(
            X_block[:, None] + tl.arange(0, BLOCK_D)[None, :] * stride_Xd +
            tl.arange(0, BLOCK_B)[:, None] * stride_Xb
        )
        # load W1 slice: [BLOCK_D, BLOCK_K]
        w1 = tl.load(
            W1_base + tl.arange(0, BLOCK_D)[None, :] * stride_W1d +
            tl.arange(k_off, k_off + BLOCK_K)[:, None] * stride_W1k
        )
        mid = tl.dot(x, w1)  # [BLOCK_B, BLOCK_K]

        # load W2 slice: [BLOCK_K, BLOCK_P]
        w2 = tl.load(
            W2_base + tl.arange(k_off, k_off + BLOCK_K)[:, None] * stride_W2k +
            tl.arange(0, BLOCK_P)[None, :] * stride_W2p
        )
        acc += tl.dot(mid, w2)

    tl.store(
        Y_block[:, None] + tl.arange(0, BLOCK_P)[None, :] * stride_Yp +
        tl.arange(0, BLOCK_B)[:, None] * stride_Yb,
        acc
    )




# choose block sizes to suit your hardware
BLOCK_B = 128
BLOCK_D = 64
BLOCK_K = 32
BLOCK_P = 64

def fused_lores_batched_gemm(X, w1_A, w1_B):
    """
    X: [B, D]
    w1_A: [N, D, K]
    w1_B: [N, K, P]
    returns Y: [N, B, P]
    """
    B, D = X.shape
    N, _, K = w1_A.shape
    _, _, P = w1_B.shape
    Y = torch.empty((N, B, P), device=X.device, dtype=X.dtype)

    grid = (N,
            triton.cdiv(B, BLOCK_B),
            triton.cdiv(P, BLOCK_P))

    fused_lores_gemm[grid](
        X, w1_A, w1_B, Y,
        B, D, K, P,
        X.stride(0), X.stride(1),
        w1_A.stride(0), w1_A.stride(1), w1_A.stride(2),
        w1_B.stride(0), w1_B.stride(1), w1_B.stride(2),
        Y.stride(0), Y.stride(1), Y.stride(2),
        BLOCK_B=BLOCK_B, BLOCK_D=BLOCK_D,
        BLOCK_K=BLOCK_K, BLOCK_P=BLOCK_P
    )
    return Y