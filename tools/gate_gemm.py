"""y = x @ W^T with W stored in bf16 and every arithmetic step in fp32, for the router gate.

`ffn.gate.weight` is BF16 [384, 5120] in the checkpoint and the engine promotes it with f32(), so
the decode step reads 7.68 MB per layer where 3.84 MB holds the same values. Rounding the GEMM's
OUTPUT to bf16 is what moved 6.67 % of the layer-0 top-6 picks, so here only the storage is bf16:
the loaded tile is upcast in registers and the accumulation, the output and the top-k that consumes
it stay fp32.

Structure copied from tools/fp32_skinny.py -- ieee input_precision (no tf32), split-K with a
deterministic second-pass reduce rather than fp32 atomics, and a per-program K span that does not
depend on M, so a row's summation order is the same in any call.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gate_kernel(X, W, Y, M, N, K,
                 stride_xm, stride_wn, stride_ys, stride_ym,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                 SPLIT_K: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rm = tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = rm < M
    n_mask = rn < N
    k_per = tl.cdiv(tl.cdiv(K, BLOCK_K), SPLIT_K) * BLOCK_K
    k_lo = pid_k * k_per
    k_hi = tl.minimum(k_lo + k_per, K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(k_lo, k_hi, BLOCK_K):
        kk = k0 + tl.arange(0, BLOCK_K)
        km = kk < k_hi
        x = tl.load(X + rm[:, None] * stride_xm + kk[None, :],
                    mask=m_mask[:, None] & km[None, :], other=0.0).to(tl.float32)
        # the one line that differs from fp32_skinny: the weight arrives in bf16 and is widened
        # here, so the multiply and the accumulator are fp32 on exactly the stored values.
        w = tl.load(W + rn[:, None] * stride_wn + kk[None, :],
                    mask=n_mask[:, None] & km[None, :], other=0.0).to(tl.float32)
        acc += tl.dot(x, tl.trans(w), input_precision="ieee", out_dtype=tl.float32)
    tl.store(Y + pid_k * stride_ys + rm[:, None] * stride_ym + rn[None, :], acc,
             mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _gate_reduce(P, Y, M, N, stride_ps, stride_pm, stride_ym,
                 SPLIT_K, SPLIT_P: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    si = tl.arange(0, SPLIT_P)
    rm = tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    msk = (si[:, None, None] < SPLIT_K) & (rm[None, :, None] < M) & (rn[None, None, :] < N)
    v = tl.load(P + si[:, None, None] * stride_ps + rm[None, :, None] * stride_pm
                + rn[None, None, :], mask=msk, other=0.0)
    o = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(Y + rm[:, None] * stride_ym + rn[None, :], tl.sum(v, 0), mask=o)


BLOCK_M = 16        # tl.dot wants 16 rows; a decode block is 2 and the rest are masked
BLOCK_MP = 8
BLOCK_K = 64
TARGET_CTAS = 48    # one per SM on GB10


def plan(N: int, K: int, block_n: int | None = None, target: int = TARGET_CTAS):
    bn = block_n
    if bn is None:
        bn = 32
        while bn < N and bn < 128:
            bn *= 2
    n_blocks = triton.cdiv(N, bn)
    n_kblocks = triton.cdiv(K, BLOCK_K)
    kb = max(1, -(-n_kblocks // max(1, target // n_blocks)))
    return bn, -(-n_kblocks // kb)


def gate_linear(x: torch.Tensor, w: torch.Tensor, block_n: int | None = None,
                target: int = TARGET_CTAS, num_warps: int = 4, num_stages: int = 3):
    """x [M, K] bf16 or fp32, w [N, K] bf16 -> y [M, N] fp32."""
    assert w.dtype == torch.bfloat16 and w.dim() == 2 and x.dim() == 2
    assert x.stride(1) == 1 and w.stride(1) == 1
    M, K = x.shape
    N = w.size(0)
    assert w.size(1) == K and M <= BLOCK_MP
    bn, sk = plan(N, K, block_n, target)
    y = torch.empty(M, N, dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(N, bn), sk)
    if sk == 1:
        _gate_kernel[grid](x, w, y, M, N, K, x.stride(0), w.stride(0), 0, y.stride(0),
                           BLOCK_M=BLOCK_M, BLOCK_N=bn, BLOCK_K=BLOCK_K, SPLIT_K=1,
                           num_warps=num_warps, num_stages=num_stages)
        return y
    # the reduce reads a [SPLIT_P, BLOCK_M, BLOCK_N] tile with tl.arange, which needs powers of
    # two; N = 384 makes bn * cdiv(N, bn) = 384, so the partial buffer is widened and the columns
    # past N are masked off there as they already were.
    nb = 1 << (bn * triton.cdiv(N, bn) - 1).bit_length()
    p = torch.empty(sk, BLOCK_MP, nb, dtype=torch.float32, device=x.device)
    _gate_kernel[grid](x, w, p, M, N, K, x.stride(0), w.stride(0), p.stride(0), p.stride(1),
                       BLOCK_M=BLOCK_M, BLOCK_N=bn, BLOCK_K=BLOCK_K, SPLIT_K=sk,
                       num_warps=num_warps, num_stages=num_stages)
    _gate_reduce[(1,)](p, y, M, N, p.stride(0), p.stride(1), y.stride(0),
                       sk, SPLIT_P=1 << (sk - 1).bit_length(), BLOCK_M=BLOCK_MP, BLOCK_N=nb,
                       num_warps=8, num_stages=1)
    return y
