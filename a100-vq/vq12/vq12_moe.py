"""Grouped MoE kernel for the VQ12 expert format (a100-vq/vq12.py).

Same slot geometry, same rate and the same [BN, 64] + [BN, 32] loads per 256 K as CB3 v2
(tools/cb3_moe.py), so the arena, the store and the routing are unchanged. The only thing that
differs is how a scale group's packed-FP4 byte tile is rebuilt:

    CB3   a 3-bit index per weight, looked up in the ROW's 8-entry codebook held in a 32-bit
          register:            ne = (cw >> (ie * 4)) & 15
    VQ12  a 12-bit index per GROUP OF FOUR weights, looked up in a global 4096-entry table:
                               v  = tl.load(lut + idx)       -> two packed bytes at once

That is the whole change, and it is worth halving the quantisation damage: over all 40 layers,
unpruned, CB3 is +8.54 % wikitext PPL against the FP4 checkpoint and VQ12 is +4.23 % (code +2.17 %
vs +0.97 %) at identical bytes. CB3's per-row subset search is already optimal for its format; the
gap is scalar vs vector quantisation.

Built on the v2 path rather than v3 (whose inline-PTX decode is specific to the 3-bit-index
arithmetic), so it starts from v2's speed, not v3's.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

import cb3_moe as C3
from cb3_moe import _split2, _split8, CB3Arena
from fp4_moe import DIM, INTER, _chunk_dot, _pick_bm, _ue8m0, build_routing


@triton.jit
def _split8w(t, BN: tl.constexpr, W: tl.constexpr):
    """[BN, 8W] -> eight contiguous [BN, W] tiles."""
    a, b = _split2(t, BN, 4 * W)
    a0, a1 = _split2(a, BN, 2 * W)
    b0, b1 = _split2(b, BN, 2 * W)
    t0, t1 = _split2(a0, BN, W)
    t2, t3 = _split2(a1, BN, W)
    t4, t5 = _split2(b0, BN, W)
    t6, t7 = _split2(b1, BN, W)
    return t0, t1, t2, t3, t4, t5, t6, t7


@triton.jit
def _grp_packed_vq(Lk, Hm, lut_ptr, BN: tl.constexpr):
    """One scale group's packed-FP4 byte tile [BN, 16] from its 8 index bytes and 4 nibble bytes.

    Byte e of the output holds K offsets 2e (low nibble) and 2e+1 (high), which is what
    `_chunk_dot` expects; a codebook entry is four E2M1 codes = exactly two such bytes.
    """
    # & 0xFF because a uint8 tile converts to int32 sign-extended; CB3 never notices (it masks the
    # result with & 3) but a 12-bit index uses the whole byte
    h = Hm.to(tl.int32) & 0xFF
    hn = tl.interleave(h & 15, (h >> 4) & 15)                     # [BN, 8] high nibbles
    idx = (Lk.to(tl.int32) & 0xFF) | (hn << 8)                    # [BN, 8] entry 0..4095
    v = tl.load(lut_ptr + idx)                                    # [BN, 8] the two bytes, packed
    return tl.interleave(v & 0xFF, (v >> 8) & 0xFF).to(tl.uint8)


@triton.jit
def _chunk_dot_lut(x_base, xk, mask_m, packed, scale_u8, tab_ptr):
    """`fp4_moe._chunk_dot` without its inline-asm decode.

    `_fp4_decode` is `tl.inline_asm_elementwise(..., pack=4)`, which reads four elements out of one
    32-bit register and therefore depends on the tile's register layout. CB3 hands it a tile derived
    from a plain load; a tile built with `tl.interleave` is laid out differently and the asm reads
    the wrong lanes -- silently, because `tl.store` of the same tile is still correct. A 16-entry
    fp16 table indexed by the nibble has no such requirement.
    """
    xe = tl.load(x_base + xk, mask=mask_m, other=0.0).to(tl.float16)
    xo = tl.load(x_base + xk + 1, mask=mask_m, other=0.0).to(tl.float16)
    p8 = packed.to(tl.int32) & 0xFF
    we = tl.load(tab_ptr + (p8 & 0xF))
    wo = tl.load(tab_ptr + (p8 >> 4))
    p = tl.dot(xe, tl.trans(we))
    p = tl.dot(xo, tl.trans(wo), acc=p)
    return p * _ue8m0(scale_u8)[None, :]


@triton.jit
def _vq_block_dot(x_base, xk, mask_m, lo_ptr, hi_ptr, s_ptr, lut_ptr, tab_ptr, BN: tl.constexpr):
    """256 logical K = 8 scale groups: 64 index bytes + 32 nibble bytes + 8 scale bytes per row."""
    L = tl.load(lo_ptr)   # [BN, 64]
    H = tl.load(hi_ptr)   # [BN, 32]
    L0, L1, L2, L3, L4, L5, L6, L7 = _split8w(L, BN, 8)
    H0, H1, H2, H3, H4, H5, H6, H7 = _split8w(H, BN, 4)
    s0, s1, s2, s3, s4, s5, s6, s7 = _split8(tl.load(s_ptr), BN)
    acc = _chunk_dot_lut(x_base, xk, mask_m, _grp_packed_vq(L0, H0, lut_ptr, BN), s0, tab_ptr)
    acc += _chunk_dot_lut(x_base + 32, xk, mask_m, _grp_packed_vq(L1, H1, lut_ptr, BN), s1, tab_ptr)
    acc += _chunk_dot_lut(x_base + 64, xk, mask_m, _grp_packed_vq(L2, H2, lut_ptr, BN), s2, tab_ptr)
    acc += _chunk_dot_lut(x_base + 96, xk, mask_m, _grp_packed_vq(L3, H3, lut_ptr, BN), s3, tab_ptr)
    acc += _chunk_dot_lut(x_base + 128, xk, mask_m, _grp_packed_vq(L4, H4, lut_ptr, BN), s4, tab_ptr)
    acc += _chunk_dot_lut(x_base + 160, xk, mask_m, _grp_packed_vq(L5, H5, lut_ptr, BN), s5, tab_ptr)
    acc += _chunk_dot_lut(x_base + 192, xk, mask_m, _grp_packed_vq(L6, H6, lut_ptr, BN), s6, tab_ptr)
    acc += _chunk_dot_lut(x_base + 224, xk, mask_m, _grp_packed_vq(L7, H7, lut_ptr, BN), s7, tab_ptr)
    return acc


@triton.jit
def _vq_up_kernel(
    x_ptr, lo1_ptr, hi1_ptr, s1_ptr, lo3_ptr, hi3_ptr, s3_ptr, h_ptr, lut_ptr, tab_ptr,
    wgt_ptr, block_slot_ptr, block_pair_ptr,
    stride_x, stride_h, limit,
    TOPK: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
):
    KL: tl.constexpr = K // 4
    KH: tl.constexpr = K // 8
    SG: tl.constexpr = K // 32
    mb = tl.program_id(0)
    nb = tl.program_id(1)
    slot = tl.load(block_slot_ptr + mb)
    if slot < 0:
        return
    slot = slot.to(tl.int64)
    offs_m = tl.load(block_pair_ptr + mb * BM + tl.arange(0, BM))
    mask_m = offs_m >= 0
    offs_m = tl.where(mask_m, offs_m, 0)
    tok = (offs_m // TOPK).to(tl.int64)
    offs_n = nb * BN + tl.arange(0, BN)
    x_base = x_ptr + tok[:, None] * stride_x
    xk = 2 * tl.arange(0, 16)[None, :]
    lo1 = lo1_ptr + slot * (N * KL) + offs_n[:, None] * KL + tl.arange(0, 64)[None, :]
    hi1 = hi1_ptr + slot * (N * KH) + offs_n[:, None] * KH + tl.arange(0, 32)[None, :]
    lo3 = lo3_ptr + slot * (N * KL) + offs_n[:, None] * KL + tl.arange(0, 64)[None, :]
    hi3 = hi3_ptr + slot * (N * KH) + offs_n[:, None] * KH + tl.arange(0, 32)[None, :]
    s1t = s1_ptr + slot * (N * SG) + offs_n[:, None] * SG + tl.arange(0, 8)[None, :]
    s3t = s3_ptr + slot * (N * SG) + offs_n[:, None] * SG + tl.arange(0, 8)[None, :]
    acc_g = tl.zeros([BM, BN], dtype=tl.float32)
    acc_u = tl.zeros([BM, BN], dtype=tl.float32)
    for b in range(0, K // 256):
        acc_g += _vq_block_dot(x_base + b * 256, xk, mask_m[:, None], lo1 + b * 64, hi1 + b * 32,
                               s1t + b * 8, lut_ptr, tab_ptr, BN)
        acc_u += _vq_block_dot(x_base + b * 256, xk, mask_m[:, None], lo3 + b * 64, hi3 + b * 32,
                               s3t + b * 8, lut_ptr, tab_ptr, BN)
    gate = tl.minimum(acc_g, limit)
    up = tl.minimum(tl.maximum(acc_u, -limit), limit)
    wgt = tl.load(wgt_ptr + offs_m, mask=mask_m, other=0.0)
    h = gate * tl.sigmoid(gate) * up * wgt[:, None]
    tl.store(h_ptr + offs_m[:, None] * stride_h + offs_n[None, :], h.to(tl.bfloat16), mask=mask_m[:, None])


@triton.jit
def _vq_down_kernel(
    h_ptr, lo2_ptr, hi2_ptr, s2_ptr, y_ptr, lut_ptr, tab_ptr,
    block_slot_ptr, block_pair_ptr,
    stride_h, stride_y,
    TOPK: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, NTOK: tl.constexpr,
):
    KL: tl.constexpr = K // 4
    KH: tl.constexpr = K // 8
    SG: tl.constexpr = K // 32
    mb = tl.program_id(0)
    nb = tl.program_id(1)
    slot = tl.load(block_slot_ptr + mb)
    if slot < 0:
        return
    slot = slot.to(tl.int64)
    offs_m = tl.load(block_pair_ptr + mb * BM + tl.arange(0, BM))
    mask_m = offs_m >= 0
    offs_m = tl.where(mask_m, offs_m, 0)
    offs_n = nb * BN + tl.arange(0, BN)
    h_base = h_ptr + offs_m[:, None].to(tl.int64) * stride_h
    xk = 2 * tl.arange(0, 16)[None, :]
    lo2 = lo2_ptr + slot * (N * KL) + offs_n[:, None] * KL + tl.arange(0, 64)[None, :]
    hi2 = hi2_ptr + slot * (N * KH) + offs_n[:, None] * KH + tl.arange(0, 32)[None, :]
    s2t = s2_ptr + slot * (N * SG) + offs_n[:, None] * SG + tl.arange(0, 8)[None, :]
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for b in range(0, K // 256):
        acc += _vq_block_dot(h_base + b * 256, xk, mask_m[:, None], lo2 + b * 64, hi2 + b * 32,
                             s2t + b * 8, lut_ptr, tab_ptr, BN)
    row = ((offs_m % TOPK) * NTOK + offs_m // TOPK).to(tl.int64)
    tl.store(y_ptr + row[:, None] * stride_y + offs_n[None, :], acc, mask=mask_m[:, None])


class VQ12Arena(CB3Arena):
    """CB3's tensors exactly (the `*_cb` planes go unused: the codebook is global)."""

    def __init__(self, slots: int, device="cuda"):
        super().__init__(slots, device)
        self.vq = None      # a100-vq/vq12.py VQ12, set by the caller
        self.lut = None     # int32 [4096], entry -> the two packed-FP4 bytes
        self.tab = None     # fp16 [16], the E2M1 grid

    def attach(self, vq):
        self.vq = vq
        self.lut = vq.lut.to(self.device).contiguous()
        from vq12 import FP4_VALS
        self.tab = FP4_VALS.to(self.device).to(torch.float16).contiguous()
        return self

    def load_slot(self, slot: int, w1, s1, w2, s2, w3, s3, non_blocking: bool = False, sim=None) -> None:
        dev = self.device
        for (w, s, lo_t, hi_t, cb_t, s_t) in ((w1, s1, self.w1_lo, self.w1_hi, self.w1_cb, self.s1),
                                              (w3, s3, self.w3_lo, self.w3_hi, self.w3_cb, self.s3),
                                              (w2, s2, self.w2_lo, self.w2_hi, self.w2_cb, self.s2)):
            wg = w.view(torch.uint8).to(dev, non_blocking=non_blocking)
            sg = s.view(torch.uint8).to(dev, non_blocking=non_blocking)
            lo, hi, cb = self.vq.pack(wg, sg)
            lo_t[slot].copy_(lo); hi_t[slot].copy_(hi); cb_t[slot].copy_(cb); s_t[slot].copy_(sg)

    def dequant_slot(self, slot: int):
        return tuple(self.vq.dequant(lo[slot], hi[slot], s[slot]) for lo, hi, s in (
            (self.w1_lo, self.w1_hi, self.s1), (self.w2_lo, self.w2_hi, self.s2),
            (self.w3_lo, self.w3_hi, self.s3)))


def moe_forward_vq(x: torch.Tensor, slots: torch.Tensor, weights: torch.Tensor, arena: VQ12Arena,
                   swiglu_limit: float = 10.0, block_m: int | None = None,
                   cfg_up=None, cfg_down=None) -> torch.Tensor:
    assert x.dtype == torch.bfloat16 and x.shape[1] == DIM and x.is_contiguous()
    assert arena.lut is not None, "VQ12Arena.attach(vq) was never called"
    T, K = slots.shape
    P = T * K
    BM = block_m or _pick_bm(P)
    bn1, nw1, ns1 = cfg_up or C3._UP_CFG[BM]
    bn2, nw2, ns2 = cfg_down or C3._DOWN_CFG[BM]
    block_slot, block_pair, NB = build_routing(slots, arena.slots, BM)
    wgt = weights.reshape(-1)
    if wgt.dtype != torch.float32 or not wgt.is_contiguous():
        wgt = wgt.float().contiguous()
    h = torch.empty((P, INTER), dtype=torch.bfloat16, device=x.device)
    parts = torch.empty((P, DIM), dtype=torch.float32, device=x.device)
    _vq_up_kernel[(NB, INTER // bn1)](
        x, arena.w1_lo, arena.w1_hi, arena.s1, arena.w3_lo, arena.w3_hi, arena.s3, h, arena.lut, arena.tab,
        wgt, block_slot, block_pair, x.stride(0), h.stride(0), float(swiglu_limit),
        TOPK=K, N=INTER, K=DIM, BM=BM, BN=bn1, num_warps=nw1, num_stages=ns1)
    _vq_down_kernel[(NB, DIM // bn2)](
        h, arena.w2_lo, arena.w2_hi, arena.s2, parts, arena.lut, arena.tab, block_slot, block_pair,
        h.stride(0), parts.stride(0), TOPK=K, N=DIM, K=INTER, BM=BM, BN=bn2, NTOK=T,
        num_warps=nw2, num_stages=ns2)
    return parts.view(K, T, DIM).sum(dim=0).to(torch.bfloat16)
