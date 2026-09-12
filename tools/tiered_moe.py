"""
tiered_moe.py -- one MoE call over an arena whose experts are stored in MORE THAN ONE format.

Why this exists. The routed experts are 288.8 GB at the checkpoint's FP4 and the box has ~100 GB
to give them, so something has to go. The two options that were already measured are "stream the
misses off NVMe" (full quality, NVMe-bound) and "drop the cold experts" (fast, and free generation
degenerates into a repeated phrase, because dropping an expert does not degrade the router's top-6,
it *replaces* it). This module is the third: keep every expert resident and spend the bit budget
unevenly -- the hot ones at the checkpoint's own FP4, the cold tail at 2 bits. A coarse expert is
still the right expert.

The arena is one flat slot space cut into per-format ranges. `moe_forward_tiered` runs one grouped
kernel per non-empty range over the pairs that land in it, all of them accumulating into the same
`parts` buffer, and sums once at the end. Pairs outside a range are masked with the routing's own
-1 sentinel, so each kernel skips them at block granularity.
"""

from __future__ import annotations

import torch
import triton

import fp4_moe as F4
import cb3_moe as C3
import cb3 as CB3
import cb2half as CBH
import fp4half as FPH
from fp4_moe import DIM, INTER, build_routing, _pick_bm


def build_routing_masked(local: torch.Tensor, BM: int):
    """`build_routing_small` for a pair list that is mostly -1.

    The upstream router groups the -1 pairs like any other slot, and its "a slot never has more
    than BM pairs at this size" invariant does not hold for that group: with a 16-pair block and
    21 masked pairs the writes run off the end of block 0 and corrupt block 1. Here the masked
    pairs are sorted to the end and scattered to a discard row instead of a block, so they take
    part in nothing. Shapes stay static and no value reaches the host, so this is still
    graph-capturable.
    """
    T, K = local.shape
    P = T * K
    dev = local.device
    flat = local.reshape(-1).to(torch.int32)
    BIG = torch.iinfo(torch.int32).max
    key = torch.where(flat >= 0, flat, torch.full_like(flat, BIG))
    order = torch.argsort(key, stable=True)
    ss = key[order]
    live = ss < BIG
    first = torch.ones(P, dtype=torch.bool, device=dev)
    first[1:] = ss[1:] != ss[:-1]
    first &= live
    blk = torch.cumsum(first.to(torch.int32), 0) - 1
    ar = torch.arange(P, device=dev, dtype=torch.int32)
    start = torch.cummax(torch.where(first, ar, torch.zeros_like(ar)), 0).values
    rank = ar - start
    pair_dst = torch.where(live, blk * BM + rank, torch.full_like(blk, P * BM))
    block_pair = torch.full((P * BM + 1,), -1, dtype=torch.int32, device=dev)
    block_pair.scatter_(0, pair_dst.long(), order.to(torch.int32))
    slot_dst = torch.where(live, blk, torch.full_like(blk, P))
    block_slot = torch.full((P + 1,), -1, dtype=torch.int32, device=dev)
    block_slot.scatter_(0, slot_dst.long(), ss.to(torch.int32))
    return block_slot[:P], block_pair[:P * BM], P


class TierSpec:
    """One format's slice of the flat slot space."""

    def __init__(self, fmt: str, base: int, slots: int, arena):
        self.fmt, self.base, self.slots, self.arena = fmt, base, slots, arena

    @property
    def end(self) -> int:
        return self.base + self.slots

    def __repr__(self) -> str:
        return f"TierSpec({self.fmt}, {self.base}:{self.end})"


class TieredArena:
    """A flat slot space backed by one arena per format.

    `tiers` is a list of (format, n_slots) in the order they occupy the space, e.g.
    [("fp4", 1200), ("cb2", 9000)] -> global slot 0..1199 is FP4, 1200..10199 is CB2.
    """

    FMT = {
        "fp4": (F4.ExpertArena, None),
        "cb3": (C3.CB3ArenaV2, C3.CB3_BYTES_PER_SLOT),
        "cb2": (C3.CB2ArenaV2, C3.CB2_BYTES_PER_SLOT),
        "cb2h": (CBH.CB2HalfArena, None),
        "fp4h": (FPH.FP4HalfArena, None),
    }

    def __init__(self, tiers, device="cuda", sims=None, inter_h: int = 1280):
        self.device = torch.device(device)
        self.tiers: list[TierSpec] = []
        base = 0
        for fmt, n in tiers:
            if n <= 0:
                continue
            cls = self.FMT[fmt][0]
            a = cls(n, self.device, inter_h=inter_h) if fmt in ("cb2h", "fp4h") else cls(n, self.device)
            if fmt in ("cb2", "cb3", "cb2h") and sims:
                a.sim = sims.get("cb2" if fmt == "cb2h" else fmt)
            self.tiers.append(TierSpec(fmt, base, n, a))
            base += n
        self.slots = base

    @property
    def bytes_per_slot(self) -> int:
        """The mean over the flat slot space. Callers that size or log the arena want one number;
        `bytes_by_fmt` has the breakdown."""
        return self.total_bytes() // max(1, self.slots)

    @property
    def bytes_by_fmt(self) -> dict:
        return [(t.fmt, t.slots, t.arena.bytes_per_slot) for t in self.tiers]

    def total_bytes(self) -> int:
        return sum(t.slots * t.arena.bytes_per_slot for t in self.tiers)

    def plan_str(self) -> str:
        return " + ".join(f"{t.slots} {t.fmt} @ {t.arena.bytes_per_slot / 1e6:.2f} MB "
                          f"({t.slots * t.arena.bytes_per_slot / 1e9:.1f} GB)" for t in self.tiers)

    def tier_of(self, gslot: int) -> TierSpec:
        for t in self.tiers:
            if t.base <= gslot < t.end:
                return t
        raise IndexError(gslot)

    def load_slot(self, gslot: int, w1, s1, w2, s2, w3, s3, **kw):
        t = self.tier_of(gslot)
        t.arena.load_slot(gslot - t.base, w1, s1, w2, s2, w3, s3, **kw)



N_ROUTED = 15360


def tail_bytes(inter_h: int, tail_fmt: str = "cb2h") -> int:
    import cb2half as _CBH
    import fp4half as _FPH
    return (_FPH.bytes_per_slot(inter_h) if tail_fmt == "fp4h" else _CBH.bytes_per_slot(inter_h))


def plan_all_resident(n_fp4: int, n_cb2: int, inter_h: int = 1280, transient_slots: int = 8,
                      n_total: int = N_ROUTED, tail_fmt: str = "cb2h"):
    """Every routed expert resident: the hottest `n_fp4` in the checkpoint's own FP4, the next
    `n_cb2` at full-width 2 bits, and the whole remaining tail at half-width 2 bits.

    This is the configuration the recipe's own dichotomy says does not exist -- neither streaming
    nor pruning. Nothing is dropped, so the router's top-6 is always the real top-6; what varies
    is only how coarsely each expert is stored, and the coarsest tier is the one the trace says is
    reached least often.
    """
    n_tail = n_total - n_fp4 - n_cb2
    assert n_tail >= 0, (n_fp4, n_cb2, n_total)
    # COLDEST FIRST. `ExpertStore._lru_slot_for` hands out `free_lru.pop()`, i.e. the HIGHEST free
    # slot, so `warm_start`'s ranked list fills the slot space from the top down: the hottest
    # expert gets the last slot. Laying the tiers out hottest-last is what puts it in the FP4 one.
    # The transient ring is the tail of the space, so it merges into that same FP4 tier and a miss
    # (which cannot happen while every expert is resident) would land in the checkpoint's format.
    tiers = [(tail_fmt, n_tail), ("cb2", n_cb2), ("fp4", n_fp4 + max(0, transient_slots))]
    return [(f, n) for f, n in tiers if n > 0]


def all_resident_bytes(n_fp4: int, n_cb2: int, inter_h: int = 1280, n_total: int = N_ROUTED,
                       tail_fmt: str = "cb2h") -> float:
    return (n_fp4 * 18800640 + n_cb2 * C3.CB2_BYTES_PER_SLOT
            + (n_total - n_fp4 - n_cb2) * tail_bytes(inter_h, tail_fmt))


def fit_all_resident(budget_bytes: float, inter_h: int = 1280, fp4_share: float = 0.4,
                     n_total: int = N_ROUTED, tail_fmt: str = "cb2h"):
    """Largest (n_fp4, n_cb2) that fits `budget_bytes`, splitting the headroom above the all-tail
    cost between the two upgrades by `fp4_share` of the SPARE bytes."""
    import cb2half as _CBH
    tail = tail_bytes(inter_h, tail_fmt)
    base = n_total * tail
    spare = budget_bytes - base
    if spare <= 0:
        raise RuntimeError(
            f"all-resident needs {base / 1e9:.1f} GB just to hold every expert at half-width "
            f"{inter_h}/2304 two-bit, and the arena budget is {budget_bytes / 1e9:.1f} GB. "
            f"Lower DSV41_TIER_INTER_H (1024 costs {n_total * _CBH.bytes_per_slot(1024) / 1e9:.1f} GB, "
            f"768 costs {n_total * _CBH.bytes_per_slot(768) / 1e9:.1f} GB), raise ARENA_GB, or use "
            f"DSV41_TIER_MODE=stream.")
    d_fp4 = 18800640 - tail
    d_cb2 = C3.CB2_BYTES_PER_SLOT - tail
    n_fp4 = int(spare * fp4_share // d_fp4)
    n_cb2 = int(spare * (1.0 - fp4_share) // d_cb2)
    while all_resident_bytes(n_fp4, n_cb2, inter_h, n_total, tail_fmt) > budget_bytes and (n_fp4 or n_cb2):
        if n_cb2 > 0:
            n_cb2 -= 16
        else:
            n_fp4 -= 16
        n_cb2 = max(0, n_cb2); n_fp4 = max(0, n_fp4)
    return n_fp4, n_cb2


def plan_tiers(arena_bytes: float, fp4_frac: float, fmt_cold: str = "cb2",
               transient_slots: int = 0):
    """Split a byte budget between the checkpoint's own FP4 and one cheaper format.

    `fp4_frac` is the share of the BYTES, not of the experts: at 0.5 the FP4 tier holds half the
    arena and, being 1.88x the size per expert, about a third of its experts.

    `transient_slots` reserves a THIRD tier, in FP4, at the very end of the slot space.
    `ExpertStore` hands out the tail of the space as its streaming ring, and a slot in the cold
    tier would make every NVMe miss pay a GPU re-pack to 2 bits before it could be used. Keeping
    the ring in the checkpoint's own format makes a miss a plain copy -- and a streamed expert is
    then exact, which is the point of not pruning.
    """
    cold_bytes = {"cb2": C3.CB2_BYTES_PER_SLOT, "cb3": C3.CB3_BYTES_PER_SLOT}[fmt_cold]
    hot_bytes = 18800640
    budget = max(0.0, arena_bytes - transient_slots * hot_bytes)
    n_hot = int(budget * fp4_frac // hot_bytes)
    n_cold = int(budget * (1.0 - fp4_frac) // cold_bytes)
    # coldest first: see plan_all_resident for why the slot space is filled from the top down.
    return [(f, n) for f, n in ((fmt_cold, n_cold), ("fp4", n_hot + max(0, transient_slots))) if n > 0]


# --------------------------------------------------------------------------- the fused forward

def _run_fp4(x, bs, bp, NB, weights_flat, arena, h, parts, BM, T, K, limit):
    bn1, nw1, ns1 = F4._UP_CFG[BM]
    bn2, nw2, ns2 = F4._DOWN_CFG[BM]
    F4._moe_up_kernel[(NB, INTER // bn1)](
        x, arena.w1, arena.s1, arena.w3, arena.s3, h, weights_flat, bs, bp,
        x.stride(0), h.stride(0), float(limit),
        TOPK=K, N=INTER, K=DIM, BM=BM, BN=bn1, num_warps=nw1, num_stages=ns1)
    F4._moe_down_kernel[(NB, DIM // bn2)](
        h, arena.w2, arena.s2, parts, bs, bp, h.stride(0), parts.stride(0),
        TOPK=K, N=DIM, K=INTER, BM=BM, BN=bn2, NTOK=T, num_warps=nw2, num_stages=ns2)


def _run_cb3(x, bs, bp, NB, weights_flat, arena, h, parts, BM, T, K, limit):
    bn1, nw1, ns1 = C3.CB3_UP_CFG[BM]
    bn2, nw2, ns2 = C3.CB3_DOWN_CFG[BM]
    C3._cb3v3_up_kernel[(NB, INTER // bn1)](
        x, arena.w1_lo, arena.w1_hi, arena.w1_cb, arena.s1,
        arena.w3_lo, arena.w3_hi, arena.w3_cb, arena.s3, h,
        weights_flat, bs, bp, x.stride(0), h.stride(0), float(limit),
        TOPK=K, N=INTER, K=DIM, BM=BM, BN=bn1,
        NB512=CB3.block_plan(DIM)[0], NB256=CB3.block_plan(DIM)[1], num_warps=nw1, num_stages=ns1)
    C3._cb3v3_down_kernel[(NB, DIM // bn2)](
        h, arena.w2_lo, arena.w2_hi, arena.w2_cb, arena.s2, parts, bs, bp,
        h.stride(0), parts.stride(0), TOPK=K, N=DIM, K=INTER, BM=BM, BN=bn2, NTOK=T,
        NB512=CB3.block_plan(INTER)[0], NB256=CB3.block_plan(INTER)[1], num_warps=nw2, num_stages=ns2)


def _run_cb2(x, bs, bp, NB, weights_flat, arena, h, parts, BM, T, K, limit):
    bn1, nw1, ns1 = C3.CB2_UP_CFG[BM]
    bn2, nw2, ns2 = C3.CB2_DOWN_CFG[BM]
    C3._cb2_up_kernel[(NB, INTER // bn1)](
        x, arena.w1_lo, arena.w1_cb, arena.s1, arena.w3_lo, arena.w3_cb, arena.s3, h,
        weights_flat, bs, bp, x.stride(0), h.stride(0), float(limit),
        TOPK=K, N=INTER, K=DIM, BM=BM, BN=bn1,
        NB512=CB3.block_plan(DIM)[0], NB256=CB3.block_plan(DIM)[1], num_warps=nw1, num_stages=ns1)
    C3._cb2_down_kernel[(NB, DIM // bn2)](
        h, arena.w2_lo, arena.w2_cb, arena.s2, parts, bs, bp,
        h.stride(0), parts.stride(0), TOPK=K, N=DIM, K=INTER, BM=BM, BN=bn2, NTOK=T,
        NB512=CB3.block_plan(INTER)[0], NB256=CB3.block_plan(INTER)[1], num_warps=nw2, num_stages=ns2)


def _run_fp4h(x, bs, bp, NB, weights_flat, arena, h, parts, BM, T, K, limit):
    """The FP4 kernels at a narrower intermediate width. Nothing about the weights changes -- the
    kept channels are the checkpoint's own e2m1 codes and UE8M0 scales -- so this is the same
    already-tested path with `N`/`K` set to `inter_h`."""
    IH = arena.inter_h
    bn1, nw1, ns1 = F4._UP_CFG[BM]
    bn2, nw2, ns2 = F4._DOWN_CFG[BM]
    F4._moe_up_kernel[(NB, IH // bn1)](
        x, arena.w1, arena.s1, arena.w3, arena.s3, h, weights_flat, bs, bp,
        x.stride(0), h.stride(0), float(limit),
        TOPK=K, N=IH, K=DIM, BM=BM, BN=bn1, num_warps=nw1, num_stages=ns1)
    F4._moe_down_kernel[(NB, DIM // bn2)](
        h, arena.w2, arena.s2, parts, bs, bp, h.stride(0), parts.stride(0),
        TOPK=K, N=DIM, K=IH, BM=BM, BN=bn2, NTOK=T, num_warps=nw2, num_stages=ns2)


def _run_cb2h(x, bs, bp, NB, weights_flat, arena, h, parts, BM, T, K, limit):
    """Same kernels as `_run_cb2`; only the intermediate width changes. `h` is allocated at the
    full 2304 and this tier writes and reads its first `inter_h` columns, so the row stride stays
    the shared one and no separate buffer is needed."""
    IH = arena.inter_h
    bn1, nw1, ns1 = C3.CB2_UP_CFG[BM]
    bn2, nw2, ns2 = C3.CB2_DOWN_CFG[BM]
    C3._cb2_up_kernel[(NB, IH // bn1)](
        x, arena.w1_lo, arena.w1_cb, arena.s1, arena.w3_lo, arena.w3_cb, arena.s3, h,
        weights_flat, bs, bp, x.stride(0), h.stride(0), float(limit),
        TOPK=K, N=IH, K=DIM, BM=BM, BN=bn1,
        NB512=CB3.block_plan(DIM)[0], NB256=CB3.block_plan(DIM)[1], num_warps=nw1, num_stages=ns1)
    C3._cb2_down_kernel[(NB, DIM // bn2)](
        h, arena.w2_lo, arena.w2_cb, arena.s2, parts, bs, bp,
        h.stride(0), parts.stride(0), TOPK=K, N=DIM, K=IH, BM=BM, BN=bn2, NTOK=T,
        NB512=CB3.block_plan(IH)[0], NB256=CB3.block_plan(IH)[1], num_warps=nw2, num_stages=ns2)


_RUN = {"fp4": _run_fp4, "cb3": _run_cb3, "cb2": _run_cb2, "cb2h": _run_cb2h, "fp4h": _run_fp4h}


def moe_forward_tiered(x: torch.Tensor, slots: torch.Tensor, weights: torch.Tensor,
                       arena: TieredArena, swiglu_limit: float = 10.0,
                       block_m: int | None = None) -> torch.Tensor:
    """Same contract as `fp4_moe.moe_forward`, over a multi-format arena.

    `slots` holds GLOBAL slot ids (or -1 for "no expert"). `parts` is zeroed once because a pair
    that belongs to another tier is skipped by this tier's kernel and would otherwise be read
    uninitialised by the final sum.
    """
    assert x.dtype == torch.bfloat16 and x.shape[1] == DIM and x.is_contiguous()
    T, K = slots.shape
    P = T * K
    dev = x.device
    BM = block_m or _pick_bm(P)
    wgt = weights.reshape(-1)
    if wgt.dtype != torch.float32 or not wgt.is_contiguous():
        wgt = wgt.float().contiguous()
    h = torch.empty((P, INTER), dtype=torch.bfloat16, device=dev)
    parts = torch.zeros((P, DIM), dtype=torch.float32, device=dev)
    gs = slots.to(torch.int32)
    # One routing build for the whole call. The blocks it produces are grouped by GLOBAL slot, and
    # the tiers own disjoint ranges of that space, so every block belongs to exactly one tier: a
    # tier's kernel only needs its own block_slot vector, re-based and with the other tiers' blocks
    # set to the -1 the kernels already skip. That is one elementwise op per tier instead of a
    # second argsort/cumsum/scatter chain, and no `.any()` sync anywhere.
    if P <= 64:
        bs, bp, NB = build_routing_masked(gs, BM)
    else:
        bs, bp, NB = build_routing(gs, arena.slots, BM)
    for t in arena.tiers:
        mine = (bs >= t.base) & (bs < t.end)
        bs_t = torch.where(mine, bs - t.base, torch.full_like(bs, -1))
        _RUN[t.fmt](x, bs_t, bp, NB, wgt, t.arena, h, parts, BM, T, K, swiglu_limit)
    return parts.view(K, T, DIM).sum(dim=0).to(torch.bfloat16)
