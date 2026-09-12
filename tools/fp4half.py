"""
fp4half.py -- an expert stored in the CHECKPOINT'S OWN FP4, over a subset of its intermediate
channels.

Why this exists. `cb2half.py` met the 1.43 bit/weight budget by keeping 2 bits per weight and
dropping channels. Measured on real prompts, that loses: the upstream keep-31 % configuration --
which removes 69 % of the experts from the router entirely -- answers "what is the highest mountain
in Japan" correctly and in one language, while the all-resident 2-bit-tail configuration gets the
altitude wrong and drifts from Japanese into Chinese. Upstream had already seen its own 3-bit CB3
format degenerate in free generation; read together, the evidence says this checkpoint's experts do
not survive being re-quantised below FP4, and that a coarsened expert can be worse than an absent
one.

So this format changes the error, not its size. No codebook, no re-quantisation: the kept channels
keep the checkpoint's exact e2m1 codes and their exact UE8M0 scales. What is lost is whole
channels. A tail expert is then "the right expert with some of its channels missing" rather than
"every weight of the right expert rounded to one of four values".

    INTER_H   MB/expert   all 15,360
    512       4.18        64.2 GB
    768       6.27        96.3 GB
    1024      8.36        128.3 GB   (does not fit)

`INTER_H` must be a multiple of 128 (`_moe_down_kernel` walks K in 128-weight quads and needs
K/32 divisible by 4) and of 32 (the UE8M0 scale group, and 32 channels is exactly 16 packed bytes
of a w2 row, so a group boundary is a byte boundary). Channel selection is `cb2half`'s: the scale
mass w1 and w3 carry, read straight out of the exponents.

The kernels are unchanged. `N` and `K` are `tl.constexpr` in `fp4_moe`, so this is the same
already-tested FP4 path at a different width.
"""

from __future__ import annotations

import torch

from fp4_moe import DIM, INTER


def bytes_per_slot(inter_h: int) -> int:
    return (2 * (inter_h * (DIM // 2) + inter_h * (DIM // 32))
            + DIM * (inter_h // 2) + DIM * (inter_h // 32))


class FP4HalfArena:
    """`fp4_moe.ExpertArena` over `inter_h` of the 2304 intermediate channels."""

    def __init__(self, slots: int, device: torch.device | str = "cuda", inter_h: int = 768):
        assert inter_h % 128 == 0 and inter_h <= INTER, inter_h
        self.slots = slots
        self.inter_h = inter_h
        self.n_groups = inter_h // 32
        self.device = torch.device(device)
        u8 = dict(dtype=torch.uint8, device=self.device)
        self.w1 = torch.empty((slots, inter_h, DIM // 2), **u8)
        self.s1 = torch.empty((slots, inter_h, DIM // 32), **u8)
        self.w3 = torch.empty((slots, inter_h, DIM // 2), **u8)
        self.s3 = torch.empty((slots, inter_h, DIM // 32), **u8)
        self.w2 = torch.empty((slots, DIM, inter_h // 2), **u8)
        self.s2 = torch.empty((slots, DIM, inter_h // 32), **u8)
        self.sim = None  # unused; kept so a TieredArena can hand every tier the same kwargs

    @property
    def bytes_per_slot(self) -> int:
        return sum(t[0].numel() for t in (self.w1, self.s1, self.w3, self.s3, self.w2, self.s2))

    def select_groups(self, s1: torch.Tensor, s3: torch.Tensor) -> torch.Tensor:
        """The `n_groups` channel groups to keep, ascending. Same criterion as cb2half: the UE8M0
        byte IS the exponent, so 2**(e-127) summed along a row is that row's scale mass, and the
        SwiGLU term is proportional to the product of what w1 and w3 carry."""
        m1 = torch.exp2(s1.float() - 127.0).sum(1)
        m3 = torch.exp2(s3.float() - 127.0).sum(1)
        score = (m1 * m3).view(INTER // 32, 32).sum(1)
        return torch.sort(torch.topk(score, self.n_groups).indices).values

    def load_slot(self, slot: int, w1, s1, w2, s2, w3, s3, non_blocking: bool = False, sim=None) -> None:
        dev = self.device
        w1 = w1.view(torch.uint8).to(dev, non_blocking=non_blocking)
        w3 = w3.view(torch.uint8).to(dev, non_blocking=non_blocking)
        w2 = w2.view(torch.uint8).to(dev, non_blocking=non_blocking)
        s1 = s1.view(torch.uint8).to(dev, non_blocking=non_blocking)
        s3 = s3.view(torch.uint8).to(dev, non_blocking=non_blocking)
        s2 = s2.view(torch.uint8).to(dev, non_blocking=non_blocking)
        g = self.select_groups(s1, s3)
        rows = (g[:, None] * 32 + torch.arange(32, device=dev)).reshape(-1)      # kept channels
        bcol = (g[:, None] * 16 + torch.arange(16, device=dev)).reshape(-1)      # their w2 bytes
        self.w1[slot].copy_(w1[rows]); self.s1[slot].copy_(s1[rows])
        self.w3[slot].copy_(w3[rows]); self.s3[slot].copy_(s3[rows])
        self.w2[slot].copy_(w2[:, bcol]); self.s2[slot].copy_(s2[:, g])
