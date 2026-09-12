"""
cb2half.py -- a half-width CB2 expert: 2-bit codebook weights over a SUBSET of the expert's
intermediate channels.

Why not a 1-bit format. Getting every one of the 15,360 routed experts resident in ~97 GB needs
about 1.5 bits per weight, and CB2 (2 bits + the UE8M0 scale = 2.25 bpw, 9.99 MB) is 153 GB for
the set. The obvious next step down is a 1-bit codebook, which needs its own packed layout and its
own PTX decode. This does the same arithmetic a different way: keep 2 bits per weight and drop
half the intermediate channels instead. A cold expert then costs 4.45-6.7 MB and -- this is the
point -- runs on the EXISTING CB2 kernel, because the only thing that changes is `INTER`.

What is dropped is chosen per expert from the checkpoint's own UE8M0 exponents: the score of an
intermediate channel is the product of the magnitudes its w1 and w3 rows carry, which is what the
SwiGLU term is proportional to before any activation is seen. Channels are selected in groups of
32 because that is the scale group -- an arbitrary subset would split a UE8M0 group and misalign
w2's packed nibbles, and 32 channels is exactly 16 packed bytes of a w2 row.

`INTER_H` must be a multiple of 256 (`cb3.block_plan`) and of 32 (the scale group): 1024, 1280 and
1536 are the useful widths, 44 %, 56 % and 67 % of the checkpoint's 2304.
"""

from __future__ import annotations

import torch

import cb3_moe as C3
from cb3 import fp4_to_cb2
from fp4_moe import DIM, INTER


def bytes_per_slot(inter_h: int) -> int:
    return (2 * (inter_h * (DIM // 4 + 4) + inter_h * (DIM // 32))
            + DIM * (inter_h // 4 + 4) + DIM * (inter_h // 32))


class CB2HalfArena:
    """CB2ArenaV2 with `inter_h` of the 2304 intermediate channels instead of all of them."""

    def __init__(self, slots: int, device: torch.device | str = "cuda", inter_h: int = 1280):
        assert inter_h % 256 == 0 and inter_h % 32 == 0 and inter_h <= INTER, inter_h
        self.slots = slots
        self.inter_h = inter_h
        self.n_groups = inter_h // 32
        self.device = torch.device(device)
        u8 = dict(dtype=torch.uint8, device=self.device)
        self.w1_lo = torch.empty((slots, inter_h, DIM // 4), **u8)
        self.w1_cb = torch.empty((slots, inter_h, 4), **u8)
        self.s1 = torch.empty((slots, inter_h, DIM // 32), **u8)
        self.w3_lo = torch.empty((slots, inter_h, DIM // 4), **u8)
        self.w3_cb = torch.empty((slots, inter_h, 4), **u8)
        self.s3 = torch.empty((slots, inter_h, DIM // 32), **u8)
        self.w2_lo = torch.empty((slots, DIM, inter_h // 4), **u8)
        self.w2_cb = torch.empty((slots, DIM, 4), **u8)
        self.s2 = torch.empty((slots, DIM, inter_h // 32), **u8)
        self.sim = None  # engine.codebook_sim.CodebookSim(2)

    @property
    def bytes_per_slot(self) -> int:
        return sum(t[0].numel() for t in (self.w1_lo, self.w1_cb, self.s1, self.w3_lo, self.w3_cb,
                                          self.s3, self.w2_lo, self.w2_cb, self.s2))

    def select_groups(self, s1: torch.Tensor, s3: torch.Tensor) -> torch.Tensor:
        """Indices of the `n_groups` channel groups to keep, ascending.

        The UE8M0 byte IS the exponent, so 2**(e-127) summed along a row is the row's scale mass
        without ever dequantising a weight. Keeping the ascending order matters: it leaves the kept
        channels in their original relative order, which is what makes the gathered w2 columns a
        run of whole 16-byte group tiles.
        """
        m1 = torch.exp2(s1.float() - 127.0).sum(1)                 # [INTER]
        m3 = torch.exp2(s3.float() - 127.0).sum(1)
        score = (m1 * m3).view(INTER // 32, 32).sum(1)             # [72]
        keep = torch.topk(score, self.n_groups).indices
        return torch.sort(keep).values

    def load_slot(self, slot: int, w1, s1, w2, s2, w3, s3, non_blocking: bool = False, sim=None) -> None:
        sim = sim or self.sim
        assert sim is not None and sim.bits == 2, "CB2HalfArena.sim must be a CodebookSim(2)"
        dev = self.device
        w1 = w1.view(torch.uint8).to(dev, non_blocking=non_blocking)
        w3 = w3.view(torch.uint8).to(dev, non_blocking=non_blocking)
        w2 = w2.view(torch.uint8).to(dev, non_blocking=non_blocking)
        s1 = s1.view(torch.uint8).to(dev, non_blocking=non_blocking)
        s3 = s3.view(torch.uint8).to(dev, non_blocking=non_blocking)
        s2 = s2.view(torch.uint8).to(dev, non_blocking=non_blocking)
        g = self.select_groups(s1, s3)                              # [n_groups]
        ar32 = torch.arange(32, device=dev)
        rows = (g[:, None] * 32 + ar32).reshape(-1)                 # [inter_h] channel ids
        ar16 = torch.arange(16, device=dev)
        bcol = (g[:, None] * 16 + ar16).reshape(-1)                 # [inter_h//2] packed w2 bytes
        for (w, s, lo_t, cb_t, s_t) in ((w1[rows], s1[rows], self.w1_lo, self.w1_cb, self.s1),
                                        (w3[rows], s3[rows], self.w3_lo, self.w3_cb, self.s3),
                                        (w2[:, bcol].contiguous(), s2[:, g].contiguous(),
                                         self.w2_lo, self.w2_cb, self.s2)):
            lo, cb = fp4_to_cb2(w.contiguous(), s.contiguous(), sim)
            lo_t[slot].copy_(lo); cb_t[slot].copy_(cb); s_t[slot].copy_(s)

    def dequant_slot(self, slot: int):
        from cb3 import dequant_cb2
        return (dequant_cb2(self.w1_lo[slot], self.w1_cb[slot], self.s1[slot]),
                dequant_cb2(self.w2_lo[slot], self.w2_cb[slot], self.s2[slot]),
                dequant_cb2(self.w3_lo[slot], self.w3_cb[slot], self.s3[slot]))
