# VQ12 — the 3-bit format that halves CB3's damage. **The kernel does not work yet.**

## Status

**Correct. Not yet fast enough.**

| | |
|---|---|
| forward vs the stock FP4 kernel on identical weights | **rel err 0.00001** |
| forward vs a hand reference | **0.00164** (CB3 v2 scores 0.00166 on the same reference: bf16 rounding) |
| up / down kernels separately | 0.00165 / 0.00157 |
| speed, 6x6 decode | **1.020 ms** against CB3 v2's 0.639 and CB3 v3's (inline PTX) 0.495 |

So the format and the kernel are right, and the kernel is **2x slower than the CB3 one it would
replace**. At the engine's warm steady state (54.5 ms/tok, hit 1.000) the MoE dominates the step, so
shipping this as-is would trade the +89 % speed win for the quality win. It needs the decode cost
back before it is worth wiring in.

### The bug that was in the way, and what it cost

The first version built each scale group's byte tile by looking the 8 group indices up once and
`tl.interleave`-ing the entry's two bytes into 16. That produces the **right values** -- storing the
tile and comparing against `vq12.unpack` matches 100 % at BN = 16/32/64/128 -- but a register layout
`tl.dot` reads wrongly. The proof is `test_layout.py`: the same tile, same dot, gives **1.678**
straight from the registers that built it and **0.00000** after a round trip through memory.

Three decode implementations (`tl.join`+`tl.reshape`, `tl.interleave`, and one avoiding
`fp4_moe._fp4_decode`'s `pack=4` inline asm) all gave bit-identical wrong output, which is what
finally pointed away from the decode and at the layout.

The fix: build the tile from **loads and elementwise ops only** -- loads always produce a layout the
dot accepts. Each output byte fetches its own group's index (`lo_ptr + j // 2`) instead of the tile
being assembled from a wide load. That is where the 2x went: 16 narrow fetches per group instead of
one wide load plus register splits. `fp4_moe`'s packed inline-asm decode was tried again once the
layout was normal and came out *slower* (1.385 ms), so a 16-entry fp16 table is used instead.

### Where the speed could come back

The redundant per-byte index fetches are L1 hits but cost instructions. Options not yet tried:
regrouping so a group's two bytes land at tile positions g and g+8 (then `we`/`wo` are concatenations
rather than interleaves, and the k mapping `xk = 2 * arange(16)` still holds -- the group would cover
k {2g, 2g+1, 2g+16, 2g+17}, which the near-independence of the source along k makes statistically
equivalent); decoding the nibble arithmetically instead of through the table; and the usual
(BN, num_warps, num_stages) sweep, which has not been done at all for this kernel.

## Why it should be worth finishing

At the same 3 bit/weight, over all 40 layers unpruned (measured on the A100, same protocol per row):

| 3-bit format | wikitext PPL | Δ | code PPL | Δ |
|---|---|---|---|---|
| baseline FP4 | 3.0765 | — | 1.2826 | — |
| CB3, per-row scalar 8-of-16 | 3.3391 | +8.54 % | 1.3104 | +2.17 % |
| **VQ12, dim-4 vector** | **3.2067** | **+4.23 %** | **1.2951** | **+0.97 %** |

Identical bytes, identical slot geometry, half the damage. CB3's subset search is already optimal
for its format; the gap is scalar vs vector quantisation.

## The format

One 12-bit index per group of 4 consecutive k into a global 4096-entry codebook whose entries are
four E2M1 codes -- 3 bit/weight, exactly CB3's rate, in exactly CB3's planes:

    lo  [N, K/4]   the low 8 bits of group g's index, one byte per group
    hi  [N, K/8]   the high 4 bits, two groups per byte
    cb  [N, 8]     unused (the codebook is global); kept so the slot stride is unchanged
    s   [N, K/32]  the UE8M0 scales, untouched

A group expands to exactly the two packed-FP4 bytes `_chunk_dot` consumes, so the kernel change is
one line of CB3's `_grp_packed`: a per-row 32-bit codebook word indexed by a 3-bit index becomes
`tl.load(lut + idx)`.

## What has been ruled out

* the encoder: `pack` round-trips idempotently and `dequant == FP4(unpack)` exactly (`vq12.py`);
* the arena: `w1_lo/w1_hi` after `load_slot` equal `vq.pack` of the source, bit for bit
  (`test_arena.py`);
* the kernel's addressing: reading the arena through the kernel's own index arithmetic and decoding
  reproduces `vq12.unpack` exactly, for blocks 0, 3 and 19 (`test_arena.py`);
* the decode tile at every block width the engine uses (`test_decode.py`, BN = 16/32/64/128);
* the hand reference: CB3 v2 matches it to 0.00166 on the same construction (`test_ref.py`);
* **the decode, conclusively**: three different implementations -- `tl.join`+`tl.reshape`,
  `tl.interleave`, and a version that avoids `fp4_moe._fp4_decode` altogether (it is
  `tl.inline_asm_elementwise(..., pack=4)`, whose four-elements-per-register PTX depends on the
  tile's register layout, so an interleaved tile was a good suspect) -- all produce **bit-identical
  output, rel err 1.94316 to five decimals**. Whatever is wrong is upstream of the decode and does
  not depend on it;
* uint8 -> int32 sign extension (masked with `& 0xFF` anyway).

The error is a factor, not a rounding: 1.94 relative against both references, which smells like a
mis-pairing of tiles with x offsets or scales inside `_vq_block_dot` that the per-tile tests cannot
see because each tile is individually right.

## Next step for whoever picks this up

`test_onehot.py` is the right idea -- one-hot x into `_vq_up_kernel` alone, sweep k, see which
weight the kernel actually used -- but as written it is **not conclusive**: it assumes `up` is 1.0
because w3 was filled with code 2, forgetting that VQ quantises 4-tuples, so the tuple (2,2,2,2)
can land on a codebook entry that is not (2,2,2,2) and `up` is then not 1. Fix it by reading the
arena's own w3 back through `vq12.unpack` (as it already does for w1) before forming the
expectation, or by setting w3 to a tuple the codebook maps to itself.

That the error is identical to five decimals across three decode implementations says the fault is
in `_vq_block_dot`'s pairing of tiles with `x_base + 32*i` / `s_i`, or in the kernel arguments, not
in how a tile is built.
