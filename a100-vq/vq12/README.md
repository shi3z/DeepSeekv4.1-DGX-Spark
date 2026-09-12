# VQ12 — the 3-bit format that halves CB3's damage. **The kernel does not work yet.**

## Status, plainly

| piece | state |
|---|---|
| the format and its packer (`vq12.py`) | **works**, verified |
| the codebook (`vq_3.0.npz`, trained on the A100) | **works** |
| the Triton decode (`_grp_packed_vq`) | **byte-exact** against `vq12.unpack` at BN = 16/32/64/128 |
| speed of the whole kernel | **0.464 ms** at 6x6 decode against CB3 v2's 0.633 and CB3 v3's (inline PTX) 0.522 |
| the end-to-end MoE forward | **WRONG** — rel err 1.94 against both a hand reference and the stock FP4 kernel on identical weights |

So the idea and the pieces check out and the kernel is *faster* than the PTX CB3 one, but something
structural in `moe_forward_vq` is wrong and is not yet localised. **Do not use this path.**

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
