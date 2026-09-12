# a100-vq — a pre-packed CB3 expert store, and what it is and is not worth

Measured 2026-09-12 on `gx10-b872` (GB10, 121 GiB unified, one 916 GB NVMe) against this repo's
engine. Every number below is one run on that box; where a measurement was attempted and not
finished it says so instead of being estimated. The companion information-theoretic study
(rate-distortion of the FP4 experts, the PPL of each format over all 40 layers) was run on an
8×A100 box and lives in `shi3z/deepseekv4.1-A100-custom`, branch `compression-study`.

## The headline, with its condition attached

Unpruned (every one of the 15,360 routed experts reachable), `ARENA_GB=94`, `DSV41_BLOCK=1`,
greedy, 300 tokens, the same Japanese prompt three times:

| run | FP4 (this repo's streaming default) | CB3 + the pre-packed store |
|---|---|---|
| 1 (cold arena) | 6.93 tok/s, 0.369 GB/tok, hit 0.9289 | 4.04, 0.116, 0.9426 |
| 2 | 9.13, 0.233, 0.9585 | 8.33, 0.036, 0.985 |
| **3** | **9.71, 0.213, 0.9625** | **18.28, 0.003, 1.000** |

FP4 plateaus at hit 0.9625: 94 GB is 5,000 FP4 slots and the working set does not fit. The same
94 GB is 6,500 CB3 slots, the working set does fit, and NVMe traffic goes to nothing.

**And the condition matters.** With five *different* prompts rotating (ja prose, Python,
translation, a proof, an English essay), so the LRU never settles:

| | FP4 | CB3 + store |
|---|---|---|
| per-prompt decode | 5.4–7.3 tok/s | **1.2–3.8 tok/s** |

The store covers trace ranks 6,500–12,587 (6,087 experts, 88 GB — what the 916 GB disk had room
for). A miss outside that band still pays the FP4 read *and* the 20.8 ms/expert GPU pack, and a
rotating workload produces many of them. **This configuration is a win for a warm, repetitive
workload and currently a loss for a diverse one.** Packing all 15,360 experts (222 GB) is what
would make it unconditional; the box has 33 GB free with the FP4 checkpoint in place.

## Why the obvious version does not work

`EXPERT_FORMAT=cb3` alone, unpruned, streaming (`results/cb3stream.log`):

```
FP4   108.7 ms/tok   NVMe 0.233 GB/tok   hit 0.9585   9.16 tok/s
CB3   148.5 ms/tok   NVMe 0.069 GB/tok   hit 0.9872   6.71 tok/s
```

The traffic collapses 3.4× and the step still gets *slower*, because every miss runs
`cb3.fp4_to_cb3_v2` on the GPU. Measured 20.8 ms/expert against the 4.6 ms the 18.8 MB read itself
takes; at ~3.7 misses/token that is ~76 ms of fill. The cost is not the 12,870-subset search (a
950 MFLOP GEMM) but the two int64 `[N, K]` intermediates it builds — 189 MB for one w13.

Two things follow, and both are here:

* `fast_fill.py` — `fp4_to_cb3_v2` rewritten to stay in the packed byte domain (256-bin histogram
  folded to 16 by a constant nibble-count matrix; a per-row 256-entry byte table for the remap).
  **Bit-identical output**, 1.5× (20.8 → 13.8 ms/expert). `fast_requant.py` does the same for
  `CodebookSim.requant_packed` (2.6×, also bit-identical). 1.5× is not enough on its own: it leaves
  ~51 ms of fill, still short of FP4.
* `pack_store.py` / `cb3_store.py` — move the pack off the miss path entirely.

## The store

One fixed-stride record per expert: the 12 slot tensors of `cb3_moe.CB3ArenaV2` concatenated as
`(w1_lo, w1_hi, w1_cb, s1, w3_lo, w3_hi, w3_cb, s3, w2_lo, w2_hi, w2_cb, s2)`. The stride is
14,454,784 B, a multiple of 4096, so a miss is **one O_DIRECT pread** into a pinned buffer and 12
H2D slice copies — no alignment slack, no fill. A sidecar JSON carries the stride, the piece
offsets and the `(layer, expert) → record` map.

Built on the box itself, in the engine's own `rank_from_trace` order, resumable:
**6,087 experts / 88.0 GB / 8.2 min** (`results/pack.log`). Packing locally beats shipping a store
in: the link to the A100 box measured 95 MB/s, so 88 GB would be 15.4 minutes of transfer plus the
remote pack, and neither box has 222 GB free to stage a full one.

`experts.py.a100-vq.patch` is the engine change: ten lines at the top of
`ExpertStore._load_into_slot`. It is opt-in — without `DSV41_CB3_STORE` nothing attaches and the
engine behaves exactly as before.

```
python a100-vq/pack_store.py --budget-gb 88          # once, ~8 min
patch -p1 < a100-vq/experts.py.a100-vq.patch

DSV41_CB3_STORE=$HOME/dsv41-spark/models/cb3_store \
DSV41_BLOCK=1 EXPERT_FORMAT=cb3 ARENA_GB=94 TRANSIENT_SLOTS=64 KEEP_FREE_GB=12 ./start.sh
```

## Quality

Greedy generation A/B, five prompts, warm pass (`results/gen_ab.log`): the CB3 arm answers all five
correctly — the Japanese seasons essay, a Fibonacci function whose docstring examples are right, an
accurate translation, Euclid's proof carried to ∎, and a sound essay on MoE serving. **No
degeneration**: none of the repeated-phrase collapse that expert pruning causes. The FP4 arm errored
on two of the five with `transient ring exhausted: more experts in one call than transient_slots` —
a `TRANSIENT_SLOTS=64` limit the CB3 arena does not reach, because it misses less during prefill.

CB3's own cost, measured on the A100 over all 40 layers with no pruning, same protocol for every
row (wikitext-2, ctx 2048, 32,752 tokens; a code corpus of 24,564 tokens):

| format, 3 bit/weight | wikitext PPL | Δ | code PPL | Δ |
|---|---|---|---|---|
| baseline FP4 | 3.0765 | — | 1.2826 | — |
| **CB3** (per-row scalar, 8-of-16) | 3.3391 | **+8.54 %** | 1.3104 | **+2.17 %** |
| dim-4 VQ, codebook snapped to the E2M1 grid | 3.2067 | **+4.23 %** | 1.2951 | **+0.97 %** |

Same bytes, same slot geometry, half the damage. CB3's subset choice is already optimal *for its
format* (exhaustive over all C(16,8)); the gap is scalar-vs-vector quantisation, not the search.
Swapping it in needs only the Triton decode to change: a per-row codebook shift becomes one lookup
in a 4096-entry table (8 kB, or 16 kB padded to u32 to avoid bank conflicts).

## Not measured

* Teacher-forced NLL on `corpus/heldout_corpus.jsonl` for these configurations. Attempted; one arm
  ran 41 minutes under streaming without finishing and was stopped. `tf_eval.sh` is the script.
* Anything about a store that covers all 15,360 experts — the disk did not have room.
* The VQ decode kernel. The quality numbers above come from simulating the format inside the FP4
  arena (the codebook entries are four E2M1 codes, so a quantised expert is still a valid FP4
  tensor and scores on the unmodified kernels); no packed VQ format or kernel exists yet.
