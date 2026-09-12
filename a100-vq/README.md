# a100-vq — a pre-packed CB3 expert store, and what it is and is not worth

Measured 2026-09-12 on `gx10-b872` (GB10, 121 GiB unified, one 916 GB NVMe) against this repo's
engine. Every number below is one run on that box; where a measurement was attempted and not
finished it says so instead of being estimated. The companion information-theoretic study
(rate-distortion of the FP4 experts, the PPL of each format over all 40 layers) was run on an
8×A100 box and lives in `shi3z/deepseekv4.1-A100-custom`, branch `compression-study`.

## The headline

Unpruned -- every one of the 15,360 routed experts reachable -- `ARENA_GB=94`, `DSV41_BLOCK=1`,
greedy. FP4 is this repo's streaming default, measured before the checkpoint's expert bytes were
punched out; CB3 is the full 222 GB store (`results/full_bench.log`, `results/fp4ctl_bench.log`).

| | FP4 | CB3 + full store |
|---|---|---|
| same prompt, run 1 / 2 / 3 | 6.93 / 9.13 / **9.71** tok/s | 9.09 / 14.16 / **18.37** tok/s |
| five different prompts, steady pass | 7.33 / 5.45 / err / 7.27 / err | **9.35 / 7.41 / 4.57 / 9.42 / 7.57** |

**+89 % on a repeated prompt, +28-36 % on the three diverse prompts FP4 could serve**, and the two
it could not -- FP4 fails them with `transient ring exhausted: more experts in one call than
transient_slots`, a limit the CB3 arena does not reach because it misses less during prefill.

FP4 plateaus at hit 0.9625: 94 GB is 5,000 FP4 slots and the working set does not fit. The same
94 GB is 6,500 CB3 slots, so it does.

A partial store (6,087 experts, trace ranks 6,500-12,587) was **not** enough: it gave the same
18.28 tok/s on the repeated prompt but 1.2-3.8 tok/s on the diverse one, because a miss outside the
band still paid the FP4 read and the 20.8 ms pack. Covering all 15,360 is what makes it
unconditional.

## Making room without breaking the checkpoint

The full store is 222 GB and the disk had 33 GB free. The shards cannot be deleted: of their
510.3 GB the routed experts are 288.8 GB, but the other 214.3 GB -- **203.1 GB of it the Engram
tables**, plus the dense/attention/head weights and the 384 DSpark draft experts -- lives in the
same files and is read every step.

`punch_fp4.py` frees the expert blocks with `fallocate --punch-hole` instead, leaving every other
tensor at its original offset so nothing that reads the checkpoint has to change. Three rules keep
it safe: only `layers.<n>.ffn.experts.<e>.*` (never `mtp.*`, whose draft experts stay FP4), only
experts already in a store, and the range is aligned **inward** to 4096 because safetensors packs
tensors 8-byte aligned and a partial edge block can hold a neighbour's bytes. It records what it
punched, and the engine patch raises on a miss for a punched expert that is not in a store rather
than feeding the model zeros.

Result: the shards still measure 476 GB apparent, 207 GB physical; 222 GB of stores; 176 GB free.

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

Built on the box itself, in the engine's own `rank_from_trace` order, resumable, in three batches
interleaved with punching so the disk never had to hold both: **15,360 experts / 222.0 GB / 30 min**
(`results/pack*.log`). Packing locally beats shipping a store
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
* The store's own NVMe traffic: `cb3_store.py` serves a miss without going through the engine's
  accounting, so `nvme_gb_per_token` reads 0.000 in the full-store rows even though misses happen
  (hit 0.82-0.95 on the diverse prompts).
* The VQ decode kernel. The quality numbers above come from simulating the format inside the FP4
  arena (the codebook entries are four E2M1 codes, so a quantised expert is still a valid FP4
  tensor and scores on the unmodified kernels); no packed VQ format or kernel exists yet.
