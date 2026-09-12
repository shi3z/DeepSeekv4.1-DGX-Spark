# The tiered arena

How all 15,360 routed experts are made to fit in ~97 GB, and what it costs.

## The arithmetic that forces it

| | |
|---|---|
| routed experts | 40 layers × 384 = 15,360 |
| one expert | 3 × 2304 × 5120 = 35.39 M weights |
| at the checkpoint's FP4 (e2m1 + one UE8M0 per 32) | 18.80 MB → **288.8 GB for the set** |
| visible memory | ~121 GiB = 130 GB |
| taken by dense weights, LM head, DSpark's own 384 experts, caches, allocator | ~33 GB |
| **left for the routed experts** | **~97 GB** |

97 GB over 543.6 B weights is **1.43 bits per weight**. That is the number every design here has to
meet. CB2 — two bits plus the UE8M0 scale, 2.25 bpw — is 153 GB for the set and does not meet it.
Nothing in the upstream repository does.

## Why coarse beats absent

The upstream default meets the budget by keeping 31 % of the experts and removing the rest from the
router. The failure mode that produces is not a noisier FFN. The router scores all 384 and takes the
top 6; if those 6 are not resident, the top 6 *of the kept set* runs — a different FFN entirely.
Upstream's own elimination (`results/htmlbug/`, `NOTES.md` 2026-09-12 00:10) traced a generation
collapsing into a repeated phrase to exactly this, and measured the cost as +0.19 nats on prose
against +0.07 on code: prose leans on the tail of the router's distribution, code does not.

A coarsely stored expert keeps the routing structure. That is the whole bet of this design, and it
is the part that still needs the held-out numbers to confirm.

## The tail format: half-width CB2

Meeting 1.43 bpw with a 1-bit codebook means a new packed layout and a new PTX decode path.
`tools/cb2half.py` gets there a different way: keep CB2's two bits per weight and store only
`INTER_H` of the 2304 intermediate channels.

```
w1, w3 : [INTER_H, 5120]   instead of [2304, 5120]
w2     : [5120, INTER_H]   instead of [5120, 2304]
```

The kernel does not change. `INTER` is a `tl.constexpr` in the CB2 up/down kernels, so the
half-width tier is the same, already-tested code at a different width. `h` stays allocated at the
full 2304 and this tier writes and reads its first `INTER_H` columns, so the row stride is shared
and no second buffer is needed.

| `INTER_H` | fraction of 2304 | MB/expert | all 15,360 | µs/expert |
|---|---|---|---|---|
| 1536 | 67 % | 6.67 | 102.4 GB | — |
| 1280 | 56 % | 5.56 | 85.4 GB | 46 |
| 1024 | 44 % | 4.45 | 68.4 GB | 40 |
| 768 | 33 % | 3.34 | 51.3 GB | 31 |

`INTER_H` must be a multiple of 256 (`cb3.block_plan` cuts a row into 512- and 256-weight blocks;
1152 is not expressible) and of 32 (the UE8M0 scale group).

### Choosing the channels

Data-free, from the checkpoint's own exponents. The UE8M0 byte *is* the exponent, so

```
m1[n] = Σ_g 2^(s1[n,g] − 127)      m3[n] = Σ_g 2^(s3[n,g] − 127)
score[n] = m1[n] · m3[n]
```

is the scale mass each intermediate channel carries in `w1` and `w3`, and their product is what the
SwiGLU term is proportional to before any activation is seen. No weight is dequantised to compute
it; `select_groups` costs 0.07 ms per expert.

Channels are kept **in groups of 32, in ascending order**. Both constraints are load-bearing:

- 32 is the UE8M0 scale group. An arbitrary subset would split a group, and `w2`'s scale column
  would no longer correspond to 32 consecutive kept channels.
- 32 channels is exactly 16 packed bytes of a `w2` row, so a group boundary is also a byte
  boundary and `w2[:, g*16 : g*16+16]` is a clean gather. An arbitrary subset would land mid-byte.
- Ascending order keeps the kept channels in their original relative order, which is what makes the
  gathered `w2` columns a run of whole group tiles.

`measure/test_cb2h2.py` checks this by restricting the test weights to four FP4 codes per row, which
makes the 2-bit codebook lossless and identical for any subset of the row: the half-width `w1`/`w3`
rows and `w2` columns then have to match the full-width ones exactly, and do.

## The arena

`TieredArena` is one flat slot space cut into per-format ranges — each tier owns a contiguous range
and its own arena object. A global slot id resolves to (tier, local slot) by range.

### Tier order is not arbitrary

`ExpertStore._lru_slot_for` hands out `free_lru.pop()` — the **highest** free slot. `warm_start`
walks the trace-ranked list from hottest to coldest, so the slot space fills **from the top down**
and the hottest expert lands on the last slot.

So the tiers are laid out **coldest first**:

```
slots 0 .. n_tail-1          CB2 half-width   <- the coldest experts
slots n_tail .. +n_cb2-1     CB2
slots .. n_slots-1           FP4              <- the hottest experts, and the transient ring
```

The transient ring is the tail of the slot space (`ExpertStore` takes the last `transient_slots`),
so it merges into the FP4 tier. A miss — which cannot happen while every expert is resident — then
lands in the checkpoint's own format rather than paying a GPU re-pack to 2 bits.

Getting this backwards is silent: the model runs, every expert is present, and the hottest 15 % of
the routing is served by the coarsest tier.

### One routing build, not one per tier

`moe_forward_tiered` builds the routing **once**, over the global slot ids. Because the tiers own
disjoint ranges of that space and the router groups pairs by slot, every block belongs to exactly
one tier. Each tier then needs only its own `block_slot` vector, re-based and with the other tiers'
blocks set to the `-1` the kernels already skip — one elementwise op per tier, not a second
argsort/cumsum/scatter chain.

There is no `if any():` guard anywhere in the path. Reading that predicate is a host sync, which
costs more than the empty launch it would save and makes the step impossible to capture in a CUDA
graph. `measure/microbench/graph_test.py` confirms capture works and that the graph re-routes when
the slot tensor changes.

### `build_routing_masked`

`fp4_moe.build_routing_small` sorts the pairs by slot and writes pair `r` of block `b` to
`block_pair[b*BM + r]`, on the documented assumption that "a slot never has more than BM pairs at
this size". True for a real expert — it appears at most once per token, so at most `T` ≤ BM times.
Not true for the `-1` sentinel, which a multi-format arena produces in quantity: at BM=16, a call
where 21 of the 36 pairs belong to other tiers writes 21 entries into block 0's 16 slots and
corrupts block 1.

`build_routing_masked` sorts the masked pairs to the end (`key = where(valid, slot, INT_MAX)`) and
scatters them to a discard row instead of a block. Shapes stay static and no value reaches the
host, so it is still graph-capturable.

## Planning a budget

`plan_all_resident(n_fp4, n_cb2, inter_h, transient_slots)` builds the tier list;
`fit_all_resident(budget_bytes, inter_h, fp4_share)` chooses `n_fp4` and `n_cb2` for a budget.

The floor is `15360 × bytes_per_slot(inter_h)` — the cost of holding every expert at the tail
format. Everything above it is headroom, split between FP4 and CB2 upgrades by `fp4_share`:

| `INTER_H` | floor | at 97 GB, `fp4_share=0.4` | at 97 GB, `fp4_share=0.8` |
|---|---|---|---|
| 768 | 51.4 GB | 1,180 FP4 + 4,118 CB2 | 2,361 FP4 + 1,372 CB2 |
| 1024 | 68.4 GB | 797 FP4 + 3,098 CB2 | 1,595 FP4 + 1,032 CB2 |
| 1280 | 85.4 GB | 350 FP4 + 1,569 CB2 | 700 FP4 + 523 CB2 |
| 1536 | 102.4 GB | does not fit | does not fit |

If the budget is below the floor, `fit_all_resident` raises and names the widths that would fit
rather than quietly returning a plan that drops experts.

### What the trace says each plan costs

Weighting the tiers by the upstream 40-layer routing histogram
(`results/trace-full-20260910/stats/coverage.json`) gives the share of routed pairs each tier
serves, and with the measured µs/expert, the expert kernel time per token:

| `INTER_H` | `fp4_share` | plan | routing served by FP4 / CB2 / tail |
|---|---|---|---|
| 768 | 0.0 | 0 + 6,863 + 8,497 | 0 % / 89.4 % / 10.6 % |
| 768 | 0.4 | 1,180 + 4,118 + 10,062 | 45.8 % / 36.8 % / 17.4 % |
| 768 | 0.8 | 2,361 + 1,372 + 11,627 | **61.0 % / 11.9 % / 27.1 %** |
| 1280 | 0.8 | 700 + 523 + 14,137 | 36.9 % / 9.6 % / 53.5 % |

`INTER_H=768` is the useful width: the tail is cheap enough that most of the budget can go back
into FP4, so the majority of the *routing* — not of the experts — is served at the checkpoint's own
precision. The shipped example (`fp4_share=0.8`) serves 61 % of routed pairs from untouched FP4
weights while the upstream default serves 79.6 % from untouched FP4 and **20.4 % from no expert at
all**.

## What this does not fix

- **Prefill.** The CB2 and half-width kernels lose to FP4 at prefill shapes. Upstream's FP4 path
  unpacks the experts it needs into a scratch arena and runs the FP4 kernel for prefill-sized
  calls; the tiered path has no equivalent, and prefill measured 1.8x slower. The unpack would be
  straightforward — the half-width tier unpacks compactly into the first `INTER_H` columns of a
  full-width scratch, which is consistent because `w1`/`w3` rows and `w2` columns carry the same
  permutation — it is simply not written.
- **Warm start.** 3x longer: all 15,360 experts are read and re-packed rather than 4,800 copied.
  Packing is 14 ms per CB2 expert and 5.7 ms per half-width one; a packed-arena disk cache would
  remove it, and there is not enough free disk for one next to a 510 GB checkpoint.
- **Acceptance.** ~3-4.5 accepted tokens per step is the single biggest lever left on this model
  and this fork does not move it deliberately. Tree or multi-candidate verification is the obvious
  next step and is blocked on the KV ring being indexed directly by position
  (`ring[pos % RING] = kv`), which makes tree siblings collide; the CSA2 compressed path, the
  indexer and the candidate pool would all need to follow.
