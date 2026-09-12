# DeepSeek-V4.1-Flash on one DGX Spark — every expert resident

> **Status: work in progress, one box, one day.** The decode numbers below are each one run on
> one machine. The quality evaluation is still being taken. Read
> [LIMITATIONS.md](LIMITATIONS.md) and the caveats in this file before quoting anything.

This is a fork of **[0xBakeer/deepseek-v41-flash-spark](https://github.com/0xBakeer/deepseek-v41-flash-spark)**
(MIT, see [LICENSE](LICENSE) and [CREDITS.md](CREDITS.md)). That repository is the engine: the
pure-PyTorch port of DeepSeek-V4.1-Flash, the Triton FP4 grouped-MoE kernel, the expert arena, the
CUDA-graph decode path, the OpenAI-compatible server, and the 40-layer routing trace. Its own
README is kept here as [README-upstream.md](README-upstream.md).

What this fork adds is one thing: **a way to keep all 15,360 routed experts resident in 121 GiB.**

## The problem it addresses

The routed experts are 15,360 × 3 × 2304 × 5120 weights, which at the checkpoint's own FP4 with
UE8M0 scales is **288.8 GB**. A DGX Spark has ~121 GiB visible. After the dense weights, the LM
head, the DSpark drafter's own experts and the caches, about **97 GB is left for them.**

The upstream repository measured both ends of the resulting dichotomy and, on 2026-09-12, concluded
there was nothing in between (`NOTES.md`, 00:10):

> At 288.8 GB of FP4 experts and 121 GiB of memory there is no arrangement that keeps every expert
> resident. Either the experts stream on a miss (full quality, NVMe-bound) or some are dropped
> (fast, and a workload the keep-set does not cover degenerates). The recipe now ships the first.

Those two ends are **3.33 tok/s at full quality** and **~20-30 tok/s with 69 % of the experts
dropped**, where free generation can collapse into a repeated phrase — the upstream
`results/htmlbug/` elimination found pruning to be the cause, not the engine.

So a tok/s number for this model on this box means nothing on its own. It has to say which side of
that line it was taken on.

## The third option

Dropping an expert is not a small error. The router picks 6 of 384 by score; if those 6 are not
resident, the top-6 *of a smaller set* runs instead, which is a different FFN, not a noisier one.
Storing an expert coarsely is a much gentler failure: the right expert still runs.

So spend the 97 GB unevenly instead of spending it on a subset:

| tier | format | what it is |
|---|---|---|
| hot | **FP4** | the checkpoint's own e2m1 + UE8M0, untouched |
| warm | **CB2** | upstream's 2-bit per-row codebook (9.99 MB/expert) |
| tail | **CB2 half-width** | 2-bit, and only `INTER_H` of the 2304 intermediate channels |

The tail format (`tools/cb2half.py`) is what makes it fit. Getting all 15,360 experts into 97 GB
needs about 1.5 bits per weight, and CB2 at 2.25 bpw is 153 GB for the set. Rather than build a
1-bit format — a new packed layout and a new PTX decode — this keeps 2 bits per weight and drops
intermediate channels, so **the existing, already-tested CB2 kernel runs it unchanged; only `INTER`
changes.** At `INTER_H=768` an expert is 3.34 MB and the whole set is 51 GB, leaving room to put
the hot experts back at FP4.

Which channels go is read out of the checkpoint's own UE8M0 exponents: a channel's score is the
product of the magnitudes its `w1` and `w3` rows carry, which is what the SwiGLU term is
proportional to before any activation is seen. Channels are selected in groups of 32 because that
is the scale group — an arbitrary subset would split a UE8M0 group and misalign `w2`'s packed
nibbles.

`tools/tiered_moe.py` runs one grouped kernel per tier over the pairs that land in it, all
accumulating into one `parts` buffer, from **one** routing build shared across the tiers.

## The numbers

One GB10 box (`sm_121a`, 128 GB unified / 121 GiB visible, 1 NVMe, CUDA 13, driver 580.159.03) with
the pool to itself. Both rows: same prompt, 200 tokens, greedy, thinking off, `MAX_SEQ=8192`, DSpark
on, CUDA graphs on, device slot LUT on. One run each.

| | upstream default (keep 31 % FP4) | **this fork (all resident)** |
|---|---|---|
| experts the router can reach | 4,800 of 15,360 — **10,560 dropped** | **15,360 of 15,360** |
| arena | 90.2 GB, one format | 94.0 GB: 2,206 FP4 + 1,282 CB2 + 11,872 CB2-half(768) |
| **decode** | 24.73 tok/s | **30.14 tok/s** |
| DSpark acceptance | 4.30 | 4.44 |
| NVMe read during decode | 0 GB | 0 GB |
| expert hit rate | 1.000 | 1.000 |
| prefill, 19-token prompt | 15.21 tok/s | **8.39 tok/s** |
| load, process start to ready | 108 s | 314 s |

**+21.9 % on decode, and nothing is dropped.** Acceptance is not identical between the rows, and
acceptance is the single biggest lever on this model's tok/s — upstream's own runs move between
20.02 tok/s at acceptance 2.99 and 30.1 at 4.72 on configuration changes that did not touch the
experts. Correcting the row to the baseline's 4.30 gives 29.2 tok/s, still +18 %.

**What is worse.** Prefill is 1.8x slower, because the CB2 and half-width kernels lose to FP4 at
prefill shapes (upstream measured 2.3-6.7x for CB3 there). The prompt those two rows were taken on
is 19 tokens, which is short enough that fixed cost dominates the rate; a long-prompt TTFT
comparison has not been taken, and the gap there could be larger. Upstream's FP4 path unpacks to a scratch
arena and runs the FP4 kernel for prefill-sized calls; the tiered path does not do that yet. Warm
start is 3x longer because all 15,360 experts are read and re-packed rather than 4,800 copied.

**What is not measured yet.** The teacher-forced held-out NLL of either configuration, and whether
the tail format changes free generation. Those runs are in progress; until they land, this fork
claims a speed result and an expert-coverage result, **not a quality result.**

### The box, measured

Every figure that the design above rests on, taken on this machine rather than from a data sheet:

| | measured |
|---|---|
| GPU read bandwidth ceiling | **234 GB/s** (vendor figure 273), flat from 1 to 32 blocks/SM |
| `cudaMalloc` / `cudaHostAlloc` / `cudaMallocManaged` read | 231 / 191 / 159 GB/s |
| NVMe O_DIRECT read, 18.8 MB / 4 MB / 1 MB | 5.82 / 4.83 / 3.78 GB/s |
| kernel launch | 3.44 µs |

Unified memory does **not** make a host allocation free to read from the GPU — it costs 17 %, and a
managed allocation 31 %. The expert arena stays in device memory.

Per expert at decode shapes (6 tokens × 6 experts, ~16 unique experts per layer):

| format | MB/expert | µs/expert | GB/s of its own bytes | all 15,360 |
|---|---|---|---|---|
| FP4 | 18.80 | 105 | 180 | 289 GB |
| CB3 | 14.45 | 85 | 171 | 222 GB |
| CB2 | 9.99 | 71 | 140 | 153 GB |
| CB2-half 1280 | 5.56 | 46 | 122 | 85 GB |
| CB2-half 1024 | 4.45 | 40 | 112 | 68 GB |
| CB2-half 768 | 3.34 | 31 | 110 | 51 GB |

Fewer bits buys less than the byte count suggests: the kernels fall from 180 GB/s to 110 GB/s as
the format narrows, so CB2 is 0.68x FP4 in wall time, not the 0.53x its bytes imply.

### Four optimizations that measured as worthless

Recorded so nobody spends a day on them again. Each was tried on this box:

1. **Fusing the "5,300 small kernels per step."** They are 2.1 ms in total. Upstream's own profile
   already said this is inter-kernel latency inside a CUDA graph, not launch overhead.
2. **Huge pages / TLB reach for a large arena.** Per-expert time is flat at 108 µs from a 0.8 GB
   arena to a 30 GB one. There is no arena-size cliff.
3. **A smaller `block_m`.** BM=8 instead of 16 is worth under 4 %; BM=32 is 10x worse.
4. **Retuning the CB2 kernel's `(BN, warps, stages)`.** A full two-stage sweep found nothing better
   than the shipped `(32, 4, 3)`.

## Running it

Install as [README-upstream.md](README-upstream.md) describes, then:

```bash
# every routed expert resident: hot at FP4, warm at CB2, tail at half-width CB2
EXPERT_FORMAT=tiered DSV41_TIER_MODE=allres \
DSV41_TIER_INTER_H=768 DSV41_TIER_FP4_SHARE=0.8 \
ARENA_GB=94 TRANSIENT_SLOTS=8 ./start.sh
```

| variable | meaning |
|---|---|
| `EXPERT_FORMAT=tiered` | use the multi-format arena |
| `DSV41_TIER_MODE` | `allres` (every expert resident) or `stream` (two resident tiers, the rest off NVMe) |
| `DSV41_TIER_INTER_H` | intermediate channels kept by the tail tier: 768, 1024, 1280 or 1536 (must be a multiple of 256) |
| `DSV41_TIER_FP4_SHARE` | how much of the headroom above the all-tail cost is spent on FP4 rather than CB2. 0.0 is the fast end, 1.0 the precise end |
| `ARENA_GB` | pin the arena. `allres` refuses to start rather than silently dropping experts if the budget cannot hold the tail tier |

The tier plan is printed at startup, and so is the line that matters:

```
tiered arena: 11872 cb2h @ 3.34 MB (39.7 GB) + 1282 cb2 @ 9.99 MB (12.8 GB)
            + 2214 fp4 @ 18.80 MB (41.6 GB) = 15368 slots (100.1 % of all routed experts resident)
every routed expert is resident (15360 slots): the decode path never touches NVMe and the slot LUT is permanent
```

`DSV41_TIER_FP4_SHARE` is the speed/precision dial. Spending the budget on FP4 makes the model
more faithful and slower (an FP4 expert is 105 µs against the tail's 31); spending none of it is
the fast end. Every setting keeps all 15,360 experts reachable.

### Measuring it yourself

```bash
python measure/drive.py <label>          # decode tok/s + acceptance + a free-generation check
MODE=both python measure/drive.py <label>  # the above plus teacher-forced held-out NLL
python measure/collect.py                # turn the logs into one comparison table
```

`measure/drive.py` never prints a tok/s number on its own: every run reports the acceptance length
it was taken at, how much of the routed expert set the configuration could reach, the NVMe bytes,
and the distinct-token ratio of a free generation, because on this model each of those can move the
headline number more than the change under test.

`measure/microbench/` holds the box measurements above — `peak_bw.cu`, `memkind.cu`, `nvme_bench.c`,
`kbench.py` (per-format µs/expert), `arena_size.py`, `graph_test.py` (CUDA-graph capture of the
tiered forward).

## What changed, file by file

| file | |
|---|---|
| `tools/tiered_moe.py` | **new.** `TieredArena`, `moe_forward_tiered`, the tier planners, and `build_routing_masked` |
| `tools/cb2half.py` | **new.** the half-width CB2 format and its channel selection |
| `engine/v41_engine.py` | `EXPERT_FORMAT=tiered`; the device slot LUT is enabled whenever every expert is resident, not only in pruned mode; pruned mode falls back to `coverage.json`'s mixed histogram when the per-layer trace `.npz` files are absent |
| `measure/` | **new.** the harness and the microbenchmarks |
| `corpus/heldout_corpus.jsonl` | rebuilt from the sources upstream ships (14 coding / 41 general sequences, 5,472 / 5,439 tokens) |

Everything else is upstream's, unmodified.

### One upstream bug found on the way

`fp4_moe.build_routing_small` groups the pair list by arena slot on the assumption that "a slot
never has more than BM pairs at this size". That holds for real experts (a slot appears at most
once per token) but not for the `-1` sentinel, which a multi-format arena produces in quantity: with
a 16-pair block and 21 masked pairs the writes run off the end of block 0 and corrupt block 1.
`tiered_moe.build_routing_masked` sorts the masked pairs to the end and scatters them to a discard
row instead. The upstream single-format path never hits this, because it never masks.

## Correctness

- `measure/test_tiered.py` — the tiered forward against the single-format kernels: exact (0.0)
  when one tier holds every pair; 0.4 % when the tiers are mixed, which is bf16 rounding of the
  reference's two separate accumulations, not the tiered path's.
- `measure/test_cb2h2.py` — the half-width gather against a full-width arena, with the row values
  restricted so the 2-bit codebook is lossless and identical for any subset: `w1`/`w3` rows and
  `w2` byte columns match exactly.
- `measure/microbench/graph_test.py` — CUDA-graph capture of `moe_forward_tiered`: replay is
  bit-identical to eager and re-routes correctly when the slot tensor changes.

## Credit

The engine, the kernels, the arena, the server, the trace and every measurement this fork builds on
are **0xBakeer**'s, under MIT — see [LICENSE](LICENSE), [CREDITS.md](CREDITS.md),
[README-upstream.md](README-upstream.md), [NOTES.md](NOTES.md) and [RESULTS.md](RESULTS.md), which
are kept intact. The model, the architecture and the reference implementation are DeepSeek's; the
weights carry DeepSeek's licence.
