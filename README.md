# DeepSeek-V4.1-Flash on one DGX Spark — one speedup, and eight measured dead ends

> **What is here.** A fork of
> [0xBakeer/deepseek-v41-flash-spark](https://github.com/0xBakeer/deepseek-v41-flash-spark) (MIT)
> that spent a day trying to make this model faster on one GB10 box. **One thing worked** — the
> speculative block length, +38 %, lossless, a configuration change — and **eight did not**, each
> closed by a measurement rather than an argument. The negative results are the larger part of the
> value: they are the things anyone attacking this box would try first.

Upstream is the engine: the pure-PyTorch port of DeepSeek-V4.1-Flash, the Triton FP4 grouped-MoE
kernel, the expert arena, the CUDA-graph decode path, the OpenAI-compatible server and the 40-layer
routing trace. Its README is kept as [README-upstream.md](README-upstream.md); its
[LICENSE](LICENSE), [CREDITS.md](CREDITS.md), [NOTES.md](NOTES.md) and [RESULTS.md](RESULTS.md) are
intact.

## The one that worked

`DSV41_BLOCK` is the number of drafted positions per speculative step. Upstream ships **5**, tuned
on an all-resident arena. On the streaming path — every expert reachable, nothing re-quantised,
misses read off NVMe — **1** is right:

| `DSV41_BLOCK` | cold | **warm** | acceptance | hit rate | NVMe GB/token |
|---|---|---|---|---|---|
| **1** | 7.02 | **9.13** | 1.65 | **0.9585** | **0.233** |
| 3 | 6.56 | 7.36 | 2.05 | 0.9428 | 0.327 |
| 5 (upstream) | 5.71 | 6.63 | 2.29 | 0.9400 | 0.386 |

**+38 %, and the output does not change** — the target still verifies every drafted token.

```bash
DSV41_BLOCK=1 ./start.sh        # Japanese prose and other low-acceptance text
DSV41_BLOCK=5 ./start.sh        # code: acceptance climbs to 4.33 there and pays for the bytes
```

Why: a verify position adds unique experts to the step, and on this path those bytes come off NVMe
at 4.1 GB/s instead of out of GPU memory at 234 GB/s — **57x more expensive per byte**, which moves
the optimum to a shorter block. A shorter block also churns the LRU less, so the hit rate *rises*
and the bytes fall again.

> Generally: **on an offloaded inference path the optimal speculative block length is set by the
> memory hierarchy, not by the model.** Speculation parameters tuned against resident GPU weights
> do not port to NVMe streaming, and here the error was 38 %.

## The eight that did not

Full numbers, methods and the tools in [docs/streaming-decode.md](docs/streaming-decode.md).

| | measured |
|---|---|
| all experts resident at 2 bits | **2 of 4** factual prompts wrong — upstream's keep-31 %, which *drops* 10,560 experts, gets **4 of 4** |
| all experts resident, FP4 over fewer channels | **1 of 4**; truncating channels hurts more than rounding weights |
| a workload-specific expert pack | 5.59 vs **6.02** tok/s for the English-ranked warm start; the LRU re-converges within one prefill |
| lossless compression of the experts | FP4 payload **0.973x** and already at its order-0 entropy limit; lz4 1.000x |
| I/O chunk size (4 / 8 / 24 MB) | 4.09 / 4.16 / **3.97** GB/s — the split was never the problem |
| a better eviction policy | **0 %** of misses were used in the previous 32 steps; the LRU has no headroom left |
| history-based prefetch | **91.2 %** of misses were never seen in a 64-step window |
| token-conditioned prefetch | 13.4 % recall on all routing slots (8.6x chance) but **0.68 % on misses** — *below* chance |

The last one closes the whole prefetch direction, and the reason is structural rather than
incidental: a predictor of "what this context usually routes to" names the frequently-routed
experts, which are exactly the ones already resident. A miss is by construction an unusual routing
choice, so any accurate predictor is anti-correlated with the thing that needs predicting. Naming
the unusual choices would mean reproducing the target router, which needs the target's hidden
state, which means running the backbone — the verify step itself.

## The quality result this fork also produced

The first thing tried here was the opposite of pruning: keep **all 15,360 routed experts** resident
by storing them coarsely, on the argument that the router's top-6 is then still the real top-6.
Two tail formats were built ([`tools/cb2half.py`](tools/cb2half.py),
[`tools/fp4half.py`](tools/fp4half.py)) and both lost to upstream's pruning on real prompts.

| | experts the router reaches | Mt. Fuji's height | capital of France | year the Tokugawa shogunate was founded | reverse a string |
|---|---|---|---|---|---|
| upstream, keep 31 % FP4 | 4,800 — **10,560 dropped** | **3,776 m** ✓ | ✓ | **1603** ✓ | ✓ |
| this fork, 2-bit tail | **all 15,360** | 3,884 m ✗ | ✓ | 「元和」✗ | ✓ |
| this fork, FP4 tail, 512/2304 channels | **all 15,360** | **1,000,000 m** ✗ | a coordinate loop ✗ | 「永暦」✗ | ✓ |

**For this checkpoint an absent expert is less harmful than a damaged one.** Pruning leaves a
smaller MoE that is still internally consistent; coarsening every expert corrupts the computation
on whatever path is taken. Code generation survived every configuration, which is why a code-only
benchmark would have missed this completely. Full reasoning in
[docs/tiered-arena.md](docs/tiered-arena.md), transcripts in [`results/battery/`](results/battery).

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

### Four kernel-level optimizations that also measured as worthless

Separate from the eight above; these are about the MoE kernels rather than the streaming path.

1. **Fusing the "5,300 small kernels per step."** They are 2.1 ms in total. Upstream's own profile
   already said this is inter-kernel latency inside a CUDA graph, not launch overhead.
2. **Huge pages / TLB reach for a large arena.** Per-expert time is flat at 108 µs from a 0.8 GB
   arena to a 30 GB one. There is no arena-size cliff.
3. **A smaller `block_m`.** BM=8 instead of 16 is worth under 4 %; BM=32 is 10x worse.
4. **Retuning the CB2 kernel's `(BN, warps, stages)`.** A full two-stage sweep found nothing better
   than the shipped `(32, 4, 3)`.

### The scales are 4 bits, and it still does not pay

An exhaustive scan of all 17.4 GB of expert UE8M0 scales finds **14 distinct values**, in the
contiguous range 115..128, so a scale is a nibble and decoding is `nibble + 115` with no lookup
table (`tools/pack_scales.py`; a *sampled* scan sees 8 and would have silently corrupted whichever
experts use the rest). It takes an expert from 18.80 to 18.25 MB, and the measured NVMe traffic
fell by exactly the predicted 3.0 % with a **bit-identical output** (matching SHA-256).

It was still **2.1 % slower**, because the GPU unpack sits on the critical path and costs more than
the bytes save. The saving only exists if the nibble is decoded *inside* the dequant kernel, which
is where the remaining +3-5 % of this idea lives.

## Running it

Install as [README-upstream.md](README-upstream.md) describes. For the streaming (full-quality)
path, the one change worth making is the block length:

```bash
DSV41_BLOCK=1 ARENA_GB=94 TRANSIENT_SLOTS=64 ./start.sh     # +38 % on Japanese prose
```

The tiered-arena flags below exist to reproduce the negative result above and to give the machinery
a home, not because the result is good.

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
| `DSV41_TIER_TAIL` | tail format: `cb2h` (2 bits over `INTER_H` channels) or `fp4h` (the checkpoint's own FP4 over them, no re-quantisation). Both measured worse than upstream's pruning |
| `DSV41_TIER_INTER_H` | intermediate channels kept by the tail tier. `cb2h` needs a multiple of 256; `fp4h` a multiple of 128 |
| `DSV41_TIER_FP4_SHARE` | how much of the headroom above the all-tail cost is spent on FP4 rather than CB2. **Lower is usually better** — see below |
| `ARENA_GB` | pin the arena. `allres` refuses to start rather than silently dropping experts if the budget cannot hold the tail tier |

The tier plan is printed at startup, and so is the line that matters:

```
tiered arena: 11872 cb2h @ 3.34 MB (39.7 GB) + 1282 cb2 @ 9.99 MB (12.8 GB)
            + 2214 fp4 @ 18.80 MB (41.6 GB) = 15368 slots (100.1 % of all routed experts resident)
every routed expert is resident (15360 slots): the decode path never touches NVMe and the slot LUT is permanent
```

**`DSV41_TIER_FP4_SHARE` is not the dial it looks like.** The obvious reading — more FP4 is more
faithful — is wrong, because what the number really controls is how many experts are left in the
*coarsest* tier. Promoting an expert from the tail to FP4 costs 14.35 MB; promoting it to CB2 costs
5.54 MB. The same bytes therefore rescue **2.6x more experts** from the tail if they are spent on
CB2, and the tail is where the damage is:

| `fp4_share` at `ARENA_GB=88`, `INTER_H=768` | routed pairs served by FP4 | by CB2 | **by the tail** | expert kernel |
|---|---|---|---|---|
| 0.8 | 55.7 % | 11.3 % | **33.0 %** | 18.42 ms/token |
| **0.4** | 41.8 % | 34.7 % | **23.6 %** | **18.19 ms/token** |
| 0.0 | 0 % | ~85 % | **~15 %** | ~19 ms/token |

`0.4` has 9.4 points less of the routing going through the coarsest tier than `0.8` **and is
marginally faster**. Every setting keeps all 15,360 experts reachable; what changes is how coarse
the least-served ones are.

### Talking to it

`start.sh` brings up the OpenAI-compatible server from upstream; `measure/chat.py` is a
standard-library terminal client for it.

```bash
EXPERT_FORMAT=tiered DSV41_TIER_MODE=allres \
DSV41_TIER_INTER_H=768 DSV41_TIER_FP4_SHARE=0.8 \
ARENA_GB=88 TRANSIENT_SLOTS=8 KEEP_FREE_GB=12 MAX_SEQ=8192 PORT=8100 ./start.sh --no-wait

python measure/chat.py --url http://127.0.0.1:8100
```

`chat.py` waits for `/health` on its own — the expert arena fills before the port opens, which
takes about four minutes for the all-resident plan — then streams, keeps the conversation, and
prints each reply's `x_engine_stats` underneath it: decode tok/s, **the acceptance length it was
achieved at**, TTFT and the expert hit rate. `/think on` puts it in reasoning mode and the
`reasoning_content` stream is shown separately. Any OpenAI client works against the same endpoint.

An interactive server wants more headroom than a benchmark: a long prompt's prefill allocates more
than a decode step, and this box stops being able to fork `sshd` if `MemAvailable` reaches zero.
`ARENA_GB=88` leaves ~13 GB free and still holds every expert — it only moves the FP4 tier from
2,206 experts to 1,903.

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
