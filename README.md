# DeepSeek-V4.1-Flash on one DGX Spark — a measured negative result

> **Read this first.** This fork set out to break the dichotomy the upstream repository arrived at
> — stream every expert (full quality, 3.33 tok/s) or drop most of them (fast, and generation
> degrades) — by keeping **all 15,360 routed experts resident at reduced precision**. The machinery
> works, it is 22 % faster than upstream's pruned default, and it touches NVMe zero times during
> decode. **And the model it produces is worse than the one that throws 69 % of the experts away.**
> The hypothesis was tested on real prompts and it failed. What follows is the design, the
> measurements that refute it, and the box characterisation that is worth keeping either way.

This is a fork of **[0xBakeer/deepseek-v41-flash-spark](https://github.com/0xBakeer/deepseek-v41-flash-spark)**
(MIT, see [LICENSE](LICENSE) and [CREDITS.md](CREDITS.md)). That repository is the engine: the
pure-PyTorch port of DeepSeek-V4.1-Flash, the Triton FP4 grouped-MoE kernel, the expert arena, the
CUDA-graph decode path, the OpenAI-compatible server, and the 40-layer routing trace. Its own
README is kept here as [README-upstream.md](README-upstream.md).

## What was tried

The routed experts are 15,360 × 3 × 2304 × 5120 weights, which at the checkpoint's own FP4 with
UE8M0 scales is **288.8 GB**. A DGX Spark has ~121 GiB visible, and after the dense weights, the LM
head, the DSpark drafter's own experts and the caches, about **97 GB is left for them** — 1.43 bits
per weight for the set.

Upstream measured both ends of the resulting dichotomy and, on 2026-09-12, concluded there was
nothing in between (`NOTES.md`, 00:10):

> At 288.8 GB of FP4 experts and 121 GiB of memory there is no arrangement that keeps every expert
> resident. Either the experts stream on a miss (full quality, NVMe-bound) or some are dropped
> (fast, and a workload the keep-set does not cover degenerates). The recipe now ships the first.

The argument for a third option went: dropping an expert is not a small error, because the router
picks 6 of 384 by score and if those 6 are not resident the top 6 *of a smaller set* runs instead —
a different FFN, not a noisier one. Storing an expert coarsely should be gentler, because the right
expert still runs. So spend the 97 GB unevenly instead of spending it on a subset: the hottest
experts at the checkpoint's own FP4, the tail compressed.

Two tail formats were built and measured (`tools/cb2half.py`, `tools/fp4half.py`): 2 bits per
weight over a third of the intermediate channels, and the checkpoint's **unmodified** FP4 codes
over 22 % of them. Both keep every expert reachable.

## What happened

Four prompts, greedy, `temperature=0`, one run each, same box and same engine — only the expert
arena differs. Full transcripts in [`results/battery/`](results/battery).

| | experts the router reaches | tail | Mt. Fuji's height | capital of France | year Tokugawa founded the shogunate | reverse a string |
|---|---|---|---|---|---|---|
| **upstream, keep 31 % FP4** | 4,800 — **10,560 dropped** | — | **3,776 m** ✓ | ✓ | **1603** ✓ | ✓ |
| this fork, CB2 half-width | **all 15,360** | 2-bit, 768/2304 ch | 3,884 m ✗ | ✓ | 「元和」✗ | ✓ |
| this fork, FP4 half-width | **all 15,360** | exact FP4, 512/2304 ch | **1,000,000 m** ✗ | 巴黎 + a coordinate loop ✗ | 「永暦」✗ | ✓ |

The configuration that throws away two thirds of the experts answers all four correctly and in one
language. Both all-resident configurations get facts wrong and drift out of Japanese into Chinese;
the FP4 half-width one degenerates into the repeated-phrase failure upstream documented. Code
generation survives everywhere, which is why a code-only benchmark would have missed this entirely.

**So: for this checkpoint, an absent expert is less harmful than a damaged one.** After the fact
the reason is easy to state — pruning leaves a smaller MoE that is still internally consistent, and
the router simply selects the best of what remains, while coarsening every expert corrupts the
computation on whatever path is taken. Before the fact I had it backwards, and I read upstream's
own evidence to suit: they recorded their 3-bit CB3 format degenerating in free generation and
later found pruning to be *a* cause of degeneration; I treated the second finding as acquitting the
first. It did not.

**Upstream's dichotomy stands.** Nothing here should be run in preference to it.

## The speed result, which is real and does not matter

Same box, same prompt, 200 greedy tokens, thinking off, `MAX_SEQ=8192`, one run each:

| | upstream default (keep 31 % FP4) | this fork (all resident, CB2 half-width) |
|---|---|---|
| experts the router can reach | 4,800 of 15,360 | **15,360 of 15,360** |
| arena | 90.2 GB, one format | 94.0 GB: 2,206 FP4 + 1,282 CB2 + 11,872 CB2-half(768) |
| **decode** | 24.73 tok/s | **30.14 tok/s** |
| DSpark acceptance | 4.30 | 4.44 |
| NVMe read during decode | 0 GB | 0 GB |
| prefill, 19-token prompt | 15.21 tok/s | 8.39 tok/s |
| load, process start to ready | 108 s | 314 s |

+21.9 % on decode with every expert resident, and it is worth nothing, because the model is worse.
It is recorded because the arithmetic and the kernels behind it are sound and reusable: if a
format is ever found that this checkpoint's experts *do* survive, the arena that holds a mixture of
formats, plans a budget across them, and captures into a CUDA graph is here and tested.

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

**Prefer upstream's configuration.** These flags exist to reproduce the negative result above and
to give the machinery a home, not because the result is good.

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
