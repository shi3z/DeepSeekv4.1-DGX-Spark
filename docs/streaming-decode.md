# Making the streaming path faster: one thing worked, eight did not

The unpruned configuration — every routed expert reachable, nothing re-quantised, misses read from
NVMe — is the only configuration of this model on this box whose output is the checkpoint's. It is
also the slow one. This is the record of what was tried to speed it up, measured rather than
argued, so that the eight dead ends are not walked again.

Box: one GB10, 121 GiB visible, one NVMe. Arena 94 GB = 4,935 FP4 expert slots (32 % of 15,360).
Prompt: a 400-character Japanese prose request, greedy, `temperature=0`. "warm" is the second
generation in the same process — the LRU is converged and the numbers are stable; "cold" is the
first, and it is 25-35 % slower, which is larger than most of the effects below and is why every
comparison here is run-matched.

## What worked: the speculative block length

`DSV41_BLOCK` is the number of drafted positions. Upstream ships 5 (a verify width of 6), tuned on
an all-resident configuration where it measured flat between 6 and 8 and worse at 4.

On the streaming path that tuning inverts:

| `DSV41_BLOCK` | cold | **warm** | acceptance | expert hit rate | NVMe GB/token |
|---|---|---|---|---|---|
| **1** | 6.75 / 7.02 | **9.16 / 9.13** | 1.65 | **0.9585** | **0.233** |
| 3 | 6.56 | 7.36 | 2.05 | 0.9428 | 0.327 |
| 5 (upstream default) | 5.71 | 6.63 | 2.29 | 0.9400 | 0.386 |

**+38 % on the shipped default, lossless, with a configuration change.** Speculation stays on and
the verification still catches every wrong draft, so the output is unchanged.

The reason is arithmetic. Performance is

    accepted tokens / ( fixed cost + expert bytes / bandwidth )

and a verify position adds unique experts to the step. On an all-resident arena those bytes come
out of GPU memory at 234 GB/s; here they come off NVMe at 4.1 GB/s, so **the same byte is 57x more
expensive** and the optimum moves to a shorter block. Fitted on the measurements above,
`T(K) = 242 + 33K` ms per step against `A(K) = 2.07 + 0.075K` accepted tokens: the time grows three
times faster in relative terms than the acceptance does, so K wants to be as small as it is allowed
to be.

There is a second effect that the arithmetic misses. A shorter block touches fewer distinct experts
per step, which **churns the LRU less**, so the hit rate itself rises (0.9400 → 0.9585) and the
bytes fall again. `B(K)` is superlinear, not linear.

Code prompts behave differently — acceptance climbs from 3.30 to 4.33 between block 3 and 5, which
is enough to pay for the extra bytes, so block 5 stays the right default there. **The block length
belongs to the workload, not to the checkpoint.**

> The general form: on an offloaded inference path, the optimal speculative block length is set by
> the memory hierarchy, not by the model. Parameters tuned on a resident GPU are not portable to
> NVMe streaming, and the error is worth tens of per cent.

## What did not work

Every row was measured on this box, run-matched where speed was the outcome.

| | result | why |
|---|---|---|
| **All experts resident, tail at 2 bits** (`tools/cb2half.py`) | 2 of 4 factual prompts wrong; upstream's keep-31 % answers 4 of 4 | see [tiered-arena.md](tiered-arena.md) — an absent expert is less harmful than a damaged one |
| **All experts resident, tail at FP4 over fewer channels** (`tools/fp4half.py`) | 1 of 4; the worst of the three | channel truncation hurts more than 2-bit rounding |
| **A Japanese expert pack** (`tools/trace_live.py`) | 5.59 tok/s vs 6.02 for the English-ranked warm start, matched | the LRU re-converges within one prefill; the warm-start ranking washes out. The arena's *size* sets the hit rate, not its initial contents |
| **Lossless compression of the experts** (`measure/entropy_probe.py`) | FP4 payload 0.973x at the order-0 entropy limit, zstd already there, lz4 1.000x | the checkpoint's per-32 block scales make the nibble stream near-uniform: 3.89 of 4 bits. Adjacent experts XOR to 0.071 nibble-zero against 0.0625 for uncorrelated — no structure to exploit |
| **I/O chunk size** (`measure/io_probe.py`, `measure/iosweep.sh`) | 4 / 8 / 24 MB → 4.09 / 4.16 / 3.97 GB/s achieved, 5.24-5.43 tok/s | the engine is already at 86 % of what the device gives at this queue depth; the split was never the problem |
| **A better eviction policy** (`tools/miss_probe.py`) | **0 %** of decode misses were used in the previous 32 steps; 8.8 % in the previous 64; **91.2 % cold** | the LRU is already at its ceiling. There is no capacity-miss headroom for a cost-aware or router-mass-weighted policy to recover |
| **History-based prefetch** | same measurement | 91.2 % of misses were never seen in a 64-step window. Nothing that looks backwards can name them |
| **Token-conditioned prefetch** (`tools/token_routing_probe.py`) | all routing slots recall@6 **13.4 %** (8.6x chance) — but **misses only 0.68 %**, *below* the 1.56 % chance rate, and flat across all 40 layers | see below |
| **4-bit scales, unpacked outside the kernel** (`tools/pack_scales.py`) | NVMe bytes −3.0 % exactly as predicted, output bit-identical (matching SHA-256), and **2.1 % slower** | the GPU unpack sits on the critical path and costs more than the bytes save. It only pays fused into the dequant kernel |

### Why no predictor can help

Token identity carries real signal: predicting a token's historical top-6 experts recovers 13.4 %
of all routing slots against a 1.56 % chance rate, an 8.6x lift. It is still useless, because of
what it predicts. A token's habitual experts are the frequently-routed ones, and the frequently-routed
ones are exactly what the LRU is already holding. A miss is by construction an *unusual* routing
choice, so the predictor is anti-correlated with it — 0.68 % recall on misses, below chance, at
every depth.

This is not a property of this predictor. Any model of "what this context usually routes to" names
the resident set. To move NVMe traffic off the critical path a predictor would have to name the
unusual choices specifically, which means reproducing the target router, which needs the target's
contextualised hidden state, which means running the backbone — the verify step itself.

The same argument closes the drafted-token direction from the other end: the verify step **already**
batches the drafted tokens. All six positions go through layer L together, which is why a layer
reads ~16 unique experts and not 6, and why lengthening the block costs bytes at all. There is no
unexploited token-direction parallelism left; the DSpark head runs its own three MTP layers, not
the forty backbone ones, so a drafted token id says nothing further about backbone routing.

## Where the time goes

Measured on the unpruned path at block 5, 164 ms per output token:

| | ms/token | |
|---|---|---|
| NVMe reads (`load_wait`) | 106 | 0.432 GB/token at an achieved 4.1 GB/s; the device gives 4.79 at this queue depth |
| host resolve (`route_s`) | 56 | of which the D2H transfer is **2.9** and the numpy set work **5.2**; the rest is waiting for the GPU to reach that point |
| staging → arena copy | 15 | |
| MoE kernels + attention | 9 | |

Two cautions, both learned the hard way here. First, these counters are accumulated across threads
into a shared dict and do not compose: an inner timer was observed exceeding the outer one that
contains it, so the *split* is indicative and only the 164 ms wall and the independently measured
device bandwidths are load-bearing. Second, there is no large "bubble" to reclaim — an earlier
version of this document computed one by comparing against the NVMe *floor* (74 ms at 5.82 GB/s)
rather than the *achieved* 106 ms, and it does not exist.

Misses are 18.5 experts per token spread over 40 layers, so the device sees a queue depth of one or
two almost all the time. Raising it needs prefetch, and prefetch is closed above.

## What is left

| | worth | cost |
|---|---|---|
| 4-bit scales **fused into the dequant kernel** | +3-5 % (the unpack disappears, and 2.94 % smaller experts means 2.94 % more of them resident) | touching the fp4, cb2 and cb3 kernels |
| a device-side slot LUT for the streaming path | at most the 8 ms/token of transfer and numpy; the rest of `route_s` is GPU wait | new code on the hot path |
| a larger arena | +0.18 points of coverage per GB, and halving the miss mass would need 128 GB | there is no memory to find: the Engram tables are on NVMe, not in RAM (a 200,000-row host cache, 53 MB), and the only sizeable resident block is the 7.2 GB DSpark drafter, which speculation needs |

None of these change the order of magnitude. **About 10 tok/s is where this architecture sits** for
full-quality Japanese prose on one DGX Spark, and the single largest step to it was noticing that a
speculation parameter tuned for resident GPU memory is wrong by 38 % when the weights come off a
disk.
