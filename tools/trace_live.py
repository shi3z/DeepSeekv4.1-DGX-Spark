#!/usr/bin/env python3
"""
trace_live.py -- record which routed experts a corpus actually uses, from the serving engine.

`tools/expert_trace.py` does this offline by streaming one 7.4 GB layer shard at a time, which is
how it was possible before the whole checkpoint was on disk. With the checkpoint local there is a
cheaper route: `ExpertStore.resolve()` is handed the expert ids of every layer of every prefill
chunk and, in the UNPRUNED configuration, of every decode step as well, because the device slot LUT
that bypasses it is only built when nothing can miss. Wrapping it gives the same per-layer
histogram from the real engine on real prompts.

The run must be unpruned. A pruned engine masks the router to its keep-set, so the histogram it
produces is a measurement of the keep-set, not of the workload.

Output is a `coverage.json` with the same `per_layer[L]["counts"]` that `experts.rank_from_trace`
reads, so it can be pointed at with TRACE_STATS and used to warm-start an arena.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arena-gb", type=float, default=94.0)
    ap.add_argument("--transient-slots", type=int, default=256)
    ap.add_argument("--keep-free-gb", type=float, default=12.0)
    ap.add_argument("--max-seq", type=int, default=8192)
    ap.add_argument("--gen-tokens", type=int, default=24, help="tokens to generate per sequence, so "
                    "the histogram reflects decode routing and not only prefill")
    ap.add_argument("--warm-trace", default="results/trace-full-20260910/stats/coverage.json",
                    help="trace used to warm-start the arena for this run; it only affects how fast "
                         "the trace is collected, not what is collected")
    a = ap.parse_args()

    os.environ.setdefault("DSV41_LUT", "0")   # belt and braces: never bypass resolve()
    from engine.v41_engine import V41Engine, log

    eng = V41Engine(a.model_dir, max_seq=a.max_seq, trace_stats=a.warm_trace, spec=True,
                    prune_keep=None, arena_gb=a.arena_gb, transient_slots=a.transient_slots,
                    keep_free_gb=a.keep_free_gb, expert_format="fp4")
    assert eng.prune_keep in (None, 1.0), "the trace must be taken unpruned"
    n_layers, n_exp = eng.args.n_layers, eng.args.n_routed_experts
    counts = np.zeros((n_layers, n_exp), dtype=np.int64)
    per_cat: dict[str, np.ndarray] = {}
    cur = {"cat": "?"}

    store = eng.store
    inner = store.resolve

    def resolve(layer, experts, prefill):
        ex = experts.detach().to("cpu").reshape(-1).numpy()
        ex = ex[ex >= 0]
        if ex.size:
            np.add.at(counts[layer], ex, 1)
            c = per_cat.setdefault(cur["cat"], np.zeros((n_layers, n_exp), dtype=np.int64))
            np.add.at(c[layer], ex, 1)
        return inner(layer, experts, prefill)

    store.resolve = resolve

    sys.path.insert(0, os.path.join(a.model_dir, "encoding"))
    from encoding import encode_messages

    seqs = [json.loads(l) for l in open(a.corpus) if l.strip()]
    log(f"tracing {len(seqs)} sequences")
    t0 = time.time()
    for i, d in enumerate(seqs):
        cur["cat"] = d.get("category", "all")
        pr = encode_messages([{"role": "user", "content": d["text"]}], thinking_mode="chat")
        ids = eng.tokenizer.encode(pr if isinstance(pr, str) else pr[0], add_special_tokens=False)
        n = 0
        for burst in eng.generate(list(ids), max_tokens=a.gen_tokens, temperature=0.0):
            n += len(burst)
        log(f"[{i + 1}/{len(seqs)}] {d.get('id', i)} prompt={len(ids)} gen={n} "
            f"pairs={int(counts.sum())} hit={eng.store.hit_rate():.3f} ({time.time() - t0:.0f}s)")

    os.makedirs(a.out, exist_ok=True)
    stats_dir = os.path.join(a.out, "stats")
    os.makedirs(stats_dir, exist_ok=True)
    per_layer = {}
    for L in range(n_layers):
        c = counts[L].astype(np.float64)
        order = np.sort(c)[::-1]
        tot = max(1.0, c.sum())
        per_layer[str(L)] = {
            "used": int((c > 0).sum()),
            "counts": counts[L].tolist(),
            "cov": (np.cumsum(order) / tot).tolist(),
            "entropy_bits": float(-(np.where(c > 0, c / tot * np.log2(c / tot + 1e-30), 0)).sum()),
        }
    out = {"per_layer": per_layer,
           "global": {"pairs": int(counts.sum()), "corpus": os.path.abspath(a.corpus),
                      "sequences": len(seqs), "gen_tokens": a.gen_tokens,
                      "categories": {k: int(v.sum()) for k, v in per_cat.items()},
                      "taken": time.strftime("%Y-%m-%dT%H:%M:%S"), "unpruned": True}}
    p = os.path.join(stats_dir, "coverage.json")
    json.dump(out, open(p, "w"))
    np.savez_compressed(os.path.join(a.out, "counts.npz"), counts=counts,
                        **{f"cat_{k}": v for k, v in per_cat.items()})
    log(f"wrote {p} ({int(counts.sum())} routed pairs, "
        f"{int((counts > 0).sum())} of {n_layers * n_exp} experts seen)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
