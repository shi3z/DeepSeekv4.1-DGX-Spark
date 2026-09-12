#!/usr/bin/env python3
"""
trace_live2.py -- routing telemetry from the serving engine, with the three things a residency
plan actually needs.

v1 counted how often each expert was picked. Frequency alone is the wrong importance measure: an
expert picked often with a small router weight contributes little, and the arena should be spent on
router MASS, not on pick counts. This version taps `route_idx`/`route_w` in both the prefill path
(`Model.moe`) and the graphed decode path (`FastDecoder`), and records per layer and per category:

  count[L, e]   how many times expert e was picked
  mass [L, e]   the router weight summed over those picks  <- what a coverage target should use
  step overlap  the fraction of a step's experts that the PREVIOUS step also used, per layer:
                the ceiling on what a "prefetch what we used last time"策 can hide

The run must be unpruned: a pruned engine masks the router to its keep-set, so its histogram
measures the keep-set and not the workload.
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
    ap.add_argument("--gen-tokens", type=int, default=32)
    ap.add_argument("--warm-trace", default="results/trace-full-20260910/stats/coverage.json")
    a = ap.parse_args()

    from engine.v41_engine import V41Engine, log

    eng = V41Engine(a.model_dir, max_seq=a.max_seq, trace_stats=a.warm_trace, spec=True,
                    prune_keep=None, arena_gb=a.arena_gb, transient_slots=a.transient_slots,
                    keep_free_gb=a.keep_free_gb, expert_format="fp4")
    assert not eng.prune_keep, "the trace must be taken unpruned"
    NL, NE = eng.args.n_layers, 384
    count = np.zeros((NL, NE), dtype=np.int64)
    mass = np.zeros((NL, NE), dtype=np.float64)
    cat_count: dict[str, np.ndarray] = {}
    cat_mass: dict[str, np.ndarray] = {}
    ov_hit = np.zeros(NL, dtype=np.int64)      # experts this step that the previous step also used
    ov_tot = np.zeros(NL, dtype=np.int64)
    prev: dict[int, set] = {}
    cur = {"cat": "all", "pend": None}

    def record(L, idx, w):
        ids = idx.reshape(-1).astype(np.int64)
        ws = w.reshape(-1).astype(np.float64)
        ok = ids >= 0
        ids, ws = ids[ok], ws[ok]
        if not ids.size:
            return
        np.add.at(count[L], ids, 1)
        np.add.at(mass[L], ids, ws)
        cc = cat_count.setdefault(cur["cat"], np.zeros((NL, NE), dtype=np.int64))
        cm = cat_mass.setdefault(cur["cat"], np.zeros((NL, NE), dtype=np.float64))
        np.add.at(cc[L], ids, 1)
        np.add.at(cm[L], ids, ws)
        s = set(ids.tolist())
        p = prev.get(L)
        if p is not None:
            ov_hit[L] += len(s & p)
            ov_tot[L] += len(s)
        prev[L] = s

    def tap_model(name, L, t):
        if name == "route_idx":
            cur["pend"] = (L, t.detach().to("cpu").numpy())
        elif name == "route_w" and cur["pend"] is not None:
            L0, idx = cur["pend"]
            cur["pend"] = None
            if L0 == L:
                record(L, idx, t.detach().to("cpu").float().numpy())

    eng.model.tap = tap_model
    fd = eng.fast
    if fd is not None:
        def tap_fast(name, L, t):
            # the fast path taps route_idx only; route_w is the buffer it just filled
            if name == "route_idx":
                record(L, t.detach().to("cpu").numpy(), fd.route_w.detach().to("cpu").float().numpy())
        fd.tap = tap_fast

    sys.path.insert(0, os.path.join(a.model_dir, "encoding"))
    from encoding import encode_messages

    seqs = [json.loads(l) for l in open(a.corpus) if l.strip()]
    log(f"tracing {len(seqs)} sequences (unpruned, tapping prefill and decode)")
    t0 = time.time()
    for i, d in enumerate(seqs):
        cur["cat"] = d.get("category", "all")
        prev.clear()
        pr = encode_messages([{"role": "user", "content": d["text"]}], thinking_mode="chat")
        ids = eng.tokenizer.encode(pr if isinstance(pr, str) else pr[0], add_special_tokens=False)
        n = 0
        for burst in eng.generate(list(ids), max_tokens=a.gen_tokens, temperature=0.0):
            n += len(burst)
        log(f"[{i + 1}/{len(seqs)}] {d.get('id', i)} prompt={len(ids)} gen={n} "
            f"picks={int(count.sum())} hit={eng.store.hit_rate():.3f} ({time.time() - t0:.0f}s)")

    os.makedirs(os.path.join(a.out, "stats"), exist_ok=True)
    per_layer = {}
    for L in range(NL):
        c = count[L].astype(np.float64)
        tot = max(1.0, c.sum())
        per_layer[str(L)] = {
            "used": int((c > 0).sum()),
            "counts": count[L].tolist(),
            "mass": mass[L].tolist(),
            "cov": (np.cumsum(np.sort(c)[::-1]) / tot).tolist(),
        }
    ov = float(ov_hit.sum()) / max(1, int(ov_tot.sum()))
    out = {"per_layer": per_layer,
           "global": {"picks": int(count.sum()), "sequences": len(seqs),
                      "corpus": os.path.abspath(a.corpus),
                      "step_overlap_with_previous": ov,
                      "step_overlap_by_layer": (ov_hit / np.maximum(1, ov_tot)).tolist(),
                      "categories": {k: int(v.sum()) for k, v in cat_count.items()},
                      "unpruned": True, "taken": time.strftime("%Y-%m-%dT%H:%M:%S")}}
    json.dump(out, open(os.path.join(a.out, "stats", "coverage.json"), "w"))
    np.savez_compressed(os.path.join(a.out, "telemetry.npz"), count=count, mass=mass,
                        ov_hit=ov_hit, ov_tot=ov_tot,
                        **{f"c_{k}": v for k, v in cat_count.items()},
                        **{f"m_{k}": v for k, v in cat_mass.items()})
    log(f"wrote {a.out}/stats/coverage.json: {int(count.sum())} picks, "
        f"{int((count > 0).sum())}/{NL * NE} experts seen, step overlap {ov:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
