#!/usr/bin/env python3
"""
How much of the expert routing is explained by the TOKEN IDENTITY alone?

History-based prefetch is closed: 91.2 % of the decode misses were not touched in the previous 64
steps, so nothing that looks backwards can anticipate them. What is not closed is looking FORWARD:
the DSpark drafter hands us the next block's token ids ~13 ms before the verify step needs their
routing. Using that window requires a map from token id to the experts that token will route to.

The cheapest possible such map is a count table -- `vocab x layer -> the experts this token has
routed to before` -- which is 62 MB at top-6 and needs no training. Whether it works is a property
of the model, not of the implementation: if the router keys on the contextualised hidden state
rather than the token, the table is worthless.

This measures it directly. Every routing decision is recorded with the token that produced it; the
first half of the generation builds the table and the second half is scored against it:

    recall(L) = | actual top-6 at layer L  &  the token's historical top-6 at layer L | / 6

A recall near 1/384*6 = 1.6 % is chance. Reported per layer, because the depth is where context
should take over from identity.
"""

from __future__ import annotations

import argparse, collections, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "tools"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--arena-gb", type=float, default=94.0)
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--topk", type=int, default=6, help="predicted experts per (token, layer)")
    ap.add_argument("--prompts", default="ja")
    a = ap.parse_args()
    from engine.v41_engine import V41Engine, log

    eng = V41Engine(a.model_dir, max_seq=8192,
                    trace_stats="results/trace-full-20260910/stats/coverage.json", spec=True,
                    prune_keep=None, arena_gb=a.arena_gb, transient_slots=64, keep_free_gb=12.0,
                    expert_format="fp4")
    NL = eng.args.n_layers
    # (layer, token) -> Counter(expert)
    table: dict[tuple, collections.Counter] = collections.defaultdict(collections.Counter)
    records: list[tuple] = []      # (layer, token, frozenset(experts))
    block = {"ids": None}

    m = eng.model
    fd = eng.fast
    prev_tap_m, prev_tap_f = getattr(m, "tap", None), (getattr(fd, "tap", None) if fd else None)

    store = eng.store

    def note(L, idx):
        ids = block["ids"]
        if ids is None:
            return
        e = idx.detach().to("cpu").numpy()
        if e.ndim == 1:
            e = e[None, :]
        n = min(len(ids), e.shape[0])
        for p in range(n):
            ex = [int(x) for x in e[p] if x >= 0]
            # a slot only needs prefetching if it is NOT resident; hits inflate a plain recall
            miss = frozenset(x for x in ex
                             if store.lru.get((L, x)) is None and store.transient_map.get((L, x)) is None)
            records.append((L, int(ids[p]), frozenset(ex), miss))

    def tap_model(name, L, t):
        if name == "route_idx":
            note(L, t)
    m.tap = tap_model
    if fd is not None:
        def tap_fast(name, L, t):
            if name == "route_idx":
                note(L, t)
        fd.tap = tap_fast
        # the verify block's token ids are the buffer the step was handed
        orig_step = fd.step
        def step(block_ids, S, rows):
            block["ids"] = block_ids.detach().to("cpu").tolist()
            return orig_step(block_ids, S, rows)
        fd.step = step

    sys.path.insert(0, os.path.join(a.model_dir, "encoding"))
    from encoding import encode_messages
    PROMPTS = {"ja": "日本の四季について、それぞれの季節の気候と代表的な行事を交えて500字程度で説明してください。",
               "code": "Write a Python class implementing an LRU cache with get/put, type hints and docstrings."}
    for which in a.prompts.split(","):
        pr = encode_messages([{"role": "user", "content": PROMPTS[which]}], thinking_mode="chat")
        ids = eng.tokenizer.encode(pr if isinstance(pr, str) else pr[0], add_special_tokens=False)
        t0 = time.time(); n = 0
        for burst in eng.generate(list(ids), max_tokens=a.tokens, temperature=0.0):
            n += len(burst)
        log(f"{which}: {n} tokens in {time.time()-t0:.0f}s, {len(records):,} routing records")

    if not records:
        print("no routing records captured"); return 1
    half = len(records) // 2
    for (L, tok, ex, ms) in records[:half]:
        table[(L, tok)].update(ex)

    hit = np.zeros(NL); tot = np.zeros(NL); seen = np.zeros(NL)
    hit12 = np.zeros(NL)
    mhit = np.zeros(NL); mtot = np.zeros(NL); mhit12 = np.zeros(NL)
    for (L, tok, ex, ms) in records[half:]:
        tot[L] += len(ex); mtot[L] += len(ms)
        c = table.get((L, tok))
        if not c:
            continue
        seen[L] += len(ex)
        pred = {e for e, _ in c.most_common(a.topk)}
        p12 = {e for e, _ in c.most_common(2 * a.topk)}
        hit[L] += len(ex & pred); hit12[L] += len(ex & p12)
        mhit[L] += len(ms & pred); mhit12[L] += len(ms & p12)
    ok = tot > 0
    chance = a.topk / 384
    print(f"\ntoken-conditioned routing prediction, top-{a.topk} from the token's own history")
    print(f"  scored on {int(tot.sum()):,} routing slots; chance = {chance*100:.2f} %")
    print(f"  {'layer':>6s} {'recall':>8s} {'recall | token seen before':>28s} {'coverage':>10s}")
    for L in range(NL):
        if not ok[L]: continue
        r = hit[L] / tot[L]
        rs = hit[L] / seen[L] if seen[L] else 0.0
        if L % 4 == 0 or L == NL - 1:
            print(f"  {L:6d} {r*100:7.2f}% {rs*100:27.2f}% {seen[L]/tot[L]*100:9.1f}%")
    R = hit.sum() / tot.sum(); R12 = hit12.sum() / tot.sum()
    MR = mhit.sum() / max(1.0, mtot.sum()); MR12 = mhit12.sum() / max(1.0, mtot.sum())
    print(f"\n  all routing slots : recall@{a.topk} {R*100:.2f} %   recall@{2*a.topk} {R12*100:.2f} %"
          f"   (chance {chance*100:.2f} %, lift {R/chance:.1f}x)")
    print(f"  MISSES only       : recall@{a.topk} {MR*100:.2f} %   recall@{2*a.topk} {MR12*100:.2f} %"
          f"   over {int(mtot.sum()):,} missing slots")
    print(f"\n  the misses are what a prefetch has to name; hits need nothing.")
    print(f"  per-layer MISS recall@{a.topk}:")
    for L in range(0, NL, 4):
        if mtot[L] > 0:
            print(f"    layer {L:2d}: {mhit[L]/mtot[L]*100:5.1f} %   ({int(mtot[L]):,} misses)")
    nvme_gb_tok = 0.233
    print(f"\n  at {nvme_gb_tok} GB/token of NVMe, naming {MR*100:.0f} % of the misses "
          f"{'moves' if MR > 0.1 else 'would move'} {nvme_gb_tok*MR:.3f} GB/token "
          f"({nvme_gb_tok*MR/4.1*1000:.0f} ms/token) off the critical path -- IF it can be "
          f"submitted early enough.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
