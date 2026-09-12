#!/usr/bin/env python3
"""Classify the expert misses: capacity or cold?

A miss whose (layer, expert) was used within the last `window` steps is a CAPACITY miss -- the LRU
had it and threw it away, so a better eviction policy can remove it without predicting anything. A
miss that was not used recently is a COLD miss, and removing it needs a prediction of routing that
the drafter cannot supply: the verify step already batches the drafted tokens (that is why a layer
reads ~16 unique experts and not 6), and the DSpark head runs its own 3 MTP layers, not the 40
backbone ones, so a drafted token id says nothing about which backbone experts it will route to.

The split decides where the remaining NVMe time can be attacked at all.
"""
import argparse, collections, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "tools"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--arena-gb", type=float, default=94.0)
    ap.add_argument("--windows", default="1,2,4,8,16,32,64")
    ap.add_argument("--tokens", type=int, default=250)
    ap.add_argument("--prompt", default="日本の四季について、それぞれの季節の気候と代表的な行事を交えて400字程度で説明してください。")
    a = ap.parse_args()
    from engine.v41_engine import V41Engine, log
    eng = V41Engine(a.model_dir, max_seq=8192,
                    trace_stats="results/trace-full-20260910/stats/coverage.json", spec=True,
                    prune_keep=None, arena_gb=a.arena_gb, transient_slots=64, keep_free_gb=12.0,
                    expert_format="fp4")
    windows = [int(w) for w in a.windows.split(",")]
    store = eng.store
    inner = store.resolve
    step = {"n": 0}
    last_seen: dict[tuple, int] = {}       # (layer, expert) -> step index of last use
    hits = collections.Counter(); miss_bytes = collections.Counter()
    n_miss = {"cold": 0, **{w: 0 for w in windows}}
    tot_miss = 0

    def resolve(layer, experts, prefill):
        nonlocal tot_miss
        if not prefill:
            ex = np.unique(experts.detach().to("cpu").numpy().reshape(-1))
            ex = ex[ex >= 0]
            s = step["n"]
            for e in ex.tolist():
                key = (layer, int(e))
                resident = store.lru.get(key) is not None or store.transient_map.get(key) is not None
                if not resident:
                    tot_miss += 1
                    seen = last_seen.get(key)
                    if seen is None:
                        n_miss["cold"] += 1
                    else:
                        age = s - seen
                        placed = False
                        for w in windows:
                            if age <= w:
                                n_miss[w] += 1; placed = True; break
                        if not placed:
                            n_miss["cold"] += 1
                last_seen[key] = s
            if layer == 39:
                step["n"] += 1
        return inner(layer, experts, prefill)

    store.resolve = resolve
    sys.path.insert(0, os.path.join(a.model_dir, "encoding"))
    from encoding import encode_messages
    pr = encode_messages([{"role": "user", "content": a.prompt}], thinking_mode="chat")
    ids = eng.tokenizer.encode(pr if isinstance(pr, str) else pr[0], add_special_tokens=False)
    for _ in eng.generate(list(ids), max_tokens=8, temperature=0.0): pass   # warm the LRU
    step["n"] = 0; last_seen.clear(); n_miss = {"cold": 0, **{w: 0 for w in windows}}; tot_miss = 0
    t0 = time.time()
    n = 0
    for burst in eng.generate(list(ids), max_tokens=a.tokens, temperature=0.0):
        n += len(burst)
    log(f"{n} tokens in {time.time()-t0:.1f}s, {step['n']} steps, {tot_miss} decode misses")
    print("\nmiss classification (decode only):")
    cum = 0
    for w in windows:
        cum += n_miss[w]
        print(f"  used within the last {w:3d} steps and evicted : {n_miss[w]:6d}  "
              f"({n_miss[w]/max(1,tot_miss)*100:5.1f} %)  cumulative {cum/max(1,tot_miss)*100:5.1f} %")
    print(f"  {'cold (not seen in window)':<40s}: {n_miss['cold']:6d}  ({n_miss['cold']/max(1,tot_miss)*100:5.1f} %)")
    print(f"\n  -> {cum/max(1,tot_miss)*100:.1f} % of the misses are CAPACITY misses: the arena had them "
          f"and evicted them.\n     Those need a better eviction policy, not a prediction.")


if __name__ == "__main__":
    sys.exit(main())
