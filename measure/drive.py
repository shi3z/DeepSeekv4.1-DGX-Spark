"""One decode measurement of one engine configuration.

Usage: drive.py <label> [--prompt code|story|html] [--tokens N]
Env selects the configuration exactly as engine/profile_fast.py does:
  EXPERT_FORMAT=fp4|cb3|tiered   PK=<prune_keep or empty>   AG=<arena GB>
  DSV41_TIER_FP4_FRAC=0.30       DSV41_TIER_COLD=cb2        TRANSIENT_SLOTS=8
Prints tok/s, acceptance, hit rate, NVMe GB and the text, so a speed number is never reported
without the evidence that the model was still answering the question.
"""
import os, sys, time, json, torch
HERE = "/home/shi3z/dsv41-spark/work"
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "tools"))
os.chdir(HERE)
from engine.v41_engine import V41Engine, log

MD = "/home/shi3z/dsv41-spark/models/DeepSeek-V4.1-Flash"
PROMPTS = {
 "code":  "Write a Python function that returns the n-th Fibonacci number, with tests.",
 "story": "Write a short story about a cartographer who discovers a map of a place that does not exist.",
 "html":  "Write a single-file HTML game. One file, no external assets.",
 "pyclass": "Write a Python class implementing an LRU cache with get/put, type hints and docstrings.",
}
label = sys.argv[1] if len(sys.argv) > 1 else "run"
which = os.environ.get("PROMPT", "code")
ntok = int(os.environ.get("TOKENS", "200"))
pk = os.environ.get("PK", "").strip()
t0 = time.time()
eng = V41Engine(MD, max_seq=int(os.environ.get("MAX_SEQ", "8192")),
                trace_stats="results/trace-full-20260910/stats/coverage.json", spec=True,
                prune_keep=(float(pk) if pk else None),
                arena_gb=(float(os.environ["AG"]) if os.environ.get("AG") else None),
                transient_slots=int(os.environ.get("TRANSIENT_SLOTS", "8")),
                keep_free_gb=float(os.environ.get("KEEP_FREE_GB", "10")),
                expert_format=os.environ.get("EXPERT_FORMAT", "fp4"))
load_s = time.time() - t0
cfg = eng.config()
print("CONFIG", json.dumps({k: cfg[k] for k in cfg if k in
      ("expert_format","prune_keep","arena_slots","arena_gb","kernel","dense_fp4","max_seq","spec")}), flush=True)
sys.path.insert(0, os.path.join(MD, "encoding"))
from encoding import encode_messages
pr = encode_messages([{"role": "user", "content": PROMPTS[which]}], thinking_mode="chat")
ids = eng.tokenizer.encode(pr if isinstance(pr, str) else pr[0], add_special_tokens=False)
MODE = os.environ.get("MODE", "speed")   # speed | quality | both
if MODE in ("quality", "both"):
    t0 = time.time()
    tf = eng.teacher_forced("corpus/heldout_corpus.jsonl", max_len=512)
    print("QUALITY " + json.dumps({k: {"nll": round(v["mean_nll"], 4), "top1": round(v["top1_acc"], 4),
                                       "n": v["n"]} for k, v in tf["summary"].items()})
          + f"  ({time.time() - t0:.0f}s)", flush=True)
    if MODE == "quality":
        raise SystemExit(0)

# A first short generation so the CUDA graphs exist and the Engram row cache is not cold: the
# repo measured that a process's first decode is slower than its second for reasons that have
# nothing to do with the configuration under test.
for _ in eng.generate(list(ids), max_tokens=8, temperature=0.0): pass

def run(n_tokens, ignore_eos):
    before = dict(eng.store.stats)
    got = []
    t0 = time.time()
    for burst in eng.generate(list(ids), max_tokens=n_tokens, temperature=0.0, ignore_eos=ignore_eos):
        got.extend(burst)
    dt = time.time() - t0
    nv = (eng.store.stats.get("bytes_read", 0) - before.get("bytes_read", 0)) / 1e9
    return got, dt, nv

# 1. the speed number: a fixed output length, so this is a decode-rate measurement and not a
#    measurement of how early each configuration decided to stop.
got, dt, nv = run(ntok, True)
n = len(got)
distinct = len(set(got)) / max(1, n)
acc = eng.last_stats.get("accept_len_mean")
print(f"RESULT {label}: {n} tok in {dt:.2f}s = {n/dt:.2f} tok/s | load {load_s:.0f}s | "
      f"acceptance {acc} | hit {eng.store.hit_rate():.3f} | NVMe {nv:.2f} GB | "
      f"distinct-token ratio {distinct:.3f}", flush=True)
print("STATS " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                             for k, v in eng.last_stats.items()}), flush=True)
print("---- text (" + which + ", ignore_eos) ----")
print(eng.tokenizer.decode(got)[:1800])
print("---- end ----", flush=True)

# 2. the degeneration check: let it stop on its own. A configuration that has dropped the experts
#    a prompt needs writes one phrase over and over; the distinct-token ratio catches that without
#    anyone having to read the text.
got2, dt2, _ = run(ntok, False)
d2 = len(set(got2)) / max(1, len(got2))
print(f"FREEGEN {label}: {len(got2)} tok, {len(got2)/max(dt2,1e-9):.2f} tok/s, "
      f"distinct-token ratio {d2:.3f} {'<-- DEGENERATE' if d2 < 0.25 else ''}", flush=True)
print("---- text (natural stop) ----")
print(eng.tokenizer.decode(got2)[:1800])
print("---- end ----", flush=True)
