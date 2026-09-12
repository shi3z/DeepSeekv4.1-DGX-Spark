import json, sys
d = json.load(sys.stdin)
try:
    t = d["choices"][0]["message"]["content"]
except Exception:
    print("ERROR", str(d)[:200]); raise SystemExit
st = d.get("x_engine_stats", {})
print("[%.2f tok/s, %d tokens]" % (st.get("decode_tok_s", 0), st.get("completion_tokens", 0)))
print(t)
