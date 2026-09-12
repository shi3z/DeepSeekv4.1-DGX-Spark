import json, sys
d = json.load(sys.stdin)
if "x_engine_stats" not in d:
    print("  ERROR", str(d)[:200]); raise SystemExit
st = d["x_engine_stats"]; n = st["completion_tokens"]
print("  run%s %6.2f tok/s | accept %5.2f | %6.1f ms/tok | nvme %.3f GB/tok | hit %s" % (
    sys.argv[1], st["decode_tok_s"], st["accept_len_mean"], st["decode_s"] / n * 1000,
    st["nvme_gb_per_token"], st["expert_hit_rate"]))
