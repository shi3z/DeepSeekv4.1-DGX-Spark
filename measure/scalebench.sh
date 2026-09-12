#!/bin/bash
cd /home/shi3z/dsv41-spark/work
run() {
  ./stop.sh >/dev/null 2>&1; sleep 8
  env $1 DSV41_BLOCK=1 EXPERT_FORMAT=fp4 ARENA_GB=94 TRANSIENT_SLOTS=64 KEEP_FREE_GB=12 \
    MAX_SEQ=8192 PORT=8100 TRACE_STATS=results/trace-full-20260910/stats/coverage.json \
    ./start.sh --no-wait >/dev/null 2>&1
  until curl -s --max-time 5 http://127.0.0.1:8100/health >/dev/null 2>&1; do sleep 8; done
  echo "=== $2 ==="
  grep "4-bit scales" logs/server.log | tail -1
  for i in 1 2; do
    timeout 400 curl -s http://127.0.0.1:8100/v1/chat/completions -H 'content-type: application/json' \
      -d '{"messages":[{"role":"user","content":"日本の四季について、それぞれの季節の気候と代表的な行事を交えて400字程度で説明してください。"}],"max_tokens":300,"temperature":0.0}' \
    | python3 -c "
import json,sys,hashlib
d=json.load(sys.stdin)
if 'x_engine_stats' not in d: print('  ERROR',str(d)[:150]); raise SystemExit
st=d['x_engine_stats']; n=st['completion_tokens']; t=d['choices'][0]['message']['content']
print(f\"  run{sys.argv[1]} {st['decode_tok_s']:6.2f} tok/s | {st['decode_s']/n*1000:6.1f} ms/tok | nvme {st['nvme_gb_per_token']:.3f} GB/tok | hit {st['expert_hit_rate']} | sha {hashlib.sha256(t.encode()).hexdigest()[:12]}\")
" $i
  done
}
run "DSV41_SCALE4=" "8-bit scales (baseline)"
run "DSV41_SCALE4=/home/shi3z/dsv41-spark/models/scales4.bin" "4-bit scales from the side file"
echo "SCALEBENCH DONE"
