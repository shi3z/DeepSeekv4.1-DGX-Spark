#!/bin/bash
cd /home/shi3z/dsv41-spark/work
for CHUNK in 4 8 24; do
  ./stop.sh >/dev/null 2>&1; sleep 8
  DSV41_READ_CHUNK_MB=$CHUNK EXPERT_FORMAT=fp4 ARENA_GB=94 TRANSIENT_SLOTS=64 KEEP_FREE_GB=12 \
    MAX_SEQ=8192 PORT=8100 TRACE_STATS=results/trace-full-20260910/stats/coverage.json \
    ./start.sh --no-wait >/dev/null 2>&1
  until curl -s --max-time 5 http://127.0.0.1:8100/health >/dev/null 2>&1; do sleep 8; done
  echo "=== DSV41_READ_CHUNK_MB=$CHUNK ==="
  cd /home/shi3z/dsv41-spark
  timeout 400 curl -s http://127.0.0.1:8100/v1/chat/completions -H 'content-type: application/json' \
    -d '{"messages":[{"role":"user","content":"日本の四季について、それぞれの季節の気候と代表的な行事を交えて400字程度で説明してください。"}],"max_tokens":400,"temperature":0.0}' \
  | python3 -c "
import json,sys
d=json.load(sys.stdin); st=d.get('x_engine_stats',{})
n=st['completion_tokens']; lw=st['load_wait_s']; gb=st['nvme_gb']
print(f\"  {st['decode_tok_s']:6.2f} tok/s | {st['decode_s']/n*1000:6.1f} ms/tok | load_wait {lw/n*1000:6.1f} ms/tok | achieved {gb/lw:5.2f} GB/s | route {st['route_s']/n*1000:5.1f} | hit {st['expert_hit_rate']}\")
"
  cd /home/shi3z/dsv41-spark/work
done
