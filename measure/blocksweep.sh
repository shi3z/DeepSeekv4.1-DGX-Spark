#!/bin/bash
# DSV41_BLOCK = drafted positions (odd; the verify width BLOCK+1 must be even for the ratio-2
# compressor's parity). Acceptance divides the per-token expert bytes directly, so on a streaming
# configuration it is the only lever left that is worth more than a few per cent.
cd /home/shi3z/dsv41-spark/work
for B in 3 5 7 9; do
  ./stop.sh >/dev/null 2>&1; sleep 8
  DSV41_BLOCK=$B EXPERT_FORMAT=fp4 ARENA_GB=94 TRANSIENT_SLOTS=64 KEEP_FREE_GB=12 \
    MAX_SEQ=8192 PORT=8100 TRACE_STATS=results/trace-full-20260910/stats/coverage.json \
    ./start.sh --no-wait >/dev/null 2>&1
  until curl -s --max-time 5 http://127.0.0.1:8100/health >/dev/null 2>&1; do sleep 8; done
  echo "=== DSV41_BLOCK=$B (verify width $((B+1))) ==="
  for P in 'ja:日本の四季について、それぞれの季節の気候と代表的な行事を交えて400字程度で説明してください。' 'code:Write a Python class implementing an LRU cache with get/put, type hints and docstrings.'; do
    TAG=${P%%:*}; Q=${P#*:}
    timeout 400 curl -s http://127.0.0.1:8100/v1/chat/completions -H 'content-type: application/json' \
      -d "$(python3 -c "import json,sys;print(json.dumps({'messages':[{'role':'user','content':sys.argv[1]}],'max_tokens':300,'temperature':0.0}))" "$Q")" \
    | TAG=$TAG python3 -c "
import json,os,sys
d=json.load(sys.stdin); st=d.get('x_engine_stats',{})
n=st['completion_tokens']
print(f\"  {os.environ['TAG']:5s} {st['decode_tok_s']:6.2f} tok/s | accept {st['accept_len_mean']:5.2f} | {st['decode_s']/n*1000:6.1f} ms/tok | nvme {st['nvme_gb_per_token']:.3f} GB/tok | hit {st['expert_hit_rate']}\")
"
  done
done
echo "SWEEP DONE"
