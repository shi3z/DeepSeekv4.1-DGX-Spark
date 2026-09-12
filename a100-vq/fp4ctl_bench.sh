#!/bin/bash
# unpruned CB3 stream served from the pre-packed store (a100-vq/pack_store.py)
cd /home/shi3z/dsv41-spark/work
./stop.sh >/dev/null 2>&1; sleep 8

DSV41_BLOCK=1 EXPERT_FORMAT=fp4 ARENA_GB=94 TRANSIENT_SLOTS=64 KEEP_FREE_GB=12 \
  MAX_SEQ=8192 PORT=8103 TRACE_STATS=results/trace-full-20260910/stats/coverage.json \
  ./start.sh --no-wait > /tmp/fp4ctl.start.log 2>&1
t0=$SECONDS
until curl -s --max-time 5 http://127.0.0.1:8103/health >/dev/null 2>&1; do
  sleep 8
  if [ $((SECONDS-t0)) -gt 1200 ]; then echo "server did not come up"; tail -25 /tmp/fp4ctl.start.log; exit 1; fi
done
echo "=== FP4 baseline, 3 runs (warm start $((SECONDS-t0))s) ==="
grep -h -m1 'a100-vq' /tmp/fp4ctl.start.log /home/shi3z/dsv41-spark/work/logs/*.log 2>/dev/null | head -1
for i in 1 2 3; do
  timeout 600 curl -s http://127.0.0.1:8103/v1/chat/completions -H 'content-type: application/json' \
    -d '{"messages":[{"role":"user","content":"日本の四季について、それぞれの季節の気候と代表的な行事を交えて400字程度で説明してください。"}],"max_tokens":300,"temperature":0.0}' \
  | python3 /tmp/statfmt.py $i
done
./stop.sh >/dev/null 2>&1
echo FP4CTL DONE
