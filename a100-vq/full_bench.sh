#!/bin/bash
# Full-store CB3: the repeated-prompt run (comparable to the earlier 18.28) and the diverse-prompt
# run (the one the partial store lost on: 1.2-3.8 tok/s against FP4's 5.4-7.3).
cd /home/shi3z/dsv41-spark/work
M=/home/shi3z/dsv41-spark/models
./stop.sh >/dev/null 2>&1; sleep 8
DSV41_CB3_STORE=$M/cb3_store:$M/cb3_store2:$M/cb3_store3 \
DSV41_BLOCK=1 EXPERT_FORMAT=cb3 ARENA_GB=94 TRANSIENT_SLOTS=64 KEEP_FREE_GB=12 \
  MAX_SEQ=8192 PORT=8106 TRACE_STATS=results/trace-full-20260910/stats/coverage.json \
  ./start.sh --no-wait > /tmp/fullbench.start.log 2>&1
t0=$SECONDS
until curl -s --max-time 5 http://127.0.0.1:8106/health >/dev/null 2>&1; do
  sleep 8
  if [ $((SECONDS-t0)) -gt 1500 ]; then echo "server did not come up"; tail -25 /tmp/fullbench.start.log; exit 1; fi
done
echo "=== full store (warm start $((SECONDS-t0))s) ==="
grep -h -m1 'a100-vq' /tmp/fullbench.start.log logs/*.log 2>/dev/null | head -1
echo "--- A: same prompt x3 ---"
for i in 1 2 3; do
  timeout 900 curl -s http://127.0.0.1:8106/v1/chat/completions -H 'content-type: application/json' \
    -d '{"messages":[{"role":"user","content":"日本の四季について、それぞれの季節の気候と代表的な行事を交えて400字程度で説明してください。"}],"max_tokens":300,"temperature":0.0}' \
  | python3 /tmp/statfmt.py $i
done
echo "--- B: five different prompts, two passes ---"
for pass in 1 2; do
  n=0
  while IFS= read -r p; do
    n=$((n+1))
    timeout 900 curl -s http://127.0.0.1:8106/v1/chat/completions -H 'content-type: application/json' \
      -d "{\"messages\":[{\"role\":\"user\",\"content\":$p}],\"max_tokens\":260,\"temperature\":0.0}" \
    | python3 /tmp/statfmt.py "pass$pass-p$n"
  done < /tmp/prompts.txt
done
./stop.sh >/dev/null 2>&1
echo FULLBENCH DONE
