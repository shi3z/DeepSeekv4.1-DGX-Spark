#!/bin/bash
# Greedy generation A/B: same prompts, FP4 vs CB3-from-store, full model (no pruning).
cd /home/shi3z/dsv41-spark/work
run_cfg () {
  CFG=$1; PORT=$2; STORE=$3
  ./stop.sh >/dev/null 2>&1; sleep 8
  if [ -n "$STORE" ]; then export DSV41_CB3_STORE=$STORE; else unset DSV41_CB3_STORE; fi
  DSV41_BLOCK=1 EXPERT_FORMAT=$CFG ARENA_GB=94 TRANSIENT_SLOTS=64 KEEP_FREE_GB=12 \
    MAX_SEQ=8192 PORT=$PORT TRACE_STATS=results/trace-full-20260910/stats/coverage.json \
    ./start.sh --no-wait > /tmp/genab_$CFG.start.log 2>&1
  t0=$SECONDS
  until curl -s --max-time 5 http://127.0.0.1:$PORT/health >/dev/null 2>&1; do
    sleep 8
    if [ $((SECONDS-t0)) -gt 1200 ]; then echo "server did not come up ($CFG)"; return 1; fi
  done
  echo "##### EXPERT_FORMAT=$CFG (warm start $((SECONDS-t0))s) #####"
  # two passes so the LRU is warm for the second (that is the state the tok/s numbers came from)
  for pass in 1 2; do
    n=0
    while IFS= read -r p; do
      n=$((n+1))
      echo "----- pass$pass prompt$n -----"
      timeout 900 curl -s http://127.0.0.1:$PORT/v1/chat/completions -H 'content-type: application/json' \
        -d "{\"messages\":[{\"role\":\"user\",\"content\":$p}],\"max_tokens\":260,\"temperature\":0.0}" \
        | python3 /tmp/genfmt.py
    done < /tmp/prompts.txt
  done
  ./stop.sh >/dev/null 2>&1
}
run_cfg fp4 8104 ""
run_cfg cb3 8105 /home/shi3z/dsv41-spark/models/cb3_store
echo GENAB DONE
