#!/bin/bash
# The engine plus a browser front end. Port 8000 belongs to something else on this box, and a
# health check alone cannot tell the two apart -- the first version of this script read another
# service's 200 as "engine up in 0s" -- so the port is checked first and the engine is identified
# by its own model id before the UI is started.
cd /home/shi3z/dsv41-spark/work || exit 1
M=/home/shi3z/dsv41-spark/models
EP=${EP:-8010}; UP=${UP:-8200}
./stop.sh >/dev/null 2>&1; sleep 6          # our own previous engine first, then the guard
if ss -ltnH "sport = :$EP" | grep -q .; then
  echo "FATAL: port $EP is held by something that is not ours:"; ss -ltnpH "sport = :$EP" | cut -c1-140; exit 1
fi
for p in $(pgrep -f "[c]hatui.py"); do kill $p; done
DSV41_DENSE_FP4=attn DSV41_HEAD_FMT=fp4 DSV41_FUSED_ATTN=0 DSV41_BLOCK=1 \
DSV41_CB3_STORE=$M/cb3_store:$M/cb3_store2:$M/cb3_store3 \
EXPERT_FORMAT=cb3 ARENA_GB=94 TRANSIENT_SLOTS=400 KEEP_FREE_GB=12 \
  MAX_SEQ=8192 PORT=$EP TRACE_STATS=results/trace-full-20260910/stats/coverage.json \
  ./start.sh --no-wait > /tmp/chat_engine.start.log 2>&1
if grep -q "^ERROR" /tmp/chat_engine.start.log; then
  echo "FATAL: start.sh refused:"; grep "^ERROR" -A2 /tmp/chat_engine.start.log; exit 1
fi
t0=$SECONDS
until curl -s --max-time 5 "http://127.0.0.1:$EP/v1/models" 2>/dev/null | grep -q "deepseek"; do
  sleep 5
  if [ $((SECONDS-t0)) -gt 1800 ]; then
    echo "FATAL: engine did not identify itself within 30 min"; tail -20 logs/server.log; exit 1
  fi
done
echo "engine up in $((SECONDS-t0))s on 127.0.0.1:$EP"
grep -m1 "fused qkv" logs/server.log
curl -s --max-time 5 "http://127.0.0.1:$EP/v1/models"; echo
nohup /usr/bin/python3 chatui.py --engine-port $EP --port $UP --host 0.0.0.0 > /tmp/chatui.log 2>&1 &
sleep 2
curl -s --max-time 10 -o /dev/null -w "UI http status %{http_code}\n" "http://127.0.0.1:$UP/"
echo "chat UI: http://100.72.123.67:$UP/   (LAN: http://192.168.120.74:$UP/)"
