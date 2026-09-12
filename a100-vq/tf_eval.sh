#!/bin/bash
# Teacher-forced NLL on the project's own held-out corpus, full model (no pruning),
# FP4 vs CB3-served-from-the-pre-packed-store. Same corpus and flags as RESULTS.md 3.1.
cd /home/shi3z/dsv41-spark/work
PY=/home/shi3z/dsv41-spark/.venv/bin/python
for CFG in fp4 cb3; do
  echo "=== teacher-forced, EXPERT_FORMAT=$CFG, no pruning ==="
  if [ "$CFG" = cb3 ]; then export DSV41_CB3_STORE=/home/shi3z/dsv41-spark/models/cb3_store; else unset DSV41_CB3_STORE; fi
  MODEL_DIR=/home/shi3z/dsv41-spark/models/DeepSeek-V4.1-Flash EXPERT_FORMAT=$CFG ARENA_GB=94 KEEP_FREE_GB=12 TRANSIENT_SLOTS=64 \
    TRACE_STATS=results/trace-full-20260910/stats/coverage.json \
    timeout 3600 $PY engine/v41_engine.py --teacher-forced corpus/heldout_corpus.jsonl \
      --tf-out /tmp/tf_$CFG.json 2>&1 | tail -8
done
echo TF DONE
