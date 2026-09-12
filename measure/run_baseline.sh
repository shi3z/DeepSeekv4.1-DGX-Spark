#!/bin/bash
# Their shipped fast configuration: keep 31 % of the experts at FP4, the rest DROPPED.
cd /home/shi3z/dsv41-spark
export EXPERT_FORMAT=fp4 PK=${PK:-0.31} AG=${AG:-90.5} TRANSIENT_SLOTS=8 KEEP_FREE_GB=10 MAX_SEQ=${MAX_SEQ:-8192}
exec ./.venv/bin/python drive.py "${1:-pruned-baseline}"
