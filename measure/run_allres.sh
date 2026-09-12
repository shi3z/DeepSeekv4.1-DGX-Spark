#!/bin/bash
# My design: every routed expert resident, hottest in the checkpoint's FP4, tail at half-width 2-bit.
cd /home/shi3z/dsv41-spark
export EXPERT_FORMAT=tiered DSV41_TIER_MODE=allres
export DSV41_TIER_INTER_H=${IH:-768} DSV41_TIER_FP4_SHARE=${SHARE:-0.8}
export AG=${AG:-95} TRANSIENT_SLOTS=8 KEEP_FREE_GB=${KF:-10} MAX_SEQ=${MAX_SEQ:-8192}
export PK=""
exec ./.venv/bin/python drive.py "${1:-allres}"
