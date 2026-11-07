#!/bin/bash
# C5 flow pilot step 3: the main model, then the three LOSO models, one GPU process at a time (FROZEN.md section 4).
# Each process checks nvidia-smi (>= 3000 MiB free) before it starts and polls every 5 min otherwise.
set -u
B=results/2026-10-17-beyond-vlm/phase2/flow_pilot
PY=python
cd $B/code
export HF_HUB_OFFLINE=1
for spec in "main none 40" "loso_waiting_line waiting_line 15" "loso_conversation_group conversation_group 15" "loso_audience audience 15"; do
  set -- $spec
  nice -n 10 $PY -I flow.py train $1 $2 $3 > $B/logs/train_$1.log 2>&1 || { echo "train $1 failed"; exit 1; }
done
echo "all trained"
