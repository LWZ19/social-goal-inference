#!/bin/bash
# Arm 4 for one model (all kept scenes, seeds 0-2), then the secondary arm on top-level variants and bases. One worker.
# Waits for >= 3,000 MiB available before starting (polls every 5 minutes, logged).
cd results/2026-10-17-beyond-vlm/phase2/i06/score
PY=.venv/bin/python
export PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
m=$1
while [ "$(free -m | awk '/Mem:/{print $7}')" -lt 3000 ]; do echo "$(date) waiting for memory" >> logs/noisy_wait.log; sleep 300; done
for s in 0 1 2; do
  nice -n 10 $PY noisy.py $m $s models > logs/noisy_${m}_seed$s.log 2>&1
done
for s in 0 1 2; do
  nice -n 10 $PY noisy.py $m $s placeholder_only top > logs/noisy_ph_${m}_seed$s.log 2>&1
done
echo "$m done" >> logs/noisy_done.log
