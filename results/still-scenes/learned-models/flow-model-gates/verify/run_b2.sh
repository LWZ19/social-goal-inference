#!/bin/bash
# waits for chain B's python (ceiling 100-106) to exit, then runs two extra subsample draws of the primary folds
while kill -0 402541 2>/dev/null; do sleep 20; done
cd .-scenes
export PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
K=results/2026-10-17-beyond-vlm/phase1d/c5_gates/verify
for salt in 1 2; do
  nice -n 10 .venv/bin/python $K/diag_sub.py draw $salt > $K/log_draw_$salt.log 2>&1
done
