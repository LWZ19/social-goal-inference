#!/bin/bash
# waits for chain B's python (ceiling 100-106) to exit, then runs two extra subsample draws of the primary folds
while kill -0 402541 2>/dev/null; do sleep 20; done
cd "$(dirname "$0")/../../../../.."
export PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
K=results/still-scenes/learned-models/flow-model-gates/verify
for salt in 1 2; do
  nice -n 10 .venv/bin/python $K/diag_sub.py draw $salt > $K/log_draw_$salt.log 2>&1
done
