#!/bin/bash
set -e
cd .-scenes
export PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
K=results/2026-10-17-beyond-vlm/phase1d/c5_gates/verify
for st in primary dev ceiling:107,108,109; do
  nice -n 10 .venv/bin/python $K/kde_v.py $st > $K/log_$(echo $st | tr ':,' '__').log 2>&1
done
