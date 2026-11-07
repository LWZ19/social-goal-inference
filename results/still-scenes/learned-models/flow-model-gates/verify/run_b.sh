#!/bin/bash
set -e
cd .-scenes
export PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
K=results/2026-10-17-beyond-vlm/phase1d/c5_gates/verify
nice -n 10 .venv/bin/python $K/kde_v.py ceiling:100,101,102,103,104,105,106 > $K/log_ceiling_100-106.log 2>&1
