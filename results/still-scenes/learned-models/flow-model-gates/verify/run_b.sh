#!/bin/bash
set -e
cd "$(dirname "$0")/../../../../.."
export PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
K=results/still-scenes/learned-models/flow-model-gates/verify
nice -n 10 .venv/bin/python $K/kde_v.py ceiling:100,101,102,103,104,105,106 > $K/log_ceiling_100-106.log 2>&1
