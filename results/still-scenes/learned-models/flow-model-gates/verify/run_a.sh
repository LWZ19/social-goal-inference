#!/bin/bash
set -e
cd "$(dirname "$0")/../../../../.."
export PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
K=results/still-scenes/learned-models/flow-model-gates/verify
for st in primary dev ceiling:107,108,109; do
  nice -n 10 .venv/bin/python $K/kde_v.py $st > $K/log_$(echo $st | tr ':,' '__').log 2>&1
done
