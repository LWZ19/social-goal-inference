#!/bin/bash
# Chain B: inner_ceiling -> outer_ceiling (outer_ceiling waits for outer_primary).
set -e
cd "$(dirname "$0")/../../../.."
PY=".venv/bin/python results/still-scenes/learned-models/flow-model-gates/gate2.py"
L=results/still-scenes/learned-models/flow-model-gates/logs
export PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
for st in inner_ceiling outer_ceiling; do
  nice -n 10 $PY $st > $L/$st.log 2>&1
done
