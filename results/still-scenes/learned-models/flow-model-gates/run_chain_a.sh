#!/bin/bash
# Chain A: inner_primary -> outer_primary -> inner_dev -> outer_dev (outer_dev waits for outer_ceiling).
set -e
cd "$(dirname "$0")/../../../.."
PY=".venv/bin/python results/still-scenes/learned-models/flow-model-gates/gate2.py"
L=results/still-scenes/learned-models/flow-model-gates/logs
export PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
for st in inner_primary outer_primary inner_dev outer_dev; do
  nice -n 10 $PY $st > $L/$st.log 2>&1
done
