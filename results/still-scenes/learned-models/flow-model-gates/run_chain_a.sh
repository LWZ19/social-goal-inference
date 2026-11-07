#!/bin/bash
# Chain A: inner_primary -> outer_primary -> inner_dev -> outer_dev (outer_dev waits for outer_ceiling).
set -e
cd .-scenes
PY=".venv/bin/python results/2026-10-17-beyond-vlm/phase1d/c5_gates/gate2.py"
L=results/2026-10-17-beyond-vlm/phase1d/c5_gates/logs
export PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
for st in inner_primary outer_primary inner_dev outer_dev; do
  nice -n 10 $PY $st > $L/$st.log 2>&1
done
