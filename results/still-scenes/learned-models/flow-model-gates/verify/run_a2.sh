#!/bin/bash
set -e
cd "$(dirname "$0")/../../../../.."
export PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
K=results/still-scenes/learned-models/flow-model-gates/verify
nice -n 10 .venv/bin/python $K/diag_sub.py their waiting_line 100,101 > $K/log_diag_their_wl.log 2>&1
nice -n 10 .venv/bin/python $K/diag_sub.py their audience 100,101 > $K/log_diag_their_au.log 2>&1
nice -n 10 .venv/bin/python $K/kde_v.py dev > $K/log_dev.log 2>&1
nice -n 10 .venv/bin/python $K/kde_v.py ceiling:107,108,109 > $K/log_ceiling_107-109.log 2>&1
