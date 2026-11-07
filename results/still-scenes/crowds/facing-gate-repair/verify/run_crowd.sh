#!/bin/bash
# Verifier arm 1: 3 seeds, at most 2 at once.
cd "$(dirname "$0")"
export PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
PY=${PY:-python}
nice -n 10 $PY -B v_crowd.py 0 > logs/crowd_seed0.log 2>&1 &
nice -n 10 $PY -B v_crowd.py 1 > logs/crowd_seed1.log 2>&1
wait
nice -n 10 $PY -B v_crowd.py 2 > logs/crowd_seed2.log 2>&1
echo done
