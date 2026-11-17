#!/bin/bash
# Arm 1 R0 pass (copy check active): two workers at most, (seed 0 then seed 2) and (seed 1); then the reproduction check.
cd "$(dirname "$0")"
PY=${PY:-python}
export PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
L=../logs
free -m | head -2 > $L/crowd_r0_free.log
( nice -n 10 $PY crowd.py r0 0 > $L/crowd_r0_seed0.log 2>&1; nice -n 10 $PY crowd.py r0 2 > $L/crowd_r0_seed2.log 2>&1 ) &
( nice -n 10 $PY crowd.py r0 1 > $L/crowd_r0_seed1.log 2>&1 ) &
wait
nice -n 10 $PY crowd.py check_r0 > $L/crowd_check_r0.log 2>&1
echo "exit $?" >> $L/crowd_check_r0.log
