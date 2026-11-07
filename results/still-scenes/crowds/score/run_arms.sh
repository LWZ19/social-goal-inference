#!/bin/bash
# Two workers at most: (seed 0 then seed 2) and (seed 1). Logs in score/logs/.
cd "$(dirname "$0")"
PY=${PY:-python}
export PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
free -m | head -2 > logs/arms_free.log
( nice -n 10 $PY arms.py 0 > logs/arms_seed0.log 2>&1; nice -n 10 $PY arms.py 2 > logs/arms_seed2.log 2>&1 ) &
( nice -n 10 $PY arms.py 1 > logs/arms_seed1.log 2>&1 ) &
wait
echo done >> logs/arms_free.log
