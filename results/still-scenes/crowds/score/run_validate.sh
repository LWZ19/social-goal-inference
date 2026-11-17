#!/bin/bash
# Validation of the noise model, one model after another in one worker (the other worker slot is used elsewhere).
cd "$(dirname "$0")"
PY=${PY:-python}
export PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
for m in "$@"; do
  free -m | head -2 >> logs/validate_free.log
  nice -n 10 $PY validate.py $m > logs/validate_$m.log 2>&1
done
