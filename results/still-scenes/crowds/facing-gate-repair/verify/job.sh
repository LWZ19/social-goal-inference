#!/bin/bash
# One verifier job: waits for >= 3000 MB available, then runs nice'd with 2 BLAS threads.
cd results/2026-10-17-beyond-vlm/phase2/i06_repair/verify
export PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
PY=.venv/bin/python
while [ "$(free -m | awk '/Mem/{print $7}')" -lt 3000 ]; do sleep 30; done
name=$1; shift
echo "$(date '+%F %T') start $name free $(free -m | awk '/Mem/{print $7}')" >> logs/jobs.log
nice -n 10 $PY -B "$@" > logs/$name.log 2>&1
echo "$(date '+%F %T') end $name rc $?" >> logs/jobs.log
