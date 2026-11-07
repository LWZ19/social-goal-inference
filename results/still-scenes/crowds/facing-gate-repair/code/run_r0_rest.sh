#!/bin/bash
# R0 passes of arm 2 (all settings) and arm 1b (3 models x 3 seeds), copy check active. Two workers at most.
cd "$(dirname "$0")"
PY=${PY:-python}
export PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
L=../logs
free -m | head -2 > $L/r0_rest_free.log
(
  nice -n 10 $PY records.py r0 > $L/records_r0.log 2>&1
  for s in 0 1 2; do nice -n 10 $PY crowd_noisy.py r0 dsv $s > $L/noisy_r0_dsv_seed$s.log 2>&1; done
) &
(
  for m in qwen glm; do for s in 0 1 2; do nice -n 10 $PY crowd_noisy.py r0 $m $s > $L/noisy_r0_${m}_seed$s.log 2>&1; done; done
) &
wait
for m in qwen glm dsv; do nice -n 10 $PY crowd_noisy.py check_r0 $m > $L/noisy_check_r0_$m.log 2>&1; done
echo "r0 rest done $(date '+%F %T %z')" >> $L/r0_rest_free.log
