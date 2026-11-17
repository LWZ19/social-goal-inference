#!/bin/bash
# Repair passes (only on sources whose R0 file exists; each script refuses otherwise): arm 1 (RF90, RF60, RN),
# arm 2 (RF90, RF60, RN in each fully reproduced setting), arm 1b (RF90). Two workers at most. Then merges and tables.
cd "$(dirname "$0")"
PY=${PY:-python}
export PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
L=../logs
free -m | head -2 > $L/repair_free.log
(
  nice -n 10 $PY crowd.py repair 0 > $L/crowd_rep_seed0.log 2>&1
  nice -n 10 $PY crowd.py repair 2 > $L/crowd_rep_seed2.log 2>&1
  nice -n 10 $PY records.py repair > $L/records_repair.log 2>&1
  for s in 0 1 2; do nice -n 10 $PY crowd_noisy.py rf90 dsv $s > $L/noisy_rf90_dsv_seed$s.log 2>&1; done
) &
(
  nice -n 10 $PY crowd.py repair 1 > $L/crowd_rep_seed1.log 2>&1
  for m in qwen glm; do for s in 0 1 2; do nice -n 10 $PY crowd_noisy.py rf90 $m $s > $L/noisy_rf90_${m}_seed$s.log 2>&1; done; done
) &
wait
nice -n 10 $PY crowd.py merge > $L/crowd_merge.log 2>&1
for m in qwen glm dsv; do nice -n 10 $PY crowd_noisy.py merge $m > $L/noisy_merge_$m.log 2>&1; done
echo "repair done $(date '+%F %T %z')" >> $L/repair_free.log
