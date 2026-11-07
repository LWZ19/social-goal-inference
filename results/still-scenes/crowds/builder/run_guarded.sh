#!/usr/bin/env bash
# Run one builder script under the shared-machine rules: wait until at least 3,000 MiB are available (poll every
# 5 minutes, logged), then run it niced with 2 threads. Usage: run_guarded.sh LOG SCRIPT [ARGS...]
set -u
LOG="$(realpath -m "$1")"; shift
I06=results/2026-10-17-beyond-vlm/phase2/i06
mkdir -p "$(dirname "$LOG")"
while true; do
  avail=$(free -m | awk '/^Mem:/ {print $7}')
  if [ "$avail" -ge 3000 ]; then
    echo "$(date -Is) free -m available ${avail} MiB >= 3000: start $*" >> "$LOG"
    break
  fi
  echo "$(date -Is) free -m available ${avail} MiB < 3000: waiting 5 min" >> "$LOG"
  sleep 300
done
cd .-scenes || exit 1
PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 \
  .venv/bin/python "$I06/builder/$@" >> "$LOG" 2>&1
status=$?
echo "$(date -Is) exit ${status}" >> "$LOG"
exit $status
