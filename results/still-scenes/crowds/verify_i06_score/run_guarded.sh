#!/usr/bin/env bash
# Verifier runner: wait for >= 3,000 MiB available (poll every 5 min, logged), then run one script niced, 2 threads.
# Usage: run_guarded.sh LOGNAME SCRIPT [ARGS...]
set -u
V=results/still-scenes/crowds/verify_i06_score
LOG="$V/logs/$1"; shift
while true; do
  avail=$(free -m | awk '/^Mem:/ {print $7}')
  if [ "$avail" -ge 3000 ]; then
    echo "$(date -Is) free -m available ${avail} MiB >= 3000: start $*" >> "$LOG"
    break
  fi
  echo "$(date -Is) free -m available ${avail} MiB < 3000: waiting 5 min" >> "$LOG"
  sleep 300
done
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 \
  ${PY:-python} "$V/$@" >> "$LOG" 2>&1
status=$?
echo "$(date -Is) exit ${status}" >> "$LOG"
exit $status
