#!/usr/bin/env bash
# Verifier runner: wait for >= 3000 MiB available (poll every 5 min, logged), then run a verifier script niced, 2 threads.
# Usage: run_guarded.sh LOG SCRIPT [ARGS...]
set -u
LOG="$(realpath -m "$1")"; shift
V=results/still-scenes/crowds/verify_i06_build
while true; do
  avail=$(free -m | awk '/^Mem:/ {print $7}')
  if [ "$avail" -ge 3000 ]; then echo "$(date -Is) available ${avail} MiB: start $*" >> "$LOG"; break; fi
  echo "$(date -Is) available ${avail} MiB < 3000: waiting 5 min" >> "$LOG"; sleep 300
done
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 ${PY:-python} "$V/$@" >> "$LOG" 2>&1
echo "$(date -Is) exit $?" >> "$LOG"
