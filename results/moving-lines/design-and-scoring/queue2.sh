#!/bin/bash
# The overnight queue (replaces queue_all.sh and queue_rtw.sh, which waited on their own launching shell).
# Starts at once: nothing else of the moving experiment is running.
cd results/2026-11-01-push/moving
P=.venv/bin/python
EVAL="cafe room_coffee_cart cafe_line2_moving cafe_line6_moving sweet_spot_line2_moving"
BREAK="cafe_line2_fast cafe_line6_stepout cafe_line_bent_moving"

echo "$(date +%H:%M:%S) set aside the DeepSeek re-ask episodes made with the bug (DEVIATIONS 4)"
for d in runs/dsv41/*/seed*/Rq runs/dsv41/*/seed*/Cq runs/dsv41/*/seed*/Rq0; do
  [ -d "$d" ] || continue
  dest="runs/dsv41_invalid_requery/${d#runs/dsv41/}"
  mkdir -p "$(dirname "$dest")"
  mv "$d" "$dest"
done

echo "$(date +%H:%M:%S) DeepSeek re-ask arms again"
$P run_moving.py --model soclaas_dsv41 --out runs/dsv41 --workers 3 --scenarios $EVAL cafe_busy --arms Rq Cq
$P run_moving.py --model soclaas_dsv41 --out runs/dsv41 --workers 3 --module sgi_habitat.experiment_controls --scenarios $EVAL --arms Rq0
$P run_moving.py --model soclaas_dsv41 --out runs/dsv41 --workers 3 --module sgi_habitat.experiment_controls --scenarios $BREAK --arms Rq

echo "$(date +%H:%M:%S) x-test-3 main"
$P run_moving.py --model soclaas_qwen38_next_low --out runs/x3 --workers 3 --scenarios $EVAL cafe_busy

echo "$(date +%H:%M:%S) x-test-3 controls"
$P run_moving.py --model soclaas_qwen38_next_low --out runs/x3 --workers 3 --module sgi_habitat.experiment_controls --scenarios $EVAL --arms RT CTv Rq0

echo "$(date +%H:%M:%S) oracle ceilings, CT0 and RTw"
$P run_moving.py --model soclaas_dsv41 --out runs/dsv41 --workers 3 --module sgi_habitat.experiment_oracle --scenarios $EVAL --arms CT0 CTo CTt
$P run_moving.py --model soclaas_qwen38_next_low --out runs/x3 --workers 3 --module sgi_habitat.experiment_oracle --scenarios $EVAL --arms CT0 CTo CTt RTw

echo "$(date +%H:%M:%S) x-test-3 break-it"
$P run_moving.py --model soclaas_qwen38_next_low --out runs/x3 --workers 3 --module sgi_habitat.experiment_controls --scenarios $BREAK --arms C1 CT CTv R1 RT Rq

echo "$(date +%H:%M:%S) queue done"
