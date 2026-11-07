#!/bin/bash
# After the DeepSeek controls batch: x-test-3 main, controls, oracle ceilings for both models, then break-it.
cd results/2026-11-01-push/moving
P=.venv/bin/python
EVAL="cafe room_coffee_cart cafe_line2_moving cafe_line6_moving sweet_spot_line2_moving"
while pgrep -f "run_moving.py --model soclaas_dsv41" > /dev/null; do sleep 20; done
echo "$(date +%H:%M:%S) start x-test-3 main"
$P run_moving.py --model soclaas_qwen38_next_low --out runs/x3 --workers 3 --scenarios $EVAL cafe_busy
echo "$(date +%H:%M:%S) start x-test-3 controls"
$P run_moving.py --model soclaas_qwen38_next_low --out runs/x3 --workers 3 --module sgi_habitat.experiment_controls --scenarios $EVAL --arms RT CTv Rq0
echo "$(date +%H:%M:%S) start oracle ceilings"
$P run_moving.py --model soclaas_dsv41 --out runs/dsv41 --workers 3 --module sgi_habitat.experiment_oracle --scenarios $EVAL --arms CTo CTt
$P run_moving.py --model soclaas_qwen38_next_low --out runs/x3 --workers 3 --module sgi_habitat.experiment_oracle --scenarios $EVAL --arms CTo CTt
echo "$(date +%H:%M:%S) start x-test-3 break-it"
$P run_moving.py --model soclaas_qwen38_next_low --out runs/x3 --workers 3 --module sgi_habitat.experiment_controls --scenarios cafe_line2_fast cafe_line6_stepout cafe_line_bent_moving --arms C1 CT CTv R1 RT Rq
echo "$(date +%H:%M:%S) queue done"
