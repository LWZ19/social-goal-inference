#!/bin/bash
# After queue_next.sh: the CT0 arm (one re-solve on tracked people, then frozen) for both models.
cd results/2026-11-01-push/moving
P=.venv/bin/python
EVAL="cafe room_coffee_cart cafe_line2_moving cafe_line6_moving sweet_spot_line2_moving"
while pgrep -f queue_next.sh > /dev/null; do sleep 30; done
echo "$(date +%H:%M:%S) start CT0"
$P run_moving.py --model soclaas_dsv41 --out runs/dsv41 --workers 3 --module sgi_habitat.experiment_oracle --scenarios $EVAL --arms CT0
$P run_moving.py --model soclaas_qwen38_next_low --out runs/x3 --workers 3 --module sgi_habitat.experiment_oracle --scenarios $EVAL --arms CT0
echo "$(date +%H:%M:%S) CT0 done"
