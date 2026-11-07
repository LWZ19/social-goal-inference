#!/bin/bash
# After queue_all.sh: x-test-3 RTw (the world-frame R-track control).
cd results/2026-11-01-push/moving
while pgrep -f queue_all.sh > /dev/null; do sleep 30; done
echo "$(date +%H:%M:%S) start x-test-3 RTw"
.venv/bin/python run_moving.py --model soclaas_qwen38_next_low --out runs/x3 --workers 3 --module sgi_habitat.experiment_oracle --scenarios cafe room_coffee_cart cafe_line2_moving cafe_line6_moving sweet_spot_line2_moving --arms RTw
echo "$(date +%H:%M:%S) RTw done"
