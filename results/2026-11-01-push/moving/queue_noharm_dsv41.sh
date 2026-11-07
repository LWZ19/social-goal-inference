#!/bin/bash
# CT4_FROZEN.md section 4, "no-harm on fresh original-scene seeds 20-29" (reported): DeepSeek (free), the original moving
# scenes without bystanders plus the six-person line, seeds 20-29, arms CT2 (first answer), CT3, CT3F, CT4.
cd results/2026-11-01-push/moving
P=.venv/bin/python
$P run_moving.py --model soclaas_dsv41 --out runs_noharm/dsv41 --workers 4 --port-base 8920 \
  --module sgi_habitat.experiment_oracle \
  --scenarios cafe_line2_moving room_coffee_cart_slots sweet_spot_line2_moving cafe_line6_moving \
  --seeds 20 21 22 23 24 25 26 27 28 29 --arms CT2 CT3 CT3F CT4 > queue_noharm_dsv41.out 2>&1
echo "done: $(grep -c 'exit 0' queue_noharm_dsv41.out) ok of 160"
