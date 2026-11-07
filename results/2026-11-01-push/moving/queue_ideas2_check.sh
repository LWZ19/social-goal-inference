#!/bin/bash
# Ideas round 2, CT4_FROZEN.md section 5: the closed-loop reproduction check of the ported arms CT3F and CT4,
# cache-only (SGI_CACHE_ONLY=1: a cache miss is an error, never a call), on design bystander cells.
cd results/2026-11-01-push/moving
P=.venv/bin/python
M=sgi_habitat.experiment_oracle
export SGI_CACHE_ONLY=1
$P run_moving.py --model soclaas_dsv41 --out runs_ideas2_check/dsv41 --workers 2 --port-base 8870 --module $M \
  --scenarios cafe_line2_bystanders sweet_spot_line2_bystanders --seeds 30 17 44 --arms CT3F CT4 > queue_ideas2_check_A.out 2>&1 &
A=$!
$P run_moving.py --model claude_sonnet55_low --out runs_ideas2_check/sonnet55_low --workers 2 --port-base 8880 --module $M \
  --scenarios room_coffee_cart_bystanders cafe_line2_bystanders sweet_spot_line2_bystanders --seeds 12 14 15 \
  --arms CT3F CT4 > queue_ideas2_check_B.out 2>&1 &
B=$!
wait $A $B
echo "A $(grep -c 'exit 0' queue_ideas2_check_A.out) ok of 12, B $(grep -c 'exit 0' queue_ideas2_check_B.out) ok of 18"
grep -hE 'exit [^0]' queue_ideas2_check_A.out queue_ideas2_check_B.out | head -5
