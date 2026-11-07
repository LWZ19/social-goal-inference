#!/bin/bash
# Sonnet = the headline model: core arms on seeds 10-12 (marker for the extras and CTd), then seeds 13-19 without C1
# (CT makes the same first answer; C1 is known to be poor). Starts when the Haiku main batch has ended.
cd results/2026-11-01-push/moving
P=.venv/bin/python
EVALF="cafe_slots room_coffee_cart_slots cafe_line2_moving cafe_line6_moving sweet_spot_line2_moving"
M=sgi_habitat.experiment_oracle
while pgrep -f "^$P run_moving.py --model claude_haiku55" > /dev/null; do sleep 20; done
echo "$(date +%H:%M:%S) sonnet seeds 10-12"
$P run_moving.py --model claude_sonnet55_low --out runs/sonnet55_low --workers 3 --port-base 8840 --module $M --scenarios $EVALF --seeds 10 11 12 --arms C1 R1 CT CT2 CT2m Rm
touch fresh_marker_sonnet55_low
echo "$(date +%H:%M:%S) sonnet seeds 13-19"
$P run_moving.py --model claude_sonnet55_low --out runs/sonnet55_low --workers 3 --port-base 8840 --module $M --scenarios $EVALF --seeds 13 14 15 16 17 18 19 --arms R1 CT CT2 CT2m Rm
echo "$(date +%H:%M:%S) pool A2 done"
