#!/bin/bash
# CT2-rat on all cached fresh cells (CT2_NONE_RAT_POSTHOC.md): Sonnet seeds 10-19 (10-12 exist and are skipped) on
# ports 8840, DeepSeek seeds 10-14 on ports 8860, in parallel. Zero calls.
cd results/2026-11-01-push/moving
P=.venv/bin/python
EVALF="cafe_slots room_coffee_cart_slots cafe_line2_moving cafe_line6_moving sweet_spot_line2_moving"
$P run_moving.py --model claude_sonnet55_low --out runs/sonnet55_low --workers 3 --port-base 8840 \
  --module sgi_habitat.experiment_oracle --scenarios $EVALF --seeds 10 11 12 13 14 15 16 17 18 19 --arms CT2-rat > runs_rat_sonnet.out 2>&1 &
A=$!
$P run_moving.py --model soclaas_dsv41 --out runs/dsv41 --workers 3 --port-base 8860 \
  --module sgi_habitat.experiment_oracle --scenarios $EVALF --seeds 10 11 12 13 14 --arms CT2-rat > runs_rat_dsv41.out 2>&1 &
B=$!
wait $A $B
echo "sonnet $(grep -c 'exit 0' runs_rat_sonnet.out) dsv41 $(grep -c 'exit 0' runs_rat_dsv41.out)"
grep -hE 'exit [^0]' runs_rat_sonnet.out runs_rat_dsv41.out | head -3
