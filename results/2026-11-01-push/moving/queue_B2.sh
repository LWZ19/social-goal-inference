#!/bin/bash
# Pool B (ports 8860+), re-planned 2026-11-02 01:15. Starts when the x-test-3 oracle batch has ended.
# DeepSeek = the free robustness model (seeds 10-14); then the one-model mechanism checks on Sonnet (ablations,
# repeats, break-it, CTs) once pool A's Sonnet marker exists; x-test-3 only for the latency check.
cd results/2026-11-01-push/moving
P=.venv/bin/python
EVALF="cafe_slots room_coffee_cart_slots cafe_line2_moving cafe_line6_moving sweet_spot_line2_moving"
M=sgi_habitat.experiment_oracle
while pgrep -f "^$P run_moving.py --model soclaas_qwen38_next_low" > /dev/null; do sleep 20; done
echo "$(date +%H:%M:%S) dsv41 seeds 10-12"
$P run_moving.py --model soclaas_dsv41 --out runs/dsv41 --workers 3 --port-base 8860 --module $M --scenarios $EVALF --seeds 10 11 12 --arms C1 R1 CT CT2 CT2m Rm
touch fresh_marker_dsv41
echo "$(date +%H:%M:%S) dsv41 seeds 13-14"
$P run_moving.py --model soclaas_dsv41 --out runs/dsv41 --workers 3 --port-base 8860 --module $M --scenarios $EVALF --seeds 13 14 --arms R1 CT CT2 CT2m Rm
while [ ! -f fresh_marker_sonnet55_low ]; do sleep 20; done
echo "$(date +%H:%M:%S) sonnet extras"
$P run_moving.py --model claude_sonnet55_low --out runs/sonnet55_low --workers 3 --port-base 8860 --module $M --scenarios $EVALF --seeds 10 11 12 --arms CT2-vac CT2-rat
$P run_moving.py --model claude_sonnet55_low --out runs/sonnet55_low_fresh_repeat --workers 3 --port-base 8860 --module $M --scenarios $EVALF --seeds 10 11 12 --arms CT CT2
$P run_moving.py --model claude_sonnet55_low --out runs/sonnet55_low --workers 3 --port-base 8860 --module $M --scenarios cafe_line6_stepout cafe_line_bent_moving --seeds 10 11 12 --arms C1 CT CT2
echo "$(date +%H:%M:%S) sonnet CTs"
$P run_moving.py --model claude_sonnet55_low --out runs/sonnet55_low --workers 3 --port-base 8860 --module $M --scenarios $EVALF --seeds 10 11 12 --arms CTs
echo "$(date +%H:%M:%S) x3 latency check"
$P run_moving.py --model soclaas_qwen38_next_low --out runs/x3 --workers 3 --port-base 8860 --module $M --scenarios $EVALF --seeds 10 11 12 --arms CT2 CT2m R1 Rm
echo "$(date +%H:%M:%S) pool B2 done"
