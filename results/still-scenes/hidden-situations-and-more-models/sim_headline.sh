#!/bin/bash
# Gemini had them. Usage: sim_headline.sh ENTRY PORT_BASE. Finished episodes are skipped, so this resumes.
D=results/still-scenes/hidden-situations-and-more-models
MV=results/moving-lines/design-and-scoring
RUNS=results/data/moving
JOBS=results/moving-lines/test-layouts
P=.venv/bin/python
m=$1; PB=$2
EVALF="cafe_slots room_coffee_cart_slots cafe_line2_moving cafe_line6_moving sweet_spot_line2_moving"
M=sgi_habitat.experiment_oracle
$P $MV/run_moving.py --model $m --out $RUNS/runs/$m --workers 2 --port-base $PB --module $M --scenarios $EVALF --seeds 10 11 12 --arms C1 R1 CT CT2 >> $D/logs/moving_${m}_a.log 2>&1; echo "$(date +%H:%M) moving $m seeds 10-12 exit $?"
$P $MV/run_moving.py --model $m --out $RUNS/runs/$m --workers 2 --port-base $PB --module $M --scenarios $EVALF --seeds 13 14 --arms R1 CT CT2 >> $D/logs/moving_${m}_b.log 2>&1; echo "$(date +%H:%M) moving $m seeds 13-14 exit $?"
echo "$(date +%H:%M) HEADLINE_DONE $m"
