#!/bin/bash
# E3 simulator track (relaunch with absolute paths; the first queue's track failed at its log redirect, nothing ran).
D=results/still-scenes/hidden-situations-and-more-models
MV=results/moving-lines/design-and-scoring
RUNS=results/data/moving
JOBS=results/moving-lines/test-layouts
P=.venv/bin/python
EVALF="cafe_slots room_coffee_cart_slots cafe_line2_moving cafe_line6_moving sweet_spot_line2_moving"
M=sgi_habitat.experiment_oracle
for m in claude_haiku55 gemini_fast; do
  O=$RUNS/runs/$m
  $P $MV/run_moving.py --model $m --out $O --workers 2 --port-base 8970 --module $M --scenarios $EVALF --seeds 10 11 12 --arms C1 R1 CT CT2 > $D/logs/moving_${m}_a.log 2>&1; echo "$(date +%H:%M) moving $m seeds 10-12 exit $?"
  $P $MV/run_moving.py --model $m --out $O --workers 2 --port-base 8970 --module $M --scenarios $EVALF --seeds 13 14 --arms R1 CT CT2 > $D/logs/moving_${m}_b.log 2>&1; echo "$(date +%H:%M) moving $m seeds 13-14 exit $?"
  $P $MV/run_moving.py --model $m --out $RUNS/runs_confirm4/$m --workers 2 --port-base 8970 --module $M --jobs $JOBS/jobs_confirm.txt --arms CT2 > $D/logs/confirm_${m}_ct2.log 2>&1; echo "$(date +%H:%M) final test $m CT2 exit $?"
  SGI_CACHE_ONLY=1 $P $MV/run_moving.py --model $m --out $RUNS/runs_confirm4/$m --workers 2 --port-base 8970 --module $M --jobs $JOBS/jobs_confirm.txt --arms CT3F CT4P CT4PJ CT4 > $D/logs/confirm_${m}_replay.log 2>&1; echo "$(date +%H:%M) final test $m replay exit $?"
done
echo "$(date +%H:%M) SIM_TRACK_DONE"
