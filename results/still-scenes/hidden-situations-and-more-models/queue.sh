#!/bin/bash
cd .
D=results/2026-11-01-push/e3_models
MV=results/2026-11-01-push/moving
P=.venv/bin/python
while pgrep -f "[e]2_driver.py --models (claude|gemini)" > /dev/null; do sleep 60; done
echo "$(date +%H:%M) E2 paid run finished; starting E3 tracks"
( $D/run_static.sh claude_sonnet46_low; $D/run_static.sh claude_sonnet5_low; $D/run_static.sh claude_haiku45;
  $D/run_held.sh claude_sonnet55_low; $D/run_held.sh claude_haiku55; echo "$(date +%H:%M) TRACK_CLAUDE_DONE" ) &
( $D/run_held.sh gemini_fast; echo "$(date +%H:%M) TRACK_GEMINI_DONE" ) &
( while pgrep -f "[s]gi.run .*--model gemini(3|31|35|36)" > /dev/null; do sleep 60; done   # memory: wait for the older Gemini runs
  echo "$(date +%H:%M) starting simulator track"; cd $MV
  EVALF="cafe_slots room_coffee_cart_slots cafe_line2_moving cafe_line6_moving sweet_spot_line2_moving"
  M=sgi_habitat.experiment_oracle
  for m in claude_haiku55 gemini_fast; do
    O=runs/$m
    $P run_moving.py --model $m --out $O --workers 2 --port-base 8970 --module $M --scenarios $EVALF --seeds 10 11 12 --arms C1 R1 CT CT2 > $D/logs/moving_${m}_a.log 2>&1
    $P run_moving.py --model $m --out $O --workers 2 --port-base 8970 --module $M --scenarios $EVALF --seeds 13 14 --arms R1 CT CT2 > $D/logs/moving_${m}_b.log 2>&1
    echo "$(date +%H:%M) moving headline $m done"
    $P run_moving.py --model $m --out runs_confirm4/$m --workers 2 --port-base 8970 --module $M --jobs sweep/jobs_confirm.txt --arms CT2 > $D/logs/confirm_${m}_ct2.log 2>&1
    SGI_CACHE_ONLY=1 $P run_moving.py --model $m --out runs_confirm4/$m --workers 2 --port-base 8970 --module $M --jobs sweep/jobs_confirm.txt --arms CT3F CT4P CT4PJ CT4 > $D/logs/confirm_${m}_replay.log 2>&1
    echo "$(date +%H:%M) final test $m done"
  done; echo "$(date +%H:%M) TRACK_SIM_DONE" ) &
wait; echo "$(date +%H:%M) E3_QUEUE_DONE"
