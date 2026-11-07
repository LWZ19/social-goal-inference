#!/bin/bash
# Fresh seeds, pool B (ports 8860+), after queue2.sh: on seeds 10-12 (once pool A's marker says their first answers
# are cached) the ablations, CTs, redraw repeats and break-it cases; then the core arms on seeds 17-19 (disjoint cells).
cd results/2026-11-01-push/moving
P=.venv/bin/python
EVALF="cafe_slots room_coffee_cart_slots cafe_line2_moving cafe_line6_moving sweet_spot_line2_moving"
M=sgi_habitat.experiment_oracle
CORE="C1 R1 CT CT2 CT2m Rm"
while pgrep -f "^/bin/bash ./queue2.sh" > /dev/null; do sleep 30; done
for pair in soclaas_dsv41:dsv41 soclaas_qwen38_next_low:x3 claude_sonnet55_low:sonnet55_low; do
  model=${pair%%:*}; name=${pair#*:}
  if [ "$name" = "sonnet55_low" ] && [ ! -f approved_sonnet_fresh ]; then echo "$(date +%H:%M:%S) Sonnet not approved: skipped"; continue; fi
  while [ ! -f fresh_marker_$name ]; do sleep 30; done
  echo "$(date +%H:%M:%S) $model extras"
  $P run_moving.py --model $model --out runs/$name --workers 3 --port-base 8860 --module $M --scenarios $EVALF --seeds 10 11 12 --arms CT2-vac CT2-rat CTs
  $P run_moving.py --model $model --out runs/${name}_fresh_repeat --workers 3 --port-base 8860 --module $M --scenarios $EVALF --seeds 10 11 12 --arms CT CT2
  $P run_moving.py --model $model --out runs/$name --workers 3 --port-base 8860 --module $M --scenarios cafe_line6_stepout cafe_line_bent_moving --seeds 10 11 12 --arms C1 CT CT2
  echo "$(date +%H:%M:%S) $model seeds 17-19"
  $P run_moving.py --model $model --out runs/$name --workers 3 --port-base 8860 --module $M --scenarios $EVALF --seeds 17 18 19 --arms $CORE
done
echo "$(date +%H:%M:%S) fresh pool B done"
