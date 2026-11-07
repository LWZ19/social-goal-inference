#!/bin/bash
# Fresh seeds, pool A (ports 8840+), after queue_claude.sh (CT2_FROZEN.md, CT2m_FROZEN.md, SCORING_STRICT.md).
# Core arms on seeds 10-12, then a marker for pool B (which reuses those cached answers), then seeds 13-16.
cd results/2026-11-01-push/moving
P=.venv/bin/python
EVALF="cafe_slots room_coffee_cart_slots cafe_line2_moving cafe_line6_moving sweet_spot_line2_moving"
M=sgi_habitat.experiment_oracle
CORE="C1 R1 CT CT2 CT2m Rm"
while pgrep -f "^/bin/bash ./queue_claude.sh" > /dev/null; do sleep 30; done
for pair in soclaas_dsv41:dsv41 soclaas_qwen38_next_low:x3 claude_sonnet55_low:sonnet55_low; do
  model=${pair%%:*}; name=${pair#*:}
  if [ "$name" = "sonnet55_low" ] && [ ! -f approved_sonnet_fresh ]; then echo "$(date +%H:%M:%S) Sonnet not approved: skipped"; continue; fi
  echo "$(date +%H:%M:%S) $model seeds 10-12"
  $P run_moving.py --model $model --out runs/$name --workers 3 --port-base 8840 --module $M --scenarios $EVALF --seeds 10 11 12 --arms $CORE
  touch fresh_marker_$name
  echo "$(date +%H:%M:%S) $model seeds 13-16"
  $P run_moving.py --model $model --out runs/$name --workers 3 --port-base 8840 --module $M --scenarios $EVALF --seeds 13 14 15 16 --arms $CORE
done
echo "$(date +%H:%M:%S) fresh pool A done"
