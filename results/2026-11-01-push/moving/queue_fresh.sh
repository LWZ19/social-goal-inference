#!/bin/bash
# CTs on fresh simulator seeds 10-12 (moving/CTs_FROZEN.md, sha256 1eef7429...). Starts when queue_claude.sh has
# ended, on its workers (ports 8840+). Per model: C1 first (makes the first answers), then CT and CTs on the same
# cached answers, then CT and CTs again into <name>_fresh_repeat for the redraw spread.
cd results/2026-11-01-push/moving
P=.venv/bin/python
EVAL="cafe room_coffee_cart cafe_line2_moving cafe_line6_moving sweet_spot_line2_moving"
while pgrep -f "^/bin/bash ./queue_claude.sh" > /dev/null; do sleep 30; done
for pair in soclaas_dsv41:dsv41 soclaas_qwen38_next_low:x3 claude_sonnet55_low:sonnet55_low; do
  model=${pair%%:*}; name=${pair#*:}
  echo "$(date +%H:%M:%S) $model fresh seeds"
  $P run_moving.py --model $model --out runs/$name --workers 3 --port-base 8840 --scenarios $EVAL --seeds 10 11 12 --arms C1 CT
  $P run_moving.py --model $model --out runs/$name --workers 3 --port-base 8840 --module sgi_habitat.experiment_oracle --scenarios $EVAL --seeds 10 11 12 --arms CTs
  $P run_moving.py --model $model --out runs/${name}_fresh_repeat --workers 3 --port-base 8840 --scenarios $EVAL --seeds 10 11 12 --arms CT
  $P run_moving.py --model $model --out runs/${name}_fresh_repeat --workers 3 --port-base 8840 --module sgi_habitat.experiment_oracle --scenarios $EVAL --seeds 10 11 12 --arms CTs
done
echo "$(date +%H:%M:%S) fresh queue done"
