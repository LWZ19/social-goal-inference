#!/bin/bash
# three more workers (ports 8840+). Evaluation scenarios only; the arms that answer the main questions. Only the
# re-ask arms (Rq, Cq) make calls beyond the first answer; CT, CTv and RTw reuse the cached first answers.
cd results/2026-11-01-push/moving
P=.venv/bin/python
EVAL="cafe room_coffee_cart cafe_line2_moving cafe_line6_moving sweet_spot_line2_moving"
for model in claude_sonnet55_low claude_haiku55; do
  name=${model#claude_}
  echo "$(date +%H:%M:%S) $model main"
  $P run_moving.py --model $model --out runs/$name --workers 3 --port-base 8840 --scenarios $EVAL --arms C1 CT R1 Rq Cq
  echo "$(date +%H:%M:%S) $model controls"
  $P run_moving.py --model $model --out runs/$name --workers 3 --port-base 8840 --module sgi_habitat.experiment_oracle --scenarios $EVAL --arms CTv RTw CT0
done
echo "$(date +%H:%M:%S) claude queue done"
