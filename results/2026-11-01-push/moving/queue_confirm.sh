#!/bin/bash
# BYSTANDERS_CONFIRM_FROZEN.md (d7dba68e): DeepSeek, cafe_line2_bystanders + sweet_spot_line2_bystanders, seeds 15-44,
# arms CT2 (makes the first answer), CT3, Gl, CT3b. Two pools of 3 workers: seeds 15-29 on 8860, 30-44 on 8840.
cd results/2026-11-01-push/moving
P=.venv/bin/python
M=sgi_habitat.experiment_oracle
SC="cafe_line2_bystanders sweet_spot_line2_bystanders"
$P run_moving.py --model soclaas_dsv41 --out runs/dsv41 --workers 3 --port-base 8860 --module $M \
  --scenarios $SC --seeds 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 --arms CT2 CT3 Gl CT3b > queue_confirm_A.out 2>&1 &
A=$!
$P run_moving.py --model soclaas_dsv41 --out runs/dsv41 --workers 3 --port-base 8840 --module $M \
  --scenarios $SC --seeds 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 --arms CT2 CT3 Gl CT3b > queue_confirm_B.out 2>&1 &
B=$!
wait $A $B
echo "A $(grep -c 'exit 0' queue_confirm_A.out) ok, B $(grep -c 'exit 0' queue_confirm_B.out) ok (of 120 each)"
grep -hE 'exit [^0]' queue_confirm_A.out queue_confirm_B.out | head -5
