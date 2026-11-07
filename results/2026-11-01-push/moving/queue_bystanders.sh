#!/bin/bash
# BYSTANDERS_FROZEN.md (sha256 683b48f0...), judged cells and no-harm cells, in two pools at once.
# Pool J (ports 8860): DeepSeek seeds 10-14 on the three bystander scenes; CT first, so that it makes the first answer
#   (about 4 free calls a cell) and C1, CT2, Gl, CT3 and CT3-j take it from the cache; R1 makes its own (3 a cell).
# Pool N (ports 8840): CT3 and CT3-j on the three originals from cached answers: DeepSeek seeds 10-14, then Sonnet
cd results/2026-11-01-push/moving
P=.venv/bin/python
M=sgi_habitat.experiment_oracle
BYST="cafe_line2_bystanders room_coffee_cart_bystanders sweet_spot_line2_bystanders"
ORIG="cafe_line2_moving room_coffee_cart_slots sweet_spot_line2_moving"

(
  echo "$(date +%H:%M:%S) J: dsv41 bystanders seeds 10-14"
  $P run_moving.py --model soclaas_dsv41 --out runs/dsv41 --workers 3 --port-base 8860 --module $M \
    --scenarios $BYST --seeds 10 11 12 13 14 --arms CT C1 CT2 Gl CT3 CT3-j R1
  echo "$(date +%H:%M:%S) J: done"
) > queue_bystanders_J.out 2>&1 &
J=$!

(
  echo "$(date +%H:%M:%S) N: dsv41 originals seeds 10-14 (cache)"
  $P run_moving.py --model soclaas_dsv41 --out runs/dsv41 --workers 3 --port-base 8840 --module $M \
    --scenarios $ORIG --seeds 10 11 12 13 14 --arms CT3 CT3-j
  echo "$(date +%H:%M:%S) N: sonnet originals seeds 10-19 (cache only)"
  SGI_CACHE_ONLY=1 $P run_moving.py --model claude_sonnet55_low --out runs/sonnet55_low --workers 3 --port-base 8840 \
    --module $M --scenarios $ORIG --seeds 10 11 12 13 14 15 16 17 18 19 --arms CT3 CT3-j
  echo "$(date +%H:%M:%S) N: done"
) > queue_bystanders_N.out 2>&1 &
N=$!

wait $J $N
echo "J $(grep -c 'exit 0' queue_bystanders_J.out) ok, N $(grep -c 'exit 0' queue_bystanders_N.out) ok"
grep -hE 'exit [^0]' queue_bystanders_J.out queue_bystanders_N.out | head -5
