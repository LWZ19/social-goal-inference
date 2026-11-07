#!/bin/bash
# CTe (E1C_FROZEN.md): Sonnet seeds 10-12, one worker (ports 8890+), after queue_ctd2.sh has ended (it owns the
# detector and tracker services until then); starts and stops its own.
cd results/2026-11-01-push/moving
P=.venv/bin/python
EVALF="cafe_slots room_coffee_cart_slots cafe_line2_moving cafe_line6_moving sweet_spot_line2_moving"
while pgrep -f "^/bin/bash ./queue_ctd2.sh" > /dev/null; do sleep 30; done
while [ ! -f fresh_marker_sonnet55_low ]; do sleep 30; done
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -I \
  ../e1/e1d/detector_service.py --port 8899 > detector_service_cte.log 2>&1 &
DP=$!
$P tracker_service.py --port 8897 > tracker_service_cte.log 2>&1 &
TP=$!
for i in $(seq 1 90); do curl -s http://127.0.0.1:8899/health > /dev/null 2>&1 && break; sleep 2; done
echo "$(date +%H:%M:%S) sonnet CTe"
SGI_TRACKER_URL=http://127.0.0.1:8897 $P run_moving.py --model claude_sonnet55_low --out runs/sonnet55_low --workers 1 --port-base 8890 \
  --module sgi_habitat.experiment_oracle --scenarios $EVALF --seeds 10 11 12 --arms CTe
kill $TP $DP
echo "$(date +%H:%M:%S) CTe queue done; services stopped"
