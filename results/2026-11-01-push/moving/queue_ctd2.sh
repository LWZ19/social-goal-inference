#!/bin/bash
# CTd (E1D_FROZEN.md), re-planned 2026-11-02 01:15: DeepSeek and Sonnet on fresh seeds 10-12, one worker (ports
# 8880+), each once its first answers are cached (the pool markers). Starts and stops its own detector and tracker.
cd results/2026-11-01-push/moving
P=.venv/bin/python
EVALF="cafe_slots room_coffee_cart_slots cafe_line2_moving cafe_line6_moving sweet_spot_line2_moving"
while [ ! -f fresh_marker_dsv41 ] && [ ! -f fresh_marker_sonnet55_low ]; do sleep 20; done
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -I \
  ../e1/e1d/detector_service.py --port 8899 > detector_service.log 2>&1 &
DP=$!
$P tracker_service.py --port 8897 > tracker_service.log 2>&1 &
TP=$!
echo "$(date +%H:%M:%S) detector pid $DP, tracker pid $TP"
for i in $(seq 1 90); do curl -s http://127.0.0.1:8899/health > /dev/null 2>&1 && break; sleep 2; done
done_dsv=0; done_son=0
while [ $done_dsv = 0 ] || [ $done_son = 0 ]; do
  if [ $done_dsv = 0 ] && [ -f fresh_marker_dsv41 ]; then
    echo "$(date +%H:%M:%S) dsv41 CTd"
    SGI_TRACKER_URL=http://127.0.0.1:8897 $P run_moving.py --model soclaas_dsv41 --out runs/dsv41 --workers 1 --port-base 8880 \
      --module sgi_habitat.experiment_oracle --scenarios $EVALF --seeds 10 11 12 --arms CTd
    done_dsv=1
  elif [ $done_son = 0 ] && [ -f fresh_marker_sonnet55_low ]; then
    echo "$(date +%H:%M:%S) sonnet CTd"
    SGI_TRACKER_URL=http://127.0.0.1:8897 $P run_moving.py --model claude_sonnet55_low --out runs/sonnet55_low --workers 1 --port-base 8880 \
      --module sgi_habitat.experiment_oracle --scenarios $EVALF --seeds 10 11 12 --arms CTd
    done_son=1
  else
    sleep 20
  fi
done
kill $TP $DP
echo "$(date +%H:%M:%S) CTd queue done; services stopped"
