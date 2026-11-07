#!/bin/bash
# CTd (CT2 on people tracked from the robot's own camera; E1D_FROZEN.md) on fresh seeds 10-12, one worker (ports
# 8880+), once pool A has cached each model's first answers (its marker). Starts the detector and tracker services
cd results/2026-11-01-push/moving
P=.venv/bin/python
EVALF="cafe_slots room_coffee_cart_slots cafe_line2_moving cafe_line6_moving sweet_spot_line2_moving"
while [ ! -f fresh_marker_dsv41 ]; do sleep 30; done
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -I \
  ../e1/e1d/detector_service.py --port 8899 > detector_service.log 2>&1 &
DP=$!
$P tracker_service.py --port 8897 > tracker_service.log 2>&1 &
TP=$!
echo "$(date +%H:%M:%S) detector pid $DP, tracker pid $TP"
for i in $(seq 1 90); do curl -s http://127.0.0.1:8899/health > /dev/null 2>&1 && break; sleep 2; done
for pair in soclaas_dsv41:dsv41 soclaas_qwen38_next_low:x3 claude_sonnet55_low:sonnet55_low; do
  model=${pair%%:*}; name=${pair#*:}
  if [ "$name" = "sonnet55_low" ] && [ ! -f approved_sonnet_fresh ]; then echo "$(date +%H:%M:%S) Sonnet not approved: skipped"; continue; fi
  while [ ! -f fresh_marker_$name ]; do sleep 30; done
  echo "$(date +%H:%M:%S) $model CTd"
  SGI_TRACKER_URL=http://127.0.0.1:8897 $P run_moving.py --model $model --out runs/$name --workers 1 --port-base 8880 \
    --module sgi_habitat.experiment_oracle --scenarios $EVALF --seeds 10 11 12 --arms CTd
done
kill $TP $DP
echo "$(date +%H:%M:%S) CTd queue done; services stopped"
