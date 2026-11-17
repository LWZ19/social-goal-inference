#!/bin/bash
# Person-snap test: the frozen detector on the 24 scenes (GPU about 60 ms a picture; no model calls).
W=results/still-scenes/group-of-four-case/snapset
E1D=results/moving-lines/own-camera/detector-on-sim-frames/e1d
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -I \
  $E1D/detector_service.py --port 8899 > $W/detector_service.log 2>&1 &
DP=$!
echo "detector pid $DP"
for i in $(seq 1 90); do curl -s http://127.0.0.1:8899/health > /dev/null 2>&1 && break; sleep 2; done
PYTHONPATH= .venv/bin/python $W/snap_detect.py
CODE=$?
kill $DP
echo "detector stopped; snap_detect exit $CODE"
