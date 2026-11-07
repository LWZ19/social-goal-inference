#!/bin/bash
# E2: A1's frozen detector on the 40 scenes (GPU about 60 ms a picture; no model calls). As group4/snapset/run_detect.sh.
cd results/2026-11-01-push/e2
E1D=results/2026-11-01-push/e1/e1d
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -I \
  $E1D/detector_service.py --port 8899 > detector_service.log 2>&1 &
DP=$!
echo "detector pid $DP"
for i in $(seq 1 90); do curl -s http://127.0.0.1:8899/health > /dev/null 2>&1 && break; sleep 2; done
PYTHONPATH= .venv/bin/python e2_detect.py --scenes run_scenes --out e2_people.json
CODE=$?
kill $DP
echo "detector stopped; e2_detect exit $CODE"
