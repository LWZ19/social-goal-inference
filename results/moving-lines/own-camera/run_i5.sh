#!/bin/bash
# Item 5 run block, copied from I5_FROZEN.md (hash-checked) with no changes but the shebang and the final echo. Paths since updated to the current folder layout.
cd "$(dirname "$0")/../../.."          # the repository root (run_moving.py runs the sim from there)
P=.venv/bin/python
D=results/moving-lines/design-and-scoring   # run_moving.py, tracker_service.py
R=results/data/moving                      # the run folders
EVALF="cafe_slots room_coffee_cart_slots cafe_line2_moving cafe_line6_moving sweet_spot_line2_moving"
M=sgi_habitat.experiment_oracle
F=results/moving-lines/own-camera
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ~/.venv/bin/python -I \
  results/moving-lines/own-camera/detector-on-sim-frames/e1d/detector_service.py --port 8899 > $F/detector_service.log 2>&1 &
DP=$!
$P $D/tracker_service.py --port 8897 > $F/tracker_service.log 2>&1 &
TP=$!
for i in $(seq 1 90); do curl -s http://127.0.0.1:8899/health > /dev/null 2>&1 && break; sleep 2; done
curl -s http://127.0.0.1:8899/health > /dev/null 2>&1 || { echo 'detector not healthy: stopping'; kill $TP $DP 2>/dev/null; exit 1; }
run() { m=$1; d=$2; shift 2
  SGI_CACHE_ONLY=1 SGI_TRACKER_URL=http://127.0.0.1:8897 $P $D/run_moving.py --model $m --out $R/runs_i5/$d --workers 2 \
    --port-base 8780 --module $M --scenarios $EVALF --seeds "$@" --arms CT2 CTd > $F/run_$d.log 2>&1
  echo "$(date +%H:%M) $d exit $?"; }
run claude_sonnet55_low sonnet55_low 10 11 12 13 14 15 16 17 18 19
run soclaas_dsv41 dsv41 10 11 12 13 14
run claude_haiku55 claude_haiku55 10 11 12 13 14
run gemini_fast gemini_fast 10 11 12 13 14
kill $TP $DP
echo I5_RUNS_DONE
