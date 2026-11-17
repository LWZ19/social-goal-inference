#!/bin/bash
# Item 4 amendment 1 run block (I4_FROZEN.md f6e3bc84), unchanged: CT2 and RTw together, cache-only, into runs_i4. Paths since updated to the current folder layout.
cd "$(dirname "$0")/../../.."          # the repository root (run_moving.py runs the sim from there)
P=.venv/bin/python
D=results/moving-lines/design-and-scoring   # run_moving.py, tracker_service.py
R=results/data/moving                      # the run folders
EVALF="cafe_slots room_coffee_cart_slots cafe_line2_moving cafe_line6_moving sweet_spot_line2_moving"
M=sgi_habitat.experiment_oracle
PB=8760
L=results/moving-lines/roles-vs-marked-spot
r() { SGI_CACHE_ONLY=1 $P $D/run_moving.py --model $1 --out $R/runs_i4/$2 --workers 2 --port-base $PB --module $M --scenarios $EVALF --seeds ${@:3} --arms CT2 RTw >> $L/run_i4_$2.log 2>&1; echo "$(date +%H:%M) i4 $1 exit $?"; }
r claude_sonnet55_low sonnet55_low 10 11 12 13 14 15 16 17 18 19
r soclaas_dsv41 dsv41 10 11 12 13 14
r claude_haiku55 claude_haiku55 10 11 12 13 14
r gemini_fast gemini_fast 10 11 12 13 14
r soclaas_qwen38_next_low x3 10 11 12
echo I4_RUNS_DONE
