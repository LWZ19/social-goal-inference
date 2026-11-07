#!/bin/bash
# Item 9 bar runs, the frozen block (I9_FROZEN.md, Run) unchanged, plus a per-minute load log (the research
# session's request). The outage passes and the 240 s secondary are run separately, after these.
cd results/2026-11-01-push/moving
P=.venv/bin/python
EVALF="cafe_slots room_coffee_cart_slots cafe_line2_moving cafe_line6_moving sweet_spot_line2_moving"
M=sgi_habitat.experiment_oracle
F=../fill/i9_qwen_glm
PB=8800
( while true; do echo "$(date +%H:%M:%S) $(cut -d' ' -f1-3 /proc/loadavg) gpu_mib $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) sim_workers $(pgrep -fc '[s]gi_habitat.experiment_oracle')"; sleep 60; done ) >> $F/load.log 2>&1 &
LP=$!
for m in soclaas_qwen38_low soclaas_glm53; do
  $P run_moving.py --model $m --out runs/$m --workers 2 --port-base $PB --module $M --scenarios $EVALF --seeds 10 11 12 --arms C1 R1 CT CT2 > $F/run_${m}_a.log 2>&1; echo "$(date +%H:%M) $m 10-12 exit $?"
  $P run_moving.py --model $m --out runs/$m --workers 2 --port-base $PB --module $M --scenarios $EVALF --seeds 13 14 --arms R1 CT CT2 > $F/run_${m}_b.log 2>&1; echo "$(date +%H:%M) $m 13-14 exit $?"
done
kill $LP
echo I9_BAR_RUNS_DONE
