#!/bin/bash
# E3 simulator work for one model: moving-line headline (skipped where done), then the final test (CT2, then replays).
# Usage: sim_model.sh ENTRY PORT_BASE. Finished episodes are skipped, so this resumes.
D=results/2026-11-01-push/e3_models
MV=results/2026-11-01-push/moving
P=.venv/bin/python
m=$1; PB=$2; cd $MV
EVALF="cafe_slots room_coffee_cart_slots cafe_line2_moving cafe_line6_moving sweet_spot_line2_moving"
M=sgi_habitat.experiment_oracle
$P run_moving.py --model $m --out runs/$m --workers 2 --port-base $PB --module $M --scenarios $EVALF --seeds 10 11 12 --arms C1 R1 CT CT2 >> $D/logs/moving_${m}_a.log 2>&1; echo "$(date +%H:%M) moving $m seeds 10-12 exit $?"
$P run_moving.py --model $m --out runs/$m --workers 2 --port-base $PB --module $M --scenarios $EVALF --seeds 13 14 --arms R1 CT CT2 >> $D/logs/moving_${m}_b.log 2>&1; echo "$(date +%H:%M) moving $m seeds 13-14 exit $?"
$P run_moving.py --model $m --out runs_confirm4/$m --workers 2 --port-base $PB --module $M --jobs sweep/jobs_confirm.txt --arms CT2 >> $D/logs/confirm_${m}_ct2.log 2>&1; echo "$(date +%H:%M) final test $m CT2 exit $?"
SGI_CACHE_ONLY=1 $P run_moving.py --model $m --out runs_confirm4/$m --workers 2 --port-base $PB --module $M --jobs sweep/jobs_confirm.txt --arms CT3F CT4P CT4PJ CT4 >> $D/logs/confirm_${m}_replay.log 2>&1; echo "$(date +%H:%M) final test $m replay exit $?"
echo "$(date +%H:%M) SIM_MODEL_DONE $m"
