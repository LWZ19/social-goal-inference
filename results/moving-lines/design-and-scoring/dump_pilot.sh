#!/bin/bash
cd .
D=results/2026-11-01-push/moving
while read scen seed; do
  name=${scen#pilot3/}
  PYTHONPATH=sim ~/miniforge3/envs/sgi-habitat/bin/python $D/ideas2_1b/dump_humans.py --scenario sim/scenarios/$scen.yaml \
    --seed $seed --out $D/sweep/pilot_dump/$name/seed$seed.json 2>&1 | grep -E "^\[dump\]|Error|Traceback"
done < $D/sweep/jobs_pilot.txt
