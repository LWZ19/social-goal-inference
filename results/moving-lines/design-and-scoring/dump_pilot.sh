#!/bin/bash
cd .
D=results/moving-lines
while read scen seed; do
  name=${scen#pilot3/}
  PYTHONPATH=sim ~/miniforge3/envs/sgi-habitat/bin/python $D/best-version-and-running-guess/dump_humans.py --scenario sim/scenarios/$scen.yaml \
    --seed $seed --out $D/test-layouts/pilot_dump/$name/seed$seed.json 2>&1 | grep -E "^\[dump\]|Error|Traceback"
done < $D/test-layouts/jobs_pilot.txt
