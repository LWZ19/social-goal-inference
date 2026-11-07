#!/bin/bash
# People dumps (robot idle, zero calls) for the confirmatory layouts (CONFIRM_CT4PJ.md; candidates 2000-2449): the
cd .
D=results/2026-11-01-push/moving
while read scen seed; do
  name=${scen#confirm4/}
  PYTHONPATH=sim ~/miniforge3/envs/sgi-habitat/bin/python $D/ideas2_1b/dump_humans.py --scenario sim/scenarios/$scen.yaml \
    --seed $seed --out $D/sweep/confirm_dump/$name/seed$seed.json 2>&1 | grep -E "^\[dump\]|Error|Traceback"
done < $D/sweep/jobs_confirm.txt
