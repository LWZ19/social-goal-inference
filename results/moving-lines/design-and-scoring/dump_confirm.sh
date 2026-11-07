#!/bin/bash
# People dumps (robot idle, zero calls) for the confirmatory layouts (CONFIRM_CT4PJ.md; candidates 2000-2449): the
cd .
D=results/moving-lines
while read scen seed; do
  name=${scen#confirm4/}
  PYTHONPATH=sim ~/miniforge3/envs/sgi-habitat/bin/python $D/best-version-and-running-guess/dump_humans.py --scenario sim/scenarios/$scen.yaml \
    --seed $seed --out $D/test-layouts/confirm_dump/$name/seed$seed.json 2>&1 | grep -E "^\[dump\]|Error|Traceback"
done < $D/test-layouts/jobs_confirm.txt
