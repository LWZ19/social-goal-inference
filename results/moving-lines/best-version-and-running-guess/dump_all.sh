#!/bin/bash
# Every (scenario, seed) design in runs/ (excluding dsv41_invalid_requery and dsv41_repeat): the scripted people at
# 2 Hz from t = 0, robot idle, no request (zero model calls). Three at a time.
cd .
D=results/moving-lines/best-version-and-running-guess
export PYTHONPATH=sim
cat $D/dump_jobs.txt | xargs -P 3 -n 2 sh -c \
  '[ -s '$D'/human_dump/$0/seed$1.json ] || ~/miniforge3/envs/sgi-habitat/bin/python '$D'/dump_humans.py --scenario sim/scenarios/$0.yaml --seed $1 --out '$D'/human_dump/$0/seed$1.json 2>&1 | grep "^\[dump\]\|Error\|Traceback"'
echo "done: $(ls $D/human_dump/*/seed*.json | wc -l) dumps"
