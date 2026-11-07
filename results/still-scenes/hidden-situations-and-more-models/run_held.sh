#!/bin/bash
# E3 hidden situations (lift, photo) for one model: the 13 Oct DeepSeek protocol (3 seeds), pointing first.
cd .
D=results/2026-11-01-push/e3_models
M=$1
scen() { python3 -c "import yaml;print(yaml.safe_load(open('$D/held_scenes_dirs.yaml'))['$1'])"; }
one() { PYTHONPATH= .venv/bin/python -m sgi.run --config $D/held_configs/$1_$2.yaml --model $M --scenes $(scen $1) --final-test --name s25_held_$1_${M}_$2 > $D/logs/held_${M}_$1_$2.log 2>&1; echo "$(date +%H:%M) held $M $1 $2 exit $?"; }
mkdir -p $D/logs
for s in seed0 seed1 seed2; do one $s cot & done; wait
for s in seed0 seed1 seed2; do one $s fsc & one $s con & done; wait
echo "$(date +%H:%M) held $M done"
