#!/bin/bash
# E3 still scenes for one model: pointing (cot) on the three sets first, then few_shot_cot and the constraint method,
# which reuse its plan and context answers through the cache (the s23 order). Usage: run_static.sh ENTRY
cd .
D=results/still-scenes/hidden-situations-and-more-models
M=$1
scen() { python3 -c "import yaml;print(yaml.safe_load(open('$D/scenes_dirs.yaml'))['$1'])"; }
one() { PYTHONPATH= .venv/bin/python -m sgi.run --config $D/configs/$1_$2.yaml --model $M --scenes $(scen $1) --name s24_$1_${M}_$2 > $D/logs/${M}_$1_$2.log 2>&1; echo "$(date +%H:%M) $M $1 $2 exit $?"; }
mkdir -p $D/logs
for v in plain staffed real; do one $v cot & done; wait
for v in plain staffed real; do one $v fsc & one $v con & done; wait
echo "$(date +%H:%M) $M done"
