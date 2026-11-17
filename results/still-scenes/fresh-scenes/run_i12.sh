#!/bin/bash
# Item 12 runs, the frozen block's one() (I12_FROZEN.md, Run) for one model over the seed folders of one assembled
# root, R first, then E and C (each seed folder in parallel). Usage: run_i12.sh MODEL ROOT (run_scenes or run_scenes_b)
cd .
I=results/still-scenes/fresh-scenes
M=$1; R=$I/$2
one() { $CAPPED env PYTHONPATH= .venv/bin/python -m sgi.run --config $I/configs/$3.yaml --model $1 --scenes $R/$2 --name s27_i12_$2_$1_$3 > $I/log_$1_$2_$3.log 2>&1; echo "$(date +%H:%M) $1 $2 $3 exit $?"; }
S=$(ls $R)
for s in $S; do one $M $s cot & done; wait
for s in $S; do one $M $s fsc & one $M $s con & done; wait
echo "$(date +%H:%M) I12_DONE $M $2"
