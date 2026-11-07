#!/bin/bash
# Item 13 runs, the frozen block's commands (I13_FROZEN.md section 6) with the models given as arguments, so lanes can
# run side by side. Usage: run_e4.sh MODEL [MODEL ...]
cd .
E=results/2026-11-01-push/e4
F=results/2026-11-01-push/fill/i13_e4
one() { $CAPPED env PYTHONPATH= .venv/bin/python -m sgi.run --config $E/configs/$3.yaml --model $1 --scenes $E/scenes/$2 \
         --final-test --name s26_e4_$2_$1_$3 > $F/log_$1_$2_$3.log 2>&1; echo "$(date +%H:%M) $1 $2 $3 exit $?"; }
for M in "$@"; do
  for s in seed600 seed601 seed602 seed603 seed604 seed605; do one $M $s cot & done; wait
  for s in seed600 seed601 seed602 seed603 seed604 seed605; do one $M $s fsc & one $M $s con & done; wait
  echo "$(date +%H:%M) E4_MODEL_DONE $M"
done
