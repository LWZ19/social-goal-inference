#!/bin/bash
# Confirmatory test (ideas2_1b/round4/CONFIRM_CT4PJ.md): the layouts of sweep/confirm_selection.json (candidates
# 2000-2449), one seed each. Stage "ct2" makes the first answers; stage "rest" replays them from the cache for CT3F,
# CT4P, CT4PJ and CT4. Model: dsv41 (DeepSeek, free) or sonnet55_low (announced with its cost before it runs).
#   queue_confirm4.sh dsv41|sonnet55_low ct2|rest
cd results/2026-11-01-push/moving
P=.venv/bin/python
case "$1" in
  dsv41) MODEL=soclaas_dsv41 ;;
  sonnet55_low) MODEL=claude_sonnet55_low ;;
  *) echo "usage: $0 dsv41|sonnet55_low ct2|rest"; exit 1 ;;
esac
case "$2" in
  ct2)  ARMS="CT2" ;;
  rest) ARMS="CT3F CT4P CT4PJ CT4" ;;
  *) echo "usage: $0 dsv41|sonnet55_low ct2|rest"; exit 1 ;;
esac
N=$(( $(wc -l < sweep/jobs_confirm.txt) * $(echo $ARMS | wc -w) ))
$P run_moving.py --model $MODEL --out runs_confirm4/$1 --workers 4 --port-base 8960 \
  --module sgi_habitat.experiment_oracle --jobs sweep/jobs_confirm.txt --arms $ARMS > queue_confirm4_$1_$2.out 2>&1
echo "done: $(grep -c 'exit 0' queue_confirm4_$1_$2.out) ok of $N"
grep -E 'exit [^0]|Error' queue_confirm4_$1_$2.out | head -5
