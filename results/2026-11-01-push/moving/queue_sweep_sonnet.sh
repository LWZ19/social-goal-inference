#!/bin/bash
# CT4_FROZEN.md (2df880ff), RUN_NOTE.md "Sonnet: committed in advance": the 40 "ours" sweep layouts
# CT2 makes the first answer; CT3, Gl, CT3F and CT4 replay it from the cache; R1 makes its own marked-point answer.
cd results/2026-11-01-push/moving
P=.venv/bin/python
$P run_moving.py --model claude_sonnet55_low --out runs_sweep/sonnet55_low --workers 4 --port-base 8940 \
  --module sgi_habitat.experiment_oracle --jobs sweep/jobs_ours.txt --arms CT2 CT3 Gl CT3F CT4 R1 \
  > queue_sweep_sonnet.out 2>&1
echo "done: $(grep -c 'exit 0' queue_sweep_sonnet.out) ok of 240"
grep -E 'exit [^0]|Error' queue_sweep_sonnet.out | head -5
