#!/bin/bash
# Diagnostic for CT4P's failed direction check: CT3F (now logging solve_direction) on the 12 high-error pilot layouts,
# into a separate folder so the stage 2 CT3F episodes stay as they are. DeepSeek, CT2's cached answers: zero calls.
cd results/2026-11-01-push/moving
P=.venv/bin/python
$P run_moving.py --model soclaas_dsv41 --out runs_pilot3/dsv41_dircheck --workers 4 --port-base 8960 \
  --module sgi_habitat.experiment_oracle --jobs sweep/jobs_pilot_high.txt --arms CT3F > queue_dircheck_ct3f.out 2>&1
echo "done: $(grep -c 'exit 0' queue_dircheck_ct3f.out) ok of 12"
