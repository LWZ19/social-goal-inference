#!/bin/bash
# Round 3 control arm CT4PJ (ideas2_1b/round4/CT4PJ.md, sha256 97053c13): CT4P with the Resolver's joined rule, as
# CT3F has it. DeepSeek, the 24 pilot layouts (design data, candidates 1000-1199), replaying CT2's cached first answers:
# zero model calls. Then: check_mech, m_ct4pj_score.py once.
cd results/2026-11-01-push/moving
P=.venv/bin/python
$P run_moving.py --model soclaas_dsv41 --out runs_pilot3/dsv41 --workers 4 --port-base 8960 \
  --module sgi_habitat.experiment_oracle --jobs sweep/jobs_pilot.txt --arms CT4PJ > queue_ct4pj_dsv41.out 2>&1
echo "done: $(grep -c 'exit 0' queue_ct4pj_dsv41.out) ok of 24"
grep -E 'exit [^0]|Error' queue_ct4pj_dsv41.out | head -5
