#!/bin/bash
# CT4_FROZEN.md (2df880ff): the fresh test on the 72 frozen sweep layouts (sweep/selection.json; scenario files
# 047f49a5), DeepSeek (free gateway), one seed per layout. CT2 makes the first answer (about 4 calls a layout);
# CT3, Gl, CT3F and CT4 replay it from the cache; R1 makes its own marked-point answer (1-2 calls).
cd results/2026-11-01-push/moving
P=.venv/bin/python
$P run_moving.py --model soclaas_dsv41 --out runs_sweep/dsv41 --workers 4 --port-base 8900 \
  --module sgi_habitat.experiment_oracle --jobs sweep/jobs_all.txt --arms CT2 CT3 Gl CT3F CT4 R1 > queue_sweep_dsv41.out 2>&1
echo "done: $(grep -c 'exit 0' queue_sweep_dsv41.out) ok of 432"
grep -E 'exit [^0]' queue_sweep_dsv41.out | head -5
