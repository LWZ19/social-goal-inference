#!/bin/bash
# Round 3 closed-loop design pilot (ROUND3.md; RANGES.md: candidates 1000-1199, design data): DeepSeek (free), the 24
# pilot layouts (sweep/jobs_pilot.txt, sim/scenarios/pilot3, sha256 ddfc5240), one seed per layout.
#   stage 1 (queue_pilot3_dsv41.sh ct2): CT2 alone, which makes the first answers (about 4 free calls a layout);
#   stage 2 (queue_pilot3_dsv41.sh rest): CT3F, CT4, W0 (CT4 + the 5 s window) and CT5 (W0 + the cost decision),
#   replaying them from the cache. Stage 2 needs the CT5 port in src and the sim arms (port_ct5_sim.py).
cd results/2026-11-01-push/moving
P=.venv/bin/python
case "$1" in
  ct2)  ARMS="CT2"; N=24 ;;
  rest) ARMS="CT3F CT4 W0 CT5"; N=96 ;;
  *) echo "usage: $0 ct2|rest"; exit 1 ;;
esac
$P run_moving.py --model soclaas_dsv41 --out runs_pilot3/dsv41 --workers 4 --port-base 8960 \
  --module sgi_habitat.experiment_oracle --jobs sweep/jobs_pilot.txt --arms $ARMS > queue_pilot3_dsv41_$1.out 2>&1
echo "done: $(grep -c 'exit 0' queue_pilot3_dsv41_$1.out) ok of $N"
grep -E 'exit [^0]|Error' queue_pilot3_dsv41_$1.out | head -5
