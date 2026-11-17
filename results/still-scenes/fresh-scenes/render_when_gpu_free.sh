#!/bin/bash
# the GPU), render item 12's missing seeds, retrying on out-of-memory up to 4 times, then resume item 4.
I=results/still-scenes/fresh-scenes
used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }
for attempt in 1 2 3 4; do
  ok=0; while [ $ok -lt 2 ]; do if [ "$(used)" -le 5400 ]; then ok=$((ok+1)); else ok=0; fi; sleep 60; done
  missing=""; for s in 512 513 514; do [ "$(find $I/renders/seed$s -name rgb.png 2>/dev/null | wc -l)" -ge 14 ] || missing="$missing $s"; done
  [ -z "$missing" ] && break
  echo "$(date +%H:%M) attempt $attempt: GPU at $(used) MiB; rendering$missing"
  for s in $missing; do [ -f $I/render_seed$s.log ] && grep -qi "out of memory" $I/render_seed$s.log && mv $I/render_seed$s.log $I/render_failed_oom/render_seed${s}_$(date +%H%M).log; done
  SEEDS="$missing" $I/render_i12.sh && break
done
for s in 510 511 512 513 514; do echo "seed$s: $(find $I/renders/seed$s -name rgb.png 2>/dev/null | wc -l) of 14 scenes"; done
results/moving-lines/roles-vs-marked-spot/run_i4.sh &
echo "$(date +%H:%M) item 4 resumed"
