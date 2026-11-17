#!/bin/bash
# Records (2 workers), then the noisy arm (9 jobs, 2 workers).
cd "$(dirname "$0")"
./job.sh records_A v_records.py old_qwen38_low old_glm53 old_dsv41 watch_qwen38_low watch_glm53 watch_dsv41 new_gemini38_low &
./job.sh records_B v_records.py new_glm53 new_dsv41 new_haiku55 new_sonnet55_low
wait
printf '%s\n' "qwen 0" "qwen 1" "qwen 2" "glm 0" "glm 1" "glm 2" "dsv 0" "dsv 1" "dsv 2" | \
  xargs -P 2 -L 1 bash -c './job.sh noisy_$0_seed$1 v_noisy.py $0 $1'
echo all done
