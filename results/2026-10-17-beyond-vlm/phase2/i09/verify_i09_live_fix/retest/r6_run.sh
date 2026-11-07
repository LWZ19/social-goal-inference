#!/bin/bash
# R6: unchanged behaviour, build 41940c3c, real command path.
#   count-only on an EMPTY cache of my own, both models  -> 84 plan misses each; no ledger, no lock, no cache write
#   count-only on the plan/context cache, both models    -> 84 constraints misses each, 0 upstream; cache unchanged
#   mock, glm53, arm real, two scenes                     -> people stage 0 calls; no ledger, no lock
source results/2026-10-17-beyond-vlm/phase2/i09/verify_i09_live_fix/retest/common.sh
T=$R/r6; rm -rf $T; mkdir -p $T/empty_cache $T/cnt_empty $T/cnt_pc; cp -r $R/cache_pc_base $T/pc_cache; rm -f $R/fake/fake_r6_*
(cd $T/pc_cache && find . -type f -printf '%p %s %T@\n' | sort | sha256sum) > $T/pc_cache_before.txt
for m in glm53 dsv41; do
  $PY $(spec $T r6_cnt_empty_$m) -- --model $m --trial 0 --out $T/cnt_empty --cache-dir $T/empty_cache --count-only > $T/cnt_empty_$m.log 2>&1 &
  $PY $(spec $T r6_cnt_pc_$m) -- --model $m --trial 0 --out $T/cnt_pc --cache-dir $T/pc_cache --count-only > $T/cnt_pc_$m.log 2>&1 &
done
$PY $(spec $T r6_mock ',"capture_mock":true') -- --model glm53 --trial 0 --out $T/mock_out --mock --arms real --only $TWO > $T/mock.log 2>&1 & PM=$!
wait $PM; XM=$?
wait
E=$T/exits.txt; : > $E
for m in glm53 dsv41; do
  for k in cnt_empty cnt_pc; do
    echo "$k $m: $(grep -h '^count-only' $T/${k}_$m.log | cut -c1-200); wrap: $(grep -h '^WRAP' $T/${k}_$m.log)" >> $E
  done
done
echo "cnt_empty folder: $(ls $T/cnt_empty | tr '\n' ' ')" >> $E
echo "cnt_pc folder: $(ls $T/cnt_pc | tr '\n' ' ')" >> $E
echo "empty cache files after: $(ls -A $T/empty_cache | wc -l)" >> $E
(cd $T/pc_cache && find . -type f -printf '%p %s %T@\n' | sort | sha256sum) > $T/pc_cache_after.txt
echo "pc cache listing unchanged: $(cmp -s $T/pc_cache_before.txt $T/pc_cache_after.txt && echo yes || echo NO); files $(ls $T/pc_cache | wc -l)" >> $E
echo "mock: exit $XM; out: $(ls $T/mock_out | tr '\n' ' '); wrap: $(grep -h '^WRAP' $T/mock.log)" >> $E
echo done >> $E
cat $E
