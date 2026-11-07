#!/bin/bash
# Extra adversarial cases, build 41940c3c, real command path, fake gateway.
#   race   two live invocations started at the same instant on a fresh --out (x3 rounds): exactly one runs
#   corrupt a ledger with a half-written line before the start: refused (exit 2), nothing sent
source results/2026-10-17-beyond-vlm/phase2/i09/verify_i09_live_fix/retest/common.sh
T=$R/r8; rm -rf $T; mkdir -p $T; rm -f $R/fake/fake_r8_*
E=$T/exits.txt; : > $E
for k in 1 2 3; do
  O=$T/race$k; mkdir -p $O; cp -r $R/cache_pc_base $T/race${k}_c1; cp -r $R/cache_pc_base $T/race${k}_c2
  S1=$(spec $T r8_race${k}_a ',"sleep_s":0.2'); S2=$(spec $T r8_race${k}_b ',"sleep_s":0.2')
  $PY $S1 -- --model glm53 --trial 0 --out $O --cache-dir $T/race${k}_c1 --env-file $NK --arms real --only $TWO > $T/race${k}_a.log 2>&1 & P1=$!
  $PY $S2 -- --model dsv41 --trial 0 --out $O --cache-dir $T/race${k}_c2 --env-file $NK --arms real --only $TWO > $T/race${k}_b.log 2>&1 & P2=$!
  wait $P1; X1=$?; wait $P2; X2=$?
  echo "race$k: exits $X1 $X2; fake $(fake r8_race${k}_a) $(fake r8_race${k}_b); ledger sent $(sent $O); refused lines: $(grep -l 'another live driver' $T/race${k}_*.log | wc -l)" >> $E
done
O=$T/corrupt; mkdir -p $O; cp -r $R/cache_pc_base $T/corrupt_c
printf '{"time": "x", "event": "sent", "model": "glm53", "total": 1, "model_total": 1}\n{"time": "x", "event": "sent", "mod' > $O/ledger.jsonl
$PY $(spec $T r8_corrupt) -- --model glm53 --trial 0 --out $O --cache-dir $T/corrupt_c --env-file $NK --arms real --only $TWO > $T/corrupt.log 2>&1; X=$?
echo "corrupt: exit $X; fake $(fake r8_corrupt); error: $(grep -h 'Error' $T/corrupt.log | tail -1 | cut -c1-150); check files $(ls $O/check_* 2>/dev/null | wc -l); run folders $(ls -d $O/*_con 2>/dev/null | wc -l)" >> $E
echo done >> $E
cat $E
