#!/bin/bash
# R1 concurrency, build 41940c3c, real command path (wrapper -> runpy driver.py __main__ -> _command), fake gateway.
# A: glm53 live, all arms, cache with plan/context only (84 constraints misses > the 60 cap), 0.3 s per fake call.
# While A sends: B and C (glm53, same --out) start together; D (dsv41, same --out); E (glm53, --out through a symlink
# to the same folder); H (count-only, same --out: takes no lock). After A ends: G (dsv41, two scenes) must take the lock.
source results/2026-10-17-beyond-vlm/phase2/i09/verify_i09_live_fix/retest/common.sh
T=$R/r1; rm -rf $T; mkdir -p $T/out; ln -s out $T/outlink; rm -f $R/fake/fake_r1_*
for x in A B C D E G H; do cp -r $R/cache_pc_base $T/cache$x; done
TR=$T/trace.txt; : > $TR
$PY $(spec $T r1_A ',"sleep_s":0.3') -- --model glm53 --trial 0 --out $T/out --cache-dir $T/cacheA --env-file $NK > $T/A.log 2>&1 &
PA=$!; echo "A pid $PA started $(date +%T.%N)" >> $TR
for i in $(seq 1 900); do [ "$(sent $T/out)" -ge 1 ] && break; sleep 0.3; done
echo "A sending at $(date +%T.%N); lock file: '$(cat $T/out/ledger.lock)'; sent $(sent $T/out)" >> $TR
$PY $(spec $T r1_B) -- --model glm53 --trial 0 --out $T/out --cache-dir $T/cacheB --env-file $NK > $T/B.log 2>&1 & PB=$!
$PY $(spec $T r1_C) -- --model glm53 --trial 0 --out $T/out --cache-dir $T/cacheC --env-file $NK > $T/C.log 2>&1 & PC=$!
$PY $(spec $T r1_D) -- --model dsv41 --trial 0 --out $T/out --cache-dir $T/cacheD --env-file $NK > $T/D.log 2>&1 & PD=$!
$PY $(spec $T r1_E) -- --model glm53 --trial 0 --out $T/outlink --cache-dir $T/cacheE --env-file $NK > $T/E.log 2>&1 & PE=$!
$PY $(spec $T r1_H) -- --model glm53 --trial 0 --out $T/out --cache-dir $T/cacheH --count-only --arms real --only $TWO > $T/H.log 2>&1 & PH=$!
for n in B C D E H; do eval "wait \$P$n"; X=$?; echo "$n exit $X at $(date +%T.%N); A alive: $(kill -0 $PA 2>/dev/null && echo yes || echo no); lock file: '$(cat $T/out/ledger.lock)'; sent $(sent $T/out)" >> $TR; done
CHG=0
while kill -0 $PA 2>/dev/null; do
  L=$(cat $T/out/ledger.lock 2>/dev/null || echo NONE)
  case "$L" in "$PA "*) ;; *) kill -0 $PA 2>/dev/null && { echo "LOCK FILE CHANGED while A alive: '$L'" >> $TR; CHG=1; };; esac
  sleep 1
done
wait $PA; EA=$?
echo "A exit $EA at $(date +%T.%N); lock file after A: '$(cat $T/out/ledger.lock 2>/dev/null || echo NONE)'; sent $(sent $T/out); lock changed while A alive: $CHG" >> $TR
echo "fake A $(fake r1_A) B $(fake r1_B) C $(fake r1_C) D $(fake r1_D) E $(fake r1_E) H $(fake r1_H)" >> $TR
$PY $(spec $T r1_G) -- --model dsv41 --trial 0 --out $T/out --cache-dir $T/cacheG --env-file $NK --arms real --only $TWO > $T/G.log 2>&1 &
PG=$!; sleep 0.1
for i in $(seq 1 600); do case "$(cat $T/out/ledger.lock)" in "$PG "*) break;; esac; kill -0 $PG 2>/dev/null || break; sleep 0.2; done
echo "G pid $PG; lock file while G runs: '$(cat $T/out/ledger.lock)'" >> $TR
wait $PG; echo "G exit $? ; lock file exists after G: $(test -f $T/out/ledger.lock && echo yes || echo no); fake G $(fake r1_G)" >> $TR
echo "ledger sent $(sent $T/out); refused $(grep -c '"refused"' $T/out/ledger.jsonl); by model: glm53 $(grep '"sent"' $T/out/ledger.jsonl | grep -c '"glm53"') dsv41 $(grep '"sent"' $T/out/ledger.jsonl | grep -c '"dsv41"')" >> $TR
for n in B C D E; do echo "$n stderr: $(grep -h 'another live driver' $T/$n.log | cut -c1-400)" >> $TR; done
echo "H line: $(grep -h '^count-only' $T/H.log | cut -c1-200)" >> $TR
echo done >> $TR
cat $TR
