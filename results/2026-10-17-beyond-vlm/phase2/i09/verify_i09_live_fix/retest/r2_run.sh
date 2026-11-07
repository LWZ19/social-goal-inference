#!/bin/bash
# during start-up, two SIGTERMs). After each: a new invocation on the same --out must take the lock at once; the exit
# code, the run folder, the check file and the ledger are recorded. Each case has its own --out; the cases run in parallel.
source results/2026-10-17-beyond-vlm/phase2/i09/verify_i09_live_fix/retest/common.sh
T=$R/r2; rm -rf $T; mkdir -p $T; rm -f $R/fake/fake_r2_*
case_run() {  # case_run NAME SIGNAL SLEEP_S MIN_SENT [startup]
  local N=$1 SIG=$2 SL=$3 MIN=$4 MODE=$5 O=$T/$1_out C=$T/$1_cache E=$T/$1.txt
  mkdir -p $O; cp -r $R/cache_pc_base $C; : > $E
  $PY $(spec $T r2_${N}_x ",\"sleep_s\":$SL") -- --model glm53 --trial 0 --out $O --cache-dir $C --env-file $NK --arms real > $T/${N}_x.log 2>&1 &
  local P=$!
  if [ "$MODE" = startup ]; then sleep 1.0; else
    for i in $(seq 1 900); do [ "$(sent $O)" -ge $MIN ] && break; kill -0 $P 2>/dev/null || break; sleep 0.2; done; fi
  echo "$N: pid $P; before kill: lock file '$(cat $O/ledger.lock 2>/dev/null || echo NONE)'; sent $(sent $O)" >> $E
  if [ "$SIG" = TERMx2 ]; then kill -TERM $P; sleep 0.05; kill -TERM $P 2>/dev/null; else kill -$SIG $P; fi
  wait $P; local X=$?
  local F=$(ls -d $O/*_con 2>/dev/null | head -1)
  echo "$N: exit $X; lock file after: '$(cat $O/ledger.lock 2>/dev/null || echo NONE)'; ledger sent $(sent $O) answered $(grep -c '"answered"' $O/ledger.jsonl 2>/dev/null) error $(grep -c '"error"' $O/ledger.jsonl 2>/dev/null); fake $(fake r2_${N}_x); check files $(ls $O/check_* 2>/dev/null | wc -l); run folders $(ls -d $O/*_con 2>/dev/null | wc -l); results lines $(cat $F/results.jsonl 2>/dev/null | wc -l); config incomplete: $(grep -h '^incomplete' $F/config.yaml 2>/dev/null); stopped_by in config: $(grep -c '^stopped_by' $F/config.yaml 2>/dev/null); summary.csv: $(test -f $F/summary.csv && echo yes || echo no); KeyboardInterrupt in log: $(grep -c KeyboardInterrupt $T/${N}_x.log); tmp files in cache: $(ls $C | grep -c tmp)" >> $E
  # a new invocation, at once, same --out and cache
  $PY $(spec $T r2_${N}_y) -- --model glm53 --trial 0 --out $O --cache-dir $C --env-file $NK --arms real --only $TWO > $T/${N}_y.log 2>&1 &
  local PY2=$!; local SEEN=NONE
  for i in $(seq 1 600); do case "$(cat $O/ledger.lock 2>/dev/null)" in "$PY2 "*) SEEN=yes; break;; esac; kill -0 $PY2 2>/dev/null || break; sleep 0.2; done
  wait $PY2; local Y=$?
  echo "$N: next invocation pid $PY2 exit $Y; it wrote its pid into the lock file: $SEEN; refused line: $(grep -h 'another live driver' $T/${N}_y.log | cut -c1-200); ledger sent now $(sent $O); fake y $(fake r2_${N}_y)" >> $E
}
case_run kill   KILL   0.5 3 &
case_run term   TERM   1.0 3 &
case_run term0  TERM   0   5 &
case_run hup    HUP    1.0 3 &
case_run early  TERM   0   0 startup &
case_run term2x TERMx2 0.5 3 &
wait
cat $T/kill.txt $T/term.txt $T/term0.txt $T/hup.txt $T/early.txt $T/term2x.txt
echo done
