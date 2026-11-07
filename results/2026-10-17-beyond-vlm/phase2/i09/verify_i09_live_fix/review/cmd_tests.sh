#!/bin/bash
# Reviewer's command-level tests of driver.py 84bab2e4 through wrap.py (fake gateway, audit hook). Zero model calls.
# Every --out, cache and env path is inside review/. The env file named does not exist.
R=results/2026-10-17-beyond-vlm/phase2/i09/verify_i09_live_fix/review
cd .-scenes
PY="env PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 .venv/bin/python $R/wrap.py"
T=$R/cmd
rm -rf $T; mkdir -p $T
ONLY="--arms real --only real_eval_08_waiting_line_4 real_eval_11_conversation_group_3 real_eval_01_no_one_place"
ENV="--env-file $R/no_such_env_file"
LOG=$T/trace.txt
spec() { echo "{\"tag\":\"$1\"${2:+,$2}}" > $T/spec_$1.json; echo $T/spec_$1.json; }
cache() { rm -rf $T/cache_$1; cp -r $R/fill/cache_cut $T/cache_$1; echo $T/cache_$1; }
sent() { local n; n=$(grep -c '"event": "sent"' $1/ledger.jsonl 2>/dev/null); echo ${n:-0}; }
refused() { local n; n=$(grep -c '"event": "refused"' $1/ledger.jsonl 2>/dev/null); echo ${n:-0}; }
lockstate() { [ -e $1/ledger.lock ] && echo "present($(cat $1/ledger.lock | tr -d '\n'))" || echo absent; }
say() { echo "$*" | tee -a $LOG; }

# T3: upstream miss in live mode (empty cache): exit 2, nothing sent, lock released.
mkdir -p $T/cache_empty
$PY $(spec t3) -- --model glm53 --trial 0 --out $T/t3 --cache-dir $T/cache_empty $ENV $ONLY > $T/t3.log 2>&1
say "T3 upstream-miss: exit $?; sent $(sent $T/t3); lock $(lockstate $T/t3); check files $(ls $T/t3/check_* 2>/dev/null | wc -l); folders $(ls -d $T/t3/*_con 2>/dev/null | wc -l); summary.csv $(ls $T/t3/*_con/summary.csv 2>/dev/null | wc -l); results lines $(cat $T/t3/*_con/results.jsonl 2>/dev/null | wc -l); cache files $(ls $T/cache_empty | wc -l)"

# T4: one constraints request fails 3 times (transient) -> vlm_error -> exit 1; the re-run sends only that one -> exit 0.
C4=$(cache t4)
$PY $(spec t4a '"transient_constraints":3') -- --model glm53 --trial 0 --out $T/t4 --cache-dir $C4 $ENV $ONLY > $T/t4a.log 2>&1
say "T4a vlm_error: exit $?; sent $(sent $T/t4); lock $(lockstate $T/t4)"
$PY $(spec t4b) -- --model glm53 --trial 0 --out $T/t4 --cache-dir $C4 $ENV $ONLY > $T/t4b.log 2>&1
say "T4b re-run: exit $?; sent $(sent $T/t4); lock $(lockstate $T/t4)"

# T5: the per-model cap (exit 3), then the total cap across models on the same ledger (exit 3).
C5=$(cache t5)
$PY $(spec t5a) -- --model glm53 --trial 0 --out $T/t5 --cache-dir $C5 $ENV $ONLY --max-calls 3 --max-calls-per-model 2 > $T/t5a.log 2>&1
say "T5a glm cap 3/2: exit $?; sent $(sent $T/t5); refused $(refused $T/t5); lock $(lockstate $T/t5)"
$PY $(spec t5b) -- --model dsv41 --trial 0 --out $T/t5 --cache-dir $C5 $ENV $ONLY --max-calls 3 --max-calls-per-model 60 > $T/t5b.log 2>&1
say "T5b dsv total cap 3: exit $?; sent $(sent $T/t5); refused $(refused $T/t5); lock $(lockstate $T/t5)"

# T1: a second invocation on the same --out is refused (exit 2) and leaves the running one's lock.
C1A=$(cache t1a); C1B=$(cache t1b)
$PY $(spec t1a '"sleep_s":3') -- --model glm53 --trial 0 --out $T/t1 --cache-dir $C1A $ENV $ONLY > $T/t1a.log 2>&1 &
PA=$!
until [ "$(sent $T/t1)" -ge 1 ] 2>/dev/null; do sleep 0.5; done
say "T1 A sending; lock $(lockstate $T/t1)"
$PY $(spec t1b) -- --model glm53 --trial 0 --out $T/t1 --cache-dir $C1B $ENV $ONLY > $T/t1b.log 2>&1
say "T1 B exit $?; lock after B $(lockstate $T/t1); sent so far $(sent $T/t1)"
wait $PA
say "T1 A exit $?; lock after A $(lockstate $T/t1); sent $(sent $T/t1)"

# T2: the lock is removed by hand while A runs (an operator's mistake); C then runs beside A on the same ledger.
#     The per-model cap (4) must still hold across A and C.
C2A=$(cache t2a); C2C=$(cache t2c)
$PY $(spec t2a '"sleep_s":3') -- --model glm53 --trial 0 --out $T/t2 --cache-dir $C2A $ENV $ONLY --max-calls-per-model 4 > $T/t2a.log 2>&1 &
PA=$!
until [ "$(sent $T/t2)" -ge 1 ] 2>/dev/null; do sleep 0.5; done
rm -f $T/t2/ledger.lock
say "T2 A sending; lock removed by hand"
$PY $(spec t2c '"sleep_s":3') -- --model glm53 --trial 0 --out $T/t2 --cache-dir $C2C $ENV $ONLY --max-calls-per-model 4 > $T/t2c.log 2>&1 &
PC=$!
wait $PA; EA=$?
say "T2 A exit $EA; lock when A exited $(lockstate $T/t2) (C still running: $(kill -0 $PC 2>/dev/null && echo yes || echo no))"
wait $PC; EC=$?
say "T2 C exit $EC; glm sent in all $(sent $T/t2) (cap 4); refused $(refused $T/t2); lock $(lockstate $T/t2)"

# T6: KeyboardInterrupt inside a send: the lock is released.
C6=$(cache t6)
$PY $(spec t6 '"kbint_at":2') -- --model glm53 --trial 0 --out $T/t6 --cache-dir $C6 $ENV $ONLY > $T/t6.log 2>&1
say "T6 KeyboardInterrupt: exit $?; sent $(sent $T/t6); lock $(lockstate $T/t6); check files $(ls $T/t6/check_* 2>/dev/null | wc -l)"

# T9: a non-transient provider error (BackendError): exit 2, the check file is written.
C9=$(cache t9)
$PY $(spec t9 '"backend_error_at":1') -- --model glm53 --trial 0 --out $T/t9 --cache-dir $C9 $ENV $ONLY > $T/t9.log 2>&1
say "T9 BackendError: exit $?; sent $(sent $T/t9); lock $(lockstate $T/t9); check files $(ls $T/t9/check_* 2>/dev/null | wc -l)"

# T7: SIGTERM during a send leaves a stale lock; the next invocation is refused with exit 2.
C7=$(cache t7)
$PY $(spec t7a '"sleep_s":5') -- --model glm53 --trial 0 --out $T/t7 --cache-dir $C7 $ENV $ONLY > $T/t7a.log 2>&1 &
PA=$!
until [ "$(sent $T/t7)" -ge 1 ] 2>/dev/null; do sleep 0.5; done
PYPID=$(cut -d' ' -f1 $T/t7/ledger.lock)
kill -TERM $PYPID
wait $PA
say "T7 SIGTERM to pid $PYPID: exit $?; lock after $(lockstate $T/t7); sent $(sent $T/t7)"
$PY $(spec t7b) -- --model glm53 --trial 0 --out $T/t7 --cache-dir $C7 $ENV $ONLY > $T/t7b.log 2>&1
say "T7 next invocation: exit $?; $(grep -h 'exists' $T/t7b.log | head -1 | cut -c1-80)"

# T8: argument errors and --help.
$PY $(spec t8a) -- --model glm53 --trial 0 --out $T/t8 --cache-dir $C7 $ENV --max-calls 121 > $T/t8a.log 2>&1; say "T8a --max-calls 121: exit $?"
$PY $(spec t8b) -- --model glm53 --trial 0 --out $T/t8 --cache-dir $C7 --env-file relative_name > $T/t8b.log 2>&1; say "T8b relative --env-file: exit $?"
$PY $(spec t8c) -- --model glm53 --trial 0 --out $T/t8 --cache-dir $C7 --count-only $ENV > $T/t8c.log 2>&1; say "T8c --env-file with --count-only: exit $?"
$PY $(spec t8d) -- --help > $T/t8d.log 2>&1; say "T8d --help: exit $?"
$PY $(spec t8e) -- --model glm53 --trial 1 --out $T/t8 --cache-dir $C7 $ENV > $T/t8e.log 2>&1; say "T8e --trial 1: exit $?"

# T10: count-only on the cut cache: exit 0, 6 constraints misses, no ledger, no lock, no cache write.
C10=$(cache t10); N0=$(ls $C10 | wc -l)
$PY $(spec t10) -- --model glm53 --trial 0 --out $T/t10 --cache-dir $C10 --count-only $ONLY > $T/t10.log 2>&1
say "T10 count-only: exit $?; $(grep -h '^count-only' $T/t10.log | cut -c1-160); ledger $(ls $T/t10/ledger.jsonl 2>/dev/null | wc -l); lock $(lockstate $T/t10); cache files $N0 -> $(ls $C10 | wc -l)"
say done
