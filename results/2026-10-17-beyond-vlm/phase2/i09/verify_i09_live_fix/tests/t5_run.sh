#!/bin/bash
# T5: exit codes through the real command path (the driver's __main__ block, so _command runs), NEW driver.
# Wrapped runs: runpy.run_path(driver.py, run_name="__main__") inside wrap_driver.py (audit hook, fake gateway).
# Direct runs: `python driver.py ...` with no wrapper, only where no back end can be built (argparse errors, a lock
# refusal with a lock file already present, count-only), each under strace (openat/connect logged) as the audit.
V=results/2026-10-17-beyond-vlm/phase2/i09/verify_i09_live_fix/tests
LIVE=results/2026-10-17-beyond-vlm/phase2/i09/live
cd .-scenes
ENV="env PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10"
PY="$ENV .venv/bin/python $V/wrap_driver.py"
DIRECT="strace -f -qq -e trace=openat,open,connect -o"
T=$V/t5; rm -rf $T; mkdir -p $T; rm -f $V/fake/fake_t5_*
TWO="real_eval_08_waiting_line_4 real_eval_11_conversation_group_3"
NK=$V/no_such_keyfile
E=$T/exits.txt; : > $E
spec() { echo "{\"tag\":\"t5_$1\",\"fake_gateway\":true,\"block_paths\":[\"no_such_keyfile\"]$2}" > $T/spec_$1.json; }

# --- lock refusal: wrapped and direct
mkdir -p $T/lock_out $T/lock_cache; echo "verifier test lock" > $T/lock_out/ledger.lock; spec lock
$PY $T/spec_lock.json -- --model glm53 --trial 0 --out $T/lock_out --cache-dir $T/lock_cache --env-file $NK > $T/lock.log 2>&1
echo "lock_wrapped exit $? ; lock content now: '$(cat $T/lock_out/ledger.lock)'; ledger file: $(ls $T/lock_out/ledger.jsonl 2>/dev/null || echo none); fake $(cat $V/fake/fake_t5_lock_*.jsonl 2>/dev/null | wc -l)" >> $E
if [ -f $T/lock_out/ledger.lock ]; then
  $DIRECT $T/strace_lock_direct.txt $ENV .venv/bin/python $LIVE/driver.py --model glm53 --trial 0 --out $T/lock_out --cache-dir $T/lock_cache --env-file $NK > $T/lock_direct.log 2>&1
  echo "lock_direct exit $? ; lock content now: '$(cat $T/lock_out/ledger.lock)'; stderr: $(grep -h 'ledger.lock exists' $T/lock_direct.log | cut -c1-160)" >> $E
fi

# --- argparse / bad arguments: direct
$DIRECT $T/strace_badarg1.txt $ENV .venv/bin/python $LIVE/driver.py --model qwen38 --trial 0 --out $T/bad_out --count-only --cache-dir $T/lock_cache > $T/badarg1.log 2>&1
echo "badarg_model_direct exit $?" >> $E
$DIRECT $T/strace_badarg2.txt $ENV .venv/bin/python $LIVE/driver.py --model glm53 --trial 1 --out $T/bad_out --count-only --cache-dir $T/lock_cache > $T/badarg2.log 2>&1
echo "badarg_trial1_direct exit $?" >> $E
$DIRECT $T/strace_badarg3.txt $ENV .venv/bin/python $LIVE/driver.py --model glm53 --trial 0 --out $T/bad_out --cache-dir $T/lock_cache > $T/badarg3.log 2>&1
echo "badarg_live_no_envfile_direct exit $?" >> $E
spec badarg
$PY $T/spec_badarg.json -- --model glm53 --trial 0 --out $T/bad_out --count-only --mock --cache-dir $T/lock_cache > $T/badarg4.log 2>&1
echo "badarg_countonly_and_mock_wrapped exit $?" >> $E

# --- unknown --only scene: count-only direct; live wrapped
mkdir -p $T/unk_cache
$DIRECT $T/strace_unknown_count.txt $ENV .venv/bin/python $LIVE/driver.py --model glm53 --trial 0 --out $T/unk_count_out --cache-dir $T/unk_cache --count-only --only no_such_scene_xyz > $T/unknown_count.log 2>&1
echo "unknown_only_countonly_direct exit $? ; $(grep -h '^count-only' $T/unknown_count.log | cut -c1-200)" >> $E
cp -r $V/cache_pc $T/unk_live_cache; spec unknown_live
$PY $T/spec_unknown_live.json -- --model glm53 --trial 0 --out $T/unk_live_out --cache-dir $T/unk_live_cache --env-file $NK --only no_such_scene_xyz > $T/unknown_live.log 2>&1
echo "unknown_only_live_wrapped exit $? ; run folders: $(ls -d $T/unk_live_out/*_con 2>/dev/null | wc -l); fake $(cat $V/fake/fake_t5_unknown_live_*.jsonl 2>/dev/null | wc -l)" >> $E

# --- SceneError (injected in the package's scene check) on the first scene: wrapped
cp -r $V/cache_pc $T/se_cache; spec sceneerror ',"scene_error_on":"real_eval_08_waiting_line_4"'
$PY $T/spec_sceneerror.json -- --model glm53 --trial 0 --out $T/se_out --cache-dir $T/se_cache --env-file $NK --arms real --only $TWO > $T/sceneerror.log 2>&1
echo "sceneerror_wrapped exit $? ; lock left: $(ls $T/se_out/ledger.lock 2>/dev/null || echo none); SceneError in stderr: $(grep -c 'SceneError' $T/sceneerror.log)" >> $E

# --- vlm_error: every attempt of the first request raises TransientError -> exit 1; then the same command again -> 0
cp -r $V/cache_pc $T/ve_cache; spec vlmerr ',"sticky_transient":true'; spec vlmerr_rerun
$PY $T/spec_vlmerr.json -- --model glm53 --trial 0 --out $T/ve_out --cache-dir $T/ve_cache --env-file $NK --arms real --only $TWO > $T/vlmerr.log 2>&1
echo "vlm_error_wrapped exit $? ; lock left: $(ls $T/ve_out/ledger.lock 2>/dev/null || echo none); ledger sent $(grep -c '"sent"' $T/ve_out/ledger.jsonl); errors $(grep -c '"error"' $T/ve_out/ledger.jsonl); fake $(cat $V/fake/fake_t5_vlmerr_*.jsonl | wc -l)" >> $E
$PY $T/spec_vlmerr_rerun.json -- --model glm53 --trial 0 --out $T/ve_out --cache-dir $T/ve_cache --env-file $NK --arms real --only $TWO > $T/vlmerr_rerun.log 2>&1
echo "vlm_error_rerun_wrapped exit $? ; ledger sent now $(grep -c '"sent"' $T/ve_out/ledger.jsonl); fake (rerun) $(cat $V/fake/fake_t5_vlmerr_rerun_*.jsonl | wc -l)" >> $E

# --- a non-transient exception mid-run (RuntimeError at fake call 3): exit 2, lock released
cp -r $V/cache_pc $T/rt_cache; spec runtime ',"raise_at":[3]'
$PY $T/spec_runtime.json -- --model glm53 --trial 0 --out $T/rt_out --cache-dir $T/rt_cache --env-file $NK --arms real > $T/runtime.log 2>&1
echo "runtime_error_wrapped exit $? ; lock left: $(ls $T/rt_out/ledger.lock 2>/dev/null || echo none); ledger sent $(grep -c '"sent"' $T/rt_out/ledger.jsonl); ledger error rows $(grep -c '"error"' $T/rt_out/ledger.jsonl); fake $(cat $V/fake/fake_t5_runtime_*.jsonl | wc -l); traceback: $(grep -c 'Traceback' $T/runtime.log)" >> $E

# --- SIGINT mid-run: lock released
cp -r $V/cache_pc $T/si_cache; spec sigint ',"sleep_s":1.0'
$PY $T/spec_sigint.json -- --model glm53 --trial 0 --out $T/si_out --cache-dir $T/si_cache --env-file $NK --arms real > $T/sigint.log 2>&1 &
P=$!
for i in $(seq 1 600); do [ -f $T/si_out/ledger.jsonl ] && [ "$(grep -c '"sent"' $T/si_out/ledger.jsonl)" -ge 3 ] && break; sleep 0.3; done
echo "sigint: before kill lock: '$(cat $T/si_out/ledger.lock 2>/dev/null || echo none)' pid $P" >> $E
kill -INT $P; wait $P; X=$?
echo "sigint_wrapped exit $X ; lock left: $(ls $T/si_out/ledger.lock 2>/dev/null || echo none); ledger sent $(grep -c '"sent"' $T/si_out/ledger.jsonl); fake $(cat $V/fake/fake_t5_sigint_*.jsonl | wc -l); KeyboardInterrupt in log: $(grep -c KeyboardInterrupt $T/sigint.log)" >> $E

# --- SIGTERM mid-run (informational): the default handler does not run finally
cp -r $V/cache_pc $T/st_cache; spec sigterm ',"sleep_s":1.0'
$PY $T/spec_sigterm.json -- --model glm53 --trial 0 --out $T/st_out --cache-dir $T/st_cache --env-file $NK --arms real > $T/sigterm.log 2>&1 &
P=$!
for i in $(seq 1 600); do [ -f $T/st_out/ledger.jsonl ] && [ "$(grep -c '"sent"' $T/st_out/ledger.jsonl)" -ge 3 ] && break; sleep 0.3; done
kill -TERM $P; wait $P; X=$?
echo "sigterm_wrapped exit $X ; lock left: $(ls $T/st_out/ledger.lock 2>/dev/null || echo none) (expected: left behind; REQUEST.md item 6 covers it)" >> $E

# --- strace audit of the direct runs
echo "strace: .env opens: $(grep -h 'open' $T/strace_*.txt | grep -c '\.env\b'); no_such_keyfile opens: $(grep -h 'no_such_keyfile' $T/strace_*.txt | grep -c open); main cache: $(grep -h '/cs6244_project/.cache' $T/strace_*.txt | wc -l); connect: $(grep -h 'connect(' $T/strace_*.txt | grep -vc 'AF_UNIX')" >> $E
echo done >> $E
cat $E
