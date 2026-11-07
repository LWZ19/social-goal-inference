#!/bin/bash
# T1: the first verifier's concurrency scenario on the NEW driver. A (glm53) runs live with a fake gateway on a cache
# that holds plan/context answers only (84 constraints misses > the 60 cap); B starts while A runs; then C starts
# while A runs. Same model, same --out. Each has its own cache copy.
V=results/2026-10-17-beyond-vlm/phase2/i09/verify_i09_live_fix/tests
cd .-scenes
PY="env PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 .venv/bin/python $V/wrap_driver.py"
T=$V/t1; rm -rf $T; mkdir -p $T/out; rm -f $V/fake/fake_t1_*
for x in A B C; do cp -r $V/cache_pc $T/cache$x; done
echo '{"tag":"t1_A","fake_gateway":true,"sleep_s":0.4,"block_paths":["no_such_keyfile"]}' > $T/spec_A.json
echo '{"tag":"t1_B","fake_gateway":true,"block_paths":["no_such_keyfile"]}' > $T/spec_B.json
echo '{"tag":"t1_C","fake_gateway":true,"block_paths":["no_such_keyfile"]}' > $T/spec_C.json
COMMON="--model glm53 --trial 0 --out $T/out --env-file $V/no_such_keyfile"
TR=$T/trace.txt; : > $TR
$PY $T/spec_A.json -- $COMMON --cache-dir $T/cacheA > $T/A.log 2>&1 &
PA=$!
echo "A pid $PA started $(date +%T.%N)" >> $TR
for i in $(seq 1 600); do [ -s $T/out/ledger.jsonl ] && grep -q '"sent"' $T/out/ledger.jsonl && break; sleep 0.5; done
echo "A sending at $(date +%T.%N); lock: $(cat $T/out/ledger.lock 2>/dev/null || echo NONE); sent so far $(grep -c '"sent"' $T/out/ledger.jsonl)" >> $TR
$PY $T/spec_B.json -- $COMMON --cache-dir $T/cacheB > $T/B.log 2>&1
EB=$?
echo "B exit $EB at $(date +%T.%N); A alive: $(kill -0 $PA 2>/dev/null && echo yes || echo no); lock after B: $(cat $T/out/ledger.lock 2>/dev/null || echo NONE); sent so far $(grep -c '"sent"' $T/out/ledger.jsonl)" >> $TR
$PY $T/spec_C.json -- $COMMON --cache-dir $T/cacheC > $T/C.log 2>&1
EC=$?
echo "C exit $EC at $(date +%T.%N); A alive: $(kill -0 $PA 2>/dev/null && echo yes || echo no); lock after C: $(cat $T/out/ledger.lock 2>/dev/null || echo NONE); sent so far $(grep -c '"sent"' $T/out/ledger.jsonl)" >> $TR
# poll the lock while A runs: it must stay A's until A ends
while kill -0 $PA 2>/dev/null; do
  L=$(cat $T/out/ledger.lock 2>/dev/null || echo NONE)
  case "$L" in "$PA "*) ;; *) kill -0 $PA 2>/dev/null && echo "LOCK CHANGED while A alive: '$L' at $(date +%T.%N)" >> $TR;; esac
  sleep 1
done
wait $PA; EA=$?
echo "A exit $EA at $(date +%T.%N); lock after A: $(cat $T/out/ledger.lock 2>/dev/null || echo NONE)" >> $TR
echo "ledger sent rows: $(grep -c '"sent"' $T/out/ledger.jsonl); refused rows: $(grep -c '"refused"' $T/out/ledger.jsonl)" >> $TR
echo "fake calls A: $(cat $V/fake/fake_t1_A_*.jsonl 2>/dev/null | wc -l); B: $(cat $V/fake/fake_t1_B_*.jsonl 2>/dev/null | wc -l); C: $(cat $V/fake/fake_t1_C_*.jsonl 2>/dev/null | wc -l)" >> $TR
echo "B stderr lock line: $(grep -h 'ledger.lock exists' $T/B.log | head -1)" >> $TR
echo "C stderr lock line: $(grep -h 'ledger.lock exists' $T/C.log | head -1)" >> $TR
echo done >> $TR
cat $TR
