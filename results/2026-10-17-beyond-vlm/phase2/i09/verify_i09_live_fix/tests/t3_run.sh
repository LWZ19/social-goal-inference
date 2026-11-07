#!/bin/bash
# T3: sequential caps on one --out (as the first build's test c), NEW driver, fake gateway, plan/context cache.
#   c1 glm53 --max-calls 5 --max-calls-per-model 5 --arms real     -> exit 3 after 5 sends
#   c2 dsv41 --max-calls 5 --arms real (same ledger)               -> exit 3 with 0 sends (total cap)
#   c3 glm53 default caps --arms real, invalid answer at fake call 3, TransientError at fake call 7
#                                                                  -> exit 0; a validation retry and a transient retry
#                                                                     each count as a call; ledger sent == sends
#   c4 --max-calls 121 / --max-calls-per-model 61                  -> exit 2 (argparse)
V=results/2026-10-17-beyond-vlm/phase2/i09/verify_i09_live_fix/tests
cd .-scenes
PY="env PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 .venv/bin/python $V/wrap_driver.py"
T=$V/t3; rm -rf $T; mkdir -p $T/out; cp -r $V/cache_pc $T/cache; rm -f $V/fake/fake_t3_*
echo '{"tag":"t3_c1","fake_gateway":true,"block_paths":["no_such_keyfile"]}' > $T/spec_c1.json
echo '{"tag":"t3_c2","fake_gateway":true,"block_paths":["no_such_keyfile"]}' > $T/spec_c2.json
echo '{"tag":"t3_c3","fake_gateway":true,"invalid_at":[3],"transient_at":[7],"block_paths":["no_such_keyfile"]}' > $T/spec_c3.json
echo '{"tag":"t3_c4","fake_gateway":true,"block_paths":["no_such_keyfile"]}' > $T/spec_c4.json
COMMON="--trial 0 --out $T/out --cache-dir $T/cache --env-file $V/no_such_keyfile --arms real"
E=$T/exits.txt; : > $E
S() { grep -c '"sent"' $T/out/ledger.jsonl 2>/dev/null || echo 0; }
$PY $T/spec_c1.json -- --model glm53 $COMMON --max-calls 5 --max-calls-per-model 5 > $T/c1.log 2>&1
echo "c1 exit $? ; lock left: $(ls $T/out/ledger.lock 2>/dev/null || echo none); ledger sent $(S); fake $(cat $V/fake/fake_t3_c1_*.jsonl 2>/dev/null | wc -l)" >> $E
$PY $T/spec_c2.json -- --model dsv41 $COMMON --max-calls 5 > $T/c2.log 2>&1
echo "c2 exit $? ; lock left: $(ls $T/out/ledger.lock 2>/dev/null || echo none); ledger sent $(S); fake $(cat $V/fake/fake_t3_c2_*.jsonl 2>/dev/null | wc -l)" >> $E
$PY $T/spec_c3.json -- --model glm53 $COMMON > $T/c3.log 2>&1
echo "c3 exit $? ; lock left: $(ls $T/out/ledger.lock 2>/dev/null || echo none); ledger sent $(S); fake $(cat $V/fake/fake_t3_c3_*.jsonl 2>/dev/null | wc -l)" >> $E
$PY $T/spec_c4.json -- --model dsv41 $COMMON --max-calls 121 > $T/c4a.log 2>&1
echo "c4a exit $? ; ledger sent $(S)" >> $E
$PY $T/spec_c4.json -- --model dsv41 $COMMON --max-calls-per-model 61 > $T/c4b.log 2>&1
echo "c4b exit $? ; ledger sent $(S)" >> $E
echo "refused rows: $(grep -c '"refused"' $T/out/ledger.jsonl)" >> $E
echo done >> $E
cat $E
