#!/bin/bash
# Consequence of the lock defect: A runs; B is refused and removes A's lock; C then runs beside A on one ledger.
V=results/2026-10-17-beyond-vlm/phase2/i09/verify_i09_live_build
cd .-scenes
PY="env PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 .venv/bin/python $V/wrap_driver.py"
rm -rf $V/e_conc $V/e_cacheA $V/e_cacheB $V/e_cacheC; mkdir -p $V/e_conc $V/e_cacheA $V/e_cacheB $V/e_cacheC
for t in A B C; do echo "{\"tag\":\"e_$t\",\"fake_gateway\":true,\"block_paths\":[\"no_such_keyfile\"]}" > $V/spec_e_$t.json; done
$PY $V/spec_e_A.json -- --model glm53 --trial 0 --out $V/e_conc --cache-dir $V/e_cacheA --env-file $V/no_such_keyfile > $V/e_A.log 2>&1 &
PA=$!
until [ -s $V/e_conc/ledger.jsonl ]; do sleep 0.5; done
echo "A sending; lock present: $(ls $V/e_conc/ledger.lock 2>/dev/null)" > $V/e_trace.txt
$PY $V/spec_e_B.json -- --model glm53 --trial 0 --out $V/e_conc --cache-dir $V/e_cacheB --env-file $V/no_such_keyfile > $V/e_B.log 2>&1
echo "B exit $? ; lock present after B: $(ls $V/e_conc/ledger.lock 2>/dev/null || echo none); ledger sent so far $(grep -c '"sent"' $V/e_conc/ledger.jsonl)" >> $V/e_trace.txt
$PY $V/spec_e_C.json -- --model glm53 --trial 0 --out $V/e_conc --cache-dir $V/e_cacheC --env-file $V/no_such_keyfile > $V/e_C.log 2>&1
echo "C exit $?" >> $V/e_trace.txt
wait $PA; echo "A exit $?" >> $V/e_trace.txt
echo "glm53 sent rows in the shared ledger: $(grep -c '"sent"' $V/e_conc/ledger.jsonl)" >> $V/e_trace.txt
echo done >> $V/e_trace.txt
