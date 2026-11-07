#!/bin/bash
V=results/2026-10-17-beyond-vlm/phase2/i09/verify_i09_live_build
cd .-scenes
PY="env PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 .venv/bin/python $V/wrap_driver.py"
COMMON="--trial 0 --out $V/c_cap --cache-dir $V/c_cache --env-file $V/no_such_keyfile"
$PY $V/spec_c1.json -- --model glm53 $COMMON > $V/c1.log 2>&1; echo "c1 $?" >> $V/c_exits.txt
$PY $V/spec_c2.json -- --model dsv41 $COMMON > $V/c2.log 2>&1; echo "c2 $?" >> $V/c_exits.txt
$PY $V/spec_c3.json -- --model glm53 $COMMON > $V/c3.log 2>&1; echo "c3 $?" >> $V/c_exits.txt
$PY $V/spec_c4.json -- --model dsv41 $COMMON --max-calls 121 > $V/c4.log 2>&1; echo "c4 $?" >> $V/c_exits.txt
$PY $V/spec_c4.json -- --model dsv41 $COMMON --max-calls-per-model 61 > $V/c4b.log 2>&1; echo "c4b $?" >> $V/c_exits.txt
echo done >> $V/c_exits.txt
