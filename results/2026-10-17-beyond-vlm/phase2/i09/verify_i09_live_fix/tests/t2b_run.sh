#!/bin/bash
# T2b: real driver invocations sharing ONE ledger file without sharing the lock: each --out is its own folder (so its
# own ledger.lock), and each folder's ledger.jsonl is a symlink to one shared file. Fake gateway; each invocation has
# its own copy of the plan/context cache, so each demands all 84 constraints calls of its model.
#   v1: 3 x glm53 + 1 x dsv41, default caps 120/60   -> expect glm 60, dsv 60, total 120
#   v2: 2 x glm53 + 2 x dsv41, --max-calls 90        -> expect total 90, each model <= 60
V=results/2026-10-17-beyond-vlm/phase2/i09/verify_i09_live_fix/tests
cd .-scenes
PY="env PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 .venv/bin/python $V/wrap_driver.py"
run_variant() {
  local name=$1; shift; local extra=$1; shift; local members="$*"
  local T=$V/t2b/$name; rm -rf $T; mkdir -p $T/shared; : > $T/shared/ledger.jsonl; rm -f $V/fake/fake_t2b_${name}_*
  local pids=() tags=()
  for m in $members; do
    local id=${m}; local model=${m%%_*}
    mkdir -p $T/o_$id; ln -s ../shared/ledger.jsonl $T/o_$id/ledger.jsonl; cp -r $V/cache_pc $T/cache_$id
    echo "{\"tag\":\"t2b_${name}_$id\",\"fake_gateway\":true,\"sleep_s\":0.05,\"block_paths\":[\"no_such_keyfile\"]}" > $T/spec_$id.json
  done
  for m in $members; do
    local model=${m%%_*}
    $PY $T/spec_$m.json -- --model $model --trial 0 --out $T/o_$m --cache-dir $T/cache_$m --env-file $V/no_such_keyfile $extra > $T/$m.log 2>&1 &
    pids+=($!); tags+=($m)
  done
  local i=0; : > $T/exits.txt
  for p in "${pids[@]}"; do wait $p; echo "${tags[$i]} exit $?" >> $T/exits.txt; i=$((i+1)); done
  for m in $members; do echo "$m lock left: $(ls $T/o_$m/ledger.lock 2>/dev/null || echo none)" >> $T/exits.txt; done
  cat $T/exits.txt
}
run_variant v1 "" glm53_1 glm53_2 glm53_3 dsv41_1
run_variant v2 "--max-calls 90" glm53_1 glm53_2 dsv41_1 dsv41_2
