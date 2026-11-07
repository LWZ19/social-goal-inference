#!/bin/bash
# b2 replication (count-only on a cache of live-identical keys made by the fake gateway) and the exit code of a refusal
V=results/2026-10-17-beyond-vlm/phase2/i09/verify_i09_live_build
cd .-scenes
PY="env PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 .venv/bin/python $V/wrap_driver.py"
ONLY="real_eval_00_no_one_place real_eval_05_no_one_person real_eval_08_waiting_line_4"
rm -rf $V/d_cache && cp -r $V/c_cache $V/d_cache
echo '{"tag":"d1_count_full","block_paths":["no_such_keyfile"]}' > $V/spec_d1.json
echo '{"tag":"d2_count_cut","block_paths":["no_such_keyfile"]}' > $V/spec_d2.json
echo '{"tag":"d3_lock","fake_gateway":true,"block_paths":["no_such_keyfile"]}' > $V/spec_d3.json
$PY $V/spec_d1.json -- --model glm53 --trial 0 --out $V/d_count1 --cache-dir $V/d_cache --count-only --arms real --only $ONLY > $V/d1.log 2>&1; echo "d1 $?" >> $V/d_exits.txt
.venv/bin/python - <<'PY' >> $V/d_exits.txt
import json, glob, os
V = "results/2026-10-17-beyond-vlm/phase2/i09/verify_i09_live_build"
rows = [json.loads(l) for l in open(sorted(glob.glob(V + "/d_count1/count_only_glm53_*.jsonl"))[-1])]
cut = sorted({r["key"] for r in rows if r["stage"] == "constraints"})
for k in cut:
    os.remove(f"{V}/d_cache/{k}.json")
json.dump(cut, open(V + "/d_removed_keys.json", "w"))
print("removed", len(cut))
PY
$PY $V/spec_d2.json -- --model glm53 --trial 0 --out $V/d_count2 --cache-dir $V/d_cache --count-only --arms real --only $ONLY > $V/d2.log 2>&1; echo "d2 $?" >> $V/d_exits.txt
mkdir -p $V/d_lock && echo "verifier test lock" > $V/d_lock/ledger.lock
$PY $V/spec_d3.json -- --model glm53 --trial 0 --out $V/d_lock --cache-dir $V/d_cache --env-file $V/no_such_keyfile --arms real --only real_eval_00_no_one_place > $V/d3.log 2>&1; echo "d3_lock_exit $?" >> $V/d_exits.txt
echo done >> $V/d_exits.txt
