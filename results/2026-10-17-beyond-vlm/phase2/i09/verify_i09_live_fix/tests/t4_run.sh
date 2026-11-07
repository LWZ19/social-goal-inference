#!/bin/bash
# T4: the upstream guard (live mode refuses plan/locate/context sends), NEW driver, fake gateway.
#   t4a glm53 live on an EMPTY cache, --arms real             -> exit 2 UpstreamMiss, 0 fake calls, 0 sent, no lock
#   t4b dsv41 live on the plan/context cache, --arms staffed  -> exit 0, only constraints requests reach the fake
#   t4c glm53 live on the plan/context cache minus ONE context key (5th record of arm real) --arms real
#                                                             -> 4 constraints sent, then exit 2 UpstreamMiss (context)
V=results/2026-10-17-beyond-vlm/phase2/i09/verify_i09_live_fix/tests
cd .-scenes
PY="env PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 .venv/bin/python $V/wrap_driver.py"
T=$V/t4; rm -rf $T; mkdir -p $T/a_out $T/a_cache $T/b_out $T/c_out; cp -r $V/cache_pc $T/b_cache; cp -r $V/cache_pc $T/c_cache
rm -f $V/fake/fake_t4_*
for t in a b c; do echo "{\"tag\":\"t4_$t\",\"fake_gateway\":true,\"block_paths\":[\"no_such_keyfile\"]}" > $T/spec_$t.json; done
.venv/bin/python -I - <<PY > $T/c_removed_context_key.json
import json
rows = json.load(open("$V/keys.json"))["rows_by_model"]["glm53"]
ctx = [r for r in rows if r["arm"] == "real" and r["stage"] == "context"]
r = ctx[4]
import os
os.remove("$T/c_cache/" + r["key"] + ".json")
print(json.dumps(r))
PY
E=$T/exits.txt; : > $E
$PY $T/spec_a.json -- --model glm53 --trial 0 --out $T/a_out --cache-dir $T/a_cache --env-file $V/no_such_keyfile --arms real > $T/a.log 2>&1
echo "t4a exit $? ; lock left: $(ls $T/a_out/ledger.lock 2>/dev/null || echo none); ledger sent $(grep -c '"sent"' $T/a_out/ledger.jsonl 2>/dev/null); ledger lines $(wc -l < $T/a_out/ledger.jsonl 2>/dev/null); fake $(cat $V/fake/fake_t4_a_*.jsonl 2>/dev/null | wc -l); cache files $(ls $T/a_cache | wc -l); UpstreamMiss in stderr: $(grep -c 'UpstreamMiss' $T/a.log)" >> $E
$PY $T/spec_b.json -- --model dsv41 --trial 0 --out $T/b_out --cache-dir $T/b_cache --env-file $V/no_such_keyfile --arms staffed > $T/b.log 2>&1
echo "t4b exit $? ; lock left: $(ls $T/b_out/ledger.lock 2>/dev/null || echo none); ledger sent $(grep -c '"sent"' $T/b_out/ledger.jsonl 2>/dev/null); fake $(cat $V/fake/fake_t4_b_*.jsonl 2>/dev/null | wc -l); fake schema titles: $(cat $V/fake/fake_t4_b_*.jsonl | .venv/bin/python -I -c 'import sys,json,collections; print(dict(collections.Counter(json.loads(l)["schema_title"] for l in sys.stdin)))')" >> $E
$PY $T/spec_c.json -- --model glm53 --trial 0 --out $T/c_out --cache-dir $T/c_cache --env-file $V/no_such_keyfile --arms real > $T/c.log 2>&1
echo "t4c exit $? ; lock left: $(ls $T/c_out/ledger.lock 2>/dev/null || echo none); ledger sent $(grep -c '"sent"' $T/c_out/ledger.jsonl 2>/dev/null); fake $(cat $V/fake/fake_t4_c_*.jsonl 2>/dev/null | wc -l); UpstreamMiss line: $(grep -h 'UpstreamMiss' $T/c.log | tail -n 1 | cut -c1-250)" >> $E
echo done >> $E
cat $E
