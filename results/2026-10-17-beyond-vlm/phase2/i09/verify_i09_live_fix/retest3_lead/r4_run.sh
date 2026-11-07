#!/bin/bash
# R4: the upstream guard, build 41940c3c, real command path, fake gateway.
#   a  glm53 live on an EMPTY cache, --arms real                -> exit 2, 0 fake calls, check file stopped_by UpstreamMiss
#   b  glm53 live, plan/context cache minus the 5th context key of arm real -> 4 constraints sent, then exit 2, check
#      file with stopped_by UpstreamMiss (context)
#   b2 dsv41 live, plan/context cache minus the plan key of the 3rd record of arm plain, --arms plain -> partial, exit 2
#   c  dsv41 live on the plan/context cache, --arms staffed     -> exit 0, only constraints requests reach the fake
source results/2026-10-17-beyond-vlm/phase2/i09/verify_i09_live_fix/retest3_lead/common.sh
T=$R/r4; rm -rf $T; mkdir -p $T/a_out $T/a_cache $T/b_out $T/b2_out $T/c_out; rm -f $R/fake/fake_r4_*
cp -r $R/cache_pc_base $T/b_cache; cp -r $R/cache_pc_base $T/b2_cache; cp -r $R/cache_pc_base $T/c_cache
.venv/bin/python -I - <<PY > $T/removed_keys.json
import json, os
rows = json.load(open("$R/keys.json"))["rows_by_model"]
ctx = [r for r in rows["glm53"] if r["arm"] == "real" and r["stage"] == "context"][4]
os.remove("$T/b_cache/" + ctx["key"] + ".json")
plan = [r for r in rows["dsv41"] if r["arm"] == "plain" and r["stage"] == "plan"][2]
os.remove("$T/b2_cache/" + plan["key"] + ".json")
print(json.dumps({"b": ctx, "b2": plan}))
PY
cf() { local CF=$(ls $1/check_* 2>/dev/null | tail -1); python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
print('check', sys.argv[1].split('/')[-1], '| stopped_by', (d['stopped_by'] or 'None')[:150], '| cache_total', d['cache_total'], '| ledger', d.get('ledger_at_start'), d.get('ledger_at_end'), '| arms', [(a['arm'], a['records'], (a['stopped_by'] or '')[:40]) for a in d['arms']])" "$CF" 2>&1 | cut -c1-600; }
cfg() { local F=$(ls -d $1/*_con 2>/dev/null | tail -1); echo "config: $(grep -h '^incomplete' $F/config.yaml) $(grep -h '^stopped_by' $F/config.yaml | cut -c1-60); results lines $(cat $F/results.jsonl | wc -l)"; }
titles() { cat $R/fake/fake_r4_$1_*.jsonl 2>/dev/null | python3 -c 'import sys,json,collections; print(dict(collections.Counter(json.loads(l).get("schema_title") for l in sys.stdin)))'; }
E=$T/exits.txt; : > $E
$PY $(spec $T r4_a) -- --model glm53 --trial 0 --out $T/a_out --cache-dir $T/a_cache --env-file $NK --arms real > $T/a.log 2>&1 & PA=$!
$PY $(spec $T r4_b) -- --model glm53 --trial 0 --out $T/b_out --cache-dir $T/b_cache --env-file $NK --arms real > $T/b.log 2>&1 & PB=$!
$PY $(spec $T r4_b2) -- --model dsv41 --trial 0 --out $T/b2_out --cache-dir $T/b2_cache --env-file $NK --arms plain > $T/b2.log 2>&1 & PB2=$!
$PY $(spec $T r4_c) -- --model dsv41 --trial 0 --out $T/c_out --cache-dir $T/c_cache --env-file $NK --arms staffed > $T/c.log 2>&1 & PC=$!
wait $PA; echo "a: exit $?; fake $(fake r4_a); ledger sent $(sent $T/a_out) lines $(cat $T/a_out/ledger.jsonl | wc -l); cache files $(ls $T/a_cache | wc -l); $(cf $T/a_out); $(cfg $T/a_out)" >> $E
wait $PB; echo "b: exit $?; fake $(fake r4_b) titles $(titles b); ledger sent $(sent $T/b_out); cache files $(ls $T/b_cache | wc -l) (335 + sends); $(cf $T/b_out); $(cfg $T/b_out)" >> $E
wait $PB2; echo "b2: exit $?; fake $(fake r4_b2) titles $(titles b2); ledger sent $(sent $T/b2_out); $(cf $T/b2_out); $(cfg $T/b2_out)" >> $E
wait $PC; echo "c: exit $?; fake $(fake r4_c) titles $(titles c); ledger sent $(sent $T/c_out); $(cf $T/c_out); $(cfg $T/c_out)" >> $E
echo done >> $E
cat $T/removed_keys.json; cat $E
