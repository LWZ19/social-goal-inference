#!/bin/bash
# R5: exit codes through the real command path (wrapper -> runpy driver.py __main__ -> _command), build 41940c3c.
source results/2026-10-17-beyond-vlm/phase2/i09/verify_i09_live_fix/retest3_lead/common.sh
T=$R/r5; rm -rf $T; mkdir -p $T/empty_cache; rm -f $R/fake/fake_r5_*
E=$T/exits.txt; : > $E
one() {  # one NAME EXPECT driver-args...
  local N=$1 X=$2; shift 2
  $PY $(spec $T r5_$N) -- "$@" > $T/$N.log 2>&1; local G=$?
  echo "$N: exit $G (expect $X) $( [ "$G" = "$X" ] && echo OK || echo MISMATCH ); fake $(fake r5_$N); out files: $(ls $T/${N}_out 2>/dev/null | tr '\n' ' '); msg: $(grep -hv '^WRAP\|INFO' $T/$N.log | tail -1 | cut -c1-170)" >> $E
}
C="--trial 0 --cache-dir $T/empty_cache"
# --- unknown --only scene
one unk_count 2 --model glm53 $C --out $T/unk_count_out --count-only --only no_such_scene_xyz
one unk_live 2  --model glm53 $C --out $T/unk_live_out --env-file $NK --only no_such_scene_xyz
one unk_mock 2  --model glm53 --trial 0 --out $T/unk_mock_out --mock --only no_such_scene_xyz
one unk_mixed 2 --model glm53 $C --out $T/unk_mixed_out --env-file $NK --only real_eval_08_waiting_line_4 no_such_scene_xyz
one unk_otherarm 2 --model glm53 $C --out $T/unk_otherarm_out --env-file $NK --arms staffed --only real_eval_08_waiting_line_4
one unk_othermodel 2 --model dsv41 $C --out $T/unk_othermodel_out --count-only --only no_such_scene_xyz
# --- bad arguments
one bad_model 2      --model qwen38 $C --out $T/bad_out --count-only
one bad_trial 2      --model glm53 --trial 1 --cache-dir $T/empty_cache --out $T/bad_out --count-only
one bad_cap_total 2  --model glm53 $C --out $T/bad_out --env-file $NK --max-calls 121
one bad_cap_model 2  --model glm53 $C --out $T/bad_out --env-file $NK --max-calls-per-model 61
one bad_cap_zero 2   --model glm53 $C --out $T/bad_out --env-file $NK --max-calls 0
one bad_no_envfile 2 --model glm53 $C --out $T/bad_out
one bad_rel_envfile 2 --model glm53 $C --out $T/bad_out --env-file no_such_keyfile
one bad_count_envfile 2 --model glm53 $C --out $T/bad_out --count-only --env-file $NK
one bad_count_mock 2 --model glm53 $C --out $T/bad_out --count-only --mock
one bad_arms 2       --model glm53 $C --out $T/bad_out --count-only --arms real,bogus
one bad_rel_cache 2  --model glm53 --trial 0 --cache-dir retest_rel_cache --out $T/bad_out --count-only
one bad_missing_cache 2 --model glm53 --trial 0 --cache-dir $T/no_such_cache --out $T/bad_out --count-only
one help 0 --help
echo "bad_out after the bad-argument runs: $(ls $T/bad_out 2>/dev/null | tr '\n' ' ') (lock or ledger here would be wrong)" >> $E
# --- vlm_error (every attempt of the first request raises TransientError) -> 1; the same command once more -> 0
mkdir -p $T/ve_out; cp -r $R/cache_pc_base $T/ve_cache
VE="--model glm53 --trial 0 --out $T/ve_out --cache-dir $T/ve_cache --env-file $NK --arms real --only $TWO"
$PY $(spec $T r5_ve1 ',"sticky_transient":true') -- $VE > $T/ve1.log 2>&1; X1=$?
S1=$(sent $T/ve_out)
$PY $(spec $T r5_ve2) -- $VE > $T/ve2.log 2>&1; X2=$?
echo "vlm_error: first exit $X1 (expect 1), ledger sent $S1, fake $(fake r5_ve1), ledger: line in log: $(grep -c 'ledger:' $T/ve1.log); re-run exit $X2 (expect 0), ledger sent now $(sent $T/ve_out), fake on re-run $(fake r5_ve2); folders: $(for f in $T/ve_out/*_con; do echo -n "$(basename $f)=$(grep -h '^incomplete' $f/config.yaml | cut -d' ' -f2),vlm_errors=$(grep -h '^vlm_errors' $f/config.yaml | cut -d' ' -f2) "; done)" >> $E
# a re-run that fails again -> 1 again (REQUEST: stop then)
mkdir -p $T/ve3_out; cp -r $R/cache_pc_base $T/ve3_cache
VE3="--model glm53 --trial 0 --out $T/ve3_out --cache-dir $T/ve3_cache --env-file $NK --arms real --only $TWO"
$PY $(spec $T r5_ve3a ',"sticky_transient":true') -- $VE3 > $T/ve3a.log 2>&1; Y1=$?
$PY $(spec $T r5_ve3b ',"sticky_transient":true') -- $VE3 > $T/ve3b.log 2>&1; Y2=$?
echo "vlm_error twice: exits $Y1 then $Y2 (expect 1, 1); ledger sent $(sent $T/ve3_out); fake $(fake r5_ve3a)+$(fake r5_ve3b)" >> $E
# --- cap: glm53 caps 5/5 -> 3; then dsv41 total cap 5 on the same ledger -> 3 with 0 sends
mkdir -p $T/cap_out; cp -r $R/cache_pc_base $T/cap_cache
$PY $(spec $T r5_cap1) -- --model glm53 --trial 0 --out $T/cap_out --cache-dir $T/cap_cache --env-file $NK --arms real --max-calls 5 --max-calls-per-model 5 > $T/cap1.log 2>&1; X=$?
CF=$(ls $T/cap_out/check_* | tail -1)
echo "cap glm53 5/5: exit $X (expect 3); ledger sent $(sent $T/cap_out) refused $(grep -c '"refused"' $T/cap_out/ledger.jsonl); fake $(fake r5_cap1); check stopped_by: $(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['stopped_by'][:110])" $CF); config: $(grep -h '^incomplete' $T/cap_out/*_real_glm53_con/config.yaml)" >> $E
$PY $(spec $T r5_cap2) -- --model dsv41 --trial 0 --out $T/cap_out --cache-dir $T/cap_cache --env-file $NK --arms real --max-calls 5 > $T/cap2.log 2>&1; X=$?
echo "cap dsv41 total 5 on the same ledger: exit $X (expect 3); ledger sent $(sent $T/cap_out) refused $(grep -c '"refused"' $T/cap_out/ledger.jsonl); fake $(fake r5_cap2)" >> $E
# --- clean
mkdir -p $T/clean_out; cp -r $R/cache_pc_base $T/clean_cache
$PY $(spec $T r5_clean) -- --model dsv41 --trial 0 --out $T/clean_out --cache-dir $T/clean_cache --env-file $NK --arms real --only $TWO > $T/clean.log 2>&1; X=$?
echo "clean: exit $X (expect 0); ledger sent $(sent $T/clean_out); fake $(fake r5_clean); check stopped_by: $(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['stopped_by'])" $(ls $T/clean_out/check_*))" >> $E
echo done >> $E
cat $E
