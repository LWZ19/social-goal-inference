#!/bin/bash
# R3 (command path): the ledger loses sent rows during a live run -> LedgerLost, exit 2, check file with stopped_by.
#   d1 the fake deletes ledger.jsonl inside its 3rd call            (the "answered" note of that call meets the loss)
#   d2 ledger.jsonl removed from outside while the fake sleeps (rm)
#   d3 the fake truncates it in place inside its 4th call
#   d4 the fake drops one "sent" row in place inside its 4th call
#   d5 ledger.jsonl moved away from outside between sends (mv), fake sleep 0
# Then, for d2: a new invocation on the same --out (documents what it finds).
source results/2026-10-17-beyond-vlm/phase2/i09/verify_i09_live_fix/retest/common.sh
T=$R/r3/cmd; rm -rf $T; mkdir -p $T; rm -f $R/fake/fake_r3_*
report() {  # report NAME EXIT
  local O=$T/$1_out; local F=$(ls -d $O/*_con 2>/dev/null | tail -1); local CF=$(ls $O/check_* 2>/dev/null | tail -1)
  echo "$1: exit $2; fake $(fake r3_$1); ledger now: exists $(test -f $O/ledger.jsonl && echo yes || echo no), sent $(sent $O), lines $(cat $O/ledger.jsonl 2>/dev/null | wc -l); check file: $(basename "$CF" 2>/dev/null); stopped_by: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['stopped_by'], '| ledger_at_start', d.get('ledger_at_start'), '| ledger_at_end', d.get('ledger_at_end'), '| arms', [(a['arm'], a['records'], a['stopped_by'] is not None) for a in d['arms']])" "$CF" 2>&1 | cut -c1-420); config: $(grep -h '^incomplete' $F/config.yaml) / $(grep -h '^stopped_by' $F/config.yaml | cut -c1-80); stderr stopped line: $(grep -h '^stopped:' $T/$1.log | cut -c1-120)" >> $T/exits.txt
}
: > $T/exits.txt
run_bg() {  # run_bg NAME EXTRA_SPEC
  mkdir -p $T/$1_out; cp -r $R/cache_pc_base $T/$1_cache
  $PY $(spec $T r3_$1 "$2") -- --model glm53 --trial 0 --out $T/$1_out --cache-dir $T/$1_cache --env-file $NK --arms real > $T/$1.log 2>&1 &
}
run_bg d1 ",\"ledger_action\":{\"at\":3,\"action\":\"delete\",\"path\":\"$T/d1_out/ledger.jsonl\"}"; P1=$!
run_bg d3 ",\"ledger_action\":{\"at\":4,\"action\":\"truncate\",\"path\":\"$T/d3_out/ledger.jsonl\"}"; P3=$!
run_bg d4 ",\"ledger_action\":{\"at\":4,\"action\":\"drop_one\",\"path\":\"$T/d4_out/ledger.jsonl\"}"; P4=$!
run_bg d2 ',"sleep_s":1.0'; P2=$!
run_bg d5 ''; P5=$!
( for i in $(seq 1 900); do [ "$(sent $T/d2_out)" -ge 3 ] && break; sleep 0.1; done; rm $T/d2_out/ledger.jsonl; echo "d2 removed at $(date +%T.%N)" >> $T/actions.txt ) &
( for i in $(seq 1 3000); do [ "$(sent $T/d5_out)" -ge 3 ] && break; sleep 0.02; done; mv $T/d5_out/ledger.jsonl $T/d5_out/moved_ledger.jsonl; echo "d5 moved at $(date +%T.%N)" >> $T/actions.txt ) &
wait $P1; report d1 $?
wait $P2; report d2 $?
wait $P3; report d3 $?
wait $P4; report d4 $?
wait $P5; report d5 $?
wait
# a new invocation on d2's --out after the loss
$PY $(spec $T r3_d2_next) -- --model glm53 --trial 0 --out $T/d2_out --cache-dir $T/d2_cache --env-file $NK --arms real --only $TWO > $T/d2_next.log 2>&1
echo "d2_next: exit $?; its start line: $(grep -h 'ledger:' $T/d2_next.log | head -1 | cut -c1-160); fake $(fake r3_d2_next)" >> $T/exits.txt
echo done >> $T/exits.txt
cat $T/actions.txt $T/exits.txt
