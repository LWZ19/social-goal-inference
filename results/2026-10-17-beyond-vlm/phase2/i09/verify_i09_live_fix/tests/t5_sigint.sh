#!/bin/bash
# T5 (SIGINT re-run): the first attempt in t5_run.sh started the run as a background job of a non-interactive shell,
# which inherits SIGINT = SIG_IGN, so the SIGINT was ignored (exit 0, a full pass). Here the wrapper restores Python's
# default SIGINT handler (as a terminal's Ctrl-C would find it), then SIGINT is sent mid-run, twice: once while the
# fake gateway sleeps inside a send, once (sleep 0) at a random point.
V=results/2026-10-17-beyond-vlm/phase2/i09/verify_i09_live_fix/tests
cd .-scenes
PY="env PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 .venv/bin/python $V/wrap_driver.py"
T=$V/t5/sigint2; rm -rf $T; mkdir -p $T; rm -f $V/fake/fake_t5_sigint2_*
NK=$V/no_such_keyfile
E=$T/exits.txt; : > $E
for k in a b; do
  cp -r $V/cache_pc $T/cache_$k
  if [ $k = a ]; then S=',"sleep_s":1.0'; else S=''; fi
  echo "{\"tag\":\"t5_sigint2_$k\",\"fake_gateway\":true,\"restore_sigint\":true,\"block_paths\":[\"no_such_keyfile\"]$S}" > $T/spec_$k.json
  $PY $T/spec_$k.json -- --model glm53 --trial 0 --out $T/out_$k --cache-dir $T/cache_$k --env-file $NK --arms real > $T/$k.log 2>&1 &
  P=$!
  for i in $(seq 1 900); do [ -f $T/out_$k/ledger.jsonl ] && [ "$(grep -c '"sent"' $T/out_$k/ledger.jsonl)" -ge 3 ] && break; sleep 0.2; done
  echo "$k: before kill lock: '$(cat $T/out_$k/ledger.lock 2>/dev/null || echo none)' pid $P; sent $(grep -c '"sent"' $T/out_$k/ledger.jsonl)" >> $E
  kill -INT $P; wait $P; X=$?
  echo "$k: sigint exit $X ; lock left: $(ls $T/out_$k/ledger.lock 2>/dev/null || echo none); ledger sent $(grep -c '"sent"' $T/out_$k/ledger.jsonl); answered $(grep -c '"answered"' $T/out_$k/ledger.jsonl); fake $(cat $V/fake/fake_t5_sigint2_${k}_*.jsonl | wc -l); KeyboardInterrupt in log: $(grep -c KeyboardInterrupt $T/$k.log); run folder incomplete: $(grep -h '^incomplete' $T/out_$k/*_con/config.yaml)" >> $E
done
echo done >> $E
cat $E
