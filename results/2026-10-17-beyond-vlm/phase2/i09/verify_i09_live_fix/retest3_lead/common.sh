# Shared settings of the re-test scripts (sourced). Zero model calls: the wrapper replaces the real back end with a
# fake gateway, and the --env-file names a file that does not exist (its open is also blocked by the audit hook).
R=results/2026-10-17-beyond-vlm/phase2/i09/verify_i09_live_fix/retest3_lead
LIVE=results/2026-10-17-beyond-vlm/phase2/i09/live
cd .-scenes
ENV="env PYTHONPATH= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10"
PY="$ENV .venv/bin/python $R/wrap_driver.py"
NK=$R/no_such_keyfile
TWO="real_eval_08_waiting_line_4 real_eval_11_conversation_group_3"
spec() {  # spec DIR TAG [extra json]
  echo "{\"tag\":\"$2\",\"fake_gateway\":true,\"block_paths\":[\"no_such_keyfile\"]$3}" > $1/spec_$2.json; echo $1/spec_$2.json; }
sent() { grep -c '"event": "sent"' $1/ledger.jsonl 2>/dev/null || echo 0; }
fake() { cat $R/fake/fake_$1_*.jsonl 2>/dev/null | grep -c '"digest"'; }
