"""Smoke test of repairs.py on a few records of s19 glm53 real (no output files)."""
import sys, time, json
sys.path.insert(0, 'results/still-scenes/replay-harness/repairs')
import repairs as R
H = R.H
ref = R.one_run('s19', 'real', 'glm53')
n = 0
t0 = time.monotonic()
for rec in sorted(H.records(ref), key=lambda r: (r.scene_id, r.episode_id, r.trial)):
    if not rec.solver_step:
        continue
    row = dict(situation=rec.situation, scene=rec.scene_id, C_rec=bool(rec.score.get('success')))
    R.solver_row(rec, row)
    print(json.dumps({k: v for k, v in row.items() if k not in ('uses_P', 'uses_C')}))
    n += 1
    if n >= 2:
        break
print('time per record', (time.monotonic() - t0) / n)
