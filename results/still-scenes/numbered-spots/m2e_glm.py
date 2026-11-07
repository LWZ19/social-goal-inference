"""POST HOC (questioner's notes on GLM's M2 result), zero calls: (1) a layout-level exact sign-flip test of per-layout
net (gain - harm); (2) GLM's gains split by A0's error class (near miss within 15 px of a right candidate, or the
wrong object); (3) what the plan-exact improvements are. m2e_glm.py runs/m2_glm53"""
import collections, itertools, json, math, sys
from pathlib import Path
import numpy as np
import m1_coverage as M
import m2lib as L
from m2_score import _scene_ctx, first_nav
X, H = M.X, M.H
run = Path(sys.argv[1])
M.check_frozen()
from sgi.evaluate import match_target
wl = H.whitelist()
rows = {(r['scene'], r['episode'], r['trial']): r for r in map(json.loads, (run / 'm2_rows.jsonl').read_text().splitlines())}
calls = [json.loads(l) for l in (run / 'calls.jsonl').read_text().splitlines() if l.strip()]
by = collections.defaultdict(dict)
for c in calls:
    by[(c['scene_id'], c['episode'], c['trial'])][c['arm']] = c
# (1) layout-level sign flip
net = collections.Counter()
for (sid, eid, t), r in rows.items():
    lay = sid.split('_', 1)[1]
    net[lay] += int(bool(not r.get('A0_right') and r.get('A2_both_right'))) - int(bool(r.get('A0_right') and not r.get('A2_both_right')))
vals = [net[k] for k in sorted(net)]
obs = sum(vals)
nz = [v for v in vals if v != 0]
cnt = sum(1 for signs in itertools.product((1, -1), repeat=len(nz)) if abs(sum(s * v for s, v in zip(signs, nz))) >= abs(obs))
print(f'(1) per-layout net (gain - harm): {dict(sorted(net.items()))}; total {obs:+d}; exact two-sided sign-flip over '
      f'{len(nz)} non-zero layouts: p = {cnt / 2 ** len(nz):.4f}')
# (2) gains by A0 error class
cache = {}
cls = collections.Counter(); examples = collections.defaultdict(list)
for (sid, eid, t), r in rows.items():
    if not (not r.get('A0_right') and r.get('A2_both_right')):
        continue
    a0 = by[(sid, eid, t)]['A0']
    sref, sc, g, cands = _scene_ctx(cache, wl, a0['scenes_dir'], sid)
    ep = sc.truth.episodes[eid]
    th = H.recorded_config(sref.runs[0])[1].thresholds
    _, s0 = first_nav(a0['subtasks'])
    if s0 is None or s0.get('point') is None:
        cls['no point in A0'] += 1; continue
    u, v = s0['point']
    right = [c for c in cands if (lambda tg: tg is not None and match_target(tg.world, sc.truth, ep, th) is not None)(
        L.resolve(g, cands, list(range(1, len(cands) + 1)), cands.index(c) + 1, None, 'number')[0])]
    d = min(L._dist_to_cand(g, c, u, v) for c in right) if right else float('inf')
    k = 'near miss (<= 15 px)' if d <= 15 else 'wrong object (> 15 px)'
    cls[k] += 1; examples[k].append(f"{sid.split('_',1)[1]} '{s0.get('target')}' d={d:.0f}px")
print('(2) GLM gains by A0 error class:', dict(cls))
for k, ex in examples.items():
    print('   ', k, '|', '; '.join(sorted(ex)[:12]))
# (3) plan exact improvements
imp = [(k, r) for k, r in rows.items() if r.get('A0_plan_exact') is False and r.get('A2_plan_exact') is True]
wors = [(k, r) for k, r in rows.items() if r.get('A0_plan_exact') is True and r.get('A2_plan_exact') is False]
with_target = sum(1 for k, r in imp if not r.get('A0_right') and r.get('A2_both_right'))
print(f'(3) plan exact: improved {len(imp)}, worsened {len(wors)}; of the improved, {with_target} are also target gains')
struct = collections.Counter()
for k, r in imp:
    a0, a2 = by[k]['A0'], by[k]['A2']
    n0 = [s['type'] for s in a0['subtasks']]; n2 = [s['type'] for s in a2['subtasks']]
    struct['same subtask types' if n0 == n2 else f'types {"-".join(x[0] for x in n0)} -> {"-".join(x[0] for x in n2)}'] += 1
print('    improved, by subtask structure:', dict(struct.most_common(8)))
