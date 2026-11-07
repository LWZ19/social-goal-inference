"""Zero calls. Where every recorded constraint-method (con) run on conversation_group_4 put its target point,
against the four group members. Target = p0, 'the person in red'. A person target matches if the target point's ground
projection is within 0.75 m of p0 AND p0 is the nearest person to it (sgi.evaluate._indicates).
Run with the scenes WT venv:  .venv/bin/python g4_targets.py"""
import sys, json, math, collections
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / '_lib' / 'paths.py').exists()) / '_lib'))
import paths  # noqa: E402  (results/_lib/paths.py)
sys.path.insert(0, str(paths.HARNESS))   # research_harness.py and its run list
import research_harness as H
wl = H.whitelist()
P = {'p0': (4.487, -4.456), 'p1': (3.770, -5.391), 'p2': (4.818, -6.096), 'p3': (5.480, -5.060),
     'p4': (-2.447, -8.609), 'p5': (-2.835, -7.502), 'p6': (0.203, -0.894), 's0': (9.726, -9.203)}
rows = []
for stage, model in [('s21', 'qwen38_low'), ('s23', 'glm53'), ('s23', 'dsv41'), ('s23', 'sonnet55_low'),
                     ('s23', 'haiku55'), ('s23', 'gemini38_low')]:
    for method in ('con', 'cot', 'fsc'):
        for run in H.runs(stage=stage, method=method, model=model, wl=wl):
            for rec in H.records(run, wl=wl):
                if 'conversation_group_4' not in rec.scene_id:
                    continue
                res = rec.result or {}
                t = res.get('target') or {}
                w = t.get('world')
                sub = rec.raw['plan']['subtasks'][rec.raw['evaluated_subtask']] if rec.raw.get('plan') and rec.raw.get('evaluated_subtask') is not None else {}
                near = None
                if w:
                    d = sorted((math.hypot(w[0] - x, w[1] - y), k) for k, (x, y) in P.items())
                    near = (d[0][1], round(d[0][0], 2), round(math.hypot(w[0] - P['p0'][0], w[1] - P['p0'][1]), 2), round(w[2], 2))
                rows.append(dict(model=model, method=method, scene=rec.scene_id.split('_')[0], ep=rec.episode_id, trial=rec.trial,
                                 status=res.get('status'), ok=rec.score.get('success'), tc=rec.score.get('target_correct'),
                                 near=near, desc=sub.get('target'), px=(t.get('point') or {}), solver=bool(getattr(rec, 'solver_step', None))))
out = collections.defaultdict(collections.Counter)
for r in rows:
    key = (r['model'], r['method'])
    out[key]['n'] += 1
    out[key]['tc'] += bool(r['tc'])
    if not r['tc'] and r['near']:
        out[key]['wrong->' + r['near'][0]] += 1
    if not r['tc'] and not r['near']:
        out[key]['no_target'] += 1
print('target_correct by model and method (all three variants, both episodes, all trials):')
for k in sorted(out):
    print(' ', k, dict(out[k]))
print()
for r in rows:
    if r['method'] == 'con' and not r['tc']:
        print(r['model'], r['scene'], r['ep'], 't', r['trial'], r['status'], 'nearest/dist/dist_p0/z', r['near'], 'px', r['px'], '|', r['desc'])
