"""Zero calls, post hoc, development scenes only (the harness skips held-out scenes). For every recorded
constraint-method record (s21 Qwen, s23 GLM / DeepSeek / Sonnet / Haiku / Gemini; cot and fsc share the plan's
target), re-decide the target's world point with a 'person snap' and re-match it with sgi.evaluate.match_target.

Snap rule (the same for every model, situation and target kind):
  candidates = detected people whose image box contains the clicked pixel;
  if any: target world := the candidate minimising |u - box centre u| / box width, at (x, y, 1.1 m);
  else  : keep the depth-lifted point (the pipeline as is).
Detectors: 'det' = I09 A1's frozen detector people (eval_people.json, E-real scenes);
           'oracle' = truth people as 0.5 m x 1.8 m upright boxes (an upper bound on the snap, not a method).
Run with the scenes WT venv:  .venv/bin/python snap_sweep.py"""
import sys, json, math, collections, hashlib
from pathlib import Path
sys.path.insert(0, 'docs/model_notes/tools')
import numpy as np
import research_harness as H
from sgi.evaluate import match_target

EP = Path('results/2026-10-17-beyond-vlm/phase1c/i09/a1/eval_people.json')
assert hashlib.sha256(EP.read_bytes()).hexdigest().startswith('ffa59545')
DET = {str(Path(k).resolve()): v for k, v in json.loads(EP.read_text())['scenes'].items() if v['domain'] == 'E-real'}
wl = H.whitelist()


def oracle_boxes(sc):
    cam = sc.observation.camera
    Tinv = np.linalg.inv(np.array(cam.T_world_camera))
    out = []
    for p in sc.truth.people:
        pts = []
        for z in (0.0, 1.8):
            for dx in (-0.25, 0.25):
                c = Tinv @ np.array([p.x, p.y, z, 1.0])
                if c[2] <= 0.1:
                    break
                # widen sideways in camera x
                pts.append((cam.fx * (c[0] + dx) / c[2] + cam.cx, cam.fy * c[1] / c[2] + cam.cy))
        if len(pts) == 4:
            us, vs = [q[0] for q in pts], [q[1] for q in pts]
            out.append(dict(x=p.x, y=p.y, box=[min(us), min(vs), max(us), max(vs)], id=p.id))
    return out


def snap(u, v, people):
    inside = [p for p in people if p['box'][0] <= u <= p['box'][2] and p['box'][1] <= v <= p['box'][3]]
    if not inside:
        return None
    best = min(inside, key=lambda p: abs(u - (p['box'][0] + p['box'][2]) / 2) / max(p['box'][2] - p['box'][0], 1.0))
    return (best['x'], best['y'], 1.1)


tally = collections.defaultdict(collections.Counter)
changes = []
for stage, model in [('s21', 'qwen38_low'), ('s23', 'glm53'), ('s23', 'dsv41'), ('s23', 'sonnet55_low'),
                     ('s23', 'haiku55'), ('s23', 'gemini38_low')]:
    for run in H.runs(stage=stage, method='con', model=model, wl=wl):
        for rec in H.records(run, wl=wl):
            t = (rec.result or {}).get('target')
            if not t or t.get('pixel') is None or t.get('world') is None:
                continue
            sc, gt, th, _ = H._against(rec)
            u, v = t['pixel']
            before = match_target(t['world'], sc.truth, gt, th) is not None
            assert before == bool(rec.score.get('target_correct')), (rec.scene_id, rec.episode_id)
            kinds = {sc.truth.targets[i].kind for i in gt.target_ids}
            kind = 'person' if kinds == {'person'} else ('object' if 'person' not in kinds else 'mixed')
            folder = str(Path(H.check_scene(rec.ref, rec.scene_id, wl=wl)).resolve())
            srcs = {'oracle': oracle_boxes(sc)}
            if folder in DET:
                srcs['det'] = DET[folder]['people']
            for src, people in srcs.items():
                w = snap(u, v, people)
                after = before if w is None else match_target(w, sc.truth, gt, th) is not None
                key = (model, src, kind)
                tally[key]['n'] += 1
                tally[key]['before'] += before
                tally[key]['after'] += after
                tally[key]['snapped'] += w is not None
                if after != before:
                    tally[key]['fixed' if after else 'broken'] += 1
                    changes.append((model, src, kind, rec.scene_id, rec.episode_id, rec.trial, 'fixed' if after else 'BROKEN'))

print('target_correct before -> after the snap; rows = (model, people source, target kind)\n')
print(f"{'model':14} {'src':6} {'kind':7} {'n':>4} {'before':>6} {'after':>6} {'snapped':>7} {'fixed':>5} {'broken':>6}")
for k in sorted(tally):
    c = tally[k]
    print(f"{k[0]:14} {k[1]:6} {k[2]:7} {c['n']:>4} {c['before']:>6} {c['after']:>6} {c['snapped']:>7} {c['fixed']:>5} {c['broken']:>6}")
print('\nby scene (det and oracle):')
by = collections.Counter((c[1], c[3].split('_', 1)[1], c[6]) for c in changes)
for k in sorted(by):
    print(' ', k, by[k])
