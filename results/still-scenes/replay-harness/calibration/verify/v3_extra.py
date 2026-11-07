"""Extra checks: people answer shared across episodes; alternative ambiguity definitions (per detection, per true person)."""
import sys, json, math, collections
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
from sgi.gt.derive import is_visible
OUT = 'results/2026-10-17-beyond-vlm/phase1/i01_calibration/verify/'
lines = []
P = lambda *a: lines.append(' '.join(str(x) for x in a))
tot = collections.Counter()
for model in ['glm53', 'ornith15', 'qwen36_35', 'qwen38', 'qwen3vl']:
    rv = H.runs(stage='s8', method='convlm', model=model)[0]
    th = H.recorded_config(rv)[1].thresholds
    w = 2 * th.person_radius_m
    recs = [r for r in H.records(rv) if r.solver_step]
    by = collections.defaultdict(list)
    c = collections.Counter()
    for r in recs:
        by[(r.scene_id, r.trial)].append(r)
        sc = H.scene(rv, r.scene_id)
        vis = [p for p in sc.truth.people if is_visible(p, sc.observation, th)]
        det = r.people
        c['det'] += len(det); c['vis'] += len(vis)
        c['det_amb'] += sum(sum(math.dist(d['xy'], (p.x, p.y)) <= w for p in vis) >= 2 for d in det)
        c['vis_amb'] += sum(sum(math.dist(d['xy'], (p.x, p.y)) <= w for d in det) >= 2 for p in vis)
        c['rec_any'] += any(sum(math.dist(d['xy'], (p.x, p.y)) <= w for p in vis) >= 2 for d in det) or any(sum(math.dist(d['xy'], (p.x, p.y)) <= w for d in det) >= 2 for p in vis)
        c['steps'] += 1
    pairs = [(k, v) for k, v in by.items() if len(v) > 1]
    same = sum(all(x.people == v[0].people for x in v) for k, v in pairs)
    c['clusters'] = len(by); c['multi'] = len(pairs); c['same_people'] = same
    fresh = sum(1 for r in recs for cl in r.result['calls'] if cl['stage'] == 'people' and not cl.get('cached'))
    c['fresh_people_calls'] = fresh
    P(model, dict(c))
    tot.update(c)
P('ALL', dict(tot))
open(OUT + 'v3_extra.txt', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
