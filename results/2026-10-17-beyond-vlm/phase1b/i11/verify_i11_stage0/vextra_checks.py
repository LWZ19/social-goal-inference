"""Small extra checks: L_S source (largest fit polygon diagonal over H.examples) and target pixels in the lift
window of the 9 primary person occluder captures."""
import itertools, json, math
import numpy as np
from vlib import H, SceneGeo, check_pins, win_slice, OUT
from sgi.config import load_config
check_pins()
th = load_config().thresholds
best = (0, None)
for d, sid in H.examples():
    sc = H.example(sid, d)
    polys = [('obstacle', o) for o in sc.truth.obstacles] + [('target', t.footprint) for t in sc.truth.targets.values() if t.footprint is not None]
    for kind, p in polys:
        P = np.asarray(p, float)
        dmax = max(math.dist(a, b) for a, b in itertools.combinations(P, 2))
        if dmax > best[0]:
            best = (dmax, (sid, kind))
print('largest fit polygon diagonal', best, '-> L_S =', 2 * round(best[0], 2) + 0.10)
rows = [json.loads(l) for l in open(OUT / 'vrows_primary.jsonl')]
occ = [r for r in rows if r['has_point'] and not r['right'] and r['kind'] == 'person' and r['lclass'] == 'occluder']
runs = {r.folder: r for r in H.runs(method='con')}
for r in occ:
    run = runs[r['run']]
    rec = [x for x in H.records(run) if x.scene_id == r['scene'] and x.episode_id == r['episode'] and x.trial == r['trial']][0]
    rth = H.recorded_config(run)[1].thresholds
    geo = SceneGeo(H.scene(run, r['scene']), rth)
    i, j = math.floor(r['v']), math.floor(r['u'])
    rs, cs = win_slice(i, j, geo.h, geo.w)
    print(r['model'], r['scene'], r['episode'], r['trial'], 'T1', r['T1_G1'], 'target px in window', int(geo.body(r['tid'])[rs, cs].sum()))
