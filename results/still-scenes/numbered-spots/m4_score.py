"""Marks probe M4 scorer (M4_FROZEN.md): per model and arm, how often the plan's target (the first navigate subtask,
grounded as M2: U by its click, M by the 'both' rule) lies within 1.0 m (horizontal) of the scenario's slot-0 face
point, the counter the line serves; the unmarked arm's trial-to-trial flips as its redraw floor. Zero calls.
    m4_score.py runs/m4_<model>
"""
from __future__ import annotations

import collections
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m4_driver as D4  # noqa: E402
import m2lib as L  # noqa: E402
import paths  # noqa: E402  (results/_lib/paths.py, on sys.path through m2_driver)

X = D4.X
SCEN = paths.REPO / 'sim/scenarios'
WITHIN_M = 1.0
GO_MIN = 5                 # judged: M within-1 m count at least U's + 5 (of 45) to run CT episodes with marked answers


def face_point(sc):
    q = next(iter(yaml.safe_load((SCEN / f'{sc}.yaml').read_text())['queues'].values()))
    return q['slots'][0]['face']


def main(run: Path):
    rows = [json.loads(l) for l in (run / 'calls.jsonl').read_text().splitlines() if l.strip()]
    obs = {(sc, s): (o, X.Geom(o.depth, o.camera)) for sc, s, f, r, c, o in D4.frames()}
    hit = collections.defaultdict(dict)
    dist = collections.defaultdict(list)
    for r in rows:
        o, g = obs[(r['scenario'], r['seed'])]
        cands = D4.candidates(g)
        assert [c['mark'] for c in r['cands']] == [list(c.mark) for c in cands]
        ok = None
        if r['status'] == 'ok':
            s = next((x for x in r['subtasks'] if x['type'] == 'navigate' and (x.get('point') or x.get('mark'))), None)
            if s is not None:
                tgt, _ = L.resolve(g, cands, r['nums'], s.get('mark') if r['arm'] == 'M' else None,
                                   None if s.get('point') is None else tuple(s['point']),
                                   'both' if r['arm'] == 'M' else 'point')
                if tgt is not None:
                    fx, fy = face_point(r['scenario'])
                    d = math.hypot(tgt.world[0] - fx, tgt.world[1] - fy)
                    dist[(r['model'], r['arm'])].append(d)
                    ok = d <= WITHIN_M
        hit[(r['model'], r['arm'])][(r['scenario'], r['seed'], r['trial'])] = bool(ok)
    lines = [f'M4 (marks on sim first frames): target within {WITHIN_M} m of the counter point']
    for model in sorted({m for m, _ in hit}):
        u, mm = hit[(model, 'U')], hit[(model, 'M')]
        keys = sorted(set(u) & set(mm))
        cu, cm = sum(u[k] for k in keys), sum(mm[k] for k in keys)
        gain = sum(1 for k in keys if mm[k] and not u[k]); harm = sum(1 for k in keys if u[k] and not mm[k])
        cells = sorted({k[:2] for k in keys})
        flips = [abs(int(u.get(c + (t1,), 0)) - int(u.get(c + (t2,), 0))) for c in cells
                 for t1, t2 in itertools.combinations((20, 21, 22), 2)]
        du = np.median(dist[(model, 'U')]) if dist[(model, 'U')] else float('nan')
        dm = np.median(dist[(model, 'M')]) if dist[(model, 'M')] else float('nan')
        go = cm - cu >= GO_MIN
        lines.append(f'{model}: U {cu}/{len(keys)}, M {cm}/{len(keys)} (gain {gain}, harm {harm}); median distance '
                     f'U {du:.2f} m, M {dm:.2f} m; U trial-to-trial flips {sum(flips)} of {len(flips)} pairs; '
                     f'JUDGED: {"run CT with marked answers" if go else "no"} (M - U {cm - cu:+d}, needs >= +{GO_MIN})')
    text = '\n'.join(lines)
    print(text)
    (run / 'm4_out.txt').write_text(text + '\n')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
