"""Verifier: layout-clustered bootstrap over all 30 base layouts (FROZEN 9 reporting), descriptive."""
import json, collections
import numpy as np
rows = {}
for s in (0, 1, 2):
    for l in open(f'results/2026-10-17-beyond-vlm/phase2/i06/verify_i06_score/out/arms_seed{s}.jsonl'):
        x = json.loads(l); rows[(s, x['scene_id'])] = x
layouts = sorted({(r['seed'], r['row']) for r in rows.values() if r['kind'] == 'base'})
assert len(layouts) == 30
def ci(vals_by_layout, n=2000):
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        pick = [layouts[i] for i in rng.integers(30, size=30)]
        xs = [v for lay in pick for v in vals_by_layout.get(lay, [])]
        out.append(np.mean(xs) if xs else np.nan)
    return np.nanpercentile(out, [2.5, 97.5])
def drop(r):
    b = rows[(r['seed'], r['base_scene_id'])]
    return 100 * (int(b['eps']['e0']['arm3']['l1']) - int(r['eps']['e0']['arm3']['l1']))
for key in [('B', 1), ('C', 1), ('D', 2), ('E', 2), 'D', 'E', 'B', 'C']:
    by = collections.defaultdict(list)
    for r in rows.values():
        if r['kind'] != 'variant': continue
        if (isinstance(key, tuple) and (r['family'], r['level']) == key) or r['family'] == key:
            by[(r['seed'], r['row'])].append(drop(r))
    lo, hi = ci(by)
    print(key, f'[{lo:.1f}, {hi:.1f}]')
