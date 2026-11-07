"""Independent verifier, C5 gate 1 (FROZEN.md section 1). Counts read from the generator's row tables; the
empirical spread recomputed from the fresh seeds 100-109 (loaded as phase1b/i03/stageA/fitdata.py does).

No scene is generated. Held-out rows (row[0] not in cfg.dev_situations) are counted only as tuples of the code's
constants for an alternative reading; nothing of theirs is instantiated or loaded."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H  # noqa: E402
from sgi.config import DEFAULT_CONFIG, load_config  # noqa: E402
from sgi.scene import load_scene  # noqa: E402
from sgi.synth import generate as G  # noqa: E402

OUT = Path('results/still-scenes/learned-models/flow-model-gates/verify')
FROZEN = OUT.parent / 'FROZEN.md'
FRESH = Path('results/still-scenes/perception/fresh_seeds')
SHA = '8b7d3ae294ad81a810c70c23706247674bbfed22cc0247445e91f6615426419c'
assert hashlib.sha256(FROZEN.read_bytes()).hexdigest() == SHA


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def main():
    cfg = load_config(DEFAULT_CONFIG)
    dev = set(cfg.dev_situations)
    tables = [('eval', 'first', G._EVAL), ('examples', 'first', G._EXAMPLES),
              ('eval', 'more', G._EVAL_MORE), ('examples', 'more', G._EXAMPLES_MORE)]
    rows, held_rows = [], []
    for split, pas, tab in tables:
        start = 0 if pas == 'first' else len(dict(G._SPLITS)[split])
        for i, row in enumerate(tab, start=start):
            (rows if row[0] in dev else held_rows).append((split, i, tuple(row)))
    out = {'n_dev_rows': len(rows), 'n_heldout_rows_named': len(held_rows)}
    prim = {r for _, _, r in rows}
    out['PRIMARY'] = len(prim)

    def lower_key(r):
        rec, amt, kind, by, sg = r
        return (rec, 0 if rec == 'no_one' else amt, by, sg)
    out['LOWER'] = len({lower_key(r) for _, _, r in rows})
    # alternative readings (all descriptive)
    alt = {}
    alt['rows_total'] = len(rows)
    alt['primary_eval_split_only'] = len({r for s, _, r in rows if s == 'eval'})
    alt['primary_examples_split_only'] = len({r for s, _, r in rows if s == 'examples'})
    alt['primary_ignoring_kind'] = len({(r[0], r[1], r[3], r[4]) for _, _, r in rows})
    alt['primary_ignoring_view_angle_for_no_one'] = len({(r[0], 0 if r[0] == 'no_one' else r[1], r[2], r[3], r[4])
                                                         for _, _, r in rows})
    alt['lower_with_kind_for_no_one'] = len({(r[0], 0 if r[0] == 'no_one' else r[1],
                                              r[2] if r[0] == 'no_one' else None, r[3], r[4]) for _, _, r in rows})
    alt['situation_and_size_only'] = len({(r[0], 0 if r[0] == 'no_one' else r[1]) for _, _, r in rows})
    alt['primary_if_heldout_tuples_were_added (upper bound, code constants only)'] = len(prim | {r for _, _, r in held_rows})
    out['alternative_readings'] = alt
    out['by_situation_primary'] = {s: len({r for _, _, r in rows if r[0] == s}) for s in sorted(dev)}
    out['by_situation_lower'] = {s: len({lower_key(r) for _, _, r in rows if r[0] == s}) for s in sorted(dev)}
    out['gate1'] = 'met' if out['PRIMARY'] >= 30 else ('borderline, not met' if out['PRIMARY'] >= 25 else 'not met')
    out['c2_source_a_kill'] = 'cleared' if out['PRIMARY'] >= 30 else ('borderline' if out['PRIMARY'] >= 25 else 'fires')

    # ---------------- EMPIRICAL (descriptive)
    man = json.loads((FRESH / 'manifest.json').read_text())
    by_row = {}
    for sd in man['seeds']:
        k = sd['seed']
        assert 100 <= k <= 109
        for s in sd['scenes']:
            by_row.setdefault((s['split'], s['index']), []).append((k, s['scene_id']))
    assert len(by_row) == 26 and all(len(v) == 10 for v in by_row.values())
    emp = {}
    front_checks = [0, 0]
    for (split, index), lst in sorted(by_row.items()):
        frames = []
        for k, sid in sorted(lst):
            folder = FRESH / f'seed{k}' / split / sid
            H.guard_scene(folder)
            inputs, truth = load_scene(folder)
            b = truth.bounds
            psi = math.atan2(b[1][1] - b[0][1], b[1][0] - b[0][0])
            (t,) = truth.targets.values()
            for tt in truth.targets.values():
                if tt.kind == 'place':
                    front_checks[1] += 1
                    if abs(wrap(psi - tt.front_yaw)) < 1e-6:
                        front_checks[0] += 1
            a = np.array(t.anchor[:2])
            c, s_ = math.cos(-psi), math.sin(-psi)
            R = np.array([[c, -s_], [s_, c]])
            eps = list(truth.episodes.values())
            gat = eps[0].group_at_target
            ppl = []
            for p in truth.people:
                q = R @ (np.array([p.x, p.y]) - a)
                member = gat is not None and isinstance(gat, str) and p.group == gat
                ppl.append((p.id, q, member))
            rob = R @ (np.array([inputs.observation.robot.x, inputs.observation.robot.y]) - a)
            frames.append({'people': ppl, 'robot': rob, 'situation': eps[0].situation})
        rec = {'situation': frames[0]['situation'], 'n_people': [len(f['people']) for f in frames]}

        def disp(sel, how):
            ds = []
            for f1, f2 in itertools.combinations(frames, 2):
                A = [(pid, q) for pid, q, m in f1['people'] if sel(m)]
                B = [(pid, q) for pid, q, m in f2['people'] if sel(m)]
                if not A or len(A) != len(B):
                    continue
                if how == 'id':
                    db = dict(B)
                    ds.extend(float(np.linalg.norm(q - db[pid])) for pid, q in A if pid in db)
                else:
                    best = None
                    for mir in ((False, True) if how == 'match_mirror' else (False,)):
                        QB = [np.array([q[0], -q[1]]) if mir else q for _, q in B]
                        for perm in itertools.permutations(range(len(B))):
                            d = [float(np.linalg.norm(A[i][1] - QB[j])) for i, j in enumerate(perm)]
                            if best is None or sum(d) < sum(best):
                                best = d
                    ds.extend(best)
            return ds
        for label, sel in (('members', lambda m: m), ('everyone', lambda m: True)):
            for how in ('id', 'match', 'match_mirror'):
                ds = disp(sel, how)
                rec[f'{label}_{how}'] = None if not ds else (round(float(np.median(ds)), 3), round(float(np.max(ds)), 3), len(ds))
        rd = [float(np.linalg.norm(f1['robot'] - f2['robot'])) for f1, f2 in itertools.combinations(frames, 2)]
        rec['robot'] = (round(float(np.median(rd)), 3), round(float(np.max(rd)), 3))
        emp[f'{split}_{index:02d}'] = rec
    out['empirical'] = emp
    out['front_yaw_checks_passed'] = front_checks
    mm = [v['members_match_mirror'] for v in emp.values() if v['members_match_mirror'] is not None]
    out['members_match_mirror_summary'] = {'rows': len(mm), 'median_range': (min(x[0] for x in mm), max(x[0] for x in mm)),
                                           'max_range': (min(x[1] for x in mm), max(x[1] for x in mm))}
    ev = [v['everyone_match_mirror'] for v in emp.values() if v['everyone_match_mirror'] is not None]
    out['everyone_match_mirror_max'] = max(x[1] for x in ev) if ev else None
    (OUT / 'gate1_v.json').write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != 'empirical'}, indent=1, default=str))
    for k, v in emp.items():
        print(k, v)


if __name__ == '__main__':
    main()
