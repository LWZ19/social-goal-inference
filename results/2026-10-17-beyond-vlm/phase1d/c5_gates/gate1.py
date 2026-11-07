"""C5 gate 1 (C2 Step 0 and its Source A kill), as FROZEN.md section 1 fixes it.

Counts the generator's content rows of the four development situations (`_EVAL`, `_EXAMPLES`, `_EVAL_MORE`,
`_EXAMPLES_MORE` of src/sgi/synth/generate.py, kept when row[0] is in cfg.dev_situations) under three readings:
PRIMARY (decides), LOWER and EMPIRICAL (descriptive; seeds 100-109 only, loaded as phase1b/i03/stageA/fitdata.py
does: H.guard_scene, then sgi.scene.load_scene). No scene is generated; no held-out row is instantiated. No model,
no network. Writes gate1.txt and gate1.json beside this file.

    cd WT && PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 .venv/bin/python <this>
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H  # noqa: E402
from sgi.config import DEFAULT_CONFIG, load_config  # noqa: E402
from sgi.scene import load_scene  # noqa: E402
from sgi.synth import generate as G  # noqa: E402

HERE = Path('results/2026-10-17-beyond-vlm/phase1d/c5_gates')
FRESH = Path('results/2026-10-17-beyond-vlm/phase1b/fresh_seeds')
SRC = Path('src/sgi/synth')
SEEDS = tuple(range(100, 110))
FROZEN_SHA = '8b7d3ae294ad81a810c70c23706247674bbfed22cc0247445e91f6615426419c'


def wrap(a):
    return math.pi - (math.pi - a) % (2 * math.pi)


def rows(cfg):
    """(pass, split, index, row) of every development row, in the generator's order; held-out rows are only named."""
    out, skipped = [], []
    for which, splits, start in (('first', G._SPLITS, None), ('more', G._MORE_SPLITS, dict(G._SPLITS))):
        for split, rws in splits:
            first = 0 if start is None else len(start[split])
            for index, row in enumerate(rws, start=first):
                if row[0] in cfg.dev_situations:
                    out.append((which, split, index, tuple(row)))
                else:
                    skipped.append(f'{split}_{index:02d} ({row[0]})')
    return out, skipped


def lower_key(row):
    recipe, amount, kind, bystanders, second = row
    size = 0 if recipe == 'no_one' else int(amount)
    return (recipe, size, int(bystanders), bool(second))


# The parameters drawn at random within a row, from the code (identifier, file, what, range). The line numbers are
# looked up below so that a reader can check each one.
RANDOM = [
    ('transform_layout(layout, rng.uniform(-math.pi, math.pi)', 'generate.py', 'rigid yaw of the whole layout', 'U(-pi, pi)'),
    ('rng.uniform(-_SHIFT_M, _SHIFT_M)', 'generate.py', 'rigid shift along x and along y', 'U(-5, 5) m each'),
    ('inside = rng.uniform(0.0, _VIEW_ANGLE_SPREAD_DEG)', 'generate.py', 'no_one: view angle inside the row\'s angle', 'U(0, 3) deg'),
    ('a = _sign(rng) * math.radians(view_angle_deg)', 'recipes.py', 'no_one: side of the front the robot looks from', '+-1'),
    ('d = rng.uniform(4.5, 5.5)', 'recipes.py', 'no_one: robot distance from the target', 'U(4.5, 5.5) m'),
    ('rng.uniform(-1.0, 1.0) * _VIEW_YAW_JITTER_RAD', 'recipes.py', 'no_one: robot yaw off the target', 'U(-5, 5) deg'),
    ('_PLACES[int(rng.integers(len(_PLACES)))]', 'recipes.py', 'place/deep_place/waiting_line: which of 4 places (kiosk, desk, ticket counter, stall)', '1 of 4'),
    ('side = _sign(rng)', 'recipes.py', 'waiting_line/audience/lift: side the robot stands on', '+-1'),
    ('b = _sign(rng) * math.radians(_GROUP_BEARING_DEG[n_people])', 'recipes.py', 'conversation_group: side of the robot bearing', '+-1'),
    ('d = rng.uniform(4.2, 4.8)', 'recipes.py', 'conversation_group: robot distance from the circle', 'U(4.2, 4.8) m'),
    ('first = rng.uniform(_AUDIENCE_ROWS_M[0]', 'recipes.py', 'audience: first row distance from the screen', 'U(2.05, 3.15 - 0.9*(rows-1)) m'),
    ('bearing, distance = side * math.radians(bearing_deg), rng.uniform(*_ATTEND_VIEW_M)', 'recipes.py', 'audience: robot distance from the people\'s middle', 'U(4.0, 4.4) m'),
    ('x=x + rng.uniform(-_JITTER_M, _JITTER_M)', 'recipes.py', 'every person: position jitter in x and y', 'U(-0.05, 0.05) m each'),
    ('yaw=wrap_angle(yaw + rng.uniform(-_JITTER_RAD, _JITTER_RAD))', 'recipes.py', 'every person: yaw jitter', 'U(-0.15, 0.15) rad'),
    ('order = [int(k) for k in rng.permutation(len(free))]', 'recipes.py', 'shirt colours of non-target people', 'permutation of 6'),
    ('rho = rng.uniform(*_SPOT_RANGE_M)', 'recipes.py', 'bystander / second group: spot distance from the robot', 'U(3, 12) m, rejection-sampled'),
    ('direction = robot.yaw + rng.uniform(-1.0, 1.0) * _SPOT_HALF_ANGLE_RAD', 'recipes.py', 'bystander / second group: spot bearing', 'U(-22, 22) deg off the robot heading'),
    ('return [_person(rng, index, x, y, rng.uniform(-math.pi, math.pi), shirts[0])]', 'recipes.py', 'bystander: facing', 'U(-pi, pi)'),
    ('along = rng.uniform(-math.pi, math.pi)', 'recipes.py', 'second group (pair): orientation of the pair', 'U(-pi, pi)'),
    ('room = Room(polygon=tuple(floor), height=_ROOM_HEIGHT_M, **_ROOMS[int(rng.integers(len(_ROOMS)))])', 'recipes.py', 'room colours', '1 of 3'),
    ('shop = dict(zip(("text", "color", "goods"), _SHOPS[int(rng.integers(len(_SHOPS)))]))', 'recipes.py', 'furnishing: the second stall\'s goods', '1 of 3'),
    ('texts = [int(k) for k in rng.permutation(len(_POSTERS))]', 'recipes.py', 'furnishing: poster order', 'permutation of 4'),
    ('s, end = _WALL_ENDS_M + rng.uniform(0.0, 1.0)', 'recipes.py', 'furnishing: start along each wall', 'U(0, 1) m'),
    ('order = rng.choice(len(_FURNISHINGS), size=len(_FURNISHINGS), replace=False, p=_FURNISHING_ODDS)', 'recipes.py', 'furnishing: kind order by odds', '11 kinds'),
    ('s += width + rng.uniform(*_FURNISH_SPACING_M)', 'recipes.py', 'furnishing: spacing along a wall', 'U(0.4, 1.2) m'),
    ('options = {"scale": rng.uniform(0.95, 1.2)}', 'recipes.py', 'furnishing: plant scale', 'U(0.95, 1.2)'),
    ('options = {"width": rng.uniform(1.6, 2.2)}', 'recipes.py', 'furnishing: window width', 'U(1.6, 2.2) m'),
    ('_, moved, footprint, frame = found[int(rng.integers(len(found)))]', 'recipes.py', 'furnishing: spot of the second stall and the first poster', 'uniform over allowed spots'),
    ('for _ in range(_MAX_ATTEMPTS)', 'generate.py', 'whole layout redrawn when the scene check rejects it', 'up to 20 attempts'),
]


def line_of(text, fname):
    for i, line in enumerate((SRC / fname).read_text().splitlines(), start=1):
        if text in line:
            return i
    return None


def frame_of(truth):
    """psi (yaw of the bounds polygon's first edge) and a (target anchor ground point); asserts psi = front_yaw of
    every place target."""
    b = truth.bounds
    psi = math.atan2(b[1][1] - b[0][1], b[1][0] - b[0][0])
    checked = 0
    for t in truth.targets.values():
        if t.kind == 'place':
            assert t.front_yaw is not None and abs(wrap(psi - t.front_yaw)) < 1e-9, (truth.scene_id, psi, t.front_yaw)
            checked += 1
    assert len(truth.targets) == 1
    t = next(iter(truth.targets.values()))
    return psi, np.array(t.anchor[:2], dtype=np.float64), checked


def local(points, psi, a):
    c, s = math.cos(-psi), math.sin(-psi)
    d = np.asarray(points, dtype=np.float64) - a
    return np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], axis=1)


def best_matching(P, Q):
    """Distances of the one-to-one matching of P to Q with the smallest sum (brute force; n <= 8)."""
    n = len(P)
    if n == 0:
        return np.zeros(0)
    C = np.hypot(P[:, None, 0] - Q[None, :, 0], P[:, None, 1] - Q[None, :, 1])
    perms = np.array(list(itertools.permutations(range(n))), dtype=np.int64)
    sums = C[np.arange(n)[None, :], perms].sum(axis=1)
    k = int(np.argmin(sums))
    return C[np.arange(n), perms[k]]


def mirror(P):
    return P * np.array([1.0, -1.0])


def summarise(d):
    d = np.asarray(d, dtype=np.float64)
    if d.size == 0:
        return None
    return {'median_m': round(float(np.median(d)), 4), 'max_m': round(float(d.max()), 4), 'n': int(d.size)}


def main():
    assert hashlib.sha256((HERE / 'FROZEN.md').read_bytes()).hexdigest() == FROZEN_SHA, 'FROZEN.md hash mismatch'
    cfg = load_config(DEFAULT_CONFIG)
    assert set(cfg.dev_situations) == {'no_one', 'waiting_line', 'conversation_group', 'audience'}
    held = set(cfg.held_out_situations)
    assert held == {'group_photo', 'lift_wait'}
    rws, skipped = rows(cfg)
    assert all(r[3][0] not in held for r in rws)

    primary = sorted({r[3] for r in rws}, key=repr)
    lower = sorted({lower_key(r[3]) for r in rws}, key=repr)
    n_primary, n_lower = len(primary), len(lower)
    verdict = 'met' if n_primary >= 30 else ('borderline (not met)' if n_primary >= 25 else 'not met')
    kill = 'cleared' if n_primary >= 30 else ('borderline' if n_primary >= 25 else 'fires')

    # ------------------------------------------------------------------------------------------- EMPIRICAL
    done = {k: json.loads((FRESH / f'seed{k}' / 'DONE.json').read_text()) for k in SEEDS}
    by_key = {k: {(s['split'], s['index']): s for s in done[k]['scenes']} for k in SEEDS}
    emp = []
    n_place_checks = 0
    for which, split, index, row in rws:
        locs, mems, robs, ids0, sid0 = [], [], [], None, None
        for k in SEEDS:
            s = by_key[k][(split, index)]
            assert tuple(s['row']) == row, (k, split, index, s['row'], row)
            folder = FRESH / f'seed{k}' / split / s['scene_id']
            H.guard_scene(folder)
            inputs, truth = load_scene(folder)
            assert inputs.scene_id == s['scene_id']
            sits = {ep.situation for ep in truth.episodes.values()}
            assert sits == {row[0]} and not (sits & held), (folder, sits)
            psi, a, chk = frame_of(truth)
            n_place_checks += chk
            ids = [p.id for p in truth.people]
            if ids0 is None:
                ids0, sid0 = ids, s['scene_id']
            assert ids == ids0, (folder, ids, ids0)
            groups = {ep.group_at_target if not isinstance(ep.group_at_target, dict) else ep.group_at_target.get('t0')
                      for ep in truth.episodes.values()}
            assert len(groups) == 1
            g = next(iter(groups))
            P = local([(p.x, p.y) for p in truth.people], psi, a) if truth.people else np.zeros((0, 2))
            member = np.array([g is not None and p.group == g for p in truth.people], dtype=bool)
            r = inputs.observation.robot
            locs.append(P)
            mems.append(member)
            robs.append(local([(r.x, r.y)], psi, a)[0])
        assert all(np.array_equal(m, mems[0]) for m in mems)
        member = mems[0]
        res = {'pass': which, 'split': split, 'index': index, 'scene_id': sid0, 'row': list(row),
               'n_people': len(ids0), 'n_members': int(member.sum())}
        for who, sel in (('members', member), ('everyone', np.ones(len(ids0), dtype=bool))):
            da, db, dm = [], [], []
            for i, j in itertools.combinations(range(len(SEEDS)), 2):
                Pi, Pj = locs[i][sel], locs[j][sel]
                if len(Pi) == 0:
                    continue
                da.extend(np.hypot(*(Pi - Pj).T).tolist())
                db.extend(best_matching(Pi, Pj).tolist())
                m0, m1 = best_matching(Pi, Pj), best_matching(Pi, mirror(Pj))
                dm.extend((m0 if m0.sum() <= m1.sum() else m1).tolist())
            res[who] = {'a_by_id': summarise(da), 'b_matching': summarise(db), 'b_matching_or_mirror': summarise(dm)}
        ra, rm = [], []
        for i, j in itertools.combinations(range(len(SEEDS)), 2):
            ra.append(float(np.hypot(*(robs[i] - robs[j]))))
            rm.append(min(ra[-1], float(np.hypot(*(robs[i] - mirror(robs[j][None])[0])))))
        res['robot'] = {'a': summarise(ra), 'mirror': summarise(rm)}
        emp.append(res)

    rand = []
    for text, fname, what, rng_ in RANDOM:
        ln = line_of(text, fname)
        rand.append({'code': text, 'where': f'src/sgi/synth/{fname}:{ln}' if ln else f'{fname}: NOT FOUND',
                     'what': what, 'range': rng_})

    out = {'frozen_sha256': FROZEN_SHA, 'rows_kept': [list(r[3]) for r in rws],
           'rows_kept_keys': [f'{r[1]}_{r[2]:02d}' for r in rws], 'rows_skipped_held_out': skipped,
           'PRIMARY': {'count': n_primary, 'tuples': [list(t) for t in primary],
                       'definition': 'distinct (recipe, amount, kind, bystanders, second group), exact equality, both splits, both passes'},
           'LOWER': {'count': n_lower, 'tuples': [list(t) for t in lower],
                     'definition': 'distinct (situation, people in the structure at the target, bystanders, second group); no_one structure 0'},
           'gate1_verdict': verdict, 'c2_source_a_kill': kill,
           'EMPIRICAL': emp, 'place_front_checks': n_place_checks, 'random_parameters': rand}
    (HERE / 'gate1.json').write_text(json.dumps(out, indent=1))

    L = []
    L.append('C5 gate 1: configuration count (FROZEN.md section 1). Generator rows from src/sgi/synth/generate.py.')
    L.append(f'Rows kept (dev situations): {len(rws)}; held-out rows named, never instantiated: {", ".join(skipped)}')
    L.append('')
    L.append(f'PRIMARY (decides): {n_primary} distinct (recipe, amount, kind, bystanders, second group)')
    for t in primary:
        L.append(f'  {t}')
    L.append(f'LOWER: {n_lower} distinct (situation, structure size, bystanders, second group)')
    for t in lower:
        L.append(f'  {t}')
    L.append('')
    L.append(f'Gate 1 (PRIMARY >= 30 met; 25-29 borderline, not met): {verdict}')
    L.append(f'C2 Source A kill (PRIMARY < 25 fires; 25-29 borderline; >= 30 cleared): {kill}')
    L.append('')
    L.append('EMPIRICAL (descriptive): per row, displacement of each person between two seeds over the 45 seed pairs '
             '(seeds 100-109), in the frame p\' = R(-psi)(p - a); psi = yaw of the bounds polygon\'s first edge '
             f'(asserted equal to front_yaw of every place target: {n_place_checks} checks), a = target anchor ground point.')
    L.append('  (a) by id; (b) best one-to-one matching (smallest sum); (b+m) best of matching and matching after the '
             'mirror y\' -> -y\'. median / max in metres; n = person-pairs.')
    hdr = (f'{"row":10s} {"recipe,amount,kind,byst,2nd":38s} {"n":>2s} {"mem":>3s} | {"members a":>13s} {"b":>13s} '
           f'{"b+m":>13s} | {"everyone a":>13s} {"b":>13s} {"b+m":>13s} | {"robot a":>13s} {"mirror":>13s}')
    L.append(hdr)

    def f(s):
        return '-' if s is None else f'{s["median_m"]:.2f}/{s["max_m"]:.2f}'
    for e in emp:
        L.append(f'{e["split"][:4]}_{e["index"]:02d}    {str(tuple(e["row"])):38s} {e["n_people"]:2d} {e["n_members"]:3d} | '
                 f'{f(e["members"]["a_by_id"]):>13s} {f(e["members"]["b_matching"]):>13s} {f(e["members"]["b_matching_or_mirror"]):>13s} | '
                 f'{f(e["everyone"]["a_by_id"]):>13s} {f(e["everyone"]["b_matching"]):>13s} {f(e["everyone"]["b_matching_or_mirror"]):>13s} | '
                 f'{f(e["robot"]["a"]):>13s} {f(e["robot"]["mirror"]):>13s}')
    L.append('')
    L.append('Random parameters within a row (from the code):')
    for r in rand:
        L.append(f'  {r["where"]:34s} {r["what"]}: {r["range"]}')
    (HERE / 'gate1.txt').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
