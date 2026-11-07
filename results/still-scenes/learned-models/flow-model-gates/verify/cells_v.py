"""Independent verifier, C5 gate 2 cells and features (FROZEN.md section 2: Data, Cells, Features).

Writes verify/vcache/<kind>/<n>.npz per scene and verify/vcache/index.json, and verify/cells_v.txt with the checks.
Fresh seeds 100-109 loaded as phase1b/i03/stageA/fitdata.py does (H.guard_scene, then sgi.scene.load_scene); the
68 development scenes only through H.distinct_scenes() and H.scene(first listed run, scene_id)."""
from __future__ import annotations

import collections
import hashlib
import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H  # noqa: E402
from sgi.config import DEFAULT_CONFIG, load_config  # noqa: E402
from sgi.freespace import connected, robot_grid  # noqa: E402
from sgi.gt.derive import prepare  # noqa: E402
from sgi.polygons import point_in_polygon  # noqa: E402
from sgi.scene import load_scene  # noqa: E402

V = Path('results/still-scenes/learned-models/flow-model-gates/verify')
FROZEN = V.parent / 'FROZEN.md'
SHA = '8b7d3ae294ad81a810c70c23706247674bbfed22cc0247445e91f6615426419c'
FRESH = Path('results/still-scenes/perception/fresh_seeds')
CACHE = V / 'vcache'
EPS = 1e-9
assert hashlib.sha256(FROZEN.read_bytes()).hexdigest() == SHA


def my_pip(px, py, poly):
    """Even-odd with the boundary (within EPS) inside, vectorised: my own re-implementation."""
    v = np.asarray([(float(a), float(b)) for a, b in poly], dtype=np.float64)
    prev = np.roll(v, 1, axis=0)
    out = np.zeros(px.size, dtype=bool)
    for lo in range(0, px.size, 4096):
        X = px[lo:lo + 4096, None]
        Y = py[lo:lo + 4096, None]
        x1, y1, x2, y2 = prev[:, 0], prev[:, 1], v[:, 0], v[:, 1]
        ex, ey = x2 - x1, y2 - y1
        L2 = ex * ex + ey * ey
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.clip(((X - x1) * ex + (Y - y1) * ey) / L2, 0.0, 1.0)
            t = np.where(L2 == 0.0, 0.0, t)
            dist = np.hypot(X - (x1 + t * ex), Y - (y1 + t * ey))
            straddle = (y1 > Y) != (y2 > Y)
            xc = x1 + (Y - y1) * (x2 - x1) / (y2 - y1)
            odd = (straddle & (xc > X)).sum(axis=1) % 2 == 1
        out[lo:lo + 4096] = (dist.min(axis=1) <= EPS) | odd
    return out


def features(cx, cy, target, truth, robot, th):
    """f1..f9 of FROZEN.md for cell centres (cx, cy)."""
    ax, ay = float(target.anchor[0]), float(target.anchor[1])
    dx, dy = cx - ax, cy - ay
    f1 = np.hypot(dx, dy)
    front = target.front_yaw
    if front is None and target.kind == 'person':
        front = truth.person(target.person_id).yaw
    if front is None:
        f2 = np.zeros_like(cx)
        f3 = np.zeros_like(cx)
    else:
        ang = np.arctan2(dy, dx) - front
        ang = (ang + np.pi) % (2 * np.pi) - np.pi
        f2, f3 = np.cos(ang), np.sin(ang)
    people = list(truth.people)
    n = cx.size
    if people:
        P = np.array([(p.x, p.y) for p in people])
        Y = np.array([p.yaw for p in people])
        D = np.hypot(cx[:, None] - P[None, :, 0], cy[:, None] - P[None, :, 1])  # (n, np)
        j = D.argmin(axis=1)
        dmin = D[np.arange(n), j]
        f4 = np.minimum(dmin, 5.0)
        ux = (cx - P[j, 0]) / np.where(dmin > 0, dmin, 1.0)
        uy = (cy - P[j, 1]) / np.where(dmin > 0, dmin, 1.0)
        f5 = np.cos(Y[j]) * ux + np.sin(Y[j]) * uy
        f5 = np.where(dmin >= 5.0, 0.0, f5)
        f6 = (D <= 1.5).sum(axis=1).astype(np.float64)
        # f8: people (target person excluded) whose centre is within person_radius_m of segment [c, a]
        keep = [i for i, p in enumerate(people) if not (target.kind == 'person' and p.id == target.person_id)]
        f8 = np.zeros(n)
        for i in keep:
            px, py = P[i]
            sx, sy = ax - cx, ay - cy             # segment from c to a
            L2 = sx * sx + sy * sy
            with np.errstate(divide='ignore', invalid='ignore'):
                t = np.clip(((px - cx) * sx + (py - cy) * sy) / L2, 0.0, 1.0)
            t = np.where(L2 == 0.0, 0.0, t)
            ds = np.hypot(px - (cx + t * sx), py - (cy + t * sy))
            f8 += (ds <= th.person_radius_m)
        da = np.hypot(P[:, 0] - ax, P[:, 1] - ay)
        near = da[da <= 2.0]
        mx = float(near.max()) if near.size else 0.0
    else:
        f4 = np.full(n, 5.0)
        f5 = np.zeros(n)
        f6 = np.zeros(n)
        f8 = np.zeros(n)
        mx = 0.0
    f7 = np.hypot(cx - robot.x, cy - robot.y)
    f9 = f1 - mx
    return np.stack([f1, f2, f3, f4, f5, f6, f7, f8, f9], axis=1)


def process(kind, n_idx, meta, inputs, truth, regions_by_ep, th, held, stats, rng_key):
    robot = inputs.observation.robot
    grid = robot_grid(truth, robot, th)
    conn = connected(grid, (robot.x, robot.y))
    floor05 = grid.free & conn
    f10 = floor05[::2, ::2]
    iy, ix = np.nonzero(f10)
    cx = grid.x0 + (2 * ix + 0.5) * grid.resolution
    cy = grid.y0 + (2 * iy + 0.5) * grid.resolution
    # the 0.10 m grid's origin is the scorer origin minus 0.025 m: the centre of 0.10 m cell I is (x0 - 0.025) + (I + .5) * .1
    assert np.allclose(cx, (grid.x0 - 0.025) + (ix + 0.5) * 0.1) and np.allclose(cy, (grid.y0 - 0.025) + (iy + 0.5) * 0.1)
    N = cx.size
    sets, ep_rows = [], []
    rng = np.random.default_rng(rng_key)
    target_of_set = []
    for eid in sorted(truth.episodes):
        ep = truth.episodes[eid]
        sit = ep.situation
        assert sit not in held, (meta, eid, sit)
        regs = regions_by_ep[eid]
        if len(ep.target_ids) != 1:
            stats['excluded_multi_target'] += 1
            ep_rows.append({'episode': eid, 'situation': sit, 'eligible': False, 'why': 'targets'})
            continue
        if not regs:
            stats['excluded_no_region'] += 1
            ep_rows.append({'episode': eid, 'situation': sit, 'eligible': False, 'why': 'no regions'})
            continue
        mask = np.zeros(N, dtype=bool)
        for r in regs:
            mask |= my_pip(cx, cy, r.polygon)
        # check against the package: every region cell plus 300 random floor cells
        chk = np.concatenate([np.nonzero(mask)[0], rng.choice(N, size=min(300, N), replace=False)])
        for i in chk:
            ref = any(point_in_polygon((float(cx[i]), float(cy[i])), r.polygon) for r in regs)
            stats['pip_tests'] += 1
            if ref != bool(mask[i]):
                stats['pip_disagree'] += 1
        if mask.sum() == 0:
            stats['excluded_no_region_cell'] += 1
            ep_rows.append({'episode': eid, 'situation': sit, 'eligible': False, 'why': 'no region cell'})
            continue
        tid = ep.target_ids[0]
        found = None
        for k, (m, t) in enumerate(zip(sets, target_of_set)):
            if t == tid and np.array_equal(m, mask):
                found = k
        if found is None:
            sets.append(mask)
            target_of_set.append(tid)
            found = len(sets) - 1
        ep_rows.append({'episode': eid, 'situation': sit, 'eligible': True, 'set': found, 'target': tid,
                        'n_region': int(mask.sum())})
    tids = sorted(set(target_of_set))
    feats = {}
    for tid in tids:
        feats[tid] = features(cx, cy, truth.targets[tid], truth, robot, th)
    assert len(tids) <= 1, (meta, tids)        # fresh/dev: one target per scene's eligible episodes
    out = CACHE / kind / f'{n_idx:03d}.npz'
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, xy=np.stack([cx, cy], 1), F=feats[tids[0]] if tids else np.zeros((N, 9)),
                        masks=np.stack(sets) if sets else np.zeros((0, N), bool))
    meta = dict(meta, N_floor=int(N), n_sets=len(sets), episodes=ep_rows, file=str(out.relative_to(V)))
    return meta


def main():
    t0 = time.time()
    cfg = load_config(DEFAULT_CONFIG)
    th = cfg.thresholds
    held = set(cfg.held_out_situations)
    index = {'fresh': [], 'dev': []}
    stats = {k: collections.Counter() for k in ('fresh', 'dev')}
    man = json.loads((FRESH / 'manifest.json').read_text())
    n = 0
    for sd in man['seeds']:
        k = sd['seed']
        assert 100 <= k <= 109
        for s in sd['scenes']:
            folder = FRESH / f'seed{k}' / s['split'] / s['scene_id']
            H.guard_scene(folder)
            inputs, truth = load_scene(folder)
            assert inputs.scene_id == s['scene_id']
            ets = prepare(inputs, truth, cfg)
            regs = {eid: list(et.gt.acceptable_regions or []) for eid, et in ets.items()}
            key = f"{k}/{s['split']}/{s['scene_id']}"
            meta = {'seed': k, 'split': s['split'], 'scene_id': s['scene_id'], 'domain': 'fresh', 'row': s['row']}
            index['fresh'].append(process('fresh', n, meta, inputs, truth, regs, th, held, stats['fresh'],
                                          [20261018, zlib.crc32(key.encode()), 7]))
            n += 1
        print(f'seed {k} done {time.time() - t0:.0f} s', flush=True)
    refs = H.distinct_scenes()
    dev_cfg_diff = 0
    dev_reg_same = collections.Counter()
    skipped = 0
    for n, ref in enumerate(refs):
        if not ref.runs:
            skipped += 1
            continue
        run = ref.runs[0]
        sc = H.scene(run, ref.scene_id)
        rth = H.recorded_config(run)[1].thresholds
        if (rth.grid_resolution_m, rth.robot_radius_m, rth.person_radius_m) != (th.grid_resolution_m, th.robot_radius_m, th.person_radius_m):
            dev_cfg_diff += 1
        regs = {eid: list(et.gt.acceptable_regions or []) for eid, et in sc.truths.items()}
        dflt = prepare(sc.inputs, sc.truth, cfg)
        for eid in regs:
            a = [r.model_dump() for r in regs[eid]]
            b = [r.model_dump() for r in (dflt[eid].gt.acceptable_regions or [])]
            dev_reg_same['same' if a == b else 'differ'] += 1
        dom = 'realistic' if 'realistic_final' in str(ref.folder) else 'placeholder'
        dom2 = 'realistic' if 'realistic_final' in str(ref.scenes_dir) else 'placeholder'
        meta = {'seed': None, 'split': None, 'scene_id': ref.scene_id, 'domain': dom, 'domain_by_scenes_dir': dom2,
                'folder': str(ref.folder), 'run': str(run.folder)}
        index['dev'].append(process('dev', n, meta, sc.inputs, sc.truth, regs, th, held, stats['dev'],
                                    [20261018, zlib.crc32(f'dev/{ref.scene_id}/{ref.folder}'.encode()), 7]))
    lines = ['C5 gate 2 cells, independent verifier']
    for kind in ('fresh', 'dev'):
        rows = index[kind]
        eps = [e for r in rows for e in r['episodes']]
        el = [e for e in eps if e['eligible']]
        nf = [r['N_floor'] for r in rows]
        nr = [e['n_region'] for e in el]
        lines.append(f"{kind}: scenes {len(rows)}, episodes {len(eps)}, eligible {len(el)}; floor cells min {min(nf)} "
                     f"median {int(np.median(nf))} max {max(nf)}; region cells per eligible episode min {min(nr)} "
                     f"median {int(np.median(nr))} max {max(nr)}; stats {dict(stats[kind])}")
        c = collections.Counter((r['domain'], e['situation'], e['eligible']) for r in rows for e in r['episodes'])
        for key in sorted(c):
            lines.append(f'   {key}: {c[key]}')
        lines.append(f"   scenes with more than one region set: {sum(1 for r in rows if r['n_sets'] > 1)}")
    lines.append(f'dev scenes skipped for no listed run: {skipped}; dev runs with different grid/robot/person radius: '
                 f'{dev_cfg_diff}; dev episodes regions recorded-config vs default: {dict(dev_reg_same)}')
    lines.append('dev domain by folder: ' + str(collections.Counter(r['domain'] for r in index['dev'])) +
                 '; by scenes_dir: ' + str(collections.Counter(r['domain_by_scenes_dir'] for r in index['dev'])))
    lines.append(f'time {time.time() - t0:.0f} s')
    (CACHE / 'index.json').write_text(json.dumps(index, indent=0, default=str))
    (V / 'cells_v.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
