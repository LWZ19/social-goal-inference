"""C5 gate 2, step 3 of FROZEN.md: cache the 0.10 m floor cells, f1-f9 and the region masks per scene and episode,
and run the cell checks. Fresh seeds 100-109 (H.guard_scene, then sgi.scene.load_scene; regions from
sgi.gt.derive.prepare under configs/default.yaml) and the 68 development scenes of H.distinct_scenes() (each with
H.scene(first listed run, scene_id)). Floor: default thresholds for every scene. No model, no network, nothing
fitted. Writes cache/*.npz, cache/index.json and cells_check.txt.

    cd WT && PYTHONPATH= OMP_NUM_THREADS=2 ... nice -n 10 .venv/bin/python <this>
"""
from __future__ import annotations

import collections
import hashlib
import json
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
sys.path.insert(0, 'results/still-scenes/learned-models/flow-model-gates')
import research_harness as H  # noqa: E402
from common import (CACHE, EPS, FRESH, FROZEN_SHA, HERE, SEEDS, SUB_SEED, coarse_floor, features,  # noqa: E402
                    in_union)
from sgi.config import DEFAULT_CONFIG, load_config  # noqa: E402
from sgi.gt.derive import prepare  # noqa: E402
from sgi.polygons import point_in_polygon  # noqa: E402
from sgi.scene import load_scene  # noqa: E402

TR_REG, TR_NON, VA_REG, VA_FLOOR, CHECK_FLOOR = 100, 200, 200, 600, 300


def episodes_of(truths, xy, held, counts, chk, scene_key, rng_check):
    """Per episode: eligibility, region polygons and region mask; runs the package check."""
    out = []
    check_idx = rng_check.choice(len(xy), size=min(CHECK_FLOOR, len(xy)), replace=False) if len(xy) else np.zeros(0, int)
    for eid in sorted(truths):
        gt = truths[eid].gt
        assert gt.situation not in held, (scene_key, eid, gt.situation)
        e = {'eid': eid, 'situation': gt.situation, 'targets': list(gt.target_ids)}
        if len(gt.target_ids) != 1:
            e.update(eligible=False, reason='several_targets')
            counts['excluded_several_targets'] += 1
            out.append((e, None))
            continue
        tid = gt.target_ids[0]
        polys = [r.polygon for r in gt.acceptable_regions or [] if r.target_id in (None, tid)]
        if not polys:
            e.update(eligible=False, reason='no_acceptable_region', target=tid)
            counts['excluded_no_acceptable_region'] += 1
            out.append((e, None))
            continue
        mask = in_union(xy, polys)
        # package check: every region cell plus the scene's 300 random floor cells
        idx = np.unique(np.concatenate([np.flatnonzero(mask), check_idx]))
        for i in idx:
            pk = any(point_in_polygon((float(xy[i, 0]), float(xy[i, 1])), poly) for poly in polys)
            chk['tested'] += 1
            if pk != bool(mask[i]):
                chk['disagree'] += 1
                chk['where'].append(f'{scene_key} {eid} cell {int(i)} replica {bool(mask[i])} package {pk}')
        e.update(target=tid, n_region=int(mask.sum()), n_polygons=len(polys))
        if mask.sum() == 0:
            e.update(eligible=False, reason='no_region_cell')
            counts['excluded_no_region_cell'] += 1
            out.append((e, None))
            continue
        e.update(eligible=True)
        counts['eligible'] += 1
        out.append((e, mask))
    return out


def save_scene(path, xy, feats, eps, subs=None):
    arrays = {'xy': xy}
    for tid, F in feats.items():
        arrays[f'F__{tid}'] = F
    for e, mask in eps:
        if mask is not None:
            arrays[f'reg__{e["eid"]}'] = np.flatnonzero(mask).astype(np.int32)
    if subs:
        for k, v in subs.items():
            arrays[k] = np.asarray(v, dtype=np.int32)
    np.savez(path, **arrays)


def main():
    assert hashlib.sha256((HERE / 'FROZEN.md').read_bytes()).hexdigest() == FROZEN_SHA
    cfg = load_config(DEFAULT_CONFIG)
    th = cfg.thresholds
    held = set(cfg.held_out_situations)
    (CACHE / 'fresh').mkdir(parents=True, exist_ok=True)
    (CACHE / 'dev').mkdir(parents=True, exist_ok=True)
    index = []
    counts = {'fresh': collections.Counter(), 'dev': collections.Counter()}
    chk = {'fresh': {'tested': 0, 'disagree': 0, 'where': []}, 'dev': {'tested': 0, 'disagree': 0, 'where': []}}
    sets_differ = []
    t0 = time.monotonic()

    # ------------------------------------------------------------------------------------------- fresh seeds
    for k in SEEDS:
        info = json.loads((FRESH / f'seed{k}' / 'DONE.json').read_text())
        for s in info['scenes']:
            key = f'seed{k}/{s["split"]}/{s["scene_id"]}'
            folder = FRESH / f'seed{k}' / s['split'] / s['scene_id']
            H.guard_scene(folder)
            inputs, truth = load_scene(folder)
            assert inputs.scene_id == s['scene_id']
            ets = prepare(inputs, truth, cfg)
            robot = inputs.observation.robot
            xy = coarse_floor(truth, robot, th)
            counts['fresh']['scenes'] += 1
            counts['fresh']['episodes'] += len(ets)
            rng_check = np.random.default_rng([SUB_SEED, zlib.crc32((key + '/check').encode())])
            eps = episodes_of(ets, xy, held, counts['fresh'], chk['fresh'], key, rng_check)
            tids = sorted({e['target'] for e, m in eps if m is not None})
            feats = {tid: features(xy, truth, truth.targets[tid], robot, th) for tid in tids}
            # region sets (identical sets in one scene pooled), in episode order
            sets, set_of = [], {}
            for e, m in eps:
                if m is None:
                    continue
                fs = np.flatnonzero(m).tobytes()
                if fs not in set_of:
                    set_of[fs] = len(sets)
                    sets.append({'mask': m, 'eids': [], 'target': e['target']})
                sets[set_of[fs]]['eids'].append(e['eid'])
                e['set'] = set_of[fs]
            if len(sets) > 1:
                sets_differ.append(key)
            # subsamples: one generator per scene, drawn once in this fixed order
            rng = np.random.default_rng([SUB_SEED, zlib.crc32(key.encode())])
            subs = {}
            for j, st in enumerate(sets):
                reg = np.flatnonzero(st['mask'])
                subs[f'trreg__{j}'] = np.sort(reg[rng.choice(len(reg), size=min(TR_REG, len(reg)), replace=False)])
            for j, st in enumerate(sets):
                non = np.flatnonzero(~st['mask'])
                subs[f'trnon__{j}'] = np.sort(non[rng.choice(len(non), size=min(TR_NON, len(non)), replace=False)])
            for j, st in enumerate(sets):
                reg = np.flatnonzero(st['mask'])
                subs[f'vareg__{j}'] = np.sort(reg[rng.choice(len(reg), size=min(VA_REG, len(reg)), replace=False)])
            subs['vafloor'] = np.sort(rng.choice(len(xy), size=min(VA_FLOOR, len(xy)), replace=False))
            fname = f'fresh/seed{k}__{s["split"]}__{s["scene_id"]}.npz'
            save_scene(CACHE / fname, xy, feats, eps, subs)
            index.append({'kind': 'fresh', 'key': key, 'seed': k, 'split': s['split'], 'scene_id': s['scene_id'],
                          'file': fname, 'domain': 'fresh', 'n_floor': int(len(xy)),
                          'episodes': [e for e, _ in eps],
                          'sets': [{'set': j, 'eids': st['eids'], 'target': st['target'],
                                    'n_region': int(st['mask'].sum()), 'pi': float(st['mask'].mean())}
                                   for j, st in enumerate(sets)]})
        print(f'seed {k} done, {time.monotonic() - t0:.0f} s', flush=True)

    # ------------------------------------------------------------------------------------------- dev scenes
    dth = []
    region_same_default = collections.Counter()
    for sr in H.distinct_scenes():
        if not sr.runs:
            counts['dev']['skipped_no_run'] += 1
            continue
        ref = sr.runs[0]
        domain = 'realistic' if 'realistic_final' in str(sr.folder) else 'placeholder'
        assert ('realistic_final' in str(sr.scenes_dir)) == (domain == 'realistic')
        sc = H.scene(ref, sr.scene_id)
        rth = H.recorded_config(ref)[1].thresholds
        diff = [f for f in ('grid_resolution_m', 'robot_radius_m', 'person_radius_m') if getattr(rth, f) != getattr(th, f)]
        if diff:
            dth.append(f'{sr.scene_id} ({ref.folder}): {diff}')
        inputs, truth, truths = sc.inputs, sc.truth, sc.truths
        assert inputs.scene_id == sr.scene_id
        # descriptive: are the regions the same under the default config?
        dflt = prepare(inputs, truth, cfg)
        for eid, et in truths.items():
            a = [(r.polygon, r.target_id) for r in et.gt.acceptable_regions or []]
            b = [(r.polygon, r.target_id) for r in dflt[eid].gt.acceptable_regions or []]
            region_same_default['same' if a == b else 'differ'] += 1
        robot = inputs.observation.robot
        xy = coarse_floor(truth, robot, th)
        key = f'{domain}/{sr.scenes_dir.parent.name}/{sr.scenes_dir.name}/{sr.scene_id}'
        counts['dev']['scenes'] += 1
        counts['dev'][f'scenes_{domain}'] += 1
        counts['dev']['episodes'] += len(truths)
        rng_check = np.random.default_rng([SUB_SEED, zlib.crc32((str(sr.folder) + '/check').encode())])
        eps = episodes_of(truths, xy, held, counts['dev'], chk['dev'], key, rng_check)
        tids = sorted({e['target'] for e, m in eps if m is not None})
        feats = {tid: features(xy, truth, truth.targets[tid], robot, th) for tid in tids}
        n_dev = counts['dev']['scenes']
        fname = f'dev/{n_dev:03d}__{sr.scene_id}.npz'
        save_scene(CACHE / fname, xy, feats, eps)
        index.append({'kind': 'dev', 'key': key, 'seed': None, 'split': None, 'scene_id': sr.scene_id,
                      'folder': str(sr.folder), 'run': ref.folder, 'file': fname, 'domain': domain,
                      'n_floor': int(len(xy)), 'episodes': [e for e, _ in eps]})
    print(f'dev done, {time.monotonic() - t0:.0f} s', flush=True)

    (CACHE / 'index.json').write_text(json.dumps(index, indent=1))
    L = ['C5 gate 2 cells: checks (FROZEN.md section 2, Cells).', '']
    for kind in ('fresh', 'dev'):
        c = counts[kind]
        L.append(f'{kind}: ' + ', '.join(f'{k} {v}' for k, v in sorted(c.items())))
        L.append(f'  point-in-polygon replica vs sgi.polygons.point_in_polygon: {chk[kind]["tested"]} cell tests '
                 f'(every region cell of every episode with polygons, plus 300 random floor cells per scene), '
                 f'{chk[kind]["disagree"]} disagreements')
        for w in chk[kind]['where'][:50]:
            L.append(f'    {w}')
    L.append(f'fresh scenes whose eligible episodes have more than one distinct region set: {len(sets_differ)} {sets_differ[:10]}')
    L.append(f'dev scenes whose first listed run recorded a different grid, robot or person radius: {len(dth)}')
    for d in dth:
        L.append(f'  {d}')
    L.append(f'dev episodes whose acceptable regions under the first run\'s recorded config equal those under '
             f'default.yaml (descriptive): {dict(region_same_default)}')
    fl = [r['n_floor'] for r in index if r['kind'] == 'fresh']
    rg = [e['n_region'] for r in index if r['kind'] == 'fresh' for e in r['episodes'] if e.get('eligible')]
    L.append(f'fresh floor cells per scene: min {min(fl)} median {int(np.median(fl))} max {max(fl)}; region cells per '
             f'eligible episode: min {min(rg)} median {int(np.median(rg))} max {max(rg)}')
    fl = [r['n_floor'] for r in index if r['kind'] == 'dev']
    rg = [e['n_region'] for r in index if r['kind'] == 'dev' for e in r['episodes'] if e.get('eligible')]
    L.append(f'dev floor cells per scene: min {min(fl)} median {int(np.median(fl))} max {max(fl)}; region cells per '
             f'eligible episode: min {min(rg)} median {int(np.median(rg))} max {max(rg)}')
    sit = collections.Counter((r['kind'], r.get('domain'), e['situation'], bool(e.get('eligible')))
                              for r in index for e in r['episodes'])
    L.append('episodes by (kind, domain, situation, eligible):')
    for k2, v in sorted(sit.items(), key=str):
        L.append(f'  {k2}: {v}')
    L.append(f'time {time.monotonic() - t0:.0f} s')
    (HERE / 'cells_check.txt').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
