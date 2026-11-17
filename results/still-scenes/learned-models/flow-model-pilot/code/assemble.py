"""C5 flow pilot step 2a, end: join the per-seed arrays into data/train.npz (seeds 110-189) and data/val.npz (seeds
190-199) for the sidecar, and write data/manifest.json (counts by situation, episodes left out and why, unlabelled
units by situation and row, seen-face and lift counts, heading exceptions, clipping by row, and the label self-check
on training data). Reads only data/seeds/. No model, no network, no torch.
"""
from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

D = Path(C.OUT) / 'data'
TRAIN = range(110, 190)
VAL = range(190, 200)


def join(seeds, name):
    R, L, HL, K, S, FR = [], [], [], [], [], []
    units, left = [], []
    for k in seeds:
        z = np.load(D / 'seeds' / f'seed{k}.npz')
        info = json.loads((D / 'seeds' / f'seed{k}.json').read_text())
        assert list(z['keys']) == [u['key'] for u in info['units']]
        R.append(z['rasters'])
        L.append(z['labels'])
        HL.append(z['has_label'])
        K.append(z['keys'])
        S.append(np.array([u['situation'] for u in info['units']]))
        FR.append(np.array([[u['frame']['ox'], u['frame']['oy'], u['frame']['yaw']] for u in info['units']]))
        units += info['units']
        left += info['left_out']
    arrays = dict(rasters=np.concatenate(R), labels=np.concatenate(L), has_label=np.concatenate(HL),
                  keys=np.concatenate(K), situation=np.concatenate(S), frame=np.concatenate(FR))
    np.savez(D / f'{name}.npz', **arrays)
    return arrays, units, left


def summary(units, left, arrays, with_selfcheck):
    by_sit = collections.defaultdict(collections.Counter)
    scenes = collections.defaultdict(set)
    unl = collections.Counter()
    clip = collections.defaultdict(lambda: [0, 0])
    for u in units:
        s = u['situation']
        b = by_sit[s]
        b['units'] += 1
        b['labelled'] += u['selfcheck'] is not None
        b['label_poses'] += C.POSES if u['selfcheck'] is not None else 0
        b['lift_failed'] += u['lift_failed']
        b['face_none'] += u['face_none']
        b['heading_not_shared'] += not u['heading_shared']
        b['multi_region'] += u['n_regions'] > 1
        b['robot_disc_in_raster'] += u['frame']['dist'] < C.HALF * C.RES + C.ROBOT_DISC_M
        scenes[s].add((u['seed'], u['split'], u['scene']))
        row = u['scene'].split('_', 2)[2]
        if u['selfcheck'] is None:
            unl[(s, row)] += 1
        c = clip[(s, row)]
        c[0] += u['n_ext_in_raster']
        c[1] += u['n_region_cells_ext']
    out = dict(n_units=len(units), n_scenes=len({(u['seed'], u['split'], u['scene']) for u in units}),
               by_situation={s: dict(v, scenes=len(scenes[s])) for s, v in sorted(by_sit.items())},
               left_out=collections.Counter(f"{x['situation']}: {x['reason']}" for x in left),
               left_out_list=left,
               unlabelled_by_situation_row={f'{s} / {r}': n for (s, r), n in sorted(unl.items())},
               region_share_inside_raster_by_row={f'{s} / {r}': round(a / b, 4) if b else None
                                                  for (s, r), (a, b) in sorted(clip.items())},
               dist_robot_target_m=dict(min=min(u['frame']['dist'] for u in units),
                                        max=max(u['frame']['dist'] for u in units)))
    if with_selfcheck:
        sc = collections.defaultdict(collections.Counter)
        for u in units:
            if u['selfcheck'] is None:
                continue
            for name in ('n', 'inreg', 'hok', 'reach', 'success'):
                sc[u['situation']][name] += u['selfcheck'][name]
                sc['ALL'][name] += u['selfcheck'][name]
        out['label_selfcheck'] = {s: dict(v, share_inreg=v['inreg'] / v['n'], share_heading_ok=v['hok'] / v['n'],
                                          share_reach=v['reach'] / v['n'], share_success=v['success'] / v['n'])
                                  for s, v in sc.items()}
    out['arrays'] = {k: list(v.shape) for k, v in arrays.items()}
    return out


def main():
    for k in list(TRAIN) + list(VAL):
        assert (D / 'seeds' / f'seed{k}.json').exists(), k
    seed_manifest = json.loads((Path(C.SEEDS_MANIFEST)).read_text())
    assert not seed_manifest['seeds_missing'], seed_manifest['seeds_missing']
    tr, tr_units, tr_left = join(TRAIN, 'train')
    va, va_units, va_left = join(VAL, 'val')
    man = dict(note='C5 flow pilot data (FROZEN.md sections 1-3). Train seeds 110-189, validation seeds 190-199; all '
                    'four development situations; units = (scene, episode) with one target and a non-empty '
                    'acceptable region; e0 and e1 both kept. Exploratory: train and test layouts share the generator '
                    'templates; labels are the project reference rules.',
               train=summary(tr_units, tr_left, tr, True), val=summary(va_units, va_left, va, False),
               sha256={f: hashlib.sha256((D / f).read_bytes()).hexdigest() for f in ('train.npz', 'val.npz')})
    (D / 'manifest.json').write_text(json.dumps(man, indent=1, default=lambda o: dict(o)))
    for part in ('train', 'val'):
        m = man[part]
        print(part, 'units', m['n_units'], 'scenes', m['n_scenes'], 'left out', dict(m['left_out']))
        for s, v in m['by_situation'].items():
            print('  ', s, dict(v))
        print('  unlabelled', m['unlabelled_by_situation_row'])
        if 'label_selfcheck' in m:
            print('  selfcheck ALL', m['label_selfcheck']['ALL'])


if __name__ == '__main__':
    main()
