"""Compare the verifier's raw sidecar outputs with the compute agent's raw_fit / raw_eval files (read as data)."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
V = str(HERE / 'raw') + '/'
A = str(HERE.parent / 'a1') + '/'
def load(p, key='rgb'):
    return {json.loads(l)[key]: json.loads(l) for l in open(p)}
out = []
for det, mine, theirs in (('rtdetr_v2_r50vd', V+'rtdetr_v2_r50vd.jsonl', [A+'raw_eval/rtdetr_v2_r50vd.jsonl', A+'raw_fit/rtdetr_v2_r50vd.jsonl']),
                          ('dfine_large', V+'dfine_large.jsonl', [A+'raw_fit/dfine_large.jsonl'])):
    M = load(mine)
    T = {}
    for p in theirs:
        T.update(load(p))
    nimg = nbox_eq = 0; maxb = maxs = maxk = maxks = 0.0; mism = []
    for rgb, m in M.items():
        t = T[rgb]
        nimg += 1
        bm, bt = np.array(m['boxes']).reshape(-1, 5), np.array(t['boxes']).reshape(-1, 5)
        if bm.shape != bt.shape:
            mism.append((m['scene_id'], bm.shape, bt.shape)); continue
        nbox_eq += 1
        if len(bm):
            maxb = max(maxb, float(np.abs(bm[:, :4]-bt[:, :4]).max())); maxs = max(maxs, float(np.abs(bm[:, 4]-bt[:, 4]).max()))
        km = np.array(m['kp']).reshape(-1, 17, 3); kt = np.array([p['kp'] for p in t['pose']]).reshape(-1, 17, 3)
        assert km.shape == kt.shape, (m['scene_id'], km.shape, kt.shape)
        if len(km):
            maxk = max(maxk, float(np.abs(km[..., :2]-kt[..., :2]).max())); maxks = max(maxks, float(np.abs(km[..., 2]-kt[..., 2]).max()))
    print(f"{det}: images {nimg}, same box count {nbox_eq}, max |box diff| {maxb:.3g} px, max |score diff| {maxs:.3g}, "
          f"max |kp diff| {maxk:.3g} px, max |kp conf diff| {maxks:.3g}; count mismatches {mism}")
