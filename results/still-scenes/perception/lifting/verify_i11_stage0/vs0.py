"""Independent verifier: S0 trigger rates at the oracle target of each fit episode (worked examples only)."""
import collections
import math

import numpy as np

from vlib import (H, SceneGeo, check_pins, check_th, surface_label, window_labels, nearest_object_component,
                  comp_lift, CLS_FLOOR, CLS_STRUCT, CLS_OBJECT, CLS_NAME, match_target, OUT)
from sgi.config import load_config
from sgi.gt.derive import oracle_target

check_pins()
cfg = load_config()
th = cfg.thresholds
check_th(th)
ex = H.examples()
c = collections.Counter()
lines = [f'S0 fit: {len(ex)} worked-example scenes']
for d, sid in ex:
    sc = H.example(sid, d)
    geo = SceneGeo(sc, th)
    for eid, et in sc.truths.items():
        gt = et.gt
        if not gt.target_ids:
            continue
        tid = gt.target_ids[0]
        ot = oracle_target(sc.observation, sc.truth.targets[tid], th)
        if ot is None:
            c['no oracle target'] += 1
            continue
        u, v = ot.pixel
        c['episodes'] += 1
        right = match_target(ot.world, sc.truth, gt, th) is not None
        c['right'] += right
        lab_s, cls_s, _ = surface_label(geo, 'G1', u, v)
        c['s ' + CLS_NAME[cls_s]] += 1
        t1 = cls_s in (CLS_FLOOR, CLS_STRUCT) or len(window_labels(geo, 'G1', u, v)) >= 2
        t2 = t1 or (cls_s == CLS_OBJECT and nearest_object_component(geo, 'G1', u, v, exclude=lab_s) is not None)
        c['T1'] += t1
        c['T2'] += t2
        v1 = cls_s in (CLS_FLOOR, CLS_STRUCT) and nearest_object_component(geo, 'G1', u, v) is not None
        c['V1 moves'] += v1
        oc = nearest_object_component(geo, 'G1', u, v)
        if oc is not None:
            res = comp_lift(geo, 'G1', oc[0], u, v)
            c['V2 moves'] += 1
            c['V2 breaks'] += right and match_target(res[1], sc.truth, gt, th) is None
lines.append(str(dict(c)))
print('\n'.join(lines))
(OUT / 'vs0_out.txt').write_text('\n'.join(lines) + '\n')
