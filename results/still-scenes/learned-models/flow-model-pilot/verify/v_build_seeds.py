"""Verifier step 1a: rebuild rasters, labels and coverage lattices for a stated sample of seeds with my own code
(v_common), from phase2/seeds/placeholder (seeds 110-199 only; H.guard_scene before every load; every scene and
episode asserted to be of development situations). Writes verify/data/vseed<k>.npz and .json.

    project venv: v_build_seeds.py K [K ...]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v_common as V  # noqa: E402
import research_harness as H  # noqa: E402
from sgi import solver  # noqa: E402
from sgi.config import load_config  # noqa: E402
from sgi.geometry import in_image, pixel_to_point, project  # noqa: E402
from sgi.gt.derive import oracle_target, prepare  # noqa: E402
from sgi.run import OraclePeople  # noqa: E402
from sgi.scene import load_scene  # noqa: E402
from sgi.types import DetectedPerson, GroundedTarget, Point  # noqa: E402

OUT = Path(V.VER) / 'data'


def anchor_gt(obs, anchor):
    pr = project(obs.camera, anchor)
    if pr is not None and in_image(obs.camera, pr[0], pr[1]):
        x, y = pixel_to_point(pr[0], pr[1], obs.camera.width, obs.camera.height)
        return GroundedTarget(point=Point(x=x, y=y), pixel=(float(pr[0]), float(pr[1])), world=tuple(map(float, anchor)))
    return GroundedTarget(point=Point(x=0, y=0), pixel=(0.0, 0.0), world=tuple(map(float, anchor)))


def build(k, cfg):
    assert 110 <= k <= 199
    th = cfg.thresholds
    sd = Path(V.SEEDS) / f'seed{k}'
    done = json.loads((sd / 'DONE.json').read_text())
    assert done['seed'] == k
    R, L, HL, K, EXT, EOFF, units, left = [], [], [], [], [], [0], [], []
    t0 = time.monotonic()
    for s in done['scenes']:
        sid, split = s['scene_id'], s['split']
        V.check_dev_scene(sid, s['situations'])
        folder = sd / split / sid
        H.guard_scene(folder)
        inputs, truth = load_scene(folder)
        assert inputs.scene_id == sid
        truths = prepare(inputs, truth, cfg)
        V.check_dev_scene(sid, [et.gt.situation for et in truths.values()])
        obs = inputs.observation
        found = OraclePeople(truth, th)(obs).found
        people = [DetectedPerson(number=i + 1, xy=f.xy, yaw=f.yaw, pixel=f.pixel, label=None) for i, f in enumerate(found)]
        fl = solver.floor(obs, people, th)
        for eid, et in truths.items():
            gt = et.gt
            key = f'seed{k}/{split}/{sid}/{eid}'
            if len(gt.target_ids) != 1:
                left.append(dict(key=key, why='targets'))
                continue
            tid = gt.target_ids[0]
            regs = [g for g in (gt.acceptable_regions or []) if g.target_id in (None, tid)]
            if not regs:
                left.append(dict(key=key, why='no region'))
                continue
            tg = truth.targets[tid]
            fr = V.RFrame(tg.anchor[:2], (obs.robot.x, obs.robot.y))
            ot = oracle_target(obs, tg, th)
            face = solver._seen_face(obs, ot if ot is not None else anchor_gt(obs, tg.anchor), th)
            ras = V.make_raster(obs, people, fl, fr, face)
            cells = V.region_cells(regs, ras, fr)
            ext = V.ext_cells(regs, fl, fr)
            same_heading = all(g.heading == regs[0].heading for g in regs)
            if len(cells):
                lab, _, _ = V.make_labels(key, cells, regs, fr)
                HL.append(True)
            else:
                lab = np.zeros((32, 4), np.float32)
                HL.append(False)
            R.append(ras)
            L.append(lab)
            K.append(key)
            EXT.append(ext)
            EOFF.append(EOFF[-1] + len(ext))
            units.append(dict(key=key, split=split, scene=sid, episode=eid, situation=gt.situation, row=sid.split('_', 2)[2],
                              target=tid, frame=[float(fr.o[0]), float(fr.o[1]), fr.yaw, fr.dist], lift_failed=ot is None,
                              face_none=face is None, n_cells=int(len(cells)), n_ext=int(len(ext)),
                              heading_shared=bool(same_heading)))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f'vseed{k}.npz', rasters=np.stack(R), labels=np.stack(L), has_label=np.array(HL),
                        keys=np.array(K), ext_xy=np.concatenate(EXT) if EXT else np.zeros((0, 2)), ext_off=np.array(EOFF))
    (OUT / f'vseed{k}.json').write_text(json.dumps(dict(seed=k, units=units, left_out=left,
                                                        seconds=round(time.monotonic() - t0, 1))))
    print(f'vseed{k}: {len(units)} units, {sum(HL)} labelled, {len(left)} left out, {time.monotonic() - t0:.1f} s', flush=True)


if __name__ == '__main__':
    cfg = load_config()
    for a in sys.argv[1:]:
        k = int(a)
        if (OUT / f'vseed{k}.json').exists():
            print(f'vseed{k}: exists', flush=True)
            continue
        build(k, cfg)
