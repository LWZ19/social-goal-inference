"""C5 flow pilot step 2a (FROZEN.md sections 1-3): rasters and labels for training seeds 110-199.

Loading follows phase1b/i03/stageA/fitdata.py: H.guard_scene, then sgi.scene.load_scene, then regions from
sgi.gt.derive.prepare under the default config. Every scene is asserted to be of development situations and no
held-out row. Only seeds 110-199 under phase2/seeds/placeholder/ are read; a seed is read only once its DONE.json
exists. No model, no network, no torch.

Per seed k writes data/seeds/seed<k>.npz (rasters, labels, ext region cells) and data/seeds/seed<k>.json (units,
left-out episodes with reasons, label self-check counts).

    cd WT && PYTHONPATH= OMP_NUM_THREADS=2 ... nice -n 10 .venv/bin/python code/build_seeds.py K [K ...]
"""
from __future__ import annotations

import collections
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import research_harness as H  # noqa: E402
import i14  # noqa: E402
from sgi import solver  # noqa: E402
from sgi.config import load_config  # noqa: E402
from sgi.freespace import connected, robot_grid  # noqa: E402
from sgi.geometry import in_image, pixel_to_point, project  # noqa: E402
from sgi.gt.derive import oracle_target, prepare  # noqa: E402
from sgi.run import OraclePeople  # noqa: E402
from sgi.scene import load_scene  # noqa: E402
from sgi.types import DetectedPerson, GroundedTarget, Point  # noqa: E402

OUTD = Path(C.OUT) / 'data' / 'seeds'


def anchor_target(obs, anchor) -> GroundedTarget:
    """A GroundedTarget at the 3-D anchor (only .world is read by _seen_face)."""
    pr = project(obs.camera, anchor)
    if pr is not None and in_image(obs.camera, pr[0], pr[1]):
        x, y = pixel_to_point(pr[0], pr[1], obs.camera.width, obs.camera.height)
        return GroundedTarget(point=Point(x=x, y=y), pixel=(float(pr[0]), float(pr[1])),
                              world=tuple(float(v) for v in anchor))
    return GroundedTarget(point=Point(x=0, y=0), pixel=(0.0, 0.0), world=tuple(float(v) for v in anchor))


def oracle_people(truth, obs, th):
    found = OraclePeople(truth, th)(obs).found
    return [DetectedPerson(number=i + 1, xy=f.xy, yaw=f.yaw, pixel=f.pixel, label=None) for i, f in enumerate(found)]


def one_seed(k: int, cfg) -> None:
    assert 110 <= k <= 199, f"seed {k} refused"
    th = cfg.thresholds
    sdir = Path(C.SEEDS_DIR) / f'seed{k}'
    done = json.loads((sdir / 'DONE.json').read_text())
    assert done['seed'] == k
    rasters, labels, has_label, keys, ext_xy, ext_off = [], [], [], [], [], [0]
    units, left_out = [], []
    sc_counts = collections.Counter()
    t0 = time.monotonic()
    for s in done['scenes']:
        split, scene_id = s['split'], s['scene_id']
        assert split in ('eval', 'examples')
        C.assert_dev_scene(scene_id, s['situations'])
        folder = sdir / split / scene_id
        H.guard_scene(folder)
        inputs, truth = load_scene(folder)
        assert inputs.scene_id == scene_id
        truths = prepare(inputs, truth, cfg)
        C.assert_dev_scene(scene_id, [et.gt.situation for et in truths.values()])
        obs = inputs.observation
        robot = obs.robot
        people = oracle_people(truth, obs, th)
        fl = solver.floor(obs, people, th)
        grid = robot_grid(truth, robot, th)
        joined = connected(grid, (robot.x, robot.y))
        for eid, et in truths.items():
            gt = et.gt
            key = f'seed{k}/{split}/{scene_id}/{eid}'
            base = dict(key=key, seed=k, split=split, scene=scene_id, episode=eid, situation=gt.situation,
                        row=s['row'])
            if len(gt.target_ids) != 1:
                left_out.append({**base, 'reason': f'{len(gt.target_ids)} targets'})
                continue
            tid = gt.target_ids[0]
            regions = [g for g in gt.acceptable_regions or [] if g.target_id in (None, tid)]
            if not regions:
                left_out.append({**base, 'reason': 'no acceptable region'})
                continue
            target = truth.targets[tid]
            frame = C.Frame(target.anchor[:2], (robot.x, robot.y))
            ot = oracle_target(obs, target, th)
            lift_failed = ot is None
            face_t = anchor_target(obs, target.anchor) if lift_failed else ot
            normal = solver._seen_face(obs, face_t, th)
            raster = C.paint(obs, people, fl, frame, normal)
            cells = C.region_cells_raster(regions, raster, frame)
            ext = C.region_cells_extended(regions, fl, frame)
            heading_shared = all(g.heading == regions[0].heading for g in regions)
            u = dict(base, target_id=tid, target_kind=target.kind, frame=frame.as_dict(),
                     robot=[robot.x, robot.y, robot.yaw], n_people=len(people), lift_failed=lift_failed,
                     face_none=normal is None, n_regions=len(regions), heading_shared=heading_shared,
                     heading_kind='yaw' if regions[0].heading.yaw is not None else 'face_point',
                     n_region_cells_raster=int(len(cells)), n_region_cells_ext=int(len(ext)),
                     n_ext_in_raster=int(np.sum((np.abs(ext[:, 0]) < C.HALF * C.RES) &
                                                (np.abs(ext[:, 1]) < C.HALF * C.RES))) if len(ext) else 0,
                     oracle_target=None if ot is None else list(ot.world))
            if len(cells):
                lab, pw, ph = C.labels_for(key, cells, regions, frame)
                rep = i14.replica(robot, gt, tid, grid, joined, pw, ph)
                u['selfcheck'] = dict(n=C.POSES, inreg=int(rep['inreg'].sum()), hok=int(rep['hok'].sum()),
                                      reach=int(rep['reach'].sum()), success=int(rep['success'].sum()))
                has_label.append(True)
            else:
                lab = np.zeros((C.POSES, 4), dtype=np.float32)
                u['selfcheck'] = None
                has_label.append(False)
            rasters.append(raster)
            labels.append(lab)
            keys.append(key)
            ext_xy.append(ext)
            ext_off.append(ext_off[-1] + len(ext))
            units.append(u)
            sc_counts[gt.situation] += 1
    OUTD.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTD / f'seed{k}.npz', rasters=np.stack(rasters), labels=np.stack(labels),
                        has_label=np.array(has_label), keys=np.array(keys),
                        ext_xy=np.concatenate(ext_xy) if ext_xy else np.zeros((0, 2)), ext_off=np.array(ext_off))
    info = dict(seed=k, n_scenes=len(done['scenes']), n_units=len(units), by_situation=dict(sc_counts),
                units=units, left_out=left_out, seconds=round(time.monotonic() - t0, 1))
    (OUTD / f'seed{k}.json').write_text(json.dumps(info))
    print(f"seed{k}: {len(done['scenes'])} scenes, {len(units)} units ({sum(has_label)} labelled), "
          f"{len(left_out)} left out, {time.monotonic() - t0:.1f} s", flush=True)


def main(argv):
    cfg = load_config()
    for k in (int(a) for a in argv):
        if (OUTD / f'seed{k}.json').exists():
            print(f"seed{k}: done already", flush=True)
            continue
        one_seed(k, cfg)


if __name__ == '__main__':
    main(sys.argv[1:])
