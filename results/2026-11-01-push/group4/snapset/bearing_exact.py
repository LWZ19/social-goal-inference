"""Zero calls, no rendering. The hashed sweep (bearing_sweep.py: same draws, rng [999, n, b, k] for k < 20, same
measure) at full precision, for every bearing the amendment names or its rule could admit, plus the generator's true
default for cg4 (72.5 deg). Then the amendment's rule: the median share is in the bin (15-40: [0.15, 0.40);
40-55: [0.40, 0.55]) and the chest is covered in at most 10 of 20 layouts.
    PYTHONPATH= .venv/bin/python results/2026-11-01-push/group4/snapset/bearing_exact.py"""
import importlib.util
from pathlib import Path

import numpy as np

import sgi.synth.recipes as R
from sgi.geometry import Camera, camera_on_robot

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("occlusion", HERE.parent / "occlusion.py")
occ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(occ)


def scene_of(layout):                              # as bearing_sweep.py
    cam = Camera(**R.CAMERA_LENS, T_world_camera=camera_on_robot(layout.robot, **R.CAMERA_MOUNT))
    return {"observation": {"T_world_camera": cam.T_world_camera.tolist(), "width": cam.width, "height": cam.height,
                            "intrinsics": {"fx": cam.fx, "fy": cam.fy, "cx": cam.cx, "cy": cam.cy}},
            "ground_truth": {"people": [p.model_dump() for p in layout.scene_gt.people]}}


def anchor_covered(scene, pid):                    # as bearing_sweep.py
    s, Tinv, K = occ.silhouettes(scene)
    p = next(p for p in scene["ground_truth"]["people"] if p["id"] == pid)
    q = occ.project(Tinv, K, (p["x"], p["y"], 1.1))
    return any(occ.covers(Tinv, K, s[o], q[0], q[1], q[2] - 0.3) for o in s if o != pid)


ASKED = {2: (10, 15, 20, 25), 3: (10, 15, 20, 45, 50, 55), 4: (10, 15, 30, 35, 40, 65, 70, 72.5, 75)}
import sys
if "--tail" in sys.argv:
    ASKED = {4: (72.5, 75)}
default = dict(R._GROUP_BEARING_DEG)
print("n  bearing  median    covered  rule")
for n, bearings in ASKED.items():
    for b in bearings:
        b = int(b) if float(b).is_integer() else b
        R._GROUP_BEARING_DEG[n] = float(b)
        shares, covered = [], 0
        for k in range(20):
            sc = scene_of(R.recipe_conversation_group(np.random.default_rng([999, n, b if float(b).is_integer() else int(round(b * 10)), k]), n))
            shares.append(occ.hidden_share(sc, "p0")[0])
            covered += anchor_covered(sc, "p0")
        m = float(np.median(shares))
        ok = covered <= 10
        rule = ("15-40" if ok and 0.15 <= m < 0.40 else "40-55" if ok and 0.40 <= m <= 0.55 else
                f"no ({'covered' if not ok else 'median'})")
        print(f"{n}  {b:6.1f}   {m:.4f}   {covered:2d}/20    {rule}", flush=True)
    R._GROUP_BEARING_DEG[n] = default[n]
