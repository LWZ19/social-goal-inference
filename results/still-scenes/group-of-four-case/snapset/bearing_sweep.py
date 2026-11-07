"""Zero calls, no rendering. What a bearing override for recipe_conversation_group would give: for 2, 3 and 4 people
and the robot's bearing from the group centre at 0-90 degrees (today fixed at 90, 60, 72.5), the target's hidden
share by occlusion.py's cylinder model, and how often its anchor (chest, 1.1 m) is covered by a person more than
0.3 m nearer, a proxy for scene check 5, which rejects the layout (the generator would draw again). The module
constant is patched in this process only; no tracked file changes.
    PYTHONPATH= .venv/bin/python results/still-scenes/group-of-four-case/snapset/bearing_sweep.py"""
import importlib.util
from pathlib import Path

import numpy as np

import sgi.synth.recipes as R
from sgi.geometry import Camera, camera_on_robot

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("occlusion", HERE.parent / "occlusion.py")
occ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(occ)


def scene_of(layout):
    cam = Camera(**R.CAMERA_LENS, T_world_camera=camera_on_robot(layout.robot, **R.CAMERA_MOUNT))
    return {"observation": {"T_world_camera": cam.T_world_camera.tolist(), "width": cam.width, "height": cam.height,
                            "intrinsics": {"fx": cam.fx, "fy": cam.fy, "cx": cam.cx, "cy": cam.cy}},
            "ground_truth": {"people": [p.model_dump() for p in layout.scene_gt.people]}}


def anchor_covered(scene, pid):
    s, Tinv, K = occ.silhouettes(scene)
    p = next(p for p in scene["ground_truth"]["people"] if p["id"] == pid)
    q = occ.project(Tinv, K, (p["x"], p["y"], 1.1))
    return any(occ.covers(Tinv, K, s[o], q[0], q[1], q[2] - 0.3) for o in s if o != pid)


default = dict(R._GROUP_BEARING_DEG)
print("n  bearing  hidden: min  median  max   anchor covered (check 5 would reject)   [today's bearing marked *]")
for n in (2, 3, 4):
    for b in range(0, 95, 5):
        R._GROUP_BEARING_DEG[n] = float(b)
        shares, covered = [], 0
        for k in range(20):
            lay = R.recipe_conversation_group(np.random.default_rng([999, n, b, k]), n)
            sc = scene_of(lay)
            h, _ = occ.hidden_share(sc, "p0")
            shares.append(h)
            covered += anchor_covered(sc, "p0")
        a = np.array(shares)
        mark = "*" if abs(b - default[n]) < 2.6 else " "
        print(f"{n}  {b:5.1f}{mark}   {a.min():.2f}  {np.median(a):.2f}  {a.max():.2f}   {covered}/20")
    R._GROUP_BEARING_DEG[n] = default[n]
