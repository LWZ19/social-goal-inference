"""Person-snap test (SNAP_FROZEN.md, amendments 1-3): the frozen I09 A1 detector people on every scene of
run_scenes_v2, in eval_people.json's format, for the snap rule. Zero model calls.

The detector service (e1/e1d/detector_service.py, A1's frozen RT-DETRv2 + ViTPose in the torch sidecar) must be up on
port 8899. Each scene's rgb.png goes to it; A1's frozen pipeline (pipeline.run_image, chosen.json's setting, facing
F5, the same path as e1d/tracker.py and evaluate.py's score_one) turns the raw answer into people with the scene's
depth and camera. Written: snap_people.json.

    PYTHONPATH= .venv/bin/python snap_detect.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402  (results/_lib/paths.py)
sys.path.insert(0, str(paths.MOVING / "own-camera" / "detector-on-sim-frames" / "e1d"))
import tracker as T  # noqa: E402  (E1d's pinned detector client and A1 loader)

from sgi.geometry import in_image, project  # noqa: E402
from sgi.scene import load_scene  # noqa: E402

SCENES = HERE / "run_scenes_v2"


def main() -> int:
    P, S = T._a1()
    out, started = {}, time.time()
    for folder in sorted(p for p in SCENES.iterdir() if (p / "scene.json").exists()):
        inputs, _ = load_scene(folder)
        obs = inputs.observation
        cam = obs.camera
        raw = T.detect((folder / "rgb.png").read_bytes(), timeout=60.0)
        people = []
        for p in P.run_image(raw, obs.depth, cam, S, "F5"):
            mk = project(cam, (p["xy"][0], p["xy"][1], 1.1))
            mark = None if mk is None or not in_image(cam, mk[0], mk[1]) else [mk[0], mk[1]]
            people.append({"x": p["xy"][0], "y": p["xy"][1], "yaw": p["yaw"], "score": p["score"], "box": p["box"],
                           "torso_pixel": list(p["torso_pixel"]), "mark_pixel": mark,
                           "box_centre": [0.5 * (p["box"][0] + p["box"][2]), 0.5 * (p["box"][1] + p["box"][3])],
                           "yaws_all_variants": p["yaws"]})
        out[str(folder.resolve())] = {"scene_id": inputs.scene_id, "set": "snap_v2", "domain": "E-real",
                                      "rgb": str((folder / "rgb.png").resolve()), "people": people,
                                      "t_det_s": raw.get("t_det_s"), "t_pose_s": raw.get("t_pose_s")}
        print(f"{folder.name:58s} people {len(people)}", flush=True)
    doc = {"setting": S, "facing_variant": "F5", "person_mark_height_m": 1.1,
           "note": "I09 A1's frozen people (as eval_people.json) on the person-snap scenes; made by snap_detect.py",
           "scenes": out}
    (HERE / "snap_people.json").write_text(json.dumps(doc, indent=1, default=float))
    print(f"{len(out)} scenes in {time.time() - started:.0f} s -> snap_people.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
