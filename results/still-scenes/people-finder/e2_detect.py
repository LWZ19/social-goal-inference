"""E2 (E2_FROZEN.md section 3): I09 A1's frozen detector people on every E2 scene, in eval_people.json's format, before
any model call. Zero model calls; GPU about 60 ms an image (announced by the run). The same path as the
person-snap test's group4/snapset/snap_detect.py, with the scene folder and the output as arguments and each image's
order kept, so the scorer can leave out the cold first image.

The detector service (e1/e1d/detector_service.py, A1's frozen RT-DETRv2 + ViTPose in the torch sidecar) must be up on
port 8899, as group4/snapset/run_detect.sh starts it:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -I \
        ../e1/e1d/detector_service.py --port 8899 &
    PYTHONPATH= .venv/bin/python e2_detect.py --scenes run_scenes --out e2_people.json
"""
from __future__ import annotations

import argparse
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists: refusing to overwrite")
    P, S = T._a1()
    out, started = {}, time.time()
    for order, folder in enumerate(sorted(p for p in a.scenes.iterdir() if (p / "scene.json").exists())):
        inputs, _ = load_scene(folder)
        obs = inputs.observation
        cam = obs.camera
        raw = T.detect((folder / "rgb.png").read_bytes(), timeout=60.0)
        people = []
        for p in P.run_image(raw, obs.depth, cam, S, "F5"):
            mk = project(cam, (p["xy"][0], p["xy"][1], 1.1))
            mark = None if mk is None or not in_image(cam, mk[0], mk[1]) else [mk[0], mk[1]]
            people.append({"x": p["xy"][0], "y": p["xy"][1], "yaw": p["yaw"], "score": p["score"], "box": p["box"],
                           "torso_pixel": list(p["torso_pixel"]), "mark_pixel": mark})
        out[str(folder.resolve())] = {"scene_id": inputs.scene_id, "set": "e2", "domain": "E-real", "order": order,
                                      "rgb": str((folder / "rgb.png").resolve()), "people": people,
                                      "t_det_s": raw.get("t_det_s"), "t_pose_s": raw.get("t_pose_s")}
        print(f"{folder.name:48s} people {len(people)}", flush=True)
    doc = {"setting": S, "facing_variant": "F5", "person_mark_height_m": 1.1,
           "note": "I09 A1's frozen people (as eval_people.json) on the E2 scenes; made by e2_detect.py",
           "scenes": out}
    a.out.write_text(json.dumps(doc, indent=1, default=float))
    print(f"{len(out)} scenes in {time.time() - started:.0f} s -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
