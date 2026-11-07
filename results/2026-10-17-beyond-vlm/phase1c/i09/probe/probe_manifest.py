"""I09 pre-freeze API probe, project side: write a one-image manifest for ONE fit image (a realistic worked
example), found and guarded through research_harness. No evaluation scene is touched. Project venv only."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H  # noqa: E402
from sgi.scene import read_scene_file  # noqa: E402

HERE = Path(__file__).resolve().parent
SCENE_ID = "real_examples_02_waiting_line_2"

folder = H.check_example(SCENE_ID)          # guarded: listed worked example, nothing held out inside
rgb = folder / read_scene_file(folder).observation.rgb
H.guard_file(rgb)
sc = H.example(SCENE_ID)
cam = sc.observation.camera
entry = {"domain": "F-real", "set": "real", "scene_id": SCENE_ID, "rgb": str(rgb),
         "sha256": hashlib.sha256(rgb.read_bytes()).hexdigest(),
         "camera": {"fx": cam.fx, "fy": cam.fy, "cx": cam.cx, "cy": cam.cy, "width": cam.width, "height": cam.height,
                    "T_world_camera": cam.T_world_camera.tolist()},
         "people": [{"id": p.id, "x": p.x, "y": p.y, "yaw": p.yaw, "role": p.role} for p in sc.truth.people]}
(HERE / "probe_manifest.json").write_text(json.dumps({"purpose": "API probe, one fit image", "images": [entry]}, indent=1))
print(json.dumps(entry, indent=1))
