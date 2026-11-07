"""Build and render one job: the script that sgi.synth.realistic.blender_renderer starts in Blender.

    blender -b --factory-startup --disable-autoexec --python-exit-code 1 -P job.py -- --job <folder>

The folder holds job.json and the pictures of the signs it names (blender_renderer.py describes them). This
script builds the description's scene once (build.py), takes every frame of the job through it (render.py),
and writes beside job.json:

    <name>.png         the picture of each frame that asks for one
    <name>.depth.npy   each frame's depth
    render.log         what Blender printed while it rendered
    facts.json         last of all: Blender's version, the device that rendered, what was built, the seconds

It reports failure by raising: Blender is started so that an exception here ends it with a status of 1.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # Blender does not put a script's folder on the path

import bpy  # noqa: E402

import build  # noqa: E402
import render  # noqa: E402


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="job.py")
    parser.add_argument("--job", type=Path, required=True, help="the job's folder, with job.json in it")
    folder = parser.parse_args(argv).job
    job = json.loads((folder / "job.json").read_text(encoding="utf-8"))
    began = time.perf_counter()
    scene = build.new_scene()
    built = build.build(scene, job, folder)
    device = render.configure(scene, job["settings"], job["spec"]["seed"], built["exposure"])
    built_at = time.perf_counter()
    print(f"sgi: built {built['objects']} objects in {built_at - began:.2f} s; rendering on {device}", flush=True)
    seconds = render.render_frames(scene, job["frames"], folder, job["settings"]["samples"], folder / "render.log")
    facts = {"blender": ".".join(str(part) for part in bpy.app.version), "device": device, **built,
             "seconds": {"build": round(built_at - began, 3), "frames": [round(each, 3) for each in seconds],
                         "all": round(time.perf_counter() - began, 3)}}
    (folder / "facts.json").write_text(json.dumps(facts, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
