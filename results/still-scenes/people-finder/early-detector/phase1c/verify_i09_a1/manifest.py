"""Verifier for I09 A1: the image manifest for the verifier's own sidecar pass (project venv).

Lists (1) the 68 evaluation scenes of research_harness.distinct_scenes() and (2) the 29 worked-example scenes of
research_harness.examples(), each guarded by the harness (guard_scene / check_example) before its scene.json is read,
the RGB path guarded as written and resolved, with its sha256. Written independently of the compute agent's code.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H  # noqa: E402
from sgi.scene import read_scene_file  # noqa: E402

OUT = Path(__file__).resolve().parent
REAL = ("real", "staffed", "plain")


def set_of_eval(scenes_dir) -> str:
    d = str(scenes_dir) + "/"
    for name in REAL:
        if f"realistic_final/{name}/" in d:
            return name
    if "dev_new/seed" in d or d.endswith("placeholder/eval/") or d.endswith("/placeholder_first14/"):
        return "placeholder"
    raise SystemExit(f"unknown set: {scenes_dir}")


def entry(folder: Path, **kw):
    rgb = folder / read_scene_file(folder).observation.rgb
    H.guard_file(rgb)
    return dict(kw, folder=str(folder), rgb=str(rgb), sha256=hashlib.sha256(rgb.read_bytes()).hexdigest())


def main():
    images = []
    for folder, scene_id in H.examples():
        s = str(folder)
        f = H.check_example(scene_id, folder)
        if 'realistic_final' in s:
            arm = s.split('realistic_final/')[1].split('/')[0]
            images.append(entry(f, part="fit", domain="F-real", set=arm, scene_id=scene_id))
        elif s.endswith('placeholder/examples'):
            images.append(entry(f, part="fit", domain="F-ph", set="ph_examples", scene_id=scene_id))
        else:
            raise SystemExit(f"unknown examples folder {folder}")
    distinct = H.distinct_scenes()
    assert len(distinct) == 68
    for sref in distinct:
        f = H.guard_scene(sref.folder)
        sset = set_of_eval(sref.scenes_dir)
        images.append(entry(f, part="eval", domain="E-real" if sset in REAL else "E-ph", set=sset,
                            scene_id=sref.scene_id, scenes_dir=str(sref.scenes_dir)))
    counts = {}
    for e in images:
        counts[e["domain"]] = counts.get(e["domain"], 0) + 1
    assert counts == {"F-real": 21, "F-ph": 8, "E-real": 42, "E-ph": 26}, counts
    out = {"purpose": "I09 A1 verifier sidecar pass", "written": time.strftime('%Y-%m-%dT%H:%M:%S%z'),
           "counts": counts, "images": images}
    (OUT / "manifest_verify.json").write_text(json.dumps(out, indent=1))
    print(counts, len(images))


if __name__ == "__main__":
    main()
