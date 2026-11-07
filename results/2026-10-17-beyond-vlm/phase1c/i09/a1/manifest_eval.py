"""I09 A1 step 6 (FROZEN.md sections 1, 2 and 12): the EVALUATION manifest, written only after FIT.sha256 (and the fix's
FIT_FIX.sha256) verify. The 68 scenes of research_harness.distinct_scenes(), each guarded (guard_scene) before its
scene.json is read; the RGB path guarded as written and resolved, and its sha256. Project venv."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import research_harness as H  # noqa: E402
from sgi.scene import read_scene_file  # noqa: E402


def verify_all():
    out = C.verify_frozen()
    for sha_file in ("FIT.sha256", "FIT_FIX.sha256"):
        for line in (HERE / sha_file).read_text().splitlines():
            want, name = line.split()
            assert C.sha(HERE / name) == want, name
            out[name] = want
    assert (HERE / "FIT.sha256").read_text() == (HERE / "CHOSEN.sha256").read_text()
    chosen = json.loads((HERE / "chosen.json").read_text())
    fix = json.loads((HERE / "chosen_fix.json").read_text())
    assert fix["setting"] == chosen["setting"]
    return out, chosen["setting"], fix["facing_variant"]


def main():
    assert not (HERE / "manifest_eval.json").exists(), "manifest_eval.json exists already"
    hashes, setting, variant = verify_all()
    distinct = H.distinct_scenes()
    assert len(distinct) == 68, len(distinct)
    images = []
    for sref in distinct:
        folder = H.guard_scene(sref.folder)
        rgb = folder / read_scene_file(folder).observation.rgb
        H.guard_file(rgb)
        sset = C.set_of(sref.scenes_dir)
        images.append({"domain": C.domain_of(sset), "set": sset, "scene_id": sref.scene_id, "folder": str(sref.folder),
                       "scenes_dir": str(sref.scenes_dir), "listed": [[str(a), b] for a, b in sref.listed],
                       "runs": [r.folder for r in sref.runs], "rgb": str(rgb),
                       "sha256": hashlib.sha256(rgb.read_bytes()).hexdigest()})
    n = {d: sum(1 for i in images if i["domain"] == d) for d in C.EVAL_DOMAINS}
    assert n == {"E-real": 42, "E-ph": 26}, n
    out = {"purpose": "I09 A1 evaluation pass (FROZEN.md sections 1, 2, 12 step 6)",
           "written": time.strftime('%Y-%m-%dT%H:%M:%S%z'), "hashes_verified": hashes,
           "chosen_setting": setting, "facing_variant": variant, "counts": n, "images": images}
    (HERE / "manifest_eval.json").write_text(json.dumps(out, indent=1))
    print(f"manifest_eval.json: {len(images)} images {n}; detector {setting['detector']}, variant {variant}; "
          f"{out['written']}")


if __name__ == "__main__":
    main()
