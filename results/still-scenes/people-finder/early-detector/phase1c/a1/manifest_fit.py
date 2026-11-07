"""I09 A1 step 3 (FROZEN.md section 12): the FIT manifest. Project venv only.

Every fit scene of phase1b/i03/stageA/fitdata.py fit_refs(), in its order: the 21 realistic and 8 placeholder worked
examples (guarded by research_harness.check_example) and the 260 fresh-seed scenes of seeds 100-109 (guarded by
research_harness.guard_scene, the only scenes read outside the harness's lists). For each: domain, set, scene id,
the RGB path (guarded as written and resolved) and its sha256. No evaluation scene is touched here.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
sys.path.insert(0, 'results/still-scenes/perception/depth-blob-detector/stageA')
import research_harness as H  # noqa: E402
import fitdata as FD  # noqa: E402
from sgi.scene import read_scene_file  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    t0 = time.time()
    images = []
    for domain, sset, scene_id, loader in FD.fit_refs():
        kind, folder, sid = loader
        assert sid == scene_id
        if kind == 'example':
            scene_folder = H.check_example(scene_id, folder)
        else:
            assert kind == 'fresh'
            scene_folder = H.guard_scene(folder / scene_id)
            assert '/still-scenes/perception/fresh_seeds/seed1' in str(scene_folder)
        rgb = scene_folder / read_scene_file(scene_folder).observation.rgb
        H.guard_file(rgb)
        images.append({"domain": domain, "set": sset, "scene_id": scene_id, "kind": kind,
                       "scene_folder": str(scene_folder), "rgb": str(rgb),
                       "sha256": hashlib.sha256(rgb.read_bytes()).hexdigest()})
    n = {d: sum(1 for i in images if i['domain'] == d) for d in ('F-real', 'F-ph')}
    assert n == {'F-real': 21, 'F-ph': 268}, n
    out = {"purpose": "I09 A1 fit pass (FROZEN.md sections 1, 2, 12 step 3)", "written": time.strftime('%Y-%m-%dT%H:%M:%S%z'),
           "counts": n, "images": images}
    (HERE / 'manifest_fit.json').write_text(json.dumps(out, indent=1))
    print(f"manifest_fit.json: {len(images)} images {n}; {time.time() - t0:.1f} s")


if __name__ == '__main__':
    main()
