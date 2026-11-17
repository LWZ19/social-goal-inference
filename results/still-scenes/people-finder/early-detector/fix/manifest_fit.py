"""I09 fix, diagnosis step 1a: the manifest of the new fit renders (seeds 110-119, 14 rows each), written once.

Every scene folder is path-guarded (pathguard_fit) and harness-guarded (H.guard_scene) before its scene.json is
read; the RGB path is guarded as written and resolved, and its sha256 recorded. Split: seeds 110-115 fit-train,
116-119 fit-validate (the sidecar runs on both; the diagnosis scores fit-train only). Project venv.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fitlib as F  # noqa: E402
from sgi.scene import read_scene_file  # noqa: E402

NAME_BAN = re.compile(r"group_photo|lift_wait|heldout|held_out|eval_1[6-9]", re.I)


def main():
    out_path = F.HERE / "manifest_fit.json"
    assert not out_path.exists(), "manifest_fit.json exists already"
    for top, dirs, files in os.walk(F.SEEDS, followlinks=False):
        for name in dirs + files:
            assert not NAME_BAN.search(name), f"banned name: {Path(top) / name}"
            assert not (Path(top) / name).is_symlink(), f"link: {Path(top) / name}"
    images = []
    for k in range(110, 120):
        ev = F.SEEDS / f"seed{k}" / "eval"
        assert sorted(os.listdir(ev)) == sorted(F.ROW_IDS), (k, sorted(os.listdir(ev)))
        for row, sid in enumerate(F.ROW_IDS):
            folder = ev / sid
            F.PG.check(folder)
            F.H.guard_scene(folder)
            sf = read_scene_file(folder)
            assert sf.scene_id == sid
            rgb = folder / sf.observation.rgb
            depth = folder / sf.observation.depth
            F.PG.check(rgb)
            F.H.guard_file(rgb)
            job = json.loads((folder / "render_job.json").read_text())
            assert job["variant"] == "real" and job["seed"] == k, (k, sid)
            images.append({"scene_id": f"seed{k}/{sid}", "row_id": sid, "seed": k, "row": row,
                           "domain": "fit-train" if k in F.PG.FIT_TRAIN else "fit-validate", "set": "real",
                           "folder": str(folder), "rgb": str(rgb), "sha256": F.sha(rgb),
                           "sha256_depth": F.sha(depth), "sha256_scene_json": F.sha(folder / "scene.json")})
    assert len({i["rgb"] for i in images}) == len(images) == 140
    out = {"purpose": "I09 fix diagnosis: new fit renders (seeds 110-119); fit-train 110-115, fit-validate 116-119",
           "written": time.strftime('%Y-%m-%dT%H:%M:%S%z'),
           "render_manifest_sha256": F.sha(F.SEEDS / "manifest.json"),
           "counts": {d: sum(i["domain"] == d for i in images) for d in ("fit-train", "fit-validate")},
           "images": images}
    out_path.write_text(json.dumps(out, indent=1))
    (F.HERE / "MANIFEST_FIT.sha256").write_text(f"{F.sha(out_path)}  manifest_fit.json\n")
    print(f"manifest_fit.json: {len(images)} images {out['counts']}; {out['written']}")


if __name__ == "__main__":
    main()
