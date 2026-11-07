#!/usr/bin/env python3
"""E4's 0.8 m sensitivity for machine use (E4_DESIGN.md; I13_FROZEN.md section 4: reported, no bar), zero calls.

The E4 scenes store their acceptable and forbidden regions, which sgi.gt.derive uses as they are. So this copies the
scenes to OUT/scenes with the machine-use episodes' stored regions removed, sets the machine-use rule to its fixed
sensitivity near limit (regions._MACHINE_NEAR_SENSITIVITY_M, 0.8 m), copies each given run's config.yaml and
results.jsonl to OUT/runs/<name>/, and rescores them with sgi.rescore against the copied scenes. The rescored records
replace the copies' results.jsonl, so e4_score.py can read OUT/runs/* unchanged. The bent-line scenes keep their
regions; their scores must come out unchanged, which is checked. The real run folders are not touched.

    PYTHONPATH= .venv/bin/python e4_rescore_0p8.py OUT RUN_DIR [RUN_DIR ...]"""
import json, shutil, sys
from pathlib import Path

import sgi.gt.regions as regions
from sgi import rescore

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402  (results/_lib/paths.py)

E4 = paths.STILL / "new-situations/design/scenes"
out, run_dirs = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
regions._GROUP_RULES["machine_use"] = lambda target, members: regions.region_machine_use(
    target, members, near=regions._MACHINE_NEAR_SENSITIVITY_M)
assert regions._MACHINE_NEAR_SENSITIVITY_M == 0.8

for seed_dir in sorted(E4.glob("seed60?")):
    for scene in sorted(seed_dir.iterdir()):
        dest = out / "scenes" / seed_dir.name / scene.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(scene, dest)
        doc = json.loads((dest / "scene.json").read_text())
        for ep in doc["episodes"]:
            if ep["ground_truth"]["situation"] == "machine_use":
                ep["ground_truth"]["acceptable_regions"] = None
                ep["ground_truth"]["forbidden_regions"] = None
        (dest / "scene.json").write_text(json.dumps(doc, indent=1))

changed = {"machine_use": 0, "waiting_line": 0}
for run in run_dirs:
    copy = out / "runs" / run.name
    if copy.exists():
        shutil.rmtree(copy)
    copy.mkdir(parents=True)
    for f in ("config.yaml", "results.jsonl"):
        shutil.copy(run / f, copy / f)
    seed = run.name.split("_s26_e4_")[1].split("_")[0]
    rescore.rescore(copy, out / "scenes" / seed)
    before = [json.loads(l) for l in open(copy / "results.jsonl")]
    after = [json.loads(l) for l in open(copy / "rescored" / "results.jsonl")]
    for b, a in zip(before, after):
        if b["score"]["success"] != a["score"]["success"]:
            changed[b["labels"]["situation"]] += 1
    shutil.copy(copy / "rescored" / "results.jsonl", copy / "results.jsonl")
print(f"rescored {len(run_dirs)} runs at near 0.8 m; records whose success changed: {changed}")
assert changed["waiting_line"] == 0, "bent-line scores changed: the copy is wrong"
