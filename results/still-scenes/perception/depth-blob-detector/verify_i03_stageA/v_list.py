"""Verifier: list the harness's examples, distinct scenes and the VLM runs (paths only)."""
import sys
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
ex = H.examples()
print("examples", len(ex))
for f, s in ex:
    print("  ", f, s)
ds = H.distinct_scenes()
print("distinct", len(ds))
for d in ds:
    print("  ", d.scenes_dir, d.scene_id, len(d.runs), sorted({r.stage + ':' + r.model + ':' + r.arm + ':' + r.method for r in d.runs})[:3])
for st in ("s19", "s20", "s23"):
    for r in H.runs(stage=st, method="con"):
        print(st, r.folder, r.arm, r.model, r.scenes_dir)
