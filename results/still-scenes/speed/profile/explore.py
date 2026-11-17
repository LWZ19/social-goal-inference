"""Exploration: list distinct scenes, the arms of the runs that load them, and their situations/groups/roles."""
import sys, collections
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H

ds = H.distinct_scenes()
print(len(ds), "distinct scenes")
for s in ds:
    arms = sorted({r.arm for r in s.runs})
    stages = sorted({r.stage for r in s.runs})
    print(s.scenes_dir.name, s.scene_id, arms, stages, len(s.runs), [str(k[0].relative_to(H.main_checkout())) for k in s.listed][:3])
