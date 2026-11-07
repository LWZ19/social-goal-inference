import sys
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
ds = H.distinct_scenes()
print(len(ds))
for s in ds:
    print(s.scene_id, '|', s.scenes_dir, '|', s.folder, '|', len(s.runs), [r.name for r in s.runs[:2]], [str(k[0]) for k in s.listed])
