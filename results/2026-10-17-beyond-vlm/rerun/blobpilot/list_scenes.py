"""List the distinct development scenes the harness gives, with the listings and runs that lead to each."""
import sys
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H

for s in H.distinct_scenes():
    stages = sorted({(r.stage, r.arm) for r in s.runs})
    print(s.scenes_dir, s.scene_id, '| listed:', [str(d).split('/')[-2:] for d, _ in s.listed], '| runs:', len(s.runs), stages[:6])
print(len(H.distinct_scenes()))
