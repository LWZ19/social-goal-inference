"""Is the convlm people answer one per (scene, trial), shared by the two episodes? (fresh people calls vs distinct
(scene, trial) among goal steps, and identical people lists within a (scene, trial))."""
import json, sys, collections
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
HERE = 'results/2026-10-17-beyond-vlm/phase1/i01_calibration'
out = []
for m in ['glm53', 'ornith15', 'qwen36_35', 'qwen38', 'qwen3vl']:
    recs = [r for r in H.records(H.runs(stage='s8', model=m, method='convlm')[0]) if r.solver_step]
    groups = collections.defaultdict(list)
    for r in recs:
        groups[(r.scene_id, r.trial)].append(r)
    fresh = sum(1 for r in recs for c in r.result['calls'] if c['stage'] == 'people' and not c.get('cached'))
    two = [g for g in groups.values() if len(g) == 2]
    same = sum(json.dumps(g[0].people, sort_keys=True) == json.dumps(g[1].people, sort_keys=True) for g in two)
    out.append(f"{m}: goal steps {len(recs)}, distinct (scene, trial) {len(groups)}, fresh people calls {fresh}; "
               f"(scene, trial) with both episodes as goal steps {len(two)}, identical people lists {same}")
open(f'{HERE}/people_share.txt', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
