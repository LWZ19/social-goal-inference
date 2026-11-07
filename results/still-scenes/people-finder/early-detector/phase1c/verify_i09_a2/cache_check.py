import sys, json
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
OLD = {"qwen38_low": "s19", "glm53": "s19", "dsv41": "s20"}
NEW = {"qwen38_low": "s21", "glm53": "s23", "dsv41": "s23"}
out = []
for m in OLD:
    lists = {}
    for w, st in (("old", OLD[m]), ("new", NEW[m])):
        d = {}
        for arm in ("real", "staffed", "plain"):
            ref = H.runs(stage=st, arm=arm, model=m, method="con")[0]
            for r in H.records(ref):
                if r.people is not None:
                    k = (arm, r.scene_id, r.trial)
                    v = json.dumps(r.people, sort_keys=True)
                    assert d.get(k, v) == v       # both episodes of a (scene, trial) share one answer
                    d[k] = v
        lists[w] = d
    shared = set(lists["old"]) & set(lists["new"])
    same = sum(lists["old"][k] == lists["new"][k] for k in shared)
    line = f"{m}: new keys with a list {len(lists['new'])}, shared with old {len(shared)}, identical {same}"
    print(line); out.append(line)
open('results/still-scenes/people-finder/early-detector/phase1c/verify_i09_a2/cache_check.txt', 'w').write("\n".join(out) + "\n")
