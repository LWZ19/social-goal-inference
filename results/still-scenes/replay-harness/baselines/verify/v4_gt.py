"""What the scene truth says about people at the group episodes (eval_10/11/12), one run (s23 real sonnet con)."""
import sys, re
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
out = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)
for arm in ("real", "staffed", "plain"):
    run = H.runs(stage="s23", arm=arm, model="sonnet55_low", method="con")[0]
    seen = set()
    for rec in H.records(run):
        if rec.situation != "conversation_group" or (rec.scene_id, rec.episode_id) in seen:
            continue
        seen.add((rec.scene_id, rec.episode_id))
        sc = H.scene(run, rec.scene_id)
        gt = sc.truths[rec.episode_id].gt
        d = gt.model_dump() if hasattr(gt, "model_dump") else vars(gt)
        short = {k: (v if not isinstance(v, (list, dict)) or len(str(v)) < 300 else str(v)[:300] + "...") for k, v in d.items()}
        p(arm, rec.scene_id, rec.episode_id, short)
open('results/still-scenes/replay-harness/baselines/verify/v4_gt.txt', "w").write("\n".join(out) + "\n")
