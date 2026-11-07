"""Reference people_matter (sgi.gt.derive.reference_people_matter: people matter when the truth standard goal
fails) at the group episodes, per arm and layout, s23 sonnet con scenes (every C run loads the same scenes)."""
import sys
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
from sgi.gt.derive import reference_people_matter
out = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)
for stage, model in (("s23", "sonnet55_low"), ("s21", "qwen38_low"), ("s20", "dsv41"), ("s19", "glm53")):
    for arm in ("real", "staffed", "plain"):
        run = H.runs(stage=stage, arm=arm, model=model, method="con")[0]
        seen = set()
        for rec in H.records(run):
            if (rec.scene_id, rec.episode_id) in seen or rec.situation == "no_one":
                continue
            seen.add((rec.scene_id, rec.episode_id))
            et = H.scene(run, rec.scene_id).truths[rec.episode_id]
            p(stage, model, arm, rec.situation, rec.scene_id, rec.episode_id, "standard_goal_ok", dict(et.standard_goal_ok),
              "reference_people_matter", reference_people_matter(et, None), "measure", rec.measures.get("reference_people_matter"))
open('results/still-scenes/replay-harness/baselines/verify/v4b_refpm.txt', "w").write("\n".join(out) + "\n")
