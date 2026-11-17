"""Verifier: role of the visible true person each B-prune-pruned detection is matched to (old wording). Harness only."""
import collections, json, sys
from pathlib import Path
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
from sgi.gt.derive import is_visible
from sgi.run import _matched_people
from sgi.types import DetectedPerson
HERE = Path(__file__).resolve().parent
for stage, model in (("s19", "qwen38_low"), ("s19", "glm53"), ("s20", "dsv41")):
    rows = {(r["arm"], r["scene"], r["episode"], r["trial"]): r for r in map(json.loads, (HERE / f"v_rows_old_{model}.jsonl").open())}
    c = collections.Counter()
    for arm in ("real", "staffed", "plain"):
        ref = H.runs(stage=stage, arm=arm, model=model, method="con")[0]
        th = H.recorded_config(ref)[1].thresholds
        for rec in H.records(ref):
            r = rows[(arm, rec.scene_id, rec.episode_id, rec.trial)]
            if not r["solver_step"] or not r["Bprune"].get("pruned_numbers"):
                continue
            sc = H.scene(ref, rec.scene_id)
            vis = [p for p in sc.truth.people if is_visible(p, sc.observation, th)]
            m = _matched_people([DetectedPerson.model_validate(p) for p in rec.people], vis, th)
            role = {p.id: p.role for p in vis}
            for n in r["Bprune"]["pruned_numbers"]:
                c["unmatched" if n not in m else ("staff" if role[m[n]] else "non-staff")] += 1
    print(model, dict(c))
