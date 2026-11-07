"""I03 Stage A step 2 (run after the fit freeze, so no evaluation scene is loaded before it): the scorer's matching
reproduces sgi.run._matched_people as the runs recorded it. For recorded records with a people list and a recorded
people_found_share (run.people = vlm), the scorer's matched count must equal people_found_share x visible and its
unmatched detections (false + exempt) must equal the recorded people_unmatched. Reads runs and scenes only through
research_harness. Writes check_matching.txt."""
from __future__ import annotations

import collections
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H  # noqa: E402
import people_scorer as PS  # noqa: E402

PLAN = [("s19", "qwen38_low", "real"), ("s19", "glm53", "staffed"), ("s20", "dsv41", "plain"),
        ("s23", "haiku55", "real"), ("s23", "sonnet55_low", "staffed"), ("s23", "gemini38_low", "plain"),
        ("s19", "glm53", "ph"), ("s20", "dsv41", "dev_seed1")]
PER_RUN = 12

lines = []
tot = collections.Counter()
for stage, model, arm in PLAN:
    (ref,) = H.runs(stage=stage, arm=arm, model=model, method="con")
    th = H.recorded_config(ref)[1].thresholds
    taken = 0
    for rec in H.records(ref):
        share = rec.measures.get("people_found_share")
        unmatched = rec.measures.get("people_unmatched")
        if rec.people is None or unmatched is None:
            continue
        sc = H.scene(ref, rec.scene_id)
        ppl = [(p["xy"][0], p["xy"][1], p["yaw"]) for p in rec.people]
        r = PS.score_people(ppl, sc, th, exempt_xy=PS.exempt_points(sc.truth))
        want = None if share is None else round(share * r["n_visible"])
        ok = (want is None and r["n_visible"] == 0 or want == r["matched"]) and r["false_raw"] == unmatched
        tot["records"] += 1
        tot["agree"] += ok
        lines.append(f"{'ok ' if ok else 'BAD'} {ref.folder:44s} {rec.scene_id:36s} {rec.episode_id:4s} t{rec.trial} "
                     f"det {r['n_det']:2d} vis {r['n_visible']:2d} | matched scorer {r['matched']:2d} recorded {want} | "
                     f"unmatched scorer {r['false_raw']:2d} recorded {unmatched}")
        taken += 1
        if taken >= PER_RUN:
            break
lines.append(f"matching check: {tot['agree']} of {tot['records']} recorded records agree (matched count and unmatched "
             f"count) with the scorer")
(HERE / "check_matching.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines[-1:]))
