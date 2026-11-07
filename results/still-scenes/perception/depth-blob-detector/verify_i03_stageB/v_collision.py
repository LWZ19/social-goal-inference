"""POST HOC check (verifier) of the compute agent's number-collision explanation: for every solver step where a
constraint names a number the recorded list does not hold, re-solve B-add with the added people numbered above every
number any constraint names, and compare with B-add as defined. Also re-solves the old GLM B-add break and B-add
flips to confirm which inputs drive them. Uses my own v_replay.run_one. No model, no network.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H  # noqa: E402
import v_replay as V  # noqa: E402


def main():
    from sgi.gt.derive import is_visible
    blobs_of = V.load_blobs()
    for wording in ("old", "new"):
        for stage, model, arms in V.PLAN[wording]:
            for arm in arms:
                ref = H.runs(stage=stage, arm=arm, model=model, method="con")[0]
                th = H.recorded_config(ref)[1].thresholds
                for rec in H.records(ref):
                    if not rec.solver_step:
                        continue
                    listed = {p["number"] for p in rec.people}
                    cnums = {n for u in rec.constraints for n in (u["constraint"].get("people") or [])}
                    dangling = sorted(cnums - listed)
                    if not dangling:
                        continue
                    sc = H.scene(ref, rec.scene_id)
                    obs = sc.observation
                    vis = [p for p in sc.truth.people if is_visible(p, obs, th)]
                    blobs = blobs_of[(str(ref.scenes_dir), rec.scene_id)]
                    arms_d, info, cons, det = V.run_one(rec, blobs, th, vis, obs)
                    a = arms_d["Badd"]
                    res = H.resolve(rec, people=a["people"], constraints=a["cons"])
                    start = max(list(listed) + list(cnums) + [0]) + 1
                    shift = start - info["add_start"]
                    added2 = [{**p, "number": p["number"] + shift} for p in a["added"]]
                    res2 = H.resolve(rec, people=det + added2, constraints=cons)
                    # and with the dangling numbers removed from the constraints (no added people renumbered)
                    taken = [n for n in dangling if any(p["number"] == n for p in a["added"])]
                    print(f"{wording} {model} {rec.scene_id} {rec.episode_id} t{rec.trial}: listed {sorted(listed)} "
                          f"dangling {dangling} taken {taken}; C_rec {bool(rec.score.get('success'))} "
                          f"Badd {bool(res.score.success)} Badd_renumbered {bool(res2.score.success)} "
                          f"(heading {res.score.heading_ok}/{res2.score.heading_ok})", flush=True)


if __name__ == "__main__":
    main()
