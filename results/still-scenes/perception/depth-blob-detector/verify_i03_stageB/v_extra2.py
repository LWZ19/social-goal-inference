"""Verifier extras 2: come_after statement in GLM's B-add class-1 line failures; the two new-wording prune breaks.
Reads records through the harness. No model, no network."""
import json, sys
from pathlib import Path
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
HERE = Path(__file__).resolve().parent
def key(r): return (r["arm"], r["scene"], r["episode"], r["trial"])
rows = {key(r): r for r in map(json.loads, (HERE / "v_rows_old_glm53.jsonl").open())}
I01 = HERE.parents[2] / "phase1" / "i01_repairs"
lfk = {key(o) for o in map(json.loads, (I01 / "rows_old_glm53.jsonl").open()) if o["line_failure"]}
c1 = {k for k in lfk if not rows[k]["Badd"]["ok"] and rows[k]["Badd"]["cls"] == 1}
none_stated = empty = other = 0
for arm in ("real", "staffed", "plain"):
    ref = H.runs(stage="s19", arm=arm, model="glm53", method="con")[0]
    for rec in H.records(ref):
        k = (arm, rec.scene_id, rec.episode_id, rec.trial)
        if k not in c1: continue
        ca = [u for u in rec.constraints if u["constraint"]["kind"] == "come_after"]
        if not ca: none_stated += 1
        elif all(not (u["constraint"].get("people") or []) for u in ca): empty += 1
        else: other += 1
        print(k, "come_after:", [(u["constraint"].get("people"), u["status"]) for u in ca], "people listed", len(rec.people))
print("class-1 GLM B-add:", len(c1), "no come_after stated", none_stated, "only empty come_after", empty, "other", other)
from sgi.gt.derive import is_visible
from sgi.run import _matched_people
from sgi.types import DetectedPerson
for stage, model, arm, scene, ep, tr in (("s23", "dsv41", "real", "real_eval_07_waiting_line_2", "e0", 2), ("s23", "haiku55", "real", "real_eval_07_waiting_line_2", "e0", 0)):
    ref = H.runs(stage=stage, arm=arm, model=model, method="con")[0]
    th = H.recorded_config(ref)[1].thresholds
    for rec in H.records(ref):
        if (rec.scene_id, rec.episode_id, rec.trial) != (scene, ep, tr): continue
        sc = H.scene(ref, scene)
        vis = [p for p in sc.truth.people if is_visible(p, sc.observation, th)]
        m = _matched_people([DetectedPerson.model_validate(p) for p in rec.people], vis, th)
        role = {p.id: p.role for p in sc.truth.people}
        print(model, scene, ep, tr, "number 3 ->", m.get(3), "role", role.get(m.get(3)), "constraints naming 3:", [u["constraint"]["kind"] for u in rec.constraints if 3 in (u["constraint"].get("people") or [])])
