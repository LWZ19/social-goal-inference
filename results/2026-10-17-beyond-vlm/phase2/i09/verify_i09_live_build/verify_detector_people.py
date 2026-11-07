"""Verifier: the driver's DetectorPeople against A1's eval_people.json on every realistic development scene, against
OraclePeople's pixel rule, and its refusals (unknown scene, another observation). No calls; writes only here.

The driver module is imported only to exercise the code under test (DetectorPeople, detector_table, run_arm); the
expected values are built here independently: my own fingerprint map through the harness, my own FoundPerson list,
and the package's OraclePeople run on the detector positions (visibility rule switched off, since a detection is
in view by construction).
"""
import importlib.util
import json
import math
import sys
import types
from pathlib import Path

V = Path(__file__).resolve().parent
RD = V.parents[2]
LIVE = V.parent / "live"
spec = importlib.util.spec_from_file_location("i09_driver_under_test", LIVE / "driver.py")
D = importlib.util.module_from_spec(spec)
sys.modules["i09_driver_under_test"] = D
spec.loader.exec_module(D)
import sgi.run as sgi_run  # noqa: E402  (the builder's src_202c165 copy, asserted by the driver)
from sgi.geometry import in_image  # noqa: E402
import research_harness as H  # noqa: E402

EP = json.loads((RD / "phase1c/i09/a1/eval_people.json").read_text())
ereal = {e["scene_id"]: (folder, e) for folder, e in EP["scenes"].items() if e["domain"] == "E-real"}
res = {"scenes": {}, "fails": []}

# my own map: every scene of the six recorded runs (3 scenes folders x 14), via check_scene, by fingerprint
mine = {}
for stage, model in (("s19", "glm53"), ("s20", "dsv41")):
    for arm in ("real", "staffed", "plain"):
        ref = H.runs(stage=stage, arm=arm, model=model, method="con")[0]
        sids = sorted({r.scene_id for r in H.records(ref) if r.trial == 0})
        for sid in sids:
            folder = H.check_scene(ref, sid)
            mine.setdefault(sid, (ref, folder, H.fingerprint(folder)))
res["n_scenes_runs"] = len(mine)
ep_fp = {sid: H.fingerprint(Path(folder)) for sid, (folder, _) in ereal.items()}
res["n_scenes_eval_people"] = len(ep_fp)
res["fingerprint_match"] = sum(mine[s][2] == ep_fp.get(s) for s in mine)

table = D.detector_table()
res["driver_table_n"] = len(table)
real_vis = sgi_run.is_visible
sgi_run.is_visible = lambda *a, **k: True
try:
    for sid, (ref, folder, fp) in sorted(mine.items()):
        sc = H.scene(ref, sid)
        obs = sc.observation
        th = H.recorded_config(ref)[1].thresholds
        assert th.person_mark_height_m == 1.1
        entry = table.get(fp)
        ok_lookup = entry is not None and entry["scene_id"] == sid
        e = ereal[sid][1]
        src = D.DetectorPeople(sid, fp, obs.sha256(), e["people"], th)
        got = src(obs)
        ordered = sorted(e["people"], key=lambda p: -p["score"])
        truth = types.SimpleNamespace(people=[types.SimpleNamespace(x=p["x"], y=p["y"], yaw=p["yaw"]) for p in ordered])
        oracle = sgi_run.OraclePeople(truth, th)(obs)
        as_tuple = lambda f: (tuple(map(float, f.xy)), f.yaw, tuple(map(float, f.pixel)), f.label)  # noqa: E731
        equal_oracle = [as_tuple(f) for f in got.found] == [as_tuple(f) for f in oracle.found]
        equal_file = [(f.xy[0], f.xy[1], f.yaw) for f in got.found] == [(p["x"], p["y"], p["yaw"]) for p in ordered]
        gap = max((math.dist(f.pixel, p["mark_pixel"]) for f, p in zip(got.found, ordered)
                   if p["mark_pixel"] is not None), default=0.0)
        inside = all(in_image(obs.camera, *f.pixel) for f in got.found)
        row = {"lookup": ok_lookup, "n": len(got.found), "n_file": len(e["people"]), "equal_oracle": equal_oracle,
               "equal_file_order": equal_file, "max_gap_a1_mark_px": gap, "all_in_image": inside,
               "calls": len(got.calls), "call_time": got.call_time_s,
               "yaw_none": sum(f.yaw is None for f in got.found)}
        res["scenes"][sid] = row
        if not (ok_lookup and equal_oracle and equal_file and len(got.found) == len(e["people"]) and inside
                and not got.calls):
            res["fails"].append(sid)
finally:
    sgi_run.is_visible = real_vis

res["people_total"] = sum(r["n"] for r in res["scenes"].values())
res["max_gap_px"] = max(r["max_gap_a1_mark_px"] for r in res["scenes"].values())

# refusal 1: another scene's observation
a, b = "real_eval_08_waiting_line_4", "real_eval_09_waiting_line_6"
ref_a = mine[a][0]
th = H.recorded_config(ref_a)[1].thresholds
src = D.DetectorPeople(a, mine[a][2], H.scene(ref_a, a).observation.sha256(), ereal[a][1]["people"], th)
try:
    src(H.scene(mine[b][0], b).observation)
    res["refuse_other_observation"] = "NOT REFUSED"
except SystemExit as exc:
    res["refuse_other_observation"] = str(exc)

# refusal 2: a scene with no entry in the table (its fingerprint removed), through run_arm in mock mode
sgi_run._given_inferrer = D._given_inferrer_detector
D._forbid_keys("verifier test")
cut = {k: v for k, v in table.items() if v["scene_id"] != a}
try:
    D.run_arm("glm53", "real", "mock", V / "unknown_scene_test", None, None, cut, D.Tally(), only=[a])
    res["refuse_unknown_scene"] = "NOT REFUSED"
except SystemExit as exc:
    res["refuse_unknown_scene"] = str(exc)
# refusal 3: a table entry whose fingerprint maps to another scene id
swapped = dict(table)
swapped[mine[a][2]] = {**table[mine[b][2]]}
try:
    D.run_arm("glm53", "real", "mock", V / "unknown_scene_test", None, None, swapped, D.Tally(), only=[a])
    res["refuse_wrong_entry"] = "NOT REFUSED"
except SystemExit as exc:
    res["refuse_wrong_entry"] = str(exc)

(V / "verify_detector_people.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != "scenes"}, indent=1))
print("yaw None total", sum(r["yaw_none"] for r in res["scenes"].values()))
