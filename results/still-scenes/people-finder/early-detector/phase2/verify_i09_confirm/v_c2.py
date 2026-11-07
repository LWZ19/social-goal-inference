"""Verifier: C2 with the verifier's own re-implementation. v_pipe.people on A1's recorded development raw pass for the
42 E-real scenes (loaded through the step-0 harness only) against a1/eval_people.json. No test data. Project venv."""
import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
A1 = Path("results/2026-10-17-beyond-vlm/phase1c/i09/a1")
sys.path.insert(0, "docs/model_notes/tools")
sys.path.insert(0, str(HERE))
import research_harness as H  # noqa: E402
import v_pipe as VP  # noqa: E402

SHAS = {"raw_eval/rtdetr_v2_r50vd.jsonl": "8a03b82ca89bb82bdd0b10a8e56944b8562f3a580967087646e408dd4862a97d",
        "eval_people.json": "ffa595453c94b444c8dd030d053d37ac103441a7e74f07e3f172836b3d0c24b3",
        "manifest_eval.json": "6c321c3800d0b76632da0711a2385dce4915a28e9cd1f563ff3f8a5782be0b22"}
for n, s in SHAS.items():
    assert hashlib.sha256((A1 / n).read_bytes()).hexdigest() == s, n
man = json.loads((A1 / "manifest_eval.json").read_text())
raw = {x["rgb"]: x for x in map(json.loads, (A1 / "raw_eval/rtdetr_v2_r50vd.jsonl").read_text().splitlines())}
ep = json.loads((A1 / "eval_people.json").read_text())
assert ep["setting"] == {"detector": "rtdetr_v2_r50vd", "tau": 0.25, "band": "wide", "flat": False, "d_dup": 0.5,
                         "d_push": 0.2} and ep["facing_variant"] == "F5"
by_folder = {e["folder"]: e for e in man["images"]}
n_sc = n_p = bad = 0
mx = 0.0
for sref in H.distinct_scenes():
    e = by_folder[str(sref.folder)]
    if e["domain"] != "E-real":
        continue
    sc = H.scene(sref.runs[0], sref.scene_id)
    obs = sc.observation
    got = VP.people(raw[e["rgb"]], obs.depth, obs.camera)
    want = ep["scenes"][str(sref.folder)]["people"]
    n_sc += 1
    if len(got) != len(want):
        bad += 1
        continue
    for g, w in zip(got, want):
        n_p += 1
        dy = 0.0 if g["yaw"] is None and w["yaw"] is None else (math.inf if (g["yaw"] is None) != (w["yaw"] is None)
                                                                 else abs(VP.wrap(g["yaw"] - w["yaw"])))
        d = max(abs(g["xy"][0] - w["x"]), abs(g["xy"][1] - w["y"]), dy)
        mx = max(mx, d)
        bad += d > 1e-9
msg = (f"v_c2: verifier re-implementation on A1's development raw pass: {n_sc} E-real scenes, {n_p} people; "
       f"mismatches beyond 1e-9: {bad}; max |diff| {mx:.2e}")
print(msg)
(HERE / "v_c2.txt").write_text(msg + "\n")
