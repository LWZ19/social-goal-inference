"""Verifier: feasibility of CONFIRM2_FROZEN.md check C2 (development part): C00's people on A1's 42 E-real scenes
equal phase1c/i09/a1/eval_people.json within 1e-9 (x, y, yaw, order). Own code (vlib)."""
import sys
sys.dont_write_bytecode = True
import json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V
s, variant = V.frozen_setting()
ep = json.loads((V.A1 / "eval_people.json").read_text())["scenes"]
mane = json.loads((V.A1 / "manifest_eval.json").read_text())
rawe = {x["rgb"]: x for x in map(json.loads, (V.A1 / "raw_eval/rtdetr_v2_r50vd.jsonl").read_text().splitlines())}
byf = {e["folder"]: e for e in mane["images"]}
n = bad = npeople = 0
for sref in V.H.distinct_scenes():
    e = byf.get(str(sref.folder))
    if e is None or e["domain"] != "E-real":
        continue
    sc = V.H.scene(sref.runs[0], sref.scene_id)
    ppl = V.run_image(rawe[e["rgb"]], sc.observation.depth, sc.observation.camera, s, variant, ["C00"])["C00"]
    ref = ep[str(sref.folder)]["people"]
    n += 1
    npeople += len(ppl)
    ok = len(ppl) == len(ref) and all(abs(p["xy"][0] - q["x"]) <= 1e-9 and abs(p["xy"][1] - q["y"]) <= 1e-9 and
                                     ((p["yaw"] is None and q["yaw"] is None) or
                                      (p["yaw"] is not None and q["yaw"] is not None and abs(p["yaw"] - q["yaw"]) <= 1e-9))
                                     for p, q in zip(ppl, ref))
    bad += not ok
msg = f"C2 development feasibility: {n} E-real scenes, {npeople} C00 people, scenes differing from eval_people.json: {bad}"
print(msg)
(Path(__file__).resolve().parent / "v_c2.txt").write_text(msg + "\n")
