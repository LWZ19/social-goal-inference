"""Verifier: non-profile facing medians (unknown = 180 deg) for the CONFIRM2_FROZEN.md section 7 comparator table,
and K3's population (all matched people within 45) on fit-validate and development, with my own code (vlib)."""
import sys
sys.dont_write_bytecode = True
import json, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V
s, variant = V.frozen_setting()
th = V.PS.default_thresholds()
raw = {x["rgb"]: x for x in map(json.loads, (V.FIXDIR / "raw_fit/rtdetr_v2_r50vd.jsonl").read_text().splitlines())}
faces = {}
def add(tag, raw_line, sc, thx):
    res = V.run_image(raw_line, sc.observation.depth, sc.observation.camera, s, variant, ["C00", "C07"])
    for cid in ("C00", "C07"):
        faces.setdefault((tag, cid), []).extend(V.score(res[cid], sc, thx)["face"])
for k in V.TRAIN + V.VALID:
    for row in V.ROWS:
        folder, sc = V.load_fit(k, row)
        add("fit-train" if k in V.TRAIN else "fit-validate", raw[str(folder / "rgb.png")], sc, th)
mane = json.loads((V.A1 / "manifest_eval.json").read_text())
rawe = {x["rgb"]: x for x in map(json.loads, (V.A1 / "raw_eval/rtdetr_v2_r50vd.jsonl").read_text().splitlines())}
byf = {e["folder"]: e for e in mane["images"]}
for sref in V.H.distinct_scenes():
    e = byf.get(str(sref.folder))
    if e is None or e["domain"] != "E-real":
        continue
    add("dev", rawe[e["rgb"]], V.H.scene(sref.runs[0], sref.scene_id), V.H.recorded_config(sref.runs[0])[1].thresholds)
out = []
for (tag, cid), f in sorted(faces.items()):
    npf = [180.0 if e is None else e for v, e in f if v != "profile"]
    allw = sum(1 for v, e in f if e is not None and e <= 45)
    out.append(f"{tag} {cid}: non-profile median {statistics.median(npf):.2f} deg (n {len(npf)}); K3 all matched within 45 "
               f"{allw}/{len(f)} = {allw / len(f):.3f}")
print("\n".join(out))
(Path(__file__).resolve().parent / "v_medians.txt").write_text("\n".join(out) + "\n")
