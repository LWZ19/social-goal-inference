"""I09 fix, diagnosis step 4: INDICATIVE what-ifs for the candidate fixes of DIAGNOSIS.md (expected effects only;
nothing here is chosen, tuned or frozen; fit-validate is not read).

Every what-if starts from A1's own candidates (P.candidates at tau 0.25, the frozen setting, F5) and changes one
step, then scores with I03's scorer unchanged (C.score_scene: exempt_points, facing full):
  base      A1 as frozen (reproduces score_fit.py)
  relift    occlusion re-lift: when the nearest confident head keypoint (nose, eyes, ears; 5x5 depth) lies more
            than 0.30 m behind the torso median Z_med, the position is the median world xy of the head keypoints
            within 0.30 m of that nearest head depth (then the usual 0.20 m push)
  imgdup    duplicate rule widened in image space: also a duplicate when the box lies >= 0.80 inside a kept box
            (intersection over the smaller box) and the world distance is <= 1.0 m
  imgonly   duplicate rule in image space only: a duplicate iff containment >= 0.50 and world distance <= 1.0 m
            (the 0.50 m world-only rule is dropped)
  dup030    A1's other grid value, d_dup 0.30 m
  tau035    score threshold 0.35
  flat      A1's flat test on (share >= 0.80 of the box's points within 0.04 m of the torso plane)
  and combinations.
Data: fit-train (seeds 110-115, 84 images; default thresholds); A1's fit data (F-real 21 worked examples, F-ph 268
placeholder scenes, RT-DETRv2 raw from a1/raw_fit; fitdata.load); A1's development E-real (42 scenes, a1/raw_eval,
through the harness as c2_repro.py). Writes whatif.txt. Project venv.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import collections  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import statistics  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fitlib as F  # noqa: E402
from sgi.geometry import _depth_at, lift  # noqa: E402

P, PS, C, H = F.P, F.PS, F.C, F.H
sys.path.insert(0, str(F.I03 / "stageA"))
import fitdata as FD  # noqa: E402

HEAD = (0, 1, 2, 3, 4)
GAP_M = 0.30
L: list[str] = []


def relift_xy(c, kp, depth, cam):
    """New body_xy for an occluded torso, or None when the rule does not fire."""
    if kp is None or c.get("reject") is not None:
        return None
    zs = []
    for j in HEAD:
        if kp[j][2] >= P.CONF:
            z = _depth_at(depth, cam, kp[j][0], kp[j][1], P.KP_WINDOW)
            if z is not None:
                zs.append((z, j))
    if not zs:
        return None
    zmin = min(z for z, _ in zs)
    if zmin - c["zmed"] <= GAP_M:
        return None
    pts = [lift(depth, cam, kp[j][0], kp[j][1], P.KP_WINDOW) for z, j in zs if z - zmin <= GAP_M]
    pts = [p for p in pts if p is not None]
    return (float(np.median([p[0] for p in pts])), float(np.median([p[1] for p in pts]))) if pts else None


def containment(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    if ix * iy <= 0:
        return 0.0
    return ix * iy / min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))


def people(cands, s, cam_xy, variant, opts, relifted):
    """P.people_for with the what-if options (same arithmetic when opts is empty)."""
    lo, hi = P.BANDS[s["band"]]
    tau = opts.get("tau", s["tau"])
    flat = opts.get("flat", s["flat"])
    dup = opts.get("dup", "world")
    d_dup = opts.get("d_dup", s["d_dup"])
    kept = []
    for c in cands:
        if c["score"] < tau:
            break
        if c["reject"] is not None:
            continue
        if c["z_top"] > hi or (not c["top_touch"] and c["z_top"] < lo):
            continue
        if flat and c["flat"]:
            continue
        body = relifted.get(c["index"], c["body_xy"]) if opts.get("relift") else c["body_xy"]
        xy = P.push(body, cam_xy, s["d_push"])
        is_dup = False
        for k in kept:
            wd = math.dist(xy, k["xy"])
            ct = containment(c["box"], k["box"])
            if dup == "world":
                is_dup = wd <= d_dup
            elif dup == "world_or_img":
                is_dup = wd <= d_dup or (ct >= 0.80 and wd <= 1.0)
            elif dup == "img":
                is_dup = ct >= 0.50 and wd <= 1.0
            if is_dup:
                break
        if is_dup:
            continue
        kept.append({"xy": xy, "yaw": c["yaw"][variant], "yaws": c["yaw"], "score": c["score"], "box": c["box"],
                     "torso_pixel": c["torso_pixel"], "index": c["index"], "facing_diag": c["facing_diag"]})
    return kept


WHATIFS = {
    "base": {},
    "relift": {"relift": True},
    "imgdup": {"dup": "world_or_img"},
    "imgonly": {"dup": "img"},
    "dup030": {"d_dup": 0.30},
    "tau035": {"tau": 0.35},
    "flat": {"flat": True},
    "relift+imgdup": {"relift": True, "dup": "world_or_img"},
    "relift+imgonly": {"relift": True, "dup": "img"},
    "relift+imgonly+tau035": {"relift": True, "dup": "img", "tau": 0.35},
    "relift+imgonly+flat": {"relift": True, "dup": "img", "flat": True},
}


def blank():
    return {"rows": [], "prows": [], "fired": 0, "fired_matched_before": 0}


def run_scene(acc, raw, sc, th, s, variant):
    obs = sc.observation
    cam = obs.camera
    cam_xy = cam.T_world_camera[:2, 3]
    cands = P.candidates(raw, obs.depth, cam, s["tau"])
    kp_of = {p["box_index"]: p["kp"] for p in raw["pose"]}
    relifted = {}
    for c in cands:
        nxy = relift_xy(c, kp_of.get(c["index"]), obs.depth, cam)
        if nxy is not None:
            relifted[c["index"]] = nxy
    ref = P.people_for(cands, s, cam_xy, variant)
    for name, opts in WHATIFS.items():
        ppl = people(cands, s, cam_xy, variant, opts, relifted)
        if name == "base":
            assert [(p["index"], p["xy"]) for p in ppl] == [(p["index"], p["xy"]) for p in ref]
        r, prow = C.score_scene(ppl, sc, th, variant)
        acc[name]["rows"].append(r)
        acc[name]["prows"] += prow
        if name == "relift":
            nums = {p["index"]: n + 1 for n, p in enumerate(ref)}
            r0, _ = C.score_scene(ref, sc, th, variant)
            for i in relifted:
                if i in nums:
                    acc[name]["fired"] += 1
                    acc[name]["fired_matched_before"] += nums[i] in r0["matches"]


def summary(a):
    rows, prows = a["rows"], a["prows"]
    Pp = PS.pool(rows)
    role = {c: (sum(1 for p in prows if p["role_class"] == c and p["matched"]),
                sum(1 for p in prows if p["role_class"] == c)) for c in C.ROLE_CLASSES}
    m = [p for p in prows if p["matched"]]
    np_e = [p["err"] for p in m if p["view"] != "profile"]
    fs = C.facing_summary(np_e)
    return (f"recall {Pp['matched']}/{Pp['n_visible']} = {Pp['recall']:.3f} (line {role['line'][0]}/{role['line'][1]}, "
            f"staff {role['staff'][0]}/{role['staff'][1]}, group {role['group'][0]}/{role['group'][1]}, other "
            f"{role['other'][0]}/{role['other'][1]}); false {Pp['false']}/{Pp['scenes']} = {Pp['fb_per_scene']:.3f}; "
            f"exempt {Pp['exempt']}; merged {Pp['merged_pairs']}/{Pp['neighbour_pairs']}; xy median "
            f"{Pp['xy_err_median']:.3f}; non-profile w45 {fs['w45']}/{fs['n']}")


def main():
    s, variant = F.frozen_setting()
    for sha_file in ("MANIFEST_FIT.sha256", "RAW_FIT.sha256"):
        for line in (F.HERE / sha_file).read_text().splitlines():
            want, name = line.split()
            assert F.sha(F.HERE / name) == want, name
    for path, want in F.DEV_RAW.items():
        assert F.sha(path) == want, path
    datasets = {}
    # fit-train
    acc = {n: blank() for n in WHATIFS}
    raw = F.raw_lines(F.HERE / "raw_fit/rtdetr_v2_r50vd.jsonl")
    th = PS.default_thresholds()
    for e in F.fit_manifest()["images"]:
        if e["domain"] != "fit-train":
            continue
        run_scene(acc, raw[e["rgb"]], F.load_fit_scene(e["folder"]), th, s, variant)
    datasets["fit-train (seeds 110-115, 84)"] = acc
    # A1 fit data
    man = json.loads((F.A1 / "manifest_fit.json").read_text())
    meta = json.loads((F.A1 / "raw_fit/meta.json").read_text())
    assert meta["manifest_sha256"] == F.sha(F.A1 / "manifest_fit.json")
    rawf = {(x["set"], x["scene_id"]): x for x in map(json.loads, (F.A1 / "raw_fit/rtdetr_v2_r50vd.jsonl")
                                                       .read_text().splitlines())}
    accs = {"F-real": {n: blank() for n in WHATIFS}, "F-ph": {n: blank() for n in WHATIFS}}
    refs = FD.fit_refs()
    assert [(r[1], r[2]) for r in refs] == [(i["set"], i["scene_id"]) for i in man["images"]]
    for domain, sset, scene_id, loader in refs:
        sc, thf = FD.load(loader)
        run_scene(accs[domain], rawf[(sset, scene_id)], sc, thf, s, variant)
    datasets["A1 F-real (seed 0 worked examples, 21)"] = accs["F-real"]
    datasets["A1 F-ph (placeholder, 268)"] = accs["F-ph"]
    # A1 development E-real
    mane = json.loads((F.A1 / "manifest_eval.json").read_text())
    rawe = {x["rgb"]: x for x in map(json.loads, (F.A1 / "raw_eval/rtdetr_v2_r50vd.jsonl").read_text().splitlines())}
    by_folder = {e["folder"]: e for e in mane["images"]}
    acc = {n: blank() for n in WHATIFS}
    n_dev = 0
    for sref in H.distinct_scenes():
        e = by_folder[str(sref.folder)]
        if e["domain"] != "E-real":
            continue
        n_dev += 1
        run_scene(acc, rawe[e["rgb"]], H.scene(sref.runs[0], sref.scene_id), C.thresholds_for(sref.runs), s,
                  variant)
    assert n_dev == 42
    datasets["A1 dev E-real (42)"] = acc
    L.append("INDICATIVE what-ifs (not a chosen, tuned or frozen fix). A1's candidates, one step changed, I03's scorer "
             "unchanged. Fit-validate (seeds 116-119) not read.")
    for dname, acc in datasets.items():
        L.append("")
        L.append(f"== {dname}")
        for name in WHATIFS:
            extra = ""
            if name == "relift":
                extra = (f" [rule fired on {acc[name]['fired']} kept boxes, of which {acc[name]['fired_matched_before']}"
                         f" were matched before]")
            L.append(f"  {name:24s} {summary(acc[name])}{extra}")
    (F.HERE / "whatif.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
