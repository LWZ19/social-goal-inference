"""I09 fix, diagnosis step 2: score fit-train (seeds 110-115, 84 images) with A1's frozen method and I03's unchanged
scorer, and record every candidate box's fate. Fit-validate (116-119) is not read.

Per image: P.candidates(raw, depth, cam, 0.25) and a trace of P.people_for's steps (asserted equal to P.run_image),
then C.score_scene (PS.score_people with exempt_xy=PS.exempt_points(truth), facing "full", default thresholds),
variant F5, setting chosen.json. The people are fixed before truth is read. Writes fit_train_rows.jsonl (per image),
fit_train_people.jsonl (per visible person), fit_train_dets.jsonl (per candidate box >= 0.25 with its fate and,
for kept ones, the match), and fit_train_summary.json. Project venv.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import collections  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import statistics  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fitlib as F  # noqa: E402
from sgi.geometry import in_image, project  # noqa: E402

C, P, PS, H = F.C, F.P, F.PS, F.H
MARK_Z = 1.1


def verify_inputs():
    for sha_file in ("MANIFEST_FIT.sha256", "RAW_FIT.sha256"):
        for line in (F.HERE / sha_file).read_text().splitlines():
            want, name = line.split()
            assert F.sha(F.HERE / name) == want, name


def trace_people(cands, s, cam_xy, variant):
    """P.people_for's steps, with each candidate's fate recorded (same arithmetic; output asserted equal)."""
    lo, hi = P.BANDS[s["band"]]
    kept, fates = [], []
    for c in cands:
        if c["score"] < s["tau"]:
            break
        if c["reject"] is not None:
            fates.append({"index": c["index"], "fate": "depth: " + c["reject"]})
            continue
        if c["z_top"] > hi:
            fates.append({"index": c["index"], "fate": "z_top above band"})
            continue
        if not c["top_touch"] and c["z_top"] < lo:
            fates.append({"index": c["index"], "fate": "z_top below band"})
            continue
        if s["flat"] and c["flat"]:
            fates.append({"index": c["index"], "fate": "flat"})
            continue
        xy = P.push(c["body_xy"], cam_xy, s["d_push"])
        near = [(math.dist(xy, k["xy"]), k["index"]) for k in kept]
        dup = [t for t in near if t[0] <= s["d_dup"]]
        if dup:
            fates.append({"index": c["index"], "fate": "duplicate", "dup_of": min(dup)[1], "dup_d": min(dup)[0],
                          "xy": xy})
            continue
        fates.append({"index": c["index"], "fate": "kept", "xy": xy,
                      "nearest_kept_d": min(near)[0] if near else None,
                      "nearest_kept_index": min(near)[1] if near else None})
        kept.append({"xy": xy, "yaw": c["yaw"][variant], "yaws": c["yaw"], "score": c["score"], "box": c["box"],
                     "torso_pixel": c["torso_pixel"], "index": c["index"], "facing_diag": c["facing_diag"]})
    return kept, fates


def best_raw_box(raw, cam, x, y):
    mk = project(cam, (x, y, MARK_Z))
    if mk is None or not in_image(cam, mk[0], mk[1]):
        return None
    for i, b in enumerate(raw["boxes"]):
        if b[0] <= mk[0] <= b[2] and b[1] <= mk[1] <= b[3]:
            return {"index": i, "score": b[4], "box": b[:4]}
    return None


def jsonable(x):
    import numpy as np
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def main():
    verify_inputs()
    s, variant = F.frozen_setting()
    man = F.fit_manifest()
    raw = F.raw_lines(F.HERE / "raw_fit/rtdetr_v2_r50vd.jsonl")
    th = PS.default_thresholds()
    PS.check_thresholds(th)
    rows, prows, drows = [], [], []
    for e in man["images"]:
        if e["domain"] != "fit-train":
            continue
        assert e["seed"] in F.PG.FIT_TRAIN
        F.PG.check(e["rgb"])
        x = raw[e["rgb"]]
        assert x["sha256"] == e["sha256"]
        sc = F.load_fit_scene(e["folder"])
        obs = sc.observation
        cam = obs.camera
        cam_xy = cam.T_world_camera[:2, 3]
        assert F.sha(Path(e["folder"]) / "depth.npy") == e["sha256_depth"]
        cands = P.candidates(x, obs.depth, cam, s["tau"])
        ppl, fates = trace_people(cands, s, cam_xy, variant)
        ref = P.run_image(x, obs.depth, cam, s, variant)
        assert [(p["index"], p["xy"]) for p in ppl] == [(p["index"], p["xy"]) for p in ref]
        # ---- the people are fixed above; truth is read only from here on
        r, prow = C.score_scene(ppl, sc, th, variant)
        info = PS.truth_info(sc, th)
        vis, hid, lh, roles = info["visible"], info["hidden"], info["lower_hidden"], info["roles"]
        meta = {"seed": e["seed"], "row_id": e["row_id"], "scene_id": e["scene_id"], "folder": e["folder"]}
        num_of = {p["index"]: n + 1 for n, p in enumerate(ppl)}
        kp_of = {p["box_index"]: p["kp"] for p in x["pose"]}
        cand_of = {c["index"]: c for c in cands}
        for fa in fates:
            c = cand_of[fa["index"]]
            d = dict(meta) | fa | {k: c.get(k) for k in ("box", "score", "region", "region_valid", "zmed", "kept",
                                                          "body_xy", "torso_pixel", "z_top", "top_touch",
                                                          "flat_share", "facing_diag", "yaw")}
            d["kp"] = kp_of.get(fa["index"])
            if fa["fate"] == "kept":
                n = num_of[fa["index"]]
                d["number"] = n
                d["matched_to"] = r["matches"].get(n)
                d["false"] = n in r["false_numbers"]
                d["exempt"] = n in r["exempt_numbers"]
                xy = fa["xy"]
                d["near_visible"] = sorted(([math.dist(xy, (p.x, p.y)), p.id, roles[p.id], bool(lh[p.id])]
                                            for p in vis), key=lambda t: t[0])[:3]
                d["near_hidden"] = sorted(([math.dist(xy, (p.x, p.y)), p.id, p.role] for p in hid),
                                          key=lambda t: t[0])[:3]
            drows.append(jsonable(d))
        for pr in prow:
            extra = {"lower_hidden": bool(lh[pr["pid"]])}
            if not pr["matched"]:
                extra["best_raw_box"] = best_raw_box(x, cam, pr["x"], pr["y"])
                near = sorted((math.dist(p["xy"], (pr["x"], pr["y"])), num_of[p["index"]]) for p in ppl)
                extra["nearest_kept"] = near[0] if near else None
            prows.append(jsonable(dict(meta) | pr | extra))
        hidden_rows = []
        for p in hid:
            mk = project(cam, (p.x, p.y, MARK_Z))
            hidden_rows.append({"pid": p.id, "role": p.role, "x": p.x, "y": p.y,
                                "mark_in_picture": bool(mk is not None and in_image(cam, mk[0], mk[1]))})
        rows.append(jsonable(dict(meta) | {k: r[k] for k in PS.POOL_KEYS} | {
            "recall": r["recall"], "xy_err": r["xy_err"], "facing": r["facing"], "misses": r["misses"],
            "false_numbers": r["false_numbers"], "exempt_numbers": r["exempt_numbers"], "matches": r["matches"],
            "pairs": r["pairs"], "people": r["people"], "n_cands": len(cands), "n_kept": len(ppl), "hidden": hidden_rows,
            "fates": dict(collections.Counter(f["fate"].split(":")[0] for f in fates))}))
    Pp = PS.pool(rows)
    by_role = {c: [sum(1 for p in prows if p["role_class"] == c and p["matched"]),
                   sum(1 for p in prows if p["role_class"] == c)] for c in C.ROLE_CLASSES}
    m = [p for p in prows if p["matched"]]
    np_errs = [p["err"] for p in m if p["view"] != "profile"]
    pf_errs = [p["err"] for p in m if p["view"] == "profile"]
    summ = {"images": Pp["scenes"], "visible": Pp["n_visible"], "matched": Pp["matched"], "recall": Pp["recall"],
            "by_role": by_role, "false": Pp["false"], "fb_per_image": Pp["fb_per_scene"], "exempt": Pp["exempt"],
            "near_hidden": Pp["near_hidden"], "merges": Pp["merges"], "merged_pairs": Pp["merged_pairs"],
            "neighbour_pairs": Pp["neighbour_pairs"], "xy_median": Pp["xy_err_median"], "n_det": Pp["n_det"],
            "hidden_people": Pp["n_hidden_people"],
            "np_facing": C.facing_summary(np_errs), "profile_facing": C.facing_summary(pf_errs),
            "all_facing": C.facing_summary([p["err"] for p in m])}
    for name, obj in (("fit_train_rows.jsonl", rows), ("fit_train_people.jsonl", prows),
                      ("fit_train_dets.jsonl", drows)):
        (F.HERE / name).write_text("".join(json.dumps(o) + "\n" for o in obj))
    F.C.jdump(summ, F.HERE / "fit_train_summary.json")
    print(json.dumps({k: v for k, v in summ.items() if not k.endswith("facing")}))
    print("np facing", summ["np_facing"]["w45"], summ["np_facing"]["n"], summ["np_facing"]["median_u"],
          "profile fb", summ["profile_facing"]["fb"], summ["profile_facing"]["n"])


if __name__ == "__main__":
    main()
