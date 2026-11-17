"""Independent verifier of I09 fix 2: my own implementation of FIX_PROTOCOL.md sections 2-3 and 5-8, written from the
protocol's text. It imports only the objects under test (A1's frozen pipeline.py, I03's people_scorer.py), the
step-0 harness and sgi; never the fix agent's fitlib.py / fix/*.py / score_fit.py. Project venv, torch-free.
Writes nothing (the caller writes into verify/)."""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402

WORKTREE = paths.REPO
A1 = paths.STILL / "people-finder/early-detector/phase1c/a1"
I03 = paths.STILL / "perception/depth-blob-detector"
FIXDIR = paths.STILL / "people-finder/early-detector/fix"
SEEDS = paths.STILL / "scene-seeds/phase3/realistic_train"
HARNESS_DIR = WORKTREE / "results/still-scenes/analysis-scripts/harness"
ROWS = ("real_eval_00_no_one_place", "real_eval_01_no_one_place", "real_eval_02_no_one_place",
        "real_eval_03_no_one_deep_place", "real_eval_04_no_one_object", "real_eval_05_no_one_person",
        "real_eval_06_waiting_line_1", "real_eval_07_waiting_line_2", "real_eval_08_waiting_line_4",
        "real_eval_09_waiting_line_6", "real_eval_10_conversation_group_2", "real_eval_11_conversation_group_3",
        "real_eval_12_conversation_group_4", "real_eval_13_waiting_line_3")
TRAIN, VALID = tuple(range(110, 116)), tuple(range(116, 120))

# The pins A1 and I03 carry in the step-0 records (values copied from phase2/i09/confirm FROZEN pins as the fix step
# restated them; checked here, and reported if different).
PINS = {
    A1 / "pipeline.py": "4826be88d6939f3f5e4d14375a2fc9698abd94973ca46243d6697e263e5d063a",
    A1 / "common.py": "cdc33d742f6db5d09c3ad9a4557c1210ef10f8909484cee101ce983f9f8b2f53",
    I03 / "people_scorer.py": "e12776eaddac1de0af75b58f8a272cd0a13332596da626bcc81b085f5900d685",
    A1 / "sidecar/infer.py": "bee1322fd692bf0e85e820871413d486fda04d5770f72b5d3d75e18266709484",
    A1 / "chosen.json": "19b1e16340d810e75f8f772eeeabeaf4b788e09614e74d78d3afc206531119ff",
    A1 / "chosen_fix.json": "fe8177165773f7208d417227c358adae7d04326bd29cb6c75d73ed87b3f83f52",
    WORKTREE / "configs/default.yaml": "c651f4c484fff15d33290e13f28bc53c9221c8edd79c0b7a24f24a128098b4fc",
    A1 / "raw_eval/rtdetr_v2_r50vd.jsonl": "8a03b82ca89bb82bdd0b10a8e56944b8562f3a580967087646e408dd4862a97d",
    A1 / "manifest_eval.json": "6c321c3800d0b76632da0711a2385dce4915a28e9cd1f563ff3f8a5782be0b22",
}


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


PIN_RESULT = {str(p): (sha(p), want) for p, want in PINS.items()}
for _p, (_got, _want) in PIN_RESULT.items():
    assert _got == _want, f"pin mismatch {_p}: {_got} != {_want}"

sys.path.insert(0, str(HARNESS_DIR))
sys.path.insert(0, str(I03))
sys.path.insert(0, str(A1))
import people_scorer as PS  # noqa: E402
import pipeline as P  # noqa: E402
import research_harness as H  # noqa: E402
from sgi.geometry import _depth_at, in_image, lift, project, wrap_angle  # noqa: E402
from sgi.gt.derive import is_visible  # noqa: E402

assert Path(P.__file__).resolve() == A1 / "pipeline.py"
assert Path(PS.__file__).resolve() == I03 / "people_scorer.py"
assert Path(H.__file__).resolve() == HARNESS_DIR / "research_harness.py"
assert "common" not in sys.modules or Path(sys.modules["common"].__file__).resolve() == A1 / "common.py"
for _m in ("fitlib", "fixlib", "score_fit", "select_candidates", "confirm_chosen"):
    assert _m not in sys.modules, _m


# ------------------------------------------------------------------------------------------------ fit scenes
_FORBID = re.compile(r"phase2/seeds|scene-seeds/phase2|held_?out|keep/heldout|(^|[_/])s18|s20_held|eval_1[6-9]|"
                     r"group_photo|lift_wait", re.I)


def fit_folder(seed: int, row: str) -> Path:
    assert seed in TRAIN + VALID and row in ROWS
    f = SEEDS / f"seed{seed}" / "eval" / row
    for s in (str(f), os.path.realpath(f)):
        assert not _FORBID.search(s), s
        assert s.startswith(str(SEEDS) + "/"), s
        nums = [int(m) for m in re.findall(r"/seed(\d+)/", s + "/")]
        assert nums == [seed], (s, nums)
    return f


def load_fit(seed: int, row: str):
    from sgi.scene import load_scene
    f = fit_folder(seed, row)
    H.guard_scene(f)
    return f, PS.FreshScene(load_scene(f))


def frozen_setting():
    ch = json.loads((A1 / "chosen.json").read_text())
    fx = json.loads((A1 / "chosen_fix.json").read_text())
    assert ch["setting"] == fx["setting"], (ch["setting"], fx["setting"])
    assert fx["fix_adopted"] is True
    return dict(fx["setting"]), fx["facing_variant"]


# ------------------------------------------------------------------------------------------------ blocks
GAP = 0.30
HEADS = (0, 1, 2, 3, 4)
BLOCKS = {"C00": (), "C01": ("HR",), "C02": ("BR",), "C03": ("HR", "d030"), "C04": ("HR", "IMG"),
          "C05": ("HR", "d030", "FLAT"), "C06": ("HR", "IMG", "FLAT"), "C07": ("HR", "d030", "FLAT", "TINY"),
          "C08": ("HR", "IMG", "FLAT", "TINY"), "C09": ("BR", "d030", "FLAT", "TINY"),
          "C10": ("BR", "IMG", "FLAT", "TINY")}
NPARAM = {"HR": 1, "BR": 1, "d030": 1, "IMG": 2, "FLAT": 1, "TINY": 1}   # FIX_PROTOCOL.md section 3


def z_top_at(box, cam, Z):
    uc = 0.5 * (box[0] + box[2])
    pc = np.array([(uc - cam.cx) * Z / cam.fx, (box[1] - cam.cy) * Z / cam.fy, Z, 1.0])
    return float((cam.T_world_camera @ pc)[2])


def a1_region(box, kp, depth, cam):
    """A1's torso region (hull of LS, RS, RH, LH when all are confident and >= 20 valid, else the central
    rectangle), rebuilt from A1's helpers: (world points, u, v, Z, name)."""
    Hh, W = depth.shape
    if kp is not None and all(kp[j][2] >= P.CONF for j in (5, 6, 12, 11)):
        cols, rows = P.hull_pixels(P.convex_hull([(kp[j][0], kp[j][1]) for j in (5, 6, 12, 11)]), Hh, W)
        if len(cols):
            pts, u, v, Z = P.lift_pixels(depth, cam, cols, rows)
            if len(Z) >= 20:
                return pts, u, v, Z, "hull"
    x0, y0, x1, y1 = box[:4]
    w, h = x1 - x0, y1 - y0
    cols, rows = P.rect_pixels(x0 + 0.3 * w, x0 + 0.7 * w, y0 + 0.2 * h, y0 + 0.5 * h, Hh, W)
    pts, u, v, Z = P.lift_pixels(depth, cam, cols, rows)
    return pts, u, v, Z, "box"


def head_fire(c, kp, depth, cam):
    if kp is None or c["reject"] is not None:
        return None
    zs = []
    for j in HEADS:
        if kp[j][2] >= 0.30:
            z = _depth_at(depth, cam, kp[j][0], kp[j][1], 5)
            if z is not None:
                zs.append((j, z))
    if not zs:
        return None
    zh = min(z for _, z in zs)
    if not (zh - c["zmed"] > GAP):
        return None
    return zh, [j for j, z in zs if z <= zh + GAP]


def relift(c, kp, depth, cam, mode):
    """New fields for a candidate on which HR/BR fires, or None."""
    fire = head_fire(c, kp, depth, cam)
    if fire is None:
        return None
    zh, hs = fire
    wpts = [lift(depth, cam, kp[j][0], kp[j][1], 5) for j in hs]
    hr_body = (float(np.median([p[0] for p in wpts])), float(np.median([p[1] for p in wpts])))
    hr_tp = (float(np.median([kp[j][0] for j in hs])), float(np.median([kp[j][1] for j in hs])))
    body, tp, zref, how = hr_body, hr_tp, zh, "HR"
    if mode == "BR":
        how = "BR->HR"
        pts, u, v, Z, name = a1_region(c["box"], kp, depth, cam)
        assert name == c["region"] and len(Z) == c["region_valid"]
        S = np.abs(Z - zh) <= 0.30
        if S.sum() >= 10:
            z2 = float(np.median(Z[S]))
            K = np.abs(Z - z2) <= 0.30
            if K.sum() >= 10:
                body = (float(np.median(pts[K, 0])), float(np.median(pts[K, 1])))
                tp = (float(np.median(u[K])), float(np.median(v[K])))
                zref, how = z2, "BR"
    cam_xy = cam.T_world_camera[:2, 3]
    yaws, _ = P.facing(kp, depth, cam, zref, body, cam_xy)
    return {"body_xy": body, "torso_pixel": tp, "z_top": z_top_at(c["box"], cam, zref), "yaw": yaws, "how": how,
            "z_h": zh, "z_ref": zref}


def push(xy, cam_xy, d):
    dx, dy = xy[0] - cam_xy[0], xy[1] - cam_xy[1]
    L = math.hypot(dx, dy)
    if L <= 0 or d == 0:
        return (float(xy[0]), float(xy[1]))
    return (float(xy[0] + d * dx / L), float(xy[1] + d * dy / L))


def contain(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    return (w * h) / min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))


def people(cands, kps, depth, cam, s, variant, blocks, relifts):
    lo, hi = P.BANDS[s["band"]]
    cam_xy = cam.T_world_camera[:2, 3]
    mode = "HR" if "HR" in blocks else "BR" if "BR" in blocks else None
    out = []
    for c in cands:
        if c["score"] < s["tau"]:
            break
        if c["reject"] is not None:
            continue
        rl = relifts[mode].get(c["index"]) if mode else None
        z_top = rl["z_top"] if rl else c["z_top"]
        if z_top > hi or (not c["top_touch"] and z_top < lo):
            continue
        if (s["flat"] or "FLAT" in blocks) and c["flat"]:
            continue
        if "TINY" in blocks and c["n_Q"] < 30:
            continue
        body = rl["body_xy"] if rl else c["body_xy"]
        xy = push(body, cam_xy, s["d_push"])
        if "IMG" in blocks:
            dup = any(contain(c["box"], k["box"]) >= 0.50 and math.dist(xy, k["xy"]) <= 1.00 for k in out)
        else:
            dd = 0.30 if "d030" in blocks else s["d_dup"]
            dup = any(math.dist(xy, k["xy"]) <= dd for k in out)
        if dup:
            continue
        yaws = rl["yaw"] if rl else c["yaw"]
        out.append({"xy": xy, "yaw": yaws[variant], "index": c["index"], "box": c["box"], "score": c["score"],
                    "m1": rl["how"] if rl else None})
    return out


def run_image(raw, depth, cam, s, variant, ids):
    cands = P.candidates(raw, depth, cam, s["tau"])
    kps = {p["box_index"]: p["kp"] for p in raw["pose"]}
    relifts = {}
    for mode in ("HR", "BR"):
        if any(mode in BLOCKS[i] for i in ids):
            relifts[mode] = {}
            for c in cands:
                r = relift(c, kps.get(c["index"]), depth, cam, mode)
                if r is not None:
                    relifts[mode][c["index"]] = r
    res = {i: people(cands, kps, depth, cam, s, variant, BLOCKS[i], relifts) for i in ids}
    ref = P.run_image(raw, depth, cam, s, variant)
    if "C00" in res:
        mine = [(p["index"], p["xy"], p["yaw"]) for p in res["C00"]]
        theirs = [(p["index"], p["xy"], p["yaw"]) for p in ref]
        assert mine == theirs, "C00 != P.run_image"
    return res


# ------------------------------------------------------------------------------------------------ scoring
def view_class(p, cam_xy):
    b = math.atan2(cam_xy[1] - p.y, cam_xy[0] - p.x)
    a = abs(math.degrees(wrap_angle(b - p.yaw)))
    return "frontal" if a < 45 else "oblique" if a < 75 else "profile" if a <= 105 else "back"


def hidden_staff_marks(sc, th):
    obs, cam = sc.observation, sc.observation.camera
    out = []
    for p in sc.truth.people:
        if not p.role or is_visible(p, obs, th):
            continue
        m = project(cam, (p.x, p.y, 1.1))
        if m is not None and in_image(cam, m[0], m[1]):
            out.append((p.id, p.x, p.y))
    return out


def score(ppl, sc, th, ref=None, ref_matched_idx=None):
    """One candidate on one scene: my per-image record."""
    pts = [(p["xy"][0], p["xy"][1], p["yaw"]) for p in ppl]
    r = PS.score_people(pts, sc, th, exempt_xy=PS.exempt_points(sc.truth), facing="full")
    cam_xy = sc.observation.camera.T_world_camera[:2, 3]
    tp = {p.id: p for p in sc.truth.people}
    vis_ids = [q["id"] for q in r["people"]]
    roles = PS.role_classes(sc.truth)
    face = []
    for num, tid in r["matches"].items():
        p = tp[tid]
        y = ppl[num - 1]["yaw"]
        e = None if y is None else abs(math.degrees(wrap_angle(y - p.yaw)))
        face.append((view_class(p, cam_xy), e))
    hs = hidden_staff_marks(sc, th)
    moved = [n for n in r["false_numbers"] if any(math.dist(ppl[n - 1]["xy"], (x, y)) <= 0.6 for _, x, y in hs)]
    # M3 audit, the fix agent's per-staff reading and the literal per-box reading
    near_ref = lost_staff = lost_box = 0
    if ref is not None:
        idx = {p["index"] for p in ppl}
        for d in ref:
            st = [(i, x, y) for i, x, y in hs if math.dist(d["xy"], (x, y)) <= 0.6]
            if not st:
                continue
            near_ref += 1
            _, x, y = st[0]
            if not any(math.dist(e["xy"], (x, y)) <= 0.6 for e in ppl):
                lost_staff += 1
            if d["index"] not in idx:
                lost_box += 1
    # post hoc classes (README table 1.5 wording)
    info = PS.truth_info(sc, th)
    hid = info["hidden"]
    vis = info["visible"]
    lh = info["lower_hidden"]
    cls = {}
    for kind, nums in (("false", r["false_numbers"]), ("exempt", r["exempt_numbers"])):
        for n in nums:
            xy = ppl[n - 1]["xy"]
            if any(p.role and lh[p.id] and math.dist(xy, (p.x, p.y)) <= 0.8 for p in vis):
                k = "A"
            elif any(math.dist(xy, (p.x, p.y)) <= 0.5 for p in hid):
                k = "B"
            elif min([math.dist(xy, (p.x, p.y)) for p in list(vis) + list(hid)] or [99]) <= 1.5:
                k = "C"
            else:
                k = "D"
            cls[f"{kind} {k}"] = cls.get(f"{kind} {k}", 0) + 1
    fired = [p for p in ppl if p["m1"]]
    return {
        "visible": r["n_visible"], "matched": r["matched"], "false": r["false"], "exempt": r["exempt"],
        "near_hidden": r["near_hidden"], "merged_pairs": r["merged_pairs"], "neighbour_pairs": r["neighbour_pairs"],
        "xy_err": r["xy_err"], "n_det": r["n_det"],
        "roles": {q["id"]: (roles[q["id"]], q["matched"]) for q in r["people"]},
        "face": face, "sec_false": r["false"] - len(moved), "sec_exempt": r["exempt"] + len(moved), "moved": len(moved),
        "m3_near": near_ref, "m3_lost_staff": lost_staff, "m3_lost_box": lost_box,
        "fired": len(fired), "fired_a1_matched": sum(p["index"] in (ref_matched_idx or set()) for p in fired),
        "classes": cls, "false_numbers": r["false_numbers"], "exempt_numbers": r["exempt_numbers"],
        "matches": r["matches"], "misses": [(m["id"], m["role_class"]) for m in r["misses"]],
        "vis_ids": vis_ids,
    }


def pool(recs):
    n = len(recs)
    vis = sum(x["visible"] for x in recs)
    mat = sum(x["matched"] for x in recs)
    role = {}
    for x in recs:
        for _, (rc, m) in x["roles"].items():
            a = role.setdefault(rc, [0, 0])
            a[0] += m
            a[1] += 1
    face = [f for x in recs for f in x["face"]]
    npf = [e for v, e in face if v != "profile"]
    pf = [e for v, e in face if v == "profile"]
    errs = sorted(e for x in recs for e in x["xy_err"])
    med = (errs[len(errs) // 2] if len(errs) % 2 else 0.5 * (errs[len(errs) // 2 - 1] + errs[len(errs) // 2])) \
        if errs else None
    cls = {}
    for x in recs:
        for k, v in x["classes"].items():
            cls[k] = cls.get(k, 0) + v
    false = sum(x["false"] for x in recs)
    sec = sum(x["sec_false"] for x in recs)
    return {
        "images": n, "visible": vis, "matched": mat, "recall": mat / vis if vis else None,
        "line": role.get("line", [0, 0]), "staff": role.get("staff", [0, 0]), "group": role.get("group", [0, 0]),
        "other": role.get("other", [0, 0]),
        "line_recall": (role["line"][0] / role["line"][1]) if role.get("line", [0, 0])[1] else None,
        "false": false, "fpi": false / n, "exempt": sum(x["exempt"] for x in recs),
        "near_hidden": sum(x["near_hidden"] for x in recs),
        "merged": sum(x["merged_pairs"] for x in recs), "pairs": sum(x["neighbour_pairs"] for x in recs),
        "xy_median": med, "n_det": sum(x["n_det"] for x in recs),
        "np_w45": sum(1 for e in npf if e is not None and e <= 45), "np_n": len(npf),
        "np_share": (sum(1 for e in npf if e is not None and e <= 45) / len(npf)) if npf else None,
        "pf_fb": sum(1 for e in pf if e is not None and e < 90), "pf_n": len(pf),
        "images_with_false": sum(1 for x in recs if x["false"] > 0),
        "sec_false": sec, "sec_exempt": sum(x["sec_exempt"] for x in recs), "sec_moved": sum(x["moved"] for x in recs),
        "m3_near": sum(x["m3_near"] for x in recs), "m3_lost_staff": sum(x["m3_lost_staff"] for x in recs),
        "m3_lost_box": sum(x["m3_lost_box"] for x in recs),
        "fired": sum(x["fired"] for x in recs), "fired_a1_matched": sum(x["fired_a1_matched"] for x in recs),
        "classes": dict(sorted(cls.items())),
        "misses": sorted(m for x in recs for m in x["misses"]),
    }


# ------------------------------------------------------------------------------------------------ intervals
def wilson95(k, n):
    z = 1.959963984540054
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = (z / d) * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return (max(0.0, c - h), min(1.0, c + h))


def _pcdf(k, mu):
    """P(X <= k), X ~ Poisson(mu), by iterated terms."""
    if k < 0:
        return 0.0
    t = math.exp(-mu)
    s = t
    for i in range(1, k + 1):
        t *= mu / i
        s += t
    return s


def garwood95(k):
    def bis(f, a, b):
        for _ in range(300):
            m = 0.5 * (a + b)
            if f(m):
                b = m
            else:
                a = m
        return 0.5 * (a + b)
    lo = 0.0 if k == 0 else bis(lambda mu: _pcdf(k - 1, mu) <= 0.975, 0.0, k + 1.0)
    hi = bis(lambda mu: _pcdf(k, mu) <= 0.025, 0.0, 10.0 * k + 50.0)
    return (lo, hi)
