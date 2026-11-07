"""I09 fix 2: the building blocks and candidates of FIX_PROTOCOL.md (sections 3-4), the secondary scorer variant
(section 8), the M3 audit (section 5), summaries and intervals. Project venv, torch-free; writes nothing.

Every candidate is a pure post-processing / lift change on A1's raw detector and pose outputs:
    P.candidates(raw, depth, cam, 0.25)  (A1, unchanged)
 -> m1_variant(...)                      (HR or BR re-lift of an occluded torso, or nothing)
 -> people_for(cands, blocks, ...)       (A1's steps 2-7 with d030 / IMG / FLAT / TINY where a block says so)
 -> C.score_scene (I03's scorer, unchanged).
No block reads a situation label, role, group id, scene name or truth. Truth is read only by the scorer, the
secondary scorer variant and the M3 audit, after the people are fixed.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import math  # noqa: E402
import statistics  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

FIX = Path(__file__).resolve().parent
sys.path.insert(0, str(FIX.parent))
import fitlib as F  # noqa: E402
from sgi.geometry import _depth_at, in_image, lift, project  # noqa: E402

P, PS, C, H = F.P, F.PS, F.C, F.H
assert FIX == F.HERE / "fix"

# ------------------------------------------------------------------------------------------------ constants
HEAD = (0, 1, 2, 3, 4)            # nose, left eye, right eye, left ear, right ear (COCO)
GAP_M = 0.30                      # HR/BR: trigger and head-keypoint filter
D030_M = 0.30                     # d030: world duplicate radius
IMG_CONT = 0.50                   # IMG: containment (intersection / smaller box)
IMG_CAP_M = 1.00                  # IMG: world cap
STAFF_EXEMPT_M = 0.6              # secondary scorer variant
MARK_Z = 1.1                      # the marked point's height

BLOCK_PARAMS = {"HR": 1, "BR": 1, "d030": 1, "IMG": 2, "FLAT": 1, "TINY": 1}
CANDIDATES = {
    "C00": (),
    "C01": ("HR",),
    "C02": ("BR",),
    "C03": ("HR", "d030"),
    "C04": ("HR", "IMG"),
    "C05": ("HR", "d030", "FLAT"),
    "C06": ("HR", "IMG", "FLAT"),
    "C07": ("HR", "d030", "FLAT", "TINY"),
    "C08": ("HR", "IMG", "FLAT", "TINY"),
    "C09": ("BR", "d030", "FLAT", "TINY"),
    "C10": ("BR", "IMG", "FLAT", "TINY"),
}
for _b in CANDIDATES.values():
    assert ("TINY" not in _b) or ("FLAT" in _b)
    assert not ({"HR", "BR"} <= set(_b)) and not ({"d030", "IMG"} <= set(_b))


def n_params(blocks) -> int:
    return sum(BLOCK_PARAMS[b] for b in blocks)


def m1_of(blocks):
    return "HR" if "HR" in blocks else "BR" if "BR" in blocks else None


# ------------------------------------------------------------------------------------------------ geometry
def region_pixels(box, kp, depth, cam):
    """A1's torso region for a box (pipeline.box_features), returning (pts, u, v, Z, region)."""
    Hh, W = depth.shape
    x0, y0, x1, y1 = box[:4]
    w, h = x1 - x0, y1 - y0
    if kp is not None and all(kp[j][2] >= P.CONF for j in (P.LS, P.RS, P.RH, P.LH)):
        cols, rows = P.hull_pixels(P.convex_hull([(kp[j][0], kp[j][1]) for j in (P.LS, P.RS, P.RH, P.LH)]), Hh, W)
        if len(cols):
            pts, u, v, Z = P.lift_pixels(depth, cam, cols, rows)
            if len(Z) >= P.HULL_MIN_VALID:
                return pts, u, v, Z, "hull"
    cols, rows = P.rect_pixels(x0 + 0.3 * w, x0 + 0.7 * w, y0 + 0.2 * h, y0 + 0.5 * h, Hh, W)
    if not len(cols):
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0), np.zeros(0), "box"
    pts, u, v, Z = P.lift_pixels(depth, cam, cols, rows)
    return pts, u, v, Z, "box"


def z_top_at(box, cam, z):
    """A1's z_top formula (box top centre lifted at camera depth z)."""
    x0, y0, x1, _ = box[:4]
    uc = 0.5 * (x0 + x1)
    p = np.array([(uc - cam.cx) * z / cam.fx, (y0 - cam.cy) * z / cam.fy, z, 1.0])
    return float(cam.T_world_camera[2] @ p)


def containment(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    if ix * iy <= 0:
        return 0.0
    return ix * iy / min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))


# ------------------------------------------------------------------------------------------------ M1 blocks
def head_trigger(c, kp, depth, cam):
    """(z_h, H') when the HR/BR trigger fires on candidate c, else None (FIX_PROTOCOL.md section 3)."""
    if kp is None or c["reject"] is not None:
        return None
    hs = []
    for j in HEAD:
        u, v, s = kp[j]
        if s >= P.CONF:
            z = _depth_at(depth, cam, u, v, P.KP_WINDOW)
            if z is not None:
                hs.append((z, j))
    if not hs:
        return None
    zh = min(z for z, _ in hs)
    if zh - c["zmed"] <= GAP_M:
        return None
    return zh, [j for z, j in hs if z <= zh + GAP_M]


def _hr(kp, depth, cam, zh, js):
    pts, uv = [], []
    for j in js:
        p = lift(depth, cam, kp[j][0], kp[j][1], P.KP_WINDOW)
        assert p is not None
        pts.append(p)
        uv.append((kp[j][0], kp[j][1]))
    body = (float(np.median([p[0] for p in pts])), float(np.median([p[1] for p in pts])))
    tp = (float(np.median([q[0] for q in uv])), float(np.median([q[1] for q in uv])))
    return body, tp, float(zh)


def _br(c, kp, depth, cam, zh, js):
    pts, u, v, Z, region = region_pixels(c["box"], kp, depth, cam)
    assert region == c["region"] and len(Z) == c["region_valid"] and abs(float(np.median(Z)) - c["zmed"]) < 1e-9
    S = np.abs(Z - zh) <= P.BAND_M
    if S.sum() >= P.MIN_KEPT:
        z2 = float(np.median(Z[S]))
        K = np.abs(Z - z2) <= P.BAND_M
        if K.sum() >= P.MIN_KEPT:
            body = (float(np.median(pts[K, 0])), float(np.median(pts[K, 1])))
            tp = (float(np.median(u[K])), float(np.median(v[K])))
            return body, tp, z2, "BR band"
    body, tp, zr = _hr(kp, depth, cam, zh, js)
    return body, tp, zr, "BR fallback HR"


def m1_variant(cands, raw, depth, cam, mode):
    """The candidate list after the M1 block (mode None, 'HR' or 'BR'); new dicts, A1's left untouched."""
    if mode is None:
        return [dict(c, m1=None) for c in cands]
    cam_xy = cam.T_world_camera[:2, 3]
    kp_of = {p["box_index"]: p["kp"] for p in raw["pose"]}
    out = []
    for c in cands:
        kp = kp_of.get(c["index"])
        trig = head_trigger(c, kp, depth, cam)
        if trig is None:
            out.append(dict(c, m1=None))
            continue
        zh, js = trig
        if mode == "HR":
            body, tp, zr = _hr(kp, depth, cam, zh, js)
            how = "HR"
        else:
            body, tp, zr, how = _br(c, kp, depth, cam, zh, js)
        assert abs(z_top_at(c["box"], cam, c["zmed"]) - c["z_top"]) < 1e-9
        yaws, diag = P.facing(kp, depth, cam, zr, body, cam_xy)
        out.append(dict(c, body_xy=body, torso_pixel=tp, z_top=z_top_at(c["box"], cam, zr), yaw=yaws,
                        facing_diag=diag, z_ref=zr, z_h=zh, m1=how))
    return out


# ------------------------------------------------------------------------------------------------ steps 2-7
def people_for(cands, blocks, s, cam_xy, variant):
    """A1's people_for with the candidate's blocks (identical arithmetic when blocks is empty)."""
    lo, hi = P.BANDS[s["band"]]
    flat_on = s["flat"] or "FLAT" in blocks
    tiny = "TINY" in blocks
    if "IMG" in blocks:
        dup_mode, d_dup = "img", None
    else:
        dup_mode, d_dup = "world", (D030_M if "d030" in blocks else s["d_dup"])
    kept = []
    for c in cands:
        if c["score"] < s["tau"]:
            break
        if c["reject"] is not None:
            continue
        if c["z_top"] > hi or (not c["top_touch"] and c["z_top"] < lo):
            continue
        if flat_on and c["flat"]:
            continue
        if tiny and c["n_Q"] < P.FLAT_MIN_Q:
            continue
        xy = P.push(c["body_xy"], cam_xy, s["d_push"])
        if dup_mode == "world":
            dup = any(math.dist(xy, k["xy"]) <= d_dup for k in kept)
        else:
            dup = any(containment(c["box"], k["box"]) >= IMG_CONT and math.dist(xy, k["xy"]) <= IMG_CAP_M
                      for k in kept)
        if dup:
            continue
        kept.append({"xy": xy, "yaw": c["yaw"][variant], "yaws": c["yaw"], "score": c["score"], "box": c["box"],
                     "torso_pixel": c["torso_pixel"], "index": c["index"], "facing_diag": c["facing_diag"],
                     "m1": c.get("m1")})
    return kept


def run_all(raw, depth, cam, s, variant, ids=None):
    """{candidate id: people} for one image; C00 asserted equal to A1's deployed path."""
    ids = list(CANDIDATES) if ids is None else ids
    cands = P.candidates(raw, depth, cam, s["tau"])
    cam_xy = cam.T_world_camera[:2, 3]
    lists = {m: m1_variant(cands, raw, depth, cam, m) for m in {m1_of(CANDIDATES[i]) for i in ids}}
    out = {i: people_for(lists[m1_of(CANDIDATES[i])], CANDIDATES[i], s, cam_xy, variant) for i in ids}
    if "C00" in out:
        ref = P.run_image(raw, depth, cam, s, variant)
        assert [(p["index"], p["xy"], p["yaw"]) for p in out["C00"]] == [(p["index"], p["xy"], p["yaw"]) for p in ref]
    return out


# ------------------------------------------------------------------------------------------------ truth-side extras
def hidden_staff_points(sc, th):
    """Counter staff (a role) failing is_visible whose marked point (x, y, 1.1 m) projects into the picture."""
    info = PS.truth_info(sc, th)
    cam = sc.observation.camera
    pts = []
    for p in info["hidden"]:
        if not p.role:
            continue
        mk = project(cam, (p.x, p.y, MARK_Z))
        if mk is not None and in_image(cam, mk[0], mk[1]):
            pts.append((p.id, float(p.x), float(p.y)))
    return pts


def secondary(r, ppl, hs_pts):
    """FIX_PROTOCOL.md section 8: false detections within 0.6 m of such staff count as exempt."""
    moved = [n for n in r["false_numbers"]
             if any(math.dist(ppl[n - 1]["xy"], (x, y)) <= STAFF_EXEMPT_M for _, x, y in hs_pts)]
    return {"false": r["false"] - len(moved), "exempt": r["exempt"] + len(moved), "moved": len(moved)}


def m3_audit(ref_ppl, ppl, hs_pts):
    """A1-kept detections within 0.6 m of such staff for which the candidate outputs nothing within 0.6 m of the
    same staff member; and the A1 total."""
    near_ref = lost = 0
    for d in ref_ppl:
        for _, x, y in hs_pts:
            if math.dist(d["xy"], (x, y)) <= STAFF_EXEMPT_M:
                near_ref += 1
                if not any(math.dist(e["xy"], (x, y)) <= STAFF_EXEMPT_M for e in ppl):
                    lost += 1
                break
    return near_ref, lost


def posthoc_class(xy, info):
    """The diagnosis's post hoc class (diagnose.py / posthoc_false.py rules) of a false or exempt detection."""
    vis, hid, lh = info["visible"], info["hidden"], info["lower_hidden"]
    if any(math.dist(xy, (p.x, p.y)) <= 0.8 and p.role and lh[p.id] for p in vis):
        return "A"
    if any(math.dist(xy, (p.x, p.y)) <= PS.NEAR_M for p in hid):
        return "B"
    near = [math.dist(xy, (p.x, p.y)) for p in list(vis) + list(hid)]
    return "C" if near and min(near) <= 1.5 else "D"


# ------------------------------------------------------------------------------------------------ accumulation
class Acc:
    """Per candidate: scorer rows, people rows, secondary counts, audits."""

    def __init__(self):
        self.rows, self.prows = [], []
        self.sec_false = self.sec_exempt = self.sec_moved = 0
        self.m3_near = self.m3_lost = 0
        self.fired = self.fired_a1_matched = 0
        self.images_with_false = 0
        self.classes = {}

    def add(self, r, prow, sec, m3, fired, fired_a1_matched, classes=None):
        self.rows.append(r)
        self.prows += prow
        self.sec_false += sec["false"]
        self.sec_exempt += sec["exempt"]
        self.sec_moved += sec["moved"]
        self.m3_near += m3[0]
        self.m3_lost += m3[1]
        self.fired += fired
        self.fired_a1_matched += fired_a1_matched
        self.images_with_false += r["false"] > 0
        for k, v in (classes or {}).items():
            self.classes[k] = self.classes.get(k, 0) + v


def score_image(accs, people, sc, th, variant, with_classes=False):
    """Score each candidate's people on one scene and add to accs (dict id -> Acc)."""
    hs = hidden_staff_points(sc, th)
    info = PS.truth_info(sc, th)
    ref = people.get("C00")
    r_ref = C.score_scene(ref, sc, th, variant)[0] if ref is not None else None
    a1_matched_idx = ({ref[n - 1]["index"] for n in r_ref["matches"]} if ref is not None else set())
    rows = {}
    for cid, ppl in people.items():
        r, prow = C.score_scene(ppl, sc, th, variant)
        sec = secondary(r, ppl, hs)
        m3 = m3_audit(ref, ppl, hs) if ref is not None else (0, 0)
        fired = [p for p in ppl if p.get("m1")]
        classes = None
        if with_classes:
            classes = {}
            for n in r["false_numbers"]:
                k = "false " + posthoc_class(ppl[n - 1]["xy"], info)
                classes[k] = classes.get(k, 0) + 1
            for n in r["exempt_numbers"]:
                k = "exempt " + posthoc_class(ppl[n - 1]["xy"], info)
                classes[k] = classes.get(k, 0) + 1
        accs[cid].add(r, prow, sec, m3, len(fired), sum(p["index"] in a1_matched_idx for p in fired), classes)
        rows[cid] = (r, prow, sec, ppl)
    return rows


def summary(a: Acc) -> dict:
    Pp = PS.pool(a.rows)
    role = {c: [sum(1 for p in a.prows if p["role_class"] == c and p["matched"]),
                sum(1 for p in a.prows if p["role_class"] == c)] for c in C.ROLE_CLASSES}
    m = [p for p in a.prows if p["matched"]]
    np_fs = C.facing_summary([p["err"] for p in m if p["view"] != "profile"])
    pf_fs = C.facing_summary([p["err"] for p in m if p["view"] == "profile"])
    n_img = Pp["scenes"]
    share = lambda k, n: (k / n) if n else None  # noqa: E731
    return {
        "images": n_img, "visible": Pp["n_visible"], "matched": Pp["matched"], "recall": Pp["recall"],
        "by_role": role, "line_recall": share(*role["line"]), "staff_recall": share(*role["staff"]),
        "false": Pp["false"], "fb_per_image": Pp["fb_per_scene"], "exempt": Pp["exempt"],
        "near_hidden": Pp["near_hidden"], "merged_pairs": Pp["merged_pairs"], "neighbour_pairs": Pp["neighbour_pairs"],
        "xy_median": Pp["xy_err_median"], "n_det": Pp["n_det"],
        "np_w45": np_fs["w45"], "np_n": np_fs["n"], "np_w45_share": np_fs["w45_share"], "np_median": np_fs["median_u"],
        "profile_fb": pf_fs["fb"], "profile_n": pf_fs["n"],
        "images_with_false": a.images_with_false,
        "secondary": {"false": a.sec_false, "exempt": a.sec_exempt, "moved": a.sec_moved,
                      "fb_per_image": (a.sec_false / n_img) if n_img else None},
        "m3_audit": {"a1_near_hidden_staff": a.m3_near, "lost": a.m3_lost},
        "m1_fired_kept": a.fired, "m1_fired_kept_a1_matched": a.fired_a1_matched,
        "posthoc_classes": dict(sorted(a.classes.items())),
        "misses": [{k: p[k] for k in ("pid", "role_class", "view")} for p in a.prows if not p["matched"]],
    }


# ------------------------------------------------------------------------------------------------ intervals
def wilson(k, n):
    lo, hi = PS.wilson(k, n)
    return [lo, hi]


def _pois_cdf(k, mu):
    if mu <= 0:
        return 1.0
    return sum(math.exp(-mu + i * math.log(mu) - math.lgamma(i + 1)) for i in range(k + 1))


def poisson_exact(k, alpha=0.05):
    """Garwood exact interval for a Poisson count k (bisection; no scipy)."""
    def solve(f, lo, hi):
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if f(mid):
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)
    lower = 0.0 if k == 0 else solve(lambda mu: 1.0 - _pois_cdf(k - 1, mu) >= alpha / 2, 0.0, float(k))
    upper = solve(lambda mu: _pois_cdf(k, mu) <= alpha / 2, float(k), float(k) + 20.0 + 10.0 * math.sqrt(k + 1))
    return [lower, upper]


def code_sha() -> dict:
    return {p.name: F.sha(p) for p in sorted(FIX.glob("*.py"))}


def fmt_summary(sm) -> str:
    r = sm["by_role"]
    return (f"recall {sm['matched']}/{sm['visible']} = {sm['recall']:.3f} (line {r['line'][0]}/{r['line'][1]}, "
            f"staff {r['staff'][0]}/{r['staff'][1]}, group {r['group'][0]}/{r['group'][1]}, other "
            f"{r['other'][0]}/{r['other'][1]}); false {sm['false']}/{sm['images']} = {sm['fb_per_image']:.3f}; "
            f"exempt {sm['exempt']}; merged {sm['merged_pairs']}/{sm['neighbour_pairs']}; xy median "
            f"{sm['xy_median']:.3f}; non-profile w45 {sm['np_w45']}/{sm['np_n']} = "
            f"{(sm['np_w45_share'] if sm['np_w45_share'] is not None else float('nan')):.3f}; profile f/b "
            f"{sm['profile_fb']}/{sm['profile_n']}")


def median_or_none(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None
