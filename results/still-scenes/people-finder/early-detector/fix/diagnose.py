"""I09 fix, diagnosis step 3: mechanisms of the false people and misses on fit-train (seeds 110-115).

Reads score_fit.py's outputs (fit_train_*.jsonl) and the fit-train scenes (guarded), never fit-validate. For every
candidate box >= 0.25 it measures, from the image, depth and keypoints only:
  metric box height (h_px * Z_med / fx), box-bottom height, flat share (A1's), head depth (median of the 5x5 depth
  at confident nose/eye/ear keypoints), shoulder depth, the torso-region depth spread, depth just outside the box's
  sides and just below its bottom.
Then, using truth (diagnosis only; nothing here is a method):
  - post hoc classes A/B/C/D exactly as phase2/i09/confirm/posthoc_false.py, plus exempt detections by the same rule;
  - for detections beside counter staff: what the torso region's pixels hit (the staff member, something nearer,
    something farther; which rendered part: render_job.json parts/models, glass flagged), the displacement split
    into along-ray and lateral parts, and the push's share;
  - for every true person who fails is_visible with the 1.1 m mark in the picture: which samples are seen and what
    the unseen ones hit;
  - for every duplicate drop and every pair of kept boxes: world distance, IoU, containment, nose distance, and
    whether the two boxes are the same true person.
Writes diag_dets.jsonl, diag_hidden.jsonl, diag_pairs.jsonl and diagnose_tables.txt. Project venv.
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
from sgi.geometry import _depth_at, in_image, project  # noqa: E402

P, PS, C = F.P, F.PS, F.C
HEAD = (0, 1, 2, 3, 4)
EDGE_PX = 2.0
VIS_Z = (0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7)
L: list[str] = []
TEMPL: list[tuple] = []


def out(s=""):
    L.append(s)


# ------------------------------------------------------------------------------------------------ geometry helpers
def region_pixels(d, depth, cam):
    """A1's torso region for a candidate (pipeline.box_features steps), returning (pts, u, v, Z, region)."""
    H, W = depth.shape
    x0, y0, x1, y1 = d["box"]
    w, h = x1 - x0, y1 - y0
    kp = d["kp"]
    if kp is not None and all(kp[j][2] >= P.CONF for j in (P.LS, P.RS, P.RH, P.LH)):
        cols, rows = P.hull_pixels(P.convex_hull([(kp[j][0], kp[j][1]) for j in (P.LS, P.RS, P.RH, P.LH)]), H, W)
        if len(cols):
            pts, u, v, Z = P.lift_pixels(depth, cam, cols, rows)
            if len(Z) >= P.HULL_MIN_VALID:
                return pts, u, v, Z, "hull"
    cols, rows = P.rect_pixels(x0 + 0.3 * w, x0 + 0.7 * w, y0 + 0.2 * h, y0 + 0.5 * h, H, W)
    pts, u, v, Z = P.lift_pixels(depth, cam, cols, rows)
    return pts, u, v, Z, "box"


def kp_depth(kp, js, depth, cam):
    zs = []
    for j in js:
        if kp is not None and kp[j][2] >= P.CONF:
            z = _depth_at(depth, cam, kp[j][0], kp[j][1], P.KP_WINDOW)
            if z is not None:
                zs.append(z)
    return statistics.median(zs) if zs else None, len(zs)


def strip_median(depth, cam, u0, u1, v0, v1):
    H, W = depth.shape
    cols, rows = P.rect_pixels(u0, u1, v0, v1, H, W)
    if not len(cols):
        return None
    Z = depth[rows, cols].astype(np.float64)
    Z = Z[np.isfinite(Z) & (Z > 0)]
    return float(np.median(Z)) if Z.size else None


def world_z_at(cam, u, v, Z):
    p = np.array([(u - cam.cx) * Z / cam.fx, (v - cam.cy) * Z / cam.fy, Z, 1.0])
    return float(cam.T_world_camera[2] @ p)


def iou_cont(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter) if inter > 0 else 0.0, inter / min(aa, bb) if inter > 0 else 0.0


def nose_px(a, b):
    ka, kb = a.get("kp"), b.get("kp")
    if ka is None or kb is None or ka[0][2] < P.CONF or kb[0][2] < P.CONF:
        return None
    return math.dist(ka[0][:2], kb[0][:2])


class Parts:
    """Rendered parts and models of a fit scene (render_job.json spec), to name what a world point lies on."""

    def __init__(self, folder):
        job = json.loads((Path(folder) / "render_job.json").read_text())
        self.parts = job["spec"]["parts"]
        self.models = job["spec"]["models"]
        self.people = job["spec"]["people"]

    @staticmethod
    def _in_box(p, cx, cy, yaw, sx, sy, z0, sz, tol):
        dx, dy = p[0] - cx, p[1] - cy
        a = dx * math.cos(yaw) + dy * math.sin(yaw)
        b = -dx * math.sin(yaw) + dy * math.cos(yaw)
        return abs(a) <= sx / 2 + tol and abs(b) <= sy / 2 + tol and z0 - tol <= p[2] <= z0 + sz + tol

    def label(self, p, tol=0.03):
        for pt in self.parts:
            s = pt["solid"]
            if s["shape"] == "box":
                hit = self._in_box(p, s["cx"], s["cy"], s["yaw"], s["sx"], s["sy"], s["z0"], s["sz"], tol)
            elif s["shape"] == "sphere":
                hit = math.dist(p, (s["x"], s["y"], s["z"])) <= s["radius"] + tol
            else:
                hit = (math.dist(p[:2], (s["x"], s["y"])) <= max(s["radius"], s.get("top_radius") or 0) + tol
                       and s["z0"] - tol <= p[2] <= s["z0"] + s["height"] + tol)
            if hit:
                kind = "glass" if pt["glass"] else pt["material"]
                if not pt["glass"] and s["shape"] == "box" and s["z0"] <= 0.1 and 0.8 <= s["z0"] + s["sz"] <= 1.25:
                    kind = "counter body (" + pt["material"] + ")"
                elif not pt["glass"] and s["shape"] == "box" and 0.8 <= s["z0"] <= 1.25 and s["sz"] <= 0.06:
                    kind = "counter top (" + pt["material"] + ")"
                return kind
        for m in self.models:
            sx, sy, sz = m["size"]
            if self._in_box(p, m["x"], m["y"], m["yaw"], sx, sy, m["z"], sz, tol):
                return "object: " + m["asset"]
        for q in self.people:
            if math.dist(p[:2], (q["x"], q["y"])) <= 0.40 and -0.05 <= p[2] <= q["height"] + 0.1:
                return "person " + q["person_id"]
        return "other (wall, floor, background)"


def identity(d, depth_obs, cam, people):
    """The true person (visible or hidden) whose axis at 1.3 m projects into the box with the smallest depth
    difference to the box's Z_med (diagnosis only)."""
    best = None
    x0, y0, x1, y1 = d["box"]
    for p in people:
        for z in (1.3, 1.5, 1.1):
            pr = project(cam, (p.x, p.y, z))
            if pr and x0 <= pr[0] <= x1 and y0 <= pr[1] <= y1:
                gap = abs(pr[2] - (d["zmed"] or 0))
                if best is None or gap < best[0]:
                    best = (gap, p.id)
                break
    return None if best is None else best[1]


# ------------------------------------------------------------------------------------------------ main
def main():
    for sha_file in ("MANIFEST_FIT.sha256", "RAW_FIT.sha256"):
        for line in (F.HERE / sha_file).read_text().splitlines():
            want, name = line.split()
            assert F.sha(F.HERE / name) == want, name
    s, variant = F.frozen_setting()
    rows = [json.loads(x) for x in (F.HERE / "fit_train_rows.jsonl").read_text().splitlines()]
    prows = [json.loads(x) for x in (F.HERE / "fit_train_people.jsonl").read_text().splitlines()]
    dets = [json.loads(x) for x in (F.HERE / "fit_train_dets.jsonl").read_text().splitlines()]
    summ = json.loads((F.HERE / "fit_train_summary.json").read_text())
    man = {e["scene_id"]: e for e in F.fit_manifest()["images"]}
    th = PS.default_thresholds()
    prow_by = {(p["scene_id"], p["pid"]): p for p in prows}
    by_scene = collections.defaultdict(list)
    for d in dets:
        by_scene[d["scene_id"]].append(d)
    ddiag, hdiag, pdiag = [], [], []
    for r in rows:
        e = man[r["scene_id"]]
        assert e["domain"] == "fit-train"
        sc = F.load_fit_scene(e["folder"])
        obs, cam = sc.observation, sc.observation.camera
        depth = obs.depth
        info = PS.truth_info(sc, th)
        vis, hid, lh, roles = info["visible"], info["hidden"], info["lower_hidden"], info["roles"]
        everyone = list(sc.truth.people)
        parts = Parts(e["folder"])
        cam_xy = cam.T_world_camera[:2, 3]
        exempt_pts = PS.exempt_points(sc.truth)
        sdets = by_scene[r["scene_id"]]
        for d in sdets:
            if d["fate"].startswith("depth"):
                continue
            x0, y0, x1, y1 = d["box"]
            w, h = x1 - x0, y1 - y0
            pts, u, v, Z, region = region_pixels(d, depth, cam)
            assert region == d["region"] and abs(float(np.median(Z)) - d["zmed"]) < 1e-9
            keep = np.abs(Z - d["zmed"]) <= P.BAND_M
            zh, nh = kp_depth(d["kp"], HEAD, depth, cam)
            zs, ns = kp_depth(d["kp"], (P.LS, P.RS), depth, cam)
            zhip, nhip = kp_depth(d["kp"], (P.LH, P.RH), depth, cam)
            hz = [_depth_at(depth, cam, d["kp"][j][0], d["kp"][j][1], P.KP_WINDOW) for j in HEAD
                  if d["kp"] is not None and d["kp"][j][2] >= P.CONF]
            hz = [z for z in hz if z is not None]
            zhmin = min(hz) if hz else None
            side = [strip_median(depth, cam, x0 - 6, x0 - 2, y0 + 0.2 * h, y0 + 0.6 * h),
                    strip_median(depth, cam, x1 + 2, x1 + 6, y0 + 0.2 * h, y0 + 0.6 * h)]
            side = [z for z in side if z is not None]
            below = strip_median(depth, cam, x0 + 0.3 * w, x0 + 0.7 * w, y1 + 2, y1 + 8)
            f = {"scene_id": d["scene_id"], "seed": d["seed"], "row_id": d["row_id"], "index": d["index"],
                 "fate": d["fate"], "number": d.get("number"), "score": d["score"], "box": d["box"],
                 "h_px": h, "w_px": w, "region": region, "region_valid": d["region_valid"], "zmed": d["zmed"],
                 "kept_px": int(keep.sum()), "kept_share": float(keep.mean()),
                 "metric_h": h * d["zmed"] / cam.fx, "z_top": d["z_top"],
                 "z_bot": world_z_at(cam, 0.5 * (x0 + x1), y1, d["zmed"]),
                 "edges": [n for n, hit in (("left", x0 <= EDGE_PX), ("top", y0 <= EDGE_PX),
                                            ("right", x1 >= cam.width - EDGE_PX),
                                            ("bottom", y1 >= cam.height - EDGE_PX)) if hit],
                 "flat_share": d["flat_share"], "z_head": zh, "n_head": nh, "z_sh": zs, "n_sh": ns,
                 "head_minus_zmed": None if zh is None else zh - d["zmed"],
                 "z_head_min": zhmin, "headmin_minus_zmed": None if zhmin is None else zhmin - d["zmed"],
                 "z_hip": zhip, "hip_minus_headmin": None if (zhip is None or zhmin is None) else zhip - zhmin,
                 "side_minus_zmed": (min(side) - d["zmed"]) if side else None,
                 "side_max_minus_zmed": (max(side) - d["zmed"]) if side else None,
                 "below_minus_zmed": None if below is None else below - d["zmed"],
                 "kp_conf": None if d["kp"] is None else [round(k[2], 3) for k in d["kp"]],
                 "identity": identity(d, depth, cam, everyone)}
            if d["fate"] == "kept":
                f |= {"matched_to": d["matched_to"], "false": d["false"], "exempt": d["exempt"], "xy": d["xy"]}
                xy = d["xy"]
                nv = min(((math.dist(xy, (p.x, p.y)), p) for p in vis), key=lambda t: t[0], default=None)
                nhd = min(((math.dist(xy, (p.x, p.y)), p) for p in hid), key=lambda t: t[0], default=None)
                if d["false"] or d["exempt"]:
                    disp = any(math.dist(xy, (p.x, p.y)) <= 0.8 and p.role and lh[p.id] for p in vis)
                    near_any = min([t[0] for t in (nv, nhd) if t is not None], default=None)
                    f["class"] = ("A" if disp else "B" if nhd and nhd[0] <= PS.NEAR_M else
                                  "C" if near_any is not None and near_any <= 1.5 else "D")
                    f["near_any_m"] = near_any
                if d["matched_to"] is not None:
                    pm = next(p for p in vis if p.id == d["matched_to"])
                    f["matched_role_class"] = roles[pm.id]
                    f["matched_lower_hidden"] = bool(lh[pm.id])
                # the staff member this box is about (diagnosis): the identity if staff, else the nearest staff
                staff = [p for p in everyone if p.role]
                ref = next((p for p in staff if p.id == f["identity"]), None)
                if ref is None and staff:
                    cand = min(staff, key=lambda p: math.dist(xy, (p.x, p.y)))
                    ref = cand if math.dist(xy, (cand.x, cand.y)) <= 1.0 else None
                if ref is not None:
                    pr = project(cam, (ref.x, ref.y, 1.2))
                    z_axis = pr[2] if pr else None
                    body = np.array(d["body_xy"])
                    sxy = np.array([ref.x, ref.y])
                    ray = sxy - cam_xy
                    ray = ray / np.linalg.norm(ray)
                    off = body - sxy
                    offp = np.array(xy) - sxy
                    hits = collections.Counter(parts.label(tuple(pts[i])) for i in np.flatnonzero(keep)[::3])
                    f["staff"] = {"pid": ref.id, "visible": any(p.id == ref.id for p in vis),
                                  "lower_hidden": bool(lh.get(ref.id)) if ref.id in lh else None,
                                  "z_axis_1p2": z_axis,
                                  "zmed_minus_axis": None if z_axis is None else d["zmed"] - z_axis,
                                  "z_head_minus_axis": None if (z_axis is None or zh is None) else zh - z_axis,
                                  "share_nearer_0p4": None if z_axis is None else float(np.mean(Z < z_axis - 0.4)),
                                  "share_on_person": None if z_axis is None else
                                  float(np.mean(np.abs(Z - (z_axis - 0.12)) <= 0.25)),
                                  "kept_hits": dict(hits.most_common(4)),
                                  "kept_world_z": [float(np.percentile(pts[keep, 2], q)) for q in (5, 50, 95)],
                                  "dist_body_xy": float(np.linalg.norm(off)),
                                  "dist_pushed": float(np.linalg.norm(offp)),
                                  "along_ray_body": float(off @ ray), "lateral_body": float(abs(off[0] * ray[1] -
                                                                                               off[1] * ray[0])),
                                  "along_ray_pushed": float(offp @ ray)}
            ddiag.append(f)
        # camera-frame position of each staff member (does the geometry repeat across seeds?)
        for p in everyone:
            if p.role:
                pr = project(cam, (p.x, p.y, 1.2))
                glass = any(pt["glass"] and pt["solid"]["shape"] == "box" and
                            math.dist((pt["solid"]["cx"], pt["solid"]["cy"]), (p.x, p.y)) < 1.2 for pt in parts.parts)
                if pr is not None and in_image(cam, pr[0], pr[1]):
                    TEMPL.append((r["row_id"], r["seed"], p.id, round(pr[0]), round(pr[1]), round(pr[2], 2), glass,
                                  any(q.id == p.id for q in vis)))
        # hidden-by-rule people
        for p in hid:
            mk = project(cam, (p.x, p.y, 1.1))
            if mk is None or not in_image(cam, mk[0], mk[1]):
                continue
            samples = []
            for z in VIS_Z:
                pr = project(cam, (p.x, p.y, z))
                if pr is None or not in_image(cam, pr[0], pr[1]):
                    samples.append({"z": z, "in": False})
                    continue
                uu, vv, zz = pr
                dz = float(depth[math.floor(vv), math.floor(uu)])
                hit = P.lift_pixels(depth, cam, [math.floor(uu)], [math.floor(vv)])[0]
                samples.append({"z": z, "in": True, "seen": abs(dz - zz) <= th.visibility_depth_tolerance_m,
                                "gap": dz - zz, "hit": parts.label(tuple(hit[0])) if len(hit) else None})
            kept = [d for d in sdets if d["fate"] == "kept"]
            near = sorted((math.dist(d["xy"], (p.x, p.y)), d["number"], d["false"], d["exempt"]) for d in kept)
            hdiag.append({"scene_id": r["scene_id"], "pid": p.id, "role": p.role, "samples": samples,
                          "n_seen": sum(1 for x in samples if x.get("seen")),
                          "nearest_kept": near[0] if near else None,
                          "near_exempt_anchor": any(math.dist((p.x, p.y), a) <= 0.5 for a in exempt_pts)})
        # pairs: duplicates and kept-kept
        idx = {d["index"]: d for d in ddiag if d["scene_id"] == r["scene_id"]}
        kept = [d for d in sdets if d["fate"] == "kept"]
        for d in sdets:
            if d["fate"] == "duplicate":
                k = next(x for x in kept if x["index"] == d["dup_of"])
                iou, cont = iou_cont(d["box"], k["box"])
                pdiag.append({"scene_id": r["scene_id"], "kind": "duplicate drop", "a": k["index"], "b": d["index"],
                              "score_a": k["score"], "score_b": d["score"], "world_d": d["dup_d"], "iou": iou,
                              "containment": cont, "nose_px": nose_px(k, d),
                              "id_a": idx[k["index"]]["identity"], "id_b": idx[d["index"]]["identity"]})
        for i in range(len(kept)):
            for j in range(i + 1, len(kept)):
                a, b = kept[i], kept[j]
                wd = math.dist(a["xy"], b["xy"])
                if wd > 1.2:
                    continue
                iou, cont = iou_cont(a["box"], b["box"])
                pdiag.append({"scene_id": r["scene_id"], "kind": "kept pair", "a": a["index"], "b": b["index"],
                              "score_a": a["score"], "score_b": b["score"], "world_d": wd, "iou": iou,
                              "containment": cont, "nose_px": nose_px(a, b),
                              "id_a": idx[a["index"]]["identity"], "id_b": idx[b["index"]]["identity"],
                              "matched_a": a["matched_to"], "matched_b": b["matched_to"]})
    for name, obj in (("diag_dets.jsonl", ddiag), ("diag_hidden.jsonl", hdiag), ("diag_pairs.jsonl", pdiag)):
        (F.HERE / name).write_text("".join(json.dumps(o, default=float) + "\n" for o in obj))
    tables(summ, rows, prows, ddiag, hdiag, pdiag, prow_by)
    (F.HERE / "diagnose_tables.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def f2(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def mmm(xs, nd=2):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "-"
    return f"{min(xs):.{nd}f} / {statistics.median(xs):.{nd}f} / {max(xs):.{nd}f} (n {len(xs)})"


def tables(summ, rows, prows, ddiag, hdiag, pdiag, prow_by):
    out("I09 fix diagnosis, fit-train (seeds 110-115, 84 images): A1's frozen detector (RT-DETRv2 tau 0.25, band "
        "1.00-2.30, flat off, d_dup 0.50, d_push 0.20, F5), I03's scorer unchanged (facing full, exempt_points, "
        "default thresholds).")
    out(f"recall {summ['matched']}/{summ['visible']} = {summ['recall']:.3f}; by role {summ['by_role']}; false "
        f"{summ['false']}/{summ['images']} = {summ['fb_per_image']:.3f} per image; exempt {summ['exempt']}; near hidden "
        f"{summ['near_hidden']}; merged pairs {summ['merged_pairs']}/{summ['neighbour_pairs']}; median xy "
        f"{summ['xy_median']:.3f} m; non-profile within 45 {summ['np_facing']['w45']}/{summ['np_facing']['n']} "
        f"(median {summ['np_facing']['median_u']:.1f}); profile front/back {summ['profile_facing']['fb']}/"
        f"{summ['profile_facing']['n']}")
    out()
    out("By row (6 images each): visible | matched | false | exempt | misses (role class)")
    byr = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        c = byr[r["row_id"]]
        c["vis"] += r["n_visible"]
        c["m"] += r["matched"]
        c["f"] += r["false"]
        c["e"] += r["exempt"]
    miss_by = collections.defaultdict(list)
    for p in prows:
        if not p["matched"]:
            miss_by[p["row_id"]].append(p["role_class"])
    for rid in F.ROW_IDS:
        c = byr[rid]
        out(f"  {rid}: {c['vis']} | {c['m']} | {c['f']} | {c['e']} | {dict(collections.Counter(miss_by[rid]))}")
    out()
    kept = [d for d in ddiag if d["fate"] == "kept"]
    fe = [d for d in kept if d["false"] or d["exempt"]]
    out("FALSE AND EXEMPT DETECTIONS by post hoc class (posthoc_false.py rules: A within 0.8 m of a visible staff "
        "member with the lower body hidden; B within 0.5 m of a true person failing is_visible; C within 1.5 m of "
        "any true person; D farther)")
    for cl in "ABCD":
        fs = [d for d in fe if d["class"] == cl and d["false"]]
        es = [d for d in fe if d["class"] == cl and d["exempt"]]
        out(f"  {cl}: false {len(fs)}, exempt {len(es)}; scores {mmm([d['score'] for d in fs + es])}; "
            f"box h px {mmm([d['h_px'] for d in fs + es], 0)}")
    out()
    out("Per false/exempt detection: seed row number F/E class score | box (h px) | region (valid px) | Z_med | "
        "head Z - Z_med | metric h m | z_top | z_bot | flat share | side Z - Z_med | below Z - Z_med | identity | "
        "nearest true m")
    for d in fe:
        b = d["box"]
        out(f"  {d['seed']} {d['row_id'][5:]} n{d['number']} {'E' if d['exempt'] else 'F'} {d['class']} "
            f"{d['score']:.3f} | [{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}] ({d['h_px']:.0f}) | {d['region']} "
            f"({d['region_valid']}) | {d['zmed']:.2f} | {f2(d['head_minus_zmed'])} | {d['metric_h']:.2f} | "
            f"{d['z_top']:.2f} | {d['z_bot']:.2f} | {f2(d['flat_share'])} | {f2(d['side_minus_zmed'])} | "
            f"{f2(d['below_minus_zmed'])} | {d['identity']} | {f2(d.get('near_any_m'))}")
    out()
    out("COUNTER STAFF: every kept detection whose identity or nearest staff (<= 1.0 m) is a staff member")
    out("  seed row n outcome | staff (visible, lower hidden) | Z_med - Z_axis(1.2 m) | head Z - Z_axis | share of "
        "region nearer than axis-0.4 | share on the person | kept pixels hit | kept world z 5/50/95 | offset body_xy "
        "(along ray, lateral) | offset after push")
    for d in kept:
        st = d.get("staff")
        if st is None:
            continue
        outc = ("matched " + str(d["matched_to"])) if d["matched_to"] else ("exempt" if d["exempt"] else
                                                                             "FALSE " + d["class"])
        out(f"  {d['seed']} {d['row_id'][5:]} n{d['number']} {outc} | {st['pid']} ({st['visible']}, "
            f"{st['lower_hidden']}) | {f2(st['zmed_minus_axis'])} | {f2(st['z_head_minus_axis'])} | "
            f"{f2(st['share_nearer_0p4'])} | {f2(st['share_on_person'])} | {st['kept_hits']} | "
            f"{'/'.join(f2(z) for z in st['kept_world_z'])} | {st['dist_body_xy']:.2f} ({st['along_ray_body']:+.2f}, "
            f"{st['lateral_body']:.2f}) | {st['dist_pushed']:.2f} ({st['along_ray_pushed']:+.2f})")
    out()
    out("COUNTER STAFF, class A only: keypoint confidences (shoulders L/R, hips L/R, knees L/R, ankles L/R) | hip depth "
        "- nearest head depth | nearest head depth - Z_med | box bottom row (image height 480)")
    for d in kept:
        if d.get("class") != "A":
            continue
        kc = d["kp_conf"]
        out(f"  {d['seed']} {d['row_id'][5:]} n{d['number']} | " + " ".join(f"{kc[j]:.2f}" for j in (5, 6, 11, 12, 13, 14,
                                                                                                 15, 16)) +
            f" | {f2(d['hip_minus_headmin'])} | {f2(d['headmin_minus_zmed'])} | {d['box'][3]:.0f}")
    out()
    out("Nearest head depth - Z_med over all kept boxes (the occlusion signal): " + "; ".join(
        f"{g}: {mmm([d['headmin_minus_zmed'] for d in ds])}" for g, ds in (
            ("matched", [d for d in kept if d["matched_to"]]),
            ("false A", [d for d in fe if d["class"] == "A"]),
            ("false/exempt B or C", [d for d in fe if d["class"] in "BC"]),
            ("false D", [d for d in fe if d["class"] == "D"]))))
    out()
    out("MISSES: seed row pid role class | lower hidden | view | best raw box over (x, y, 1.1 m) | nearest kept "
        "detection (m, number)")
    for p in prows:
        if not p["matched"]:
            br = p.get("best_raw_box")
            out(f"  {p['seed']} {p['row_id'][5:]} {p['pid']} {p['role_class']} | {p['lower_hidden']} | {p['view']} "
                f"{p['view_angle']:.0f} | {f2(br['score'], 3) if br else None} | "
                f"{[round(p['nearest_kept'][0], 2), p['nearest_kept'][1]] if p.get('nearest_kept') else None}")
    out()
    out("DUPLICATE DROPS (d_dup 0.50 m): seed row kept idx -> dropped idx | scores | world d | IoU | containment | "
        "nose px | identity kept / dropped | same person")
    for q in pdiag:
        if q["kind"] != "duplicate drop":
            continue
        out(f"  {q['scene_id']} {q['a']} -> {q['b']} | {q['score_a']:.2f} {q['score_b']:.2f} | {q['world_d']:.2f} | "
            f"{q['iou']:.2f} | {q['containment']:.2f} | {f2(q['nose_px'], 1)} | {q['id_a']} / {q['id_b']} | "
            f"{q['id_a'] == q['id_b']}")
    out()
    out("KEPT PAIRS within 1.2 m (both kept): same person vs different people")
    for same in (True, False):
        qs = [q for q in pdiag if q["kind"] == "kept pair" and (q["id_a"] == q["id_b"]) == same
              and q["id_a"] is not None]
        out(f"  same person {same}: n {len(qs)}; world d {mmm([q['world_d'] for q in qs])}; IoU "
            f"{mmm([q['iou'] for q in qs])}; containment {mmm([q['containment'] for q in qs])}; nose px "
            f"{mmm([q['nose_px'] for q in qs], 1)}")
        if same:
            for q in qs:
                out(f"    {q['scene_id']} {q['a']}/{q['b']} d {q['world_d']:.2f} IoU {q['iou']:.2f} cont "
                    f"{q['containment']:.2f} nose {f2(q['nose_px'], 1)} matched {q['matched_a']}/{q['matched_b']}")
    diff = [q for q in pdiag if q["kind"] == "kept pair" and q["id_a"] != q["id_b"]]
    out(f"  different people, closest 8 by world d: " + "; ".join(
        f"{q['world_d']:.2f} m IoU {q['iou']:.2f} cont {q['containment']:.2f}"
        for q in sorted(diff, key=lambda q: q["world_d"])[:8]))
    out()
    out("HIDDEN BY RULE with the 1.1 m mark in the picture: scene pid role | samples seen (of 9) | unseen samples' "
        "gap (depth - sample Z) min/max | what unseen samples hit | nearest kept detection (m, number, false, exempt) "
        "| near an exempt anchor")
    for h in hdiag:
        uns = [x for x in h["samples"] if x.get("in") and not x.get("seen")]
        hits = collections.Counter(x["hit"] for x in uns)
        gaps = [x["gap"] for x in uns]
        nk = h["nearest_kept"]
        out(f"  {h['scene_id']} {h['pid']} {h['role']} | {h['n_seen']} | {f2(min(gaps)) if gaps else '-'}/"
            f"{f2(max(gaps)) if gaps else '-'} | {dict(hits)} | "
            f"{None if nk is None else [round(nk[0], 2), nk[1], nk[2], nk[3]]} | {h['near_exempt_anchor']}")
    out()
    out("STAFF CAMERA GEOMETRY by row (staff in the picture at 1.2 m): seed pid (u, v, Z) [glass booth] "
        "[hidden by rule]; identical (u, v, Z) across seeds = the same camera-staff geometry re-dressed")
    for rid in F.ROW_IDS:
        xs = [t for t in TEMPL if t[0] == rid]
        if xs:
            out(f"  {rid[5:]}: " + "; ".join(f"{t[1]} {t[2]} ({t[3]}, {t[4]}, {t[5]:.2f}){' glass' if t[6] else ''}"
                                              f"{' HIDDEN' if not t[7] else ''}" for t in xs))
    out()
    out("FEATURE DISTRIBUTIONS of kept detections (min / median / max (n)): matched vs false class D vs false A vs "
        "B+exempt")
    groups = {"matched": [d for d in kept if d["matched_to"]],
              "matched, not lower-hidden staff": [d for d in kept if d["matched_to"] and not
                                                  d.get("matched_lower_hidden")],
              "matched lower-hidden staff": [d for d in kept if d["matched_to"] and d.get("matched_lower_hidden")],
              "false A": [d for d in fe if d["class"] == "A"],
              "false/exempt B": [d for d in fe if d["class"] == "B"],
              "false D": [d for d in fe if d["class"] == "D"]}
    for key in ("score", "h_px", "metric_h", "z_bot", "flat_share", "head_minus_zmed", "side_minus_zmed",
                "side_max_minus_zmed", "below_minus_zmed", "kept_share"):
        out(f"  {key}:")
        for g, ds in groups.items():
            out(f"    {g}: {mmm([d[key] for d in ds])}")
    out()
    m = groups["matched"]
    out("Matched detections below simple cut-offs (side effects of candidate D filters):")
    for name, test in (("score < 0.30", lambda d: d["score"] < 0.30), ("score < 0.35", lambda d: d["score"] < 0.35),
                       ("score < 0.55", lambda d: d["score"] < 0.55),
                       ("metric h < 0.50 m", lambda d: d["metric_h"] < 0.50),
                       ("metric h < 0.90 m", lambda d: d["metric_h"] < 0.90),
                       ("flat share >= 0.80", lambda d: (d["flat_share"] or 0) >= 0.80),
                       ("box h < 40 px", lambda d: d["h_px"] < 40),
                       ("max side contrast < 0.30 m", lambda d: d["side_max_minus_zmed"] is not None and
                        d["side_max_minus_zmed"] < 0.30)):
        mm = [d for d in m if test(d)]
        ff = {cl: sum(1 for d in fe if d["class"] == cl and test(d)) for cl in "ABD"}
        out(f"  {name}: matched {len(mm)}/{len(m)} {[(d['scene_id'][4:7], d['row_id'][10:16], d['matched_to'], round(d['score'], 2), round(d['metric_h'], 2)) for d in mm][:8]}; false/exempt by class {ff}")


if __name__ == "__main__":
    main()
