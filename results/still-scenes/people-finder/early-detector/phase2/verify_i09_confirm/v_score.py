"""Verifier: independent recomputation of the I09 confirmation's step 6 (project venv).

People: the verifier's own re-implementation (v_pipe.people) on the verifier's own raw pass (bit-identical to the
compute agent's, v_compare_raw.txt), cross-checked against phase1c's P.run_image and the compute agent's rows.jsonl.
Scoring: the verifier's own code for FROZEN section 4's definitions (visibility = the package's is_visible; matching =
greedy nearest first, one to one, within 2 x person_radius_m, ties by detection number then visible order; false =
unmatched detection not within 0.5 m of an anchor of a target with no person; view class; facing error; role class),
its own Wilson and its own cluster bootstrap (same procedure as FROZEN section 6: sorted cluster keys,
random.Random(0), 2000 draws, percentile). Compared against the compute agent's people_rows.jsonl and summary.json.
Writes v_tables.txt, v_people_rows.jsonl, v_summary.json here. Reads no image pixels except through load_scene.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import collections  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402
RESEARCH = paths.STILL                             # results/still-scenes
WT = paths.REPO                                    # the repository root
CONFIRM = RESEARCH / "people-finder/early-detector/phase2/confirm"
A1 = RESEARCH / "people-finder/early-detector/phase1c/a1"
sys.path.insert(0, str(WT / "results/still-scenes/analysis-scripts/harness"))
sys.path.insert(0, str(HERE))
import research_harness as H  # noqa: E402
import v_pipe as VP  # noqa: E402
from sgi.config import load_config  # noqa: E402
from sgi.gt.derive import is_visible  # noqa: E402
from sgi.scene import load_scene  # noqa: E402

spec = importlib.util.spec_from_file_location("v_guard", HERE / "v_guard.py")
VG = importlib.util.module_from_spec(spec)
spec.loader.exec_module(VG)
PIPE_SHA = "4826be88d6939f3f5e4d14375a2fc9698abd94973ca46243d6697e263e5d063a"
spec2 = importlib.util.spec_from_file_location("a1_pipeline", A1 / "pipeline.py")
P = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(P)

SETTING = {"detector": "rtdetr_v2_r50vd", "tau": 0.25, "band": "wide", "flat": False, "d_dup": 0.5, "d_push": 0.2}
L = []


def out(*a):
    s = " ".join(str(x) for x in a)
    L.append(s)
    print(s, flush=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def wrapd(a):
    return abs(math.degrees(VP.wrap(a)))


# ------------------------------------------------------------------------------------------------ truth helpers
def roles(truth):
    gs = collections.defaultdict(set)
    for ep in truth.episodes.values():
        g = ep.group_at_target
        for gid in (g.values() if isinstance(g, dict) else [g]):
            if gid is not None:
                gs[gid].add(ep.situation)
    r = {}
    for p in truth.people:
        if p.role:
            r[p.id] = "staff"
        elif p.group is not None and "waiting_line" in gs.get(p.group, ()):
            r[p.id] = "line"
        elif p.group is not None and gs.get(p.group, set()) & {"conversation_group", "audience", "group_photo"}:
            r[p.id] = "group"
        else:
            r[p.id] = "other"
    return r


def project(cam, xyz):
    T = np.asarray(cam.T_world_camera, dtype=np.float64)
    X, Y, Z = (np.linalg.inv(T) @ np.array([xyz[0], xyz[1], xyz[2], 1.0]))[:3]
    if Z <= 0:
        return None
    return cam.fx * X / Z + cam.cx, cam.fy * Y / Z + cam.cy, Z


def inimg(cam, u, v):
    return 0 <= u < cam.width and 0 <= v < cam.height


def lower_hidden(p, obs):
    hits = 0
    for z in (0.3, 0.5, 0.7):
        q = project(obs.camera, (p.x, p.y, z))
        if q is None or not inimg(obs.camera, q[0], q[1]):
            continue
        d = float(obs.depth[int(math.floor(q[1])), int(math.floor(q[0]))])
        if math.isfinite(d) and d > 0 and d < q[2] - 0.3:
            hits += 1
    return hits >= 2


def view_of(p, cam_xy):
    a = wrapd(math.atan2(cam_xy[1] - p.y, cam_xy[0] - p.x) - p.yaw)
    return a, ("frontal" if a < 45 else "oblique" if a < 75 else "profile" if a <= 105 else "back")


def best_box(raw, cam, x, y):
    q = project(cam, (x, y, 1.1))
    if q is None or not inimg(cam, q[0], q[1]):
        return None
    for b in raw["boxes"]:
        if b[0] <= q[0] <= b[2] and b[1] <= q[1] <= b[3]:
            return b[4]
    return None


def match(dets, vis, width):
    pairs = sorted((math.dist(d, (p.x, p.y)), i, j) for i, d in enumerate(dets) for j, p in enumerate(vis))
    m, used = {}, set()
    for gap, i, j in pairs:
        if gap <= width and i not in m and j not in used:
            m[i] = j
            used.add(j)
    return m


def score(dets, truth, obs, th):
    """dets: [(x, y, yaw or None, score)]. Returns the scene row and person rows."""
    vis = [p for p in truth.people if is_visible(p, obs, th)]
    hid = [p for p in truth.people if not is_visible(p, obs, th)]
    rc = roles(truth)
    width = 2 * th.person_radius_m
    m = match([(d[0], d[1]) for d in dets], vis, width)
    ex = [(t.anchor[0], t.anchor[1]) for t in truth.targets.values() if t.person_id is None]
    false, exempt, near_hid, fdetail = 0, 0, 0, []
    for i, d in enumerate(dets):
        if i in m:
            continue
        if any(math.dist(d[:2], e) <= 0.5 for e in ex):
            exempt += 1
            continue
        false += 1
        nh = any(math.dist(d[:2], (p.x, p.y)) <= 0.5 for p in hid)
        near_hid += nh
        fdetail.append({"i": i, "score": d[3], "near_hidden": nh,
                        "nearest_visible_m": min((math.dist(d[:2], (p.x, p.y)) for p in vis), default=None)})
    pairs = merged = 0
    matched_j = set(m.values())
    for a in range(len(vis)):
        for b in range(a + 1, len(vis)):
            pa, pb = vis[a], vis[b]
            if math.dist((pa.x, pa.y), (pb.x, pb.y)) >= 0.8:
                continue
            pairs += 1
            both = a in matched_j and b in matched_j
            shared = any(math.dist(d[:2], (pa.x, pa.y)) <= 0.5 and math.dist(d[:2], (pb.x, pb.y)) <= 0.5
                         for d in dets)
            merged += (not both) and shared
    cam_xy = np.asarray(obs.camera.T_world_camera)[:2, 3]
    inv = {j: i for i, j in m.items()}
    prow = []
    for j, p in enumerate(vis):
        a, vc = view_of(p, cam_xy)
        r = {"pid": p.id, "role_class": rc[p.id], "view": vc, "view_angle": a, "matched": j in inv,
             "lower_hidden": lower_hidden(p, obs), "role": p.role, "x": p.x, "y": p.y}
        if j in inv:
            d = dets[inv[j]]
            r["xy_err"] = math.dist(d[:2], (p.x, p.y))
            r["err"] = None if d[2] is None else wrapd(d[2] - p.yaw)
            r["det_score"] = d[3]
        prow.append(r)
    row = {"n_det": len(dets), "n_visible": len(vis), "n_hidden": len(hid), "matched": len(m), "false": false,
           "exempt": exempt, "near_hidden": near_hid, "pairs": pairs, "merged_pairs": merged, "false_detail": fdetail}
    return row, prow


# ------------------------------------------------------------------------------------------------ statistics
def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def boot(groups, stat, n=2000, seed=0):
    keys = sorted(groups)
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        items = []
        for _k in keys:
            items.extend(groups[rng.choice(keys)])
        v = stat(items)
        if v is not None:
            vals.append(v)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))]


def med_u(errs):
    return statistics.median([180.0 if e is None else e for e in errs]) if errs else None


def share(errs, f):
    return sum(1 for e in errs if e is not None and f(e)) / len(errs) if errs else None


def st_recall(items):
    n = sum(it["n_visible"] for it in items)
    return sum(it["matched"] for it in items) / n if n else None


def st_fb(items):
    return sum(it["false"] for it in items) / len(items) if items else None


def st_line(items):
    xs = [p for p in items if p["role_class"] == "line"]
    return sum(p["matched"] for p in xs) / len(xs) if xs else None


def st_xy(items):
    xs = [p["xy_err"] for p in items if p["matched"]]
    return statistics.median(xs) if xs else None


def errs_of(items, sel):
    return [p["err"] for p in items if p["matched"] and sel(p["view"])]


STATS = {"S1": ("s", st_recall), "S2": ("p", st_line), "S3": ("s", st_fb), "S4": ("p", st_xy),
         "S5": ("p", lambda it: med_u(errs_of(it, lambda v: v != "profile"))),
         "S6": ("p", lambda it: share(errs_of(it, lambda v: v != "profile"), lambda e: e <= 45)),
         "S7": ("p", lambda it: share(errs_of(it, lambda v: v == "profile"), lambda e: e < 90)),
         "K3": ("p", lambda it: share(errs_of(it, lambda v: True), lambda e: e <= 45))}


def measures(rows, prows):
    m = {}
    m["images"] = len(rows)
    m["visible"] = sum(r["n_visible"] for r in rows)
    m["matched"] = sum(r["matched"] for r in rows)
    m["false"] = sum(r["false"] for r in rows)
    m["exempt"] = sum(r["exempt"] for r in rows)
    m["near_hidden"] = sum(r["near_hidden"] for r in rows)
    m["pairs"] = sum(r["pairs"] for r in rows)
    m["merged_pairs"] = sum(r["merged_pairs"] for r in rows)
    m["recall"] = m["matched"] / m["visible"]
    m["fb"] = m["false"] / m["images"]
    m["by_role"] = {c: [sum(p["matched"] for p in prows if p["role_class"] == c),
                        sum(1 for p in prows if p["role_class"] == c)] for c in ("line", "group", "staff", "other")}
    m["line"] = m["by_role"]["line"]
    m["xy_median"] = st_xy(prows)
    m["n_xy"] = sum(p["matched"] for p in prows)
    np_ = errs_of(prows, lambda v: v != "profile")
    pf = errs_of(prows, lambda v: v == "profile")
    al = errs_of(prows, lambda v: True)
    m["np"] = [sum(1 for e in np_ if e is not None and e <= 45), len(np_)]
    m["np_median_u"] = med_u(np_)
    m["pf_fb"] = [sum(1 for e in pf if e is not None and e < 90), len(pf)]
    m["all_w45"] = [sum(1 for e in al if e is not None and e <= 45), len(al)]
    m["coverage"] = [sum(1 for e in al if e is not None), len(al)]
    m["by_view"] = {}
    for vc in ("frontal", "oblique", "profile", "back"):
        e = errs_of(prows, lambda v, vc=vc: v == vc)
        m["by_view"][vc] = {"n": len(e), "w45": sum(1 for x in e if x is not None and x <= 45),
                            "median_u": med_u(e), "fb": sum(1 for x in e if x is not None and x < 90),
                            "unknown": sum(1 for x in e if x is None)}
    m["staff_lower_hidden"] = sum(1 for p in prows if p["role_class"] == "staff" and p["lower_hidden"])
    return m


def bars(m):
    s = {"S1": m["recall"] >= 0.95, "S2": m["line"][0] / m["line"][1] >= 0.95, "S3": m["fb"] <= 0.10,
         "S4": m["xy_median"] <= 0.15, "S5": m["np_median_u"] <= 25, "S6": m["np"][0] / m["np"][1] >= 0.85,
         "S7": m["pf_fb"][0] / m["pf_fb"][1] >= 0.90}
    k = {"K1": m["recall"] < 0.90, "K2": m["fb"] > 0.30, "K3": m["all_w45"][0] / m["all_w45"][1] < 0.70}
    v = "kill" if any(k.values()) else "success" if all(s.values()) else "neither"
    return s, k, v


# ------------------------------------------------------------------------------------------------ main
def main():
    t0 = time.time()
    assert sha(CONFIRM / "FROZEN.md") == "f55341c03cd5e1d38c91b7e0142ef10f0870ccb2b43c5bf4edfac07d1996e751"
    assert sha(A1 / "pipeline.py") == PIPE_SHA
    assert sha(CONFIRM / "manifest.json") == "cf2937ab1ccb769ce89fe1b38c8827696c39cc204ef092ef0556f3cd1ac608f4"
    for line in (CONFIRM / "SCORE.sha256").read_text().splitlines():
        want, name = line.split()
        assert sha(CONFIRM / name) == want, name
    chosen = json.loads((A1 / "chosen.json").read_text())
    fix = json.loads((A1 / "chosen_fix.json").read_text())
    assert chosen["setting"] == fix["setting"] == SETTING and fix["facing_variant"] == "F5" and fix["fix_adopted"]
    th = load_config().thresholds
    assert th.person_radius_m == 0.25 and th.visibility_min_samples == 3 and th.visibility_depth_tolerance_m == 0.3
    man = json.loads((CONFIRM / "manifest.json").read_text())["entries"]
    mine_raw = [json.loads(x) for x in (HERE / "raw/v_rtdetr_v2_r50vd.jsonl").read_text().splitlines()]
    their_raw = [json.loads(x) for x in (CONFIRM / "raw/rtdetr_v2_r50vd.jsonl").read_text().splitlines()]
    assert [r["rgb"] for r in mine_raw] == [e["rgb"] for e in man] == [r["rgb"] for r in their_raw]
    crows = [json.loads(x) for x in (CONFIRM / "rows.jsonl").read_text().splitlines()]
    cprows = [json.loads(x) for x in (CONFIRM / "people_rows.jsonl").read_text().splitlines()]
    csum = json.loads((CONFIRM / "summary.json").read_text())

    rows, prows, truth_rows = [], [], []
    pp_diff_ref = pp_diff_compute = 0
    max_d_ref = max_d_cmp = 0.0
    n_people = 0
    for e, raw, traw, cr in zip(man, mine_raw, their_raw, crows):
        assert raw["boxes"] == traw["boxes"] and raw["pose"] == traw["pose"]
        f = Path(e["folder"])
        assert VG.check(str(f / "scene.json")) == e["seed"]
        H.guard_scene(f)
        assert sha(f / "rgb.png") == e["sha256_rgb_png"] and sha(f / "depth.npy") == e["sha256_depth_npy"]
        assert sha(f / "scene.json") == e["sha256_scene_json"]
        assert cr["seed"] == e["seed"] and cr["scene_id"] == e["scene_id"]
        inputs, truth = load_scene(f)
        obs = inputs.observation
        mine = VP.people(raw, obs.depth, obs.camera)                       # people fixed before truth is used
        ref = P.run_image(raw, obs.depth, obs.camera, SETTING, "F5")
        cmp_ = cr["det_people"]
        n_people += len(mine)

        def dyaw(a, b):
            if a is None or b is None:
                return 0.0 if a is None and b is None else math.inf
            return abs(VP.wrap(a - b))
        if len(mine) != len(ref):
            pp_diff_ref += 1
        else:
            for a, b in zip(mine, ref):
                d = max(abs(a["xy"][0] - b["xy"][0]), abs(a["xy"][1] - b["xy"][1]), dyaw(a["yaw"], b["yaw"]))
                max_d_ref = max(max_d_ref, d)
                pp_diff_ref += d > 1e-9
        if len(mine) != len(cmp_):
            pp_diff_compute += 1
        else:
            for a, b in zip(mine, cmp_):
                d = max(abs(a["xy"][0] - b["x"]), abs(a["xy"][1] - b["y"]), dyaw(a["yaw"], b["yaw"]))
                max_d_cmp = max(max_d_cmp, d)
                pp_diff_compute += d > 1e-9
        dets = [(p["xy"][0], p["xy"][1], p["yaw"], p["score"]) for p in mine]
        row, pr = score(dets, truth, obs, th)
        lay = f"seed{e['seed']}/{e['scene_id']}"
        row.update(seed=e["seed"], row=e["row"], scene_id=e["scene_id"], layout=lay)
        for p in pr:
            p.update(seed=e["seed"], row=e["row"], scene_id=e["scene_id"], layout=lay)
            if not p["matched"]:
                p["best_raw_box"] = best_box(raw, obs.camera, p["x"], p["y"])
        rows.append(row)
        prows += pr
        # C3: truth self-check (true visible people with true yaws)
        vis = [p for p in truth.people if is_visible(p, obs, th)]
        tr, tpr = score([(p.x, p.y, p.yaw, 1.0) for p in vis], truth, obs, th)
        truth_rows.append((tr["matched"] == tr["n_visible"], tr["false"] == 0,
                           all(p["err"] is not None and p["err"] <= 45 for p in tpr)))

    out(f"v_score {time.strftime('%Y-%m-%dT%H:%M:%S%z')}; {len(rows)} test images; {n_people} detector people")
    out(f"people: verifier re-implementation vs phase1c P.run_image (F5): scenes/people differing beyond 1e-9: "
        f"{pp_diff_ref} (max |diff| {max_d_ref:.2e}); vs compute agent's rows.jsonl det_people: {pp_diff_compute} "
        f"(max |diff| {max_d_cmp:.2e})")
    c3 = [all(t) for t in truth_rows]
    out(f"C3 (verifier's scorer): {sum(c3)}/{len(c3)} scenes with recall 1, 0 false, all within 45 deg")

    # person-row comparison with the compute agent
    ck = {(p["seed"], p["scene_id"], p["pid"]): p for p in cprows}
    mk = {(p["seed"], p["scene_id"], p["pid"]): p for p in prows}
    diffs = collections.Counter()
    for k, p in mk.items():
        c = ck.get(k)
        if c is None:
            diffs["missing in compute"] += 1
            continue
        for fld in ("matched", "role_class", "view", "lower_hidden"):
            if p[fld] != c[fld]:
                diffs[fld] += 1
        if p["matched"]:
            if abs(p["xy_err"] - c["xy_err"]) > 1e-9:
                diffs["xy_err"] += 1
            if (p["err"] is None) != (c["err"] is None) or (p["err"] is not None and abs(p["err"] - c["err"]) > 1e-6):
                diffs["err"] += 1
        else:
            cb = c.get("best_raw_box")
            if (cb["score"] if cb else None) != p["best_raw_box"]:
                diffs["best_raw_box"] += 1
    out(f"person rows: verifier {len(mk)}, compute {len(ck)}; keys equal {set(mk) == set(ck)}; field differences "
        f"{dict(diffs) or 'none'}")
    sdiff = collections.Counter()
    for r, c in zip(rows, crows):
        for fld, cf in (("n_visible", "n_visible"), ("matched", "matched"), ("false", "false"), ("exempt", "exempt"),
                        ("near_hidden", "near_hidden"), ("n_det", "n_det"), ("n_hidden", "n_hidden_people"),
                        ("pairs", "neighbour_pairs"), ("merged_pairs", "merged_pairs")):
            if r[fld] != c[cf]:
                sdiff[fld] += 1
    out(f"scene rows: field differences vs compute rows.jsonl {dict(sdiff) or 'none'}")

    m = measures(rows, prows)
    s, k, verdict = bars(m)
    gl_s, gl_p = collections.defaultdict(list), collections.defaultdict(list)
    gs_s, gs_p = collections.defaultdict(list), collections.defaultdict(list)
    for r in rows:
        gl_s[r["layout"]].append(r)
        gs_s[r["seed"]].append(r)
        gl_p.setdefault(r["layout"], [])
        gs_p.setdefault(r["seed"], [])
    for p in prows:
        gl_p[p["layout"]].append(p)
        gs_p[p["seed"]].append(p)
    ci_l = {b: boot(gl_s if lv == "s" else gl_p, f) for b, (lv, f) in STATS.items()}
    ci_s = {b: boot(gs_s if lv == "s" else gs_p, f) for b, (lv, f) in STATS.items()}

    out("")
    out("== Criteria, pooled test set (verifier), point estimates ==")
    rl, li, npw, pff, aw = m["by_role"], m["line"], m["np"], m["pf_fb"], m["all_w45"]
    vals = {"S1": (m["recall"], f"{m['matched']}/{m['visible']}", wilson(m["matched"], m["visible"])),
            "S2": (li[0] / li[1], f"{li[0]}/{li[1]}", wilson(*li)),
            "S3": (m["fb"], f"{m['false']}/{m['images']}", None),
            "S4": (m["xy_median"], f"n {m['n_xy']}", None),
            "S5": (m["np_median_u"], f"n {npw[1]}", None),
            "S6": (npw[0] / npw[1], f"{npw[0]}/{npw[1]}", wilson(*npw)),
            "S7": (pff[0] / pff[1], f"{pff[0]}/{pff[1]}", wilson(*pff)),
            "K1": (m["recall"], f"{m['matched']}/{m['visible']}", wilson(m["matched"], m["visible"])),
            "K2": (m["fb"], f"{m['false']}/{m['images']}", None),
            "K3": (aw[0] / aw[1], f"{aw[0]}/{aw[1]}", wilson(*aw))}
    cmp_bar = {**csum["success_bars"], **csum["kill_clauses"]}
    agree = {}
    for b, (v, kn, w) in vals.items():
        flag = s.get(b, k.get(b))
        cflag = next(val for key, val in cmp_bar.items() if key.startswith(b + " "))
        ci = ci_l.get(b if b not in ("K1", "K2") else {"K1": "S1", "K2": "S3"}[b])
        cs = ci_s.get(b if b not in ("K1", "K2") else {"K1": "S1", "K2": "S3"}[b])
        cl_c = csum["layout_bootstrap"][b]
        cs_c = csum["seed_bootstrap"][b]
        agree[b] = (flag == cflag and abs(ci[0] - cl_c[0]) < 1e-12 and abs(ci[1] - cl_c[1]) < 1e-12
                    and abs(cs[0] - cs_c[0]) < 1e-12 and abs(cs[1] - cs_c[1]) < 1e-12)
        out(f"{b}: {v:.4f} ({kn}){'' if w is None else f' Wilson [{w[0]:.3f}, {w[1]:.3f}]'}; layout boot "
            f"[{ci[0]:.4f}, {ci[1]:.4f}]; seed boot [{cs[0]:.4f}, {cs[1]:.4f}]; "
            f"{'met' if b[0] == 'S' and flag else 'NOT met' if b[0] == 'S' else 'FIRES' if flag else 'no fire'}; "
            f"compute flag {cflag}; flags and both intervals equal to compute: {agree[b]}")
    out(f"VERDICT (verifier): {verdict}; compute: {csum['verdict']}")
    cm = csum["measures"]
    eqm = {"visible": cm["visible"] == m["visible"], "matched": cm["matched"] == m["matched"],
           "false": cm["false"] == m["false"], "exempt": cm["exempt"] == m["exempt"],
           "near_hidden": cm["near_hidden"] == m["near_hidden"], "by_role": cm["by_role"] == m["by_role"],
           "xy_median": abs(cm["xy_median"] - m["xy_median"]) < 1e-12,
           "np": [cm["np_w45"], cm["np_n"]] == m["np"], "np_median": abs(cm["np_median_u"] - m["np_median_u"]) < 1e-9,
           "pf": [cm["pf_fb"], cm["pf_n"]] == m["pf_fb"], "all": [cm["all_w45"], cm["all_n"]] == m["all_w45"],
           "coverage": cm["coverage"] == m["coverage"], "merged_pairs": cm["merged_pairs"] == m["merged_pairs"],
           "neighbour_pairs": cm["neighbour_pairs"] == m["pairs"]}
    out(f"measures equal to compute summary.json: {eqm}")
    out("")
    out(f"by role: {m['by_role']}; staff Wilson {wilson(*m['by_role']['staff'])}")
    out(f"by view: {json.dumps(m['by_view'])}")
    cf = csum["facing"]
    vok = all(cf[vc]["n"] == m["by_view"][vc]["n"] and cf[vc]["w45"] == m["by_view"][vc]["w45"]
              and cf[vc]["fb"] == m["by_view"][vc]["fb"] and abs(cf[vc]["median_u"] - m["by_view"][vc]["median_u"]) < 1e-9
              for vc in ("frontal", "oblique", "profile", "back"))
    out(f"by view equal to compute: {vok}; coverage {m['coverage']}; unknown by view "
        f"{ {vc: m['by_view'][vc]['unknown'] for vc in m['by_view']} }")
    out(f"visible staff {m['by_role']['staff'][1]}, of whom lower body hidden {m['staff_lower_hidden']}; hidden people "
        f"{sum(r['n_hidden'] for r in rows)}; neighbour pairs {m['pairs']}, merged {m['merged_pairs']}")
    misses = [p for p in prows if not p["matched"]]
    out(f"misses {len(misses)}: " + "; ".join(
        f"{p['seed']} {p['scene_id'][5:12]} {p['pid']} {p['role_class']} lower_hidden={p['lower_hidden']} "
        f"view={p['view']} best_raw_box={p['best_raw_box'] if p['best_raw_box'] is None else round(p['best_raw_box'], 3)}"
        for p in misses))
    fd = [(r["seed"], r["scene_id"], f) for r in rows for f in r["false_detail"]]
    out(f"false people {len(fd)} (near a hidden person {sum(f['near_hidden'] for _, _, f in fd)}); scores "
        f"{min(f['score'] for _, _, f in fd):.2f}-{max(f['score'] for _, _, f in fd):.2f}; by row "
        f"{dict(sorted(collections.Counter(sid[5:12] for _, sid, _ in fd).items()))}")
    for name, rng in (("200-209", range(200, 210)), ("210-219", range(210, 220))):
        rr = [r for r in rows if r["seed"] in rng]
        pr = [p for p in prows if p["seed"] in rng]
        mm = measures(rr, pr)
        out(f"batch {name}: images {mm['images']}, recall {mm['matched']}/{mm['visible']}, line {mm['line']}, false "
            f"{mm['false']}/{mm['images']} = {mm['fb']:.3f}, xy {mm['xy_median']:.4f}, np {mm['np']} median "
            f"{mm['np_median_u']:.2f}, pf {mm['pf_fb']}, all {mm['all_w45']}")
    tt = [r["t_det_s"] + r["t_pose_s"] for r in their_raw]
    out(f"runtime (compute pass) median {statistics.median(tt) * 1000:.1f} ms, p90 (nearest rank) "
        f"{sorted(tt)[math.ceil(0.9 * len(tt)) - 1] * 1000:.1f} ms; n {len(tt)}")
    out(f"elapsed {time.time() - t0:.0f} s")
    (HERE / "v_tables.txt").write_text("\n".join(L) + "\n")
    with open(HERE / "v_people_rows.jsonl", "w") as fh:
        for p in prows:
            fh.write(json.dumps(p) + "\n")
    (HERE / "v_summary.json").write_text(json.dumps(
        {"verdict": verdict, "success": s, "kill": k, "measures": {kk: vv for kk, vv in m.items()},
         "layout_bootstrap": ci_l, "seed_bootstrap": ci_s, "agree_with_compute": agree, "measures_equal": eqm,
         "people_vs_P_run_image_diffs": pp_diff_ref, "people_vs_compute_diffs": pp_diff_compute,
         "person_row_diffs": dict(diffs), "scene_row_diffs": dict(sdiff), "c3_pass_scenes": sum(c3)}, indent=1,
        default=str))


if __name__ == "__main__":
    main()
