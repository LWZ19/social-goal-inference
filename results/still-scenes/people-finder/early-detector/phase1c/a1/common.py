"""I09 A1 shared helpers (project venv): the frozen grid, the selection order, view classes and facing summaries
(FROZEN.md sections 4-6 and 10; DEVIATIONS.md C11-C14)."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
I09 = HERE.parent
sys.path.insert(0, 'docs/model_notes/tools')
sys.path.insert(0, 'results/2026-10-17-beyond-vlm/phase1b/i03')
sys.path.insert(0, 'results/2026-10-17-beyond-vlm/phase1b/i03/stageA')
import people_scorer as PS  # noqa: E402
from sgi.geometry import wrap_angle  # noqa: E402

DETECTORS = ("rtdetr_v2_r50vd", "dfine_large")
TAUS = (0.25, 0.40, 0.55, 0.70)
BAND_NAMES = ("wide", "narrow")             # [1.00, 2.30], [1.30, 2.10]
FLATS = (False, True)
D_DUPS = (0.30, 0.50)
D_PUSHES = (0.00, 0.10, 0.20)
GRID = [dict(detector=d, tau=t, band=b, flat=f, d_dup=dd, d_push=dp)
        for d, t, b, f, dd, dp in itertools.product(DETECTORS, TAUS, BAND_NAMES, FLATS, D_DUPS, D_PUSHES)]
assert len(GRID) == 192
FIT_DOMAINS = ("F-real", "F-ph")
VIEW_CLASSES = ("frontal", "oblique", "profile", "back")
ROLE_CLASSES = ("line", "group", "staff", "other")


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(file, sha_file):
    want = Path(sha_file).read_text().split()[0]
    got = sha(file)
    assert got == want, f"{file}: sha256 {got} != {want} in {sha_file}"
    return got


def verify_frozen():
    return {"FROZEN.md": verify(I09 / "FROZEN.md", I09 / "FROZEN.sha256"),
            "DEVIATIONS.md": verify(I09 / "DEVIATIONS.md", I09 / "DEVIATIONS.sha256")}


def setting_key(s) -> str:
    return (f"{s['detector']} tau={s['tau']:.2f} band={s['band']} flat={'on' if s['flat'] else 'off'} "
            f"d_dup={s['d_dup']:.2f} d_push={s['d_push']:.2f}")


def restrictive_key(s):
    """Least restrictive first: lower tau, wider band, flat off, smaller d_dup, smaller d_push."""
    return (s["tau"], BAND_NAMES.index(s["band"]), int(s["flat"]), s["d_dup"], s["d_push"])


def view_angle_deg(p, cam_xy) -> float:
    b = math.atan2(cam_xy[1] - p.y, cam_xy[0] - p.x)
    return abs(math.degrees(wrap_angle(b - p.yaw)))


def view_class(a: float) -> str:
    if a < 45:
        return "frontal"
    if a < 75:
        return "oblique"
    if a <= 105:
        return "profile"
    return "back"


def err_deg(yaw, true_yaw):
    return None if yaw is None else abs(math.degrees(wrap_angle(yaw - true_yaw)))


def facing_summary(errs) -> dict:
    """errs: list of error (deg) or None (unknown). Unknown counted as wrong (180 deg in medians) and among known;
    axis versions (min(e, 180 - e)) too."""
    n = len(errs)
    known = [e for e in errs if e is not None]
    axis = [min(e, 180.0 - e) for e in known]
    full_u = [180.0 if e is None else e for e in errs]
    axis_u = [180.0 if e is None else min(e, 180.0 - e) for e in errs]
    return {
        "n": n, "known": len(known), "unknown": n - len(known),
        "w45": sum(e <= 45 for e in known), "w45_share": (sum(e <= 45 for e in known) / n) if n else None,
        "w45_known_share": (sum(e <= 45 for e in known) / len(known)) if known else None,
        "median_u": statistics.median(full_u) if n else None,
        "median_known": statistics.median(known) if known else None,
        "fb": sum(e < 90 for e in known), "fb_share": (sum(e < 90 for e in known) / n) if n else None,
        "fb_known_share": (sum(e < 90 for e in known) / len(known)) if known else None,
        "axis_w45": sum(a <= 45 for a in axis), "axis_w45_share": (sum(a <= 45 for a in axis) / n) if n else None,
        "axis_w45_known_share": (sum(a <= 45 for a in axis) / len(known)) if known else None,
        "axis_median_u": statistics.median(axis_u) if n else None,
        "axis_median_known": statistics.median(axis) if axis else None,
    }


def score_scene(people, sc, th, variant):
    """The I03 scorer, unchanged, on one scene's people (people_for output), and per-person rows for every visible
    true person: matched or not, the detection's yaws (all variants), errors, view class, role class."""
    pts = [(p["xy"][0], p["xy"][1], p["yaws"][variant] if "yaws" in p else p["yaw"]) for p in people]
    r = PS.score_people(pts, sc, th, exempt_xy=PS.exempt_points(sc.truth), facing="full")
    info = PS.truth_info(sc, th)
    roles = info["roles"]
    cam_xy = sc.observation.camera.T_world_camera[:2, 3]
    by_tid = {tid: num for num, tid in r["matches"].items()}
    rows = []
    for p in info["visible"]:
        a = view_angle_deg(p, cam_xy)
        row = {"pid": p.id, "role_class": roles[p.id], "role": p.role, "view_angle": a, "view": view_class(a),
               "true_yaw": p.yaw, "x": p.x, "y": p.y, "matched": p.id in by_tid}
        if p.id in by_tid:
            d = people[by_tid[p.id] - 1]
            row["det_number"] = by_tid[p.id]
            row["xy_err"] = math.dist(d["xy"], (p.x, p.y))
            row["score"] = d["score"]
            if "yaws" in d:
                row["yaws"] = d["yaws"]
                row["errs"] = {v: err_deg(y, p.yaw) for v, y in d["yaws"].items()}
                row["facing_diag"] = d.get("facing_diag")
            row["yaw"] = d["yaws"][variant] if "yaws" in d else d["yaw"]
            row["err"] = err_deg(row["yaw"], p.yaw)
        rows.append(row)
    return r, rows


def jdump(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1, default=str))


# ------------------------------------------------------------------------------------------------ evaluation helpers
REAL_SETS = ("real", "staffed", "plain")
SETS = ("real", "staffed", "plain", "placeholder")
EVAL_DOMAINS = ("E-real", "E-ph")


def set_of(scenes_dir) -> str:
    """As phase1b/i03/stageA/evaluate.py set_of."""
    d = str(scenes_dir) + "/"
    for name in REAL_SETS:
        if f"realistic_final/{name}/" in d:
            return name
    if "dev_new/seed" in d or d.endswith("placeholder/eval/") or d.endswith("/placeholder_first14/"):
        return "placeholder"
    raise SystemExit(f"unknown set: {scenes_dir}")


def domain_of(sset: str) -> str:
    return "E-real" if sset in REAL_SETS else "E-ph"


def thresholds_for(refs):
    """As phase1b/i03/stageA/evaluate.py thresholds_for: the runs' recorded thresholds, asserted equal on the keys
    the scorer uses."""
    import research_harness as H
    ths = [H.recorded_config(r)[1].thresholds for r in refs]
    for th in ths:
        PS.check_thresholds(th)
    keys = ("person_radius_m", "visibility_min_samples", "visibility_depth_tolerance_m", "grid_resolution_m")
    for th in ths[1:]:
        assert all(getattr(th, k) == getattr(ths[0], k) for k in keys)
    return ths[0]


def layout_of(scene_id: str) -> str:
    for a in REAL_SETS:
        if scene_id.startswith(a + "_"):
            return scene_id[len(a) + 1:]
    return scene_id


def p90(xs):
    """Nearest-rank 90th percentile."""
    s = sorted(xs)
    return s[max(0, math.ceil(0.9 * len(s)) - 1)] if s else None


def boot_ci(groups: dict, stat, n_boot: int = 2000, seed: int = 0):
    """Percentile 95% interval of stat(list of items) over cluster draws: whole groups (layouts) resampled with
    replacement (as phase1/a0_facing/a0_facing.py boot_ci)."""
    import random
    keys = sorted(groups)
    if len(keys) < 2:
        return (None, None)
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        draw = [it for _ in keys for it in groups[rng.choice(keys)]]
        v = stat(draw)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            vals.append(v)
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))])
