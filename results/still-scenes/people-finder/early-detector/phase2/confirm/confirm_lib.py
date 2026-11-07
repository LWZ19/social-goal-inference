"""I09 unseen-layout confirmation: shared project-venv code (FROZEN.md sections 3, 4, 6, 7; DEVIATIONS_1.md).

- C1: every pinned file of FROZEN section 3 (and section 7's development rows) is asserted by sha256 at import.
- A1's code is reused by import, never copied: P = phase1c/i09/a1/pipeline.py, C = common.py, PS = people_scorer.py.
- score_image(): P.run_image (lift, filters, duplicates, F5) then C.score_scene (PS.score_people, unchanged) and the
  per-scene and per-person rows; the people are produced before any situation, label or row is read.
- measures() / tabulation helpers shared by C2 (development reproduction), the dry run and score.py.
Torch-free. Writes nothing.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import collections  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent                     # phase2/i09/confirm
RESEARCH = Path("results/2026-10-17-beyond-vlm")
WORKTREE = Path(".-scenes")
assert HERE == RESEARCH / "phase2/i09/confirm", HERE
A1 = RESEARCH / "phase1c/i09/a1"
SEEDS = RESEARCH / "phase2/seeds/realistic"

PINNED = {   # FROZEN.md section 3
    A1 / "pipeline.py": "4826be88d6939f3f5e4d14375a2fc9698abd94973ca46243d6697e263e5d063a",
    A1 / "common.py": "cdc33d742f6db5d09c3ad9a4557c1210ef10f8909484cee101ce983f9f8b2f53",
    RESEARCH / "phase1b/i03/people_scorer.py": "e12776eaddac1de0af75b58f8a272cd0a13332596da626bcc81b085f5900d685",
    A1 / "sidecar/infer.py": "bee1322fd692bf0e85e820871413d486fda04d5770f72b5d3d75e18266709484",
    A1 / "chosen.json": "19b1e16340d810e75f8f772eeeabeaf4b788e09614e74d78d3afc206531119ff",
    A1 / "chosen_fix.json": "fe8177165773f7208d417227c358adae7d04326bd29cb6c75d73ed87b3f83f52",
    A1 / "ADDENDUM_FIX.md": "3a7d185526e3d9801bd2adcdaf529bfbe744c1a636dc8e03d0f53de45d2e208c",
    RESEARCH / "phase1c/i09/FROZEN.md": "f315e96c1145ad9f2c86128d16711bb5305690d99f5d8513c0177290c474df15",
    WORKTREE / "configs/default.yaml": "c651f4c484fff15d33290e13f28bc53c9221c8edd79c0b7a24f24a128098b4fc",
}
DEV_ROWS = {  # FROZEN.md section 7 (as in a1/EVAL.sha256)
    A1 / "people_rows.jsonl": "96384dab0cb313715aa27c04fe1f8f4274b5b6ef3a8f83d6e1aa6cdf248f9f86",
    A1 / "rows.jsonl": "d0d889051e97720f93df32a6dfbeab1126438385cf438322fa49daeb2331d51f",
    A1 / "summary.json": "f2abcf10dc74623f43ebfd8f4f332a8b43320161d28b73064ad63afa703df60c",
}
DEV_RAW = {A1 / "raw_eval/rtdetr_v2_r50vd.jsonl": "8a03b82ca89bb82bdd0b10a8e56944b8562f3a580967087646e408dd4862a97d",
           A1 / "eval_people.json": "ffa595453c94b444c8dd030d053d37ac103441a7e74f07e3f172836b3d0c24b3",
           A1 / "manifest_eval.json": "6c321c3800d0b76632da0711a2385dce4915a28e9cd1f563ff3f8a5782be0b22"}
FROZEN_SETTING = {"detector": "rtdetr_v2_r50vd", "tau": 0.25, "band": "wide", "flat": False, "d_dup": 0.5,
                  "d_push": 0.2}
VARIANT = "F5"
ROW_IDS = ("real_eval_00_no_one_place", "real_eval_01_no_one_place", "real_eval_02_no_one_place",
           "real_eval_03_no_one_deep_place", "real_eval_04_no_one_object", "real_eval_05_no_one_person",
           "real_eval_06_waiting_line_1", "real_eval_07_waiting_line_2", "real_eval_08_waiting_line_4",
           "real_eval_09_waiting_line_6", "real_eval_10_conversation_group_2", "real_eval_11_conversation_group_3",
           "real_eval_12_conversation_group_4", "real_eval_13_waiting_line_3")
SEED_RANGE = range(200, 220)
REQUIRED_SEEDS = range(200, 210)
MARK_Z = 1.1


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sha_file(sha_file: Path) -> dict:
    out = {}
    for line in Path(sha_file).read_text().splitlines():
        if not line.strip():
            continue
        want, name = line.split()
        got = sha(sha_file.parent / name)
        assert got == want, f"{name}: sha256 {got} != {want} in {sha_file}"
        out[name] = want
    return out


def verify_protocol() -> dict:
    """FROZEN.md and every DEVIATIONS_<n>.md with its own sha file, then the pinned files (C1)."""
    out = verify_sha_file(HERE / "FROZEN.sha256")
    assert out["FROZEN.md"] == "f55341c03cd5e1d38c91b7e0142ef10f0870ccb2b43c5bf4edfac07d1996e751", out
    for dev in sorted(HERE.glob("DEVIATIONS_*.md")):
        out |= verify_sha_file(HERE / (dev.stem + ".sha256"))
    assert "DEVIATIONS_1.md" in out
    for path, want in PINNED.items():
        got = sha(path)
        assert got == want, f"C1 code identity: {path} sha256 {got} != {want}"
        out[str(path.relative_to(RESEARCH) if path.is_relative_to(RESEARCH) else path.relative_to(WORKTREE))] = got
    return out


PROTOCOL_HASHES = verify_protocol()

sys.path.insert(0, str(A1))
import common as C  # noqa: E402  (inserts the harness and phase1b/i03 paths)
import pipeline as P  # noqa: E402
import people_scorer as PS  # noqa: E402
import research_harness as H  # noqa: E402
from sgi.geometry import in_image, project  # noqa: E402

assert Path(C.__file__).resolve() == A1 / "common.py" and Path(P.__file__).resolve() == A1 / "pipeline.py"
assert Path(PS.__file__).resolve() == RESEARCH / "phase1b/i03/people_scorer.py"
assert Path(H.__file__).resolve() == WORKTREE / "docs/model_notes/tools/research_harness.py"


def frozen_setting():
    chosen = json.loads((A1 / "chosen.json").read_text())
    fix = json.loads((A1 / "chosen_fix.json").read_text())
    assert fix["setting"] == chosen["setting"] == FROZEN_SETTING, (fix["setting"], chosen["setting"])
    assert fix["fix_adopted"] is True and fix["facing_variant"] == VARIANT
    assert P.BANDS[FROZEN_SETTING["band"]] == (1.00, 2.30)
    return dict(FROZEN_SETTING), VARIANT


# ------------------------------------------------------------------------------------------------ one image
def best_raw_box(raw, cam, x, y):
    """DEVIATIONS_1.md C7: the highest-scoring raw box (score >= 0.05) containing the projection of (x, y, 1.1 m)."""
    mk = project(cam, (x, y, MARK_Z))
    if mk is None or not in_image(cam, mk[0], mk[1]):
        return None
    for b in raw["boxes"]:                                   # descending score
        if b[0] <= mk[0] <= b[2] and b[1] <= mk[1] <= b[3]:
            return {"score": b[4], "box": b[:4]}
    return None


def filter_trace(cands, s):
    lo, hi = P.BANDS[s["band"]]
    out = collections.Counter()
    for c in cands:
        if c["reject"] is not None:
            out["depth: " + c["reject"]] += 1
        elif c["z_top"] > hi:
            out["z_top above band"] += 1
        elif not c["top_touch"] and c["z_top"] < lo:
            out["z_top below band"] += 1
        elif s["flat"] and c["flat"]:
            out["flat"] += 1
        else:
            out["passed filters (kept or duplicate)"] += 1
    return dict(out)


def score_image(raw, sc, th, s, variant, meta):
    """People from P.run_image (timed), then scoring. Returns (scene row, per-person rows, people out)."""
    obs = sc.observation
    cam = obs.camera
    t0 = time.perf_counter()
    ppl = P.run_image(raw, obs.depth, cam, s, variant)
    t_proj = time.perf_counter() - t0
    # ---- the people are fixed above; truth is read only from here on
    r, prow = C.score_scene(ppl, sc, th, variant)
    info = PS.truth_info(sc, th)
    lh = info["lower_hidden"]
    vis = info["visible"]
    false_rows = []
    for n in r["false_numbers"] + r["exempt_numbers"]:
        d = ppl[n - 1]
        near_vis = min((math.dist(d["xy"], (p.x, p.y)) for p in vis), default=None)
        false_rows.append({"number": n, "exempt": n in r["exempt_numbers"], "xy": d["xy"], "score": d["score"],
                           "box": d["box"], "nearest_visible_m": near_vis,
                           "near_hidden": any(math.dist(d["xy"], (p.x, p.y)) <= PS.NEAR_M for p in info["hidden"])})
    cands = P.candidates(raw, obs.depth, cam, s["tau"])          # diagnostics only
    row = dict(meta) | {k: r[k] for k in PS.POOL_KEYS} | {
        "recall": r["recall"], "xy_err": r["xy_err"], "facing": r["facing"], "misses": r["misses"],
        "false_detail": false_rows, "pairs": r["pairs"], "people": r["people"], "matches": r["matches"],
        "n_candidates": len(cands), "filter_trace": filter_trace(cands, s), "n_kept": len(ppl),
        "t_project_s": t_proj, "n_boxes_raw": len(raw["boxes"])}
    people_rows = []
    for pr in prow:
        extra = {"lower_hidden": bool(lh[pr["pid"]]), "staff_behind": bool(pr["role"] and lh[pr["pid"]])}
        if not pr["matched"]:
            extra["best_raw_box"] = best_raw_box(raw, cam, pr["x"], pr["y"])
        people_rows.append(dict(meta) | pr | extra)
    out = [{"x": p["xy"][0], "y": p["xy"][1], "yaw": p["yaw"], "score": p["score"], "box": p["box"],
            "torso_pixel": list(p["torso_pixel"]), "yaws_all_variants": p["yaws"]} for p in ppl]
    return row, people_rows, out


# ------------------------------------------------------------------------------------------------ measures
def facing_groups(prows):
    m = [p for p in prows if p["matched"]]
    out = {vc: C.facing_summary([p["err"] for p in m if p["view"] == vc]) for vc in C.VIEW_CLASSES}
    out["non-profile"] = C.facing_summary([p["err"] for p in m if p["view"] != "profile"])
    out["all matched"] = C.facing_summary([p["err"] for p in m])
    return out


def measures(rows, prows) -> dict:
    """The bar statistics and section 7's measures from scene rows and per-person rows (F5 'err')."""
    Pp = PS.pool(rows)
    by_role = {c: [sum(1 for p in prows if p["role_class"] == c and p["matched"]),
                   sum(1 for p in prows if p["role_class"] == c)] for c in C.ROLE_CLASSES}
    assert Pp["n_visible"] == len(prows) and Pp["matched"] == sum(p["matched"] for p in prows)
    fg = facing_groups(prows)
    xy = [p["xy_err"] for p in prows if p["matched"]]
    assert sorted(xy) == sorted(Pp["xy_err"])
    return {"images": Pp["scenes"], "visible": Pp["n_visible"], "matched": Pp["matched"],
            "recall": Pp["matched"] / Pp["n_visible"] if Pp["n_visible"] else None,
            "by_role": by_role, "false": Pp["false"], "fb_per_image": Pp["fb_per_scene"],
            "exempt": Pp["exempt"], "near_hidden": Pp["near_hidden"], "merges": Pp["merges"],
            "merged_pairs": Pp["merged_pairs"], "neighbour_pairs": Pp["neighbour_pairs"],
            "xy_median": Pp["xy_err_median"], "n_xy": len(Pp["xy_err"]),
            "np_n": fg["non-profile"]["n"], "np_w45": fg["non-profile"]["w45"],
            "np_median_u": fg["non-profile"]["median_u"],
            "pf_n": fg["profile"]["n"], "pf_fb": fg["profile"]["fb"],
            "all_n": fg["all matched"]["n"], "all_w45": fg["all matched"]["w45"],
            "coverage": [fg["all matched"]["known"], fg["all matched"]["n"]],
            "staff_visible": by_role["staff"][1], "facing": fg}


def bars(m) -> tuple[dict, dict]:
    """FROZEN.md section 5, on point estimates: >= / <= for 'at least' / 'at most', < / > for 'below' / 'more than'."""
    line = m["by_role"]["line"]
    s = {"S1 recall >= 0.95": m["recall"] >= 0.95,
         "S2 line recall >= 0.95": (line[0] / line[1] >= 0.95) if line[1] else None,
         "S3 false per image <= 0.10": m["fb_per_image"] <= 0.10,
         "S4 median position error <= 0.15 m": m["xy_median"] <= 0.15,
         "S5 non-profile facing median <= 25 deg": (m["np_median_u"] <= 25) if m["np_n"] else None,
         "S6 non-profile within 45 >= 0.85": (m["np_w45"] / m["np_n"] >= 0.85) if m["np_n"] else None,
         "S7 profile front/back >= 0.90": (m["pf_fb"] / m["pf_n"] >= 0.90) if m["pf_n"] else None}
    k = {"K1 recall < 0.90": m["recall"] < 0.90,
         "K2 false per image > 0.30": m["fb_per_image"] > 0.30,
         "K3 all matched within 45 < 0.70": (m["all_w45"] / m["all_n"] < 0.70) if m["all_n"] else None}
    return s, k


def verdict_of(s, k, complete=True) -> str:
    if not complete:
        return "incomplete"
    if any(v is True for v in k.values()):
        return "kill"
    if all(v is True for v in s.values()):
        return "success"
    return "neither"


def section7_numbers(rows, prows) -> dict:
    """The comparable section-7 numbers (exact values)."""
    m = measures(rows, prows)
    return {"recall": [m["matched"], m["visible"]], "line": m["by_role"]["line"], "group": m["by_role"]["group"],
            "staff": m["by_role"]["staff"], "other": m["by_role"]["other"], "false": [m["false"], m["images"]],
            "xy_median": m["xy_median"], "n_xy": m["n_xy"], "np_w45": [m["np_w45"], m["np_n"]],
            "np_median_u": m["np_median_u"], "pf_fb": [m["pf_fb"], m["pf_n"]], "all_w45": [m["all_w45"], m["all_n"]]}


def load_dev_recorded():
    """A1's recorded rows (section 7), hash-checked: (E-real rows, E-real people rows)."""
    for path, want in DEV_ROWS.items():
        assert sha(path) == want, path
    rows = [json.loads(x) for x in (A1 / "rows.jsonl").read_text().splitlines()]
    prows = [json.loads(x) for x in (A1 / "people_rows.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["domain"] == "E-real"]
    prows = [p for p in prows if p["domain"] == "E-real"]
    assert len(rows) == 42, len(rows)
    return rows, prows


def jdump(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1, default=str))


def fmt_k(k, n):
    if not n:
        return f"{k}/{n} -"
    lo, hi = PS.wilson(k, n)
    return f"{k}/{n} = {k / n:.3f} [{lo:.3f}, {hi:.3f}]"


def f3(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def median_or_none(xs):
    return statistics.median(xs) if xs else None
