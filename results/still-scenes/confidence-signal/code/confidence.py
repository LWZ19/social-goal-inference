"""Phase 3 confidence, pass 1 (FROZEN.md section 9, step 2): every signal and checks R1-R7. No AUROC, no rate, no
criterion is computed here.

Zero calls. Runs and scenes only through docs/model_notes/tools/research_harness.py. Earlier code is imported
unchanged: phase1/i14_v0/i14.py (plan, key), phase1c/i09/a2/replay_d.py (verify_hashes, load_detector_people,
match_xy, build, MATCH_M; loaded by path) and, through it, phase1/i01_repairs/repairs.py (scene_info).

    PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 \
        .venv/bin/python -B code/confidence.py run CELL [CELL ...]    -> rows_<cell>.jsonl, appends pass1.log
        .venv/bin/python -B code/confidence.py checks                   -> checks.txt (R1-R7), appends pass1.log

Cells: old_qwen old_glm old_deepseek watch_qwen watch_glm watch_deepseek new_glm new_deepseek new_haiku new_sonnet
new_gemini. The results do not depend on how the cells are split between processes.
"""
from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
from fractions import Fraction
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

OUT = Path(__file__).resolve().parents[1]                       # .../phase3/confidence
RD = OUT.parents[1]                                             # .../2026-10-17-beyond-vlm
assert RD.name == "2026-10-17-beyond-vlm", RD
FROZEN_SHA = "2a3a890a3b23ff466e8b72c63e6dbef9143d7738f68badc34a4fad6a17ccb4b3"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_frozen() -> dict:
    """FROZEN.md against FROZEN.sha256 (and the hash given to this agent); DEVIATIONS.md against its hash if any."""
    listed = (OUT / "FROZEN.sha256").read_text().split()
    assert listed[1] == "FROZEN.md" and listed[0] == FROZEN_SHA, listed
    got = sha256(OUT / "FROZEN.md")
    assert got == FROZEN_SHA, f"FROZEN.md hash mismatch: {got}"
    out = {"FROZEN.md": got}
    if (OUT / "DEVIATIONS.sha256").exists():
        h, name = (OUT / "DEVIATIONS.sha256").read_text().split()[:2]
        got_d = sha256(OUT / name)
        assert got_d == h, f"{name} hash mismatch: {got_d} != {h}"
        out[name] = got_d
    return out


sys.path.insert(0, "docs/model_notes/tools")
sys.path.insert(0, str(RD / "phase1" / "i14_v0"))
import research_harness as H  # noqa: E402
import i14  # noqa: E402  (unchanged: plan, key, REAL_ARMS)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RDD = _load("i09_a2_replay_d", RD / "phase1c" / "i09" / "a2" / "replay_d.py")      # unchanged
RP = RDD.R                                                                          # repairs.py, unchanged

S23 = [("s23", m) for m in ("glm53", "dsv41", "haiku55", "sonnet55_low", "gemini38_low")]
CELL_NAME = {"qwen38_low": "qwen", "glm53": "glm", "dsv41": "deepseek", "haiku55": "haiku",
             "sonnet55_low": "sonnet", "gemini38_low": "gemini"}
CELLS = ("old_qwen", "old_glm", "old_deepseek", "watch_qwen", "watch_glm", "watch_deepseek",
         "new_glm", "new_deepseek", "new_haiku", "new_sonnet", "new_gemini")
FLAGS = ("exclusion_broken", "far_from_target", "constraint_unusable")
LAYOUT = re.compile(r"eval_[0-9]{2}")
TARGET_SPREAD_M = 1.0                                           # S4: targets more than this apart differ
INF = float("inf")

# FROZEN.md section 2: n, successes (I14 top-1), wrong target (I14 T1)
EXPECTED = {"old_qwen": (143, 134, 6), "old_glm": (141, 96, 14), "old_deepseek": (157, 109, 14),
            "watch_qwen": (36, 32, 0), "watch_glm": (36, 23, 2), "watch_deepseek": (36, 23, 0),
            "new_glm": (141, 86, 22), "new_deepseek": (151, 106, 15), "new_haiku": (146, 134, 1),
            "new_sonnet": (143, 134, 7), "new_gemini": (135, 109, 26)}
A2_FILE = {"old_qwen": "rows_old_qwen38_low.jsonl", "old_glm": "rows_old_glm53.jsonl",
           "old_deepseek": "rows_old_dsv41.jsonl", "new_glm": "rows_new_glm53.jsonl",
           "new_deepseek": "rows_new_dsv41.jsonl", "new_haiku": "rows_new_haiku55.jsonl",
           "new_sonnet": "rows_new_sonnet55_low.jsonl", "new_gemini": "rows_new_gemini38_low.jsonl"}


def now():
    return time.strftime("%F %T %z")


def log(msg, fh):
    line = f"{now()} [{os.getpid()}] {msg}"
    print(line, flush=True)
    with open(OUT / "pass1.log", "a") as f:
        f.write(line + "\n")


def plan():
    """(set label, RunRef): i14.plan('oldwatch') then the s23 con runs on the realistic arms (FROZEN.md section 2)."""
    out = [(label, ref) for label, ref in i14.plan("oldwatch")]
    for stage, model in S23:
        for arm in i14.REAL_ARMS:
            refs = H.runs(stage=stage, arm=arm, model=model, method="con")
            assert len(refs) == 1, (stage, arm, model, refs)
            out.append(("new", refs[0]))
    return out


def cell_of(label, ref):
    return f"{label}_{CELL_NAME[ref.model]}"


def layout_of(label, ref, rec):
    if label == "watch":
        return f"{ref.arm}/{rec.scene_id}"
    m = LAYOUT.search(rec.scene_id)
    assert m is not None, rec.scene_id
    return m.group(0)


def wrap_deg(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi) * 180.0 / math.pi


def r6(x):
    return x if x == INF else round(x, 6)


# ------------------------------------------------------------------------------------------------ the side data
def load_i04():
    out = {}
    for p in sorted((RD / "phase1d" / "i04").glob("rows_*.jsonl")):
        for line in open(p):
            r = json.loads(line)
            assert r["key"] not in out
            out[r["key"]] = {"cell": r["cell"], "layout": r["layout"], "flags": r["flags"], "top1": r["top1"],
                             "conc": r["V1"]["conc"]}
    return out


def load_i14():
    out = {}
    for name in ("rows_oldwatch.jsonl", "rows_new.jsonl"):
        for line in open(RD / "phase1" / "i14_v0" / name):
            r = json.loads(line)
            if r["set"] == "new" and r["stage"] == "s21":
                continue
            k = f"{r['run']}/{r['scene']}/{r['ep']}/{r['trial']}"
            assert k not in out
            out[k] = {"rec_success": r["rec_success"], "target_ok": r["target_ok"]}
    return out


def load_a2(cell):
    out = {}
    for line in open(RD / "phase1c" / "i09" / "a2" / A2_FILE[cell]):
        r = json.loads(line)
        out[(r["run"], r["scene"], r["episode"], r["trial"])] = r
    return out


# ------------------------------------------------------------------------------------------------ signals
def s2_s3(det, cons, dets):
    """S2 = 1 - F1 (one-to-one greedy within 0.5 m, replay_d.match_xy); S3 = share of named numbers unsupported."""
    pairs = RDD.match_xy(det, dets)
    n_v, n_d, m = len(det), len(dets), len(pairs)
    s2 = Fraction(0) if n_v + n_d == 0 else 1 - Fraction(2 * m, n_v + n_d)
    by_num = {d.number: d for d in det}
    named = sorted({n for c in cons for n in (c.get("people") or [])})
    unsup, absent = [], []
    for n in named:
        d = by_num.get(n)
        if d is None:
            unsup.append(n)
            absent.append(n)
            continue
        if not any(math.dist(d.xy, (q["x"], q["y"])) <= RDD.MATCH_M for q in dets):
            unsup.append(n)
    s3 = Fraction(0) if not named else Fraction(len(unsup), len(named))
    return pairs, dict(S2=float(s2), S2_frac=f"{s2.numerator}/{s2.denominator}", n_v=n_v, n_d=n_d, n_match=m,
                       S3=float(s3), S3_frac=f"{s3.numerator}/{s3.denominator}", named=named,
                       unsupported=unsup, named_absent=absent)


def s4_groups(recs):
    """{(scene, episode): S4 fields} over every allowed record of one run (FROZEN.md section 4, S4)."""
    groups = collections.defaultdict(list)
    for rec in recs:
        groups[(rec.scene_id, rec.episode_id)].append(rec)
    out = {}
    for key, members in groups.items():
        assert len({r.situation for r in members}) == 1, key
        trials = sorted(r.trial for r in members)
        poses = [r.pose_world for r in members]
        tws = [((r.target or {}).get("world") if r.target else None) for r in members]
        for p, t in zip(poses, tws):
            assert (p is None) == (t is None), key        # place and target world co-occur (format check)
        missing = sum(p is None for p in poses)
        if missing:
            out[key] = dict(S4=INF, S4_place=INF, S4_reason="missing_place", S4_trials=trials, S4_missing=missing,
                            S4_target_spread=None, S4_place_spread=None)
            continue
        idx = range(len(members))
        pairs = [(i, j) for i in idx for j in idx if i < j]
        tspread = max((math.dist(tws[i][:2], tws[j][:2]) for i, j in pairs), default=0.0)
        pspread = max((math.dist(poses[i][:2], poses[j][:2]) for i, j in pairs), default=0.0)
        s4_place = r6(pspread)
        s4 = INF if tspread > TARGET_SPREAD_M else s4_place
        out[key] = dict(S4=s4, S4_place=s4_place, S4_reason="target_spread" if s4 == INF else None,
                        S4_trials=trials, S4_missing=0, S4_target_spread=round(tspread, 6),
                        S4_place_spread=round(pspread, 6))
    return out


def p1_fields(rec, det, cons, dets, pairs):
    """P1 / P1b and the D re-solve (A2's setting D, built with replay_d.build, unchanged)."""
    from sgi.solver import target_person
    from sgi.types import GroundedTarget
    sc, th, _vis, _oracle = RP.scene_info(rec)
    obs = sc.observation
    nums = [d.number for d in det]
    cnums = [n for c in cons for n in (c.get("people") or [])]
    M = max(nums + cnums, default=0)
    people, out_cons, dropped, added, info = RDD.build(det, cons, dets, pairs, False, M, obs, th)
    gt = GroundedTarget.model_validate(rec.target)
    tnum_c = target_person(obs, gt, det, th)
    tnum_d = target_person(obs, gt, people, th)
    row = dict(M=M, D_dropped=sorted(dropped), D_added=len(added), tnum_C=tnum_c, tnum_D=tnum_d,
               D_exception=None, D_pose=None, D_success=None, D_heading_diff_deg=None)
    try:
        res = H.resolve(rec, people=[p.model_dump() for p in people], constraints=out_cons)
        pose = res.pose_world
    except Exception as e:                                         # listed with its exception (FROZEN.md, P1)
        res, pose = None, None
        row["D_exception"] = repr(e)
    if pose is None:
        p1 = INF
        if row["D_exception"] is None:
            row["D_exception"] = "no pose_world"
    else:
        p1 = r6(math.dist(rec.pose_world[:2], pose[:2]))
        row.update(D_pose=list(pose), D_success=bool(res.score.success),
                   D_heading_diff_deg=wrap_deg(pose[2], rec.pose_world[2]))
    row["P1"] = p1
    row["P1b"] = INF if (p1 == INF or tnum_c != tnum_d) else p1
    return row


# ------------------------------------------------------------------------------------------------ run
def mode_run(cells):
    from sgi.types import DetectedPerson
    t_all = time.monotonic()
    fz = verify_frozen()
    hashes = RDD.verify_hashes()
    log(f"run {cells}: FROZEN verified {fz}; replay_d.verify_hashes {hashes}", None)
    det_of = RDD.load_detector_people()
    i04, i14r = load_i04(), load_i14()
    wanted = set(cells)
    outputs = {c: [] for c in cells}
    for label, ref in plan():
        cell = cell_of(label, ref)
        if cell not in wanted:
            continue
        t0 = time.monotonic()
        realistic = label in ("old", "new")
        a2 = load_a2(cell) if realistic else {}
        recs = list(H.records(ref))
        if label == "watch":
            recs = [r for r in recs if r.situation == "audience"]
        g4 = s4_groups(recs)
        n_goal = 0
        for rec in recs:
            if not rec.solver_step:
                continue
            n_goal += 1
            k = i14.key(rec)
            det = [DetectedPerson.model_validate(p) for p in rec.people]
            assert len({d.number for d in det}) == len(det)
            cons = [u["constraint"] for u in rec.constraints]
            dk = (str(ref.scenes_dir), rec.scene_id)
            det_present = dk in det_of
            dets = det_of[dk][1] if det_present else []
            row = dict(key=k, cell=cell, set=label, stage=ref.stage, model=ref.model, arm=ref.arm, run=ref.folder,
                       scene=rec.scene_id, episode=rec.episode_id, trial=rec.trial, situation=rec.situation,
                       layout=layout_of(label, ref, rec), success=bool(rec.score.get("success")),
                       target_correct=bool(rec.score.get("target_correct")),
                       pose=list(rec.pose_world), det_present=det_present,
                       det_folder=det_of[dk][0] if det_present else None)
            pairs, f23 = s2_s3(det, cons, dets)
            row.update(f23)
            row.update(g4[(rec.scene_id, rec.episode_id)])
            fl = rec.result["flags"]
            row["flags"] = {f: bool(fl[f]) for f in FLAGS}
            row["S5a"] = int(any(row["flags"].values()))
            # checks R2, R3, R6 (I14, I04)
            row["rescore_success"] = bool(H.rescore(rec).success)
            a = i14r.get(k)
            row["i14_join"] = a is not None
            row["i14_rec_success"] = None if a is None else a["rec_success"]
            row["i14_target_ok"] = None if a is None else a["target_ok"]
            b = i04.get(k)
            row["i04_join"] = b is not None
            row["i04_top1"] = None if b is None else b["top1"]
            row["i04_flags_equal"] = None if b is None else all(bool(b["flags"][f]) == row["flags"][f] for f in FLAGS)
            row["i04_cell_equal"] = None if b is None else b["cell"] == cell
            row["i04_layout_equal"] = None if b is None else b["layout"] == row["layout"]
            row["c_V1"] = None if b is None else b["conc"]
            row["S5b"] = None if b is None else 1.0 - b["conc"]
            if realistic:
                rr = H.resolve(rec)                                   # R4, step 0
                row["step0_moved_m"] = rr.moved_m
                row["step0_reproduced"] = bool(rr.reproduced)
                row.update(p1_fields(rec, det, cons, dets, pairs))
                ar = a2.get((ref.folder, rec.scene_id, rec.episode_id, rec.trial))
                ok = ar is not None and ar["solver_step"] and "D_d" in ar
                row["a2_join"] = bool(ok)
                row["a2_D"] = ar["D"] if ok else None
                row["a2_moved_from_C_m"] = ar["D_d"].get("moved_from_C_m") if ok else None
                row["a2_tnum_C"] = ar.get("tnum_C") if ok else None
                row["a2_tnum_D"] = ar["D_d"].get("tnum") if ok else None
            else:
                row.update(P1=None, P1b=None)
            outputs[cell].append(row)
        log(f"{cell:15s} {ref.folder}: allowed {len(recs)}, goal steps {n_goal}, {time.monotonic() - t0:.1f} s", None)
    for cell, rows in outputs.items():
        with open(OUT / f"rows_{cell}.jsonl", "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        log(f"wrote rows_{cell}.jsonl: {len(rows)} rows", None)
    log(f"run {cells} done in {time.monotonic() - t_all:.1f} s", None)


# ------------------------------------------------------------------------------------------------ checks
def mode_checks():
    fz = verify_frozen()
    hashes = RDD.verify_hashes()
    rows = []
    for cell in CELLS:
        p = OUT / f"rows_{cell}.jsonl"
        assert p.exists(), f"missing {p}"
        rows += [json.loads(line) for line in open(p)]
    lines, ok = [], {}
    by_cell = collections.defaultdict(list)
    for r in rows:
        by_cell[r["cell"]].append(r)
    real = [r for r in rows if r["set"] in ("old", "new")]

    # R1
    i14_keys = set(load_i14())
    keys = [r["key"] for r in rows]
    bad_n = {c: (len(by_cell[c]), EXPECTED[c][0]) for c in CELLS if len(by_cell[c]) != EXPECTED[c][0]}
    r1 = not bad_n and len(set(keys)) == len(keys) and set(keys) == i14_keys
    ok["R1"] = r1
    lines.append(f"R1 I14 goal-step counts and key set: {'PASS' if r1 else 'FAIL'}; total {len(rows)} (unique "
                 f"{len(set(keys))}), I14 {len(i14_keys)}; key sets equal {set(keys) == i14_keys}; per-cell n "
                 + ", ".join(f"{c} {len(by_cell[c])}" for c in CELLS) + (f"; MISMATCH {bad_n}" if bad_n else ""))

    # R2
    bad2 = [r["key"] for r in rows if not (r["success"] == r["rescore_success"] == r["i14_rec_success"]
                                           == r["i04_top1"])]
    succ = {c: sum(r["success"] for r in by_cell[c]) for c in CELLS}
    bad_s = {c: (succ[c], EXPECTED[c][1]) for c in CELLS if succ[c] != EXPECTED[c][1]}
    r2 = not bad2 and not bad_s
    ok["R2"] = r2
    lines.append(f"R2 recorded success = rescore = I14 rec_success = I04 top1: {'PASS' if r2 else 'FAIL'}; "
                 f"{len(rows) - len(bad2)} of {len(rows)} agree; per-cell successes "
                 + ", ".join(f"{c} {succ[c]}" for c in CELLS) + (f"; MISMATCH {bad_s}" if bad_s else "")
                 + (f"; disagreeing keys {bad2[:10]}" if bad2 else ""))

    # R3
    bad3 = [r["key"] for r in rows if r["target_correct"] != r["i14_target_ok"]]
    wt = {c: sum(not r["target_correct"] for r in by_cell[c]) for c in CELLS}
    bad_w = {c: (wt[c], EXPECTED[c][2]) for c in CELLS if wt[c] != EXPECTED[c][2]}
    r3 = not bad3 and not bad_w
    ok["R3"] = r3
    lines.append(f"R3 target_correct = I14 target_ok: {'PASS' if r3 else 'FAIL'}; {len(rows) - len(bad3)} of "
                 f"{len(rows)} agree; per-cell wrong target " + ", ".join(f"{c} {wt[c]}" for c in CELLS)
                 + (f"; MISMATCH {bad_w}" if bad_w else "") + (f"; disagreeing keys {bad3[:10]}" if bad3 else ""))

    # R4
    bad4 = [(r["key"], r["step0_moved_m"]) for r in real if not r["step0_reproduced"]]
    r4 = not bad4 and len(real) == 1157
    ok["R4"] = r4
    lines.append(f"R4 step 0 (H.resolve(rec) within 1e-6 m): {'PASS' if r4 else 'FAIL'}; {len(real) - len(bad4)} of "
                 f"{len(real)} realistic records reproduced; max moved "
                 f"{max(r['step0_moved_m'] for r in real):.3e} m" + (f"; FAILED {bad4[:10]}" if bad4 else ""))

    # R5
    bad5s, bad5m, bad5t, inf_p1 = [], [], [], []
    max_gap = 0.0
    for r in real:
        if r["P1"] == INF:
            inf_p1.append((r["key"], r["D_exception"]))
        if not r["a2_join"]:
            continue
        if r["D_success"] != r["a2_D"]:
            bad5s.append(r["key"])
        if r["P1"] == INF or r["a2_moved_from_C_m"] is None:
            bad5m.append((r["key"], r["P1"], r["a2_moved_from_C_m"]))
        else:
            gap = abs(r["P1"] - r["a2_moved_from_C_m"])
            max_gap = max(max_gap, gap)
            if gap > 1e-4:
                bad5m.append((r["key"], r["P1"], r["a2_moved_from_C_m"]))
        if r["tnum_C"] != r["a2_tnum_C"] or r["tnum_D"] != r["a2_tnum_D"]:
            bad5t.append((r["key"], r["tnum_C"], r["a2_tnum_C"], r["tnum_D"], r["a2_tnum_D"]))
    n_a2 = sum(r["a2_join"] for r in real)
    r5 = not bad5s and not bad5m and not bad5t and n_a2 == len(real)
    ok["R5"] = r5
    lines.append(f"R5 A2's D per record: {'PASS' if r5 else 'FAIL'}; joined {n_a2} of {len(real)}; D success equal "
                 f"in {n_a2 - len(bad5s)}; |P1 - A2 moved_from_C_m| <= 1e-4 in {n_a2 - len(bad5m)} (max gap "
                 f"{max_gap:.2e} m); target numbers (C and D) equal in {n_a2 - len(bad5t)}; D successes "
                 f"{sum(bool(r['D_success']) for r in real)}, A2 D successes {sum(bool(r['a2_D']) for r in real)}"
                 + (f"; success mismatch {bad5s[:10]}" if bad5s else "") + (f"; move mismatch {bad5m[:10]}" if bad5m
                                                                             else "")
                 + (f"; target-number mismatch {bad5t[:10]}" if bad5t else ""))
    lines.append(f"   P1 = +inf (D re-solve gives no place): {len(inf_p1)} records" +
                 "".join(f"\n     {k}: {e}" for k, e in inf_p1))

    # R6
    bad6 = {"i04_join": [r["key"] for r in rows if not r["i04_join"]],
            "i04_flags_equal": [r["key"] for r in rows if not r["i04_flags_equal"]],
            "i04_cell_equal": [r["key"] for r in rows if not r["i04_cell_equal"]],
            "a2_join": [r["key"] for r in real if not r["a2_join"]],
            "det_present": [r["key"] for r in rows if not r["det_present"]],
            "i14_join": [r["key"] for r in rows if not r["i14_join"]]}
    r6_ok = not any(bad6.values())
    ok["R6"] = r6_ok
    lines.append(f"R6 joins: {'PASS' if r6_ok else 'FAIL'}; I04 joined {len(rows) - len(bad6['i04_join'])} of "
                 f"{len(rows)}, flags equal {len(rows) - len(bad6['i04_flags_equal'])}, cell equal "
                 f"{len(rows) - len(bad6['i04_cell_equal'])}; A2 solver-step join {len(real) - len(bad6['a2_join'])} "
                 f"of {len(real)}; detector people present {len(rows) - len(bad6['det_present'])} of {len(rows)}; "
                 f"I14 joined {len(rows) - len(bad6['i14_join'])}"
                 + "".join(f"; {k} missing {v[:5]}" for k, v in bad6.items() if v))
    lay_bad = [r["key"] for r in rows if not r["i04_layout_equal"]]
    lines.append(f"   (FROZEN.md section 2 statement) layout equals I04's layout field: {len(rows) - len(lay_bad)} of "
                 f"{len(rows)}" + (f"; differ {lay_bad[:5]}" if lay_bad else ""))

    # R7
    a2_listing = {}
    for line in (RD / "phase1c" / "i09" / "a2" / "A2.sha256").read_text().splitlines():
        h, f = line.split(None, 1)
        a2_listing[f.strip()] = h
    a2_files = ["replay_d.py"] + sorted(set(A2_FILE.values()))
    a2_bad = [f for f in a2_files if a2_listing.get(f) != sha256(RD / "phase1c" / "i09" / "a2" / f)]
    ok["R7"] = True                                                  # verify_frozen / verify_hashes assert above
    lines.append(f"R7 hashes: PASS; FROZEN.md {fz['FROZEN.md']}; replay_d.verify_hashes {hashes}")
    lines.append(f"   (provenance, not in R7) replay_d.py and the 8 A2 row files against A2.sha256: "
                 f"{'all match' if not a2_bad else 'MISMATCH ' + str(a2_bad)}")
    inputs = {f"phase1d/i04/{p.name}": sha256(p) for p in sorted((RD / "phase1d" / "i04").glob("rows_*.jsonl"))}
    inputs.update({f"phase1/i14_v0/{n}": sha256(RD / "phase1" / "i14_v0" / n)
                   for n in ("rows_oldwatch.jsonl", "rows_new.jsonl", "i14.py")})
    inputs["phase1/i01_repairs/repairs.py"] = sha256(RD / "phase1" / "i01_repairs" / "repairs.py")
    inputs["phase1c/i09/a1/eval_people.json"] = sha256(RD / "phase1c" / "i09" / "a1" / "eval_people.json")
    lines.append("   input file hashes (provenance): " + "".join(f"\n     {h}  {k}" for k, h in inputs.items()))
    lines.append(f"   pass-1 row file hashes: " + "".join(f"\n     {sha256(OUT / f'rows_{c}.jsonl')}  rows_{c}.jsonl"
                                                       for c in CELLS))

    allpass = all(ok.values())
    head = f"Phase 3 confidence, pass 1 checks R1-R7 ({now()}): {'ALL PASS' if allpass else 'FAIL: ' + str([k for k, v in ok.items() if not v])}"
    (OUT / "checks.txt").write_text(head + "\n" + "\n".join(lines) + "\n")
    log(head, None)
    print("\n".join(lines))
    return 0 if allpass else 1


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "run":
        cells = sys.argv[2:]
        assert cells and all(c in CELLS for c in cells), cells
        mode_run(cells)
    elif mode == "checks":
        raise SystemExit(mode_checks())
    else:
        raise SystemExit(f"unknown mode {mode}")
