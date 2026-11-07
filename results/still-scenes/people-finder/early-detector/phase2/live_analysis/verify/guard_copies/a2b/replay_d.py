"""I09 A2: solver replays with A1's frozen detector people (FROZEN.md section 11; a2/CLARIFICATIONS.md).

Zero calls; runs and scenes only through research_harness; the solver is the package's, unchanged, through
H.resolve, exactly as phase1/i01_repairs/repairs.py runs it (its one_run, scene_info, strip and solver_row are
imported and used unchanged). Reused from phase1b/i03/stageB/replay.py (loaded by path, unchanged): base_row,
score_fields, close_pairs, layout_of.

Modes:
  base  old|new|ph [MODEL ...]  step 0 (H.resolve(rec), reproduced <= 1e-6 m, verdict vs recorded) and I01's P, R3,
                                R4 from the unchanged repairs.solver_row; asserted equal to phase1/i01_repairs/rows_*
                                (realistic) and compared with phase1b/i03/stageB/rows_ph_* (placeholder).
                                -> base_<w>_<model>.jsonl
  arms  old|new|ph [MODEL ...]  D (primary), D-pix and D-keep on every solver step (needs base_*; refuses to run if
                                any realistic baseline disagrees or any place fails to reproduce).
                                -> rows_<w>_<model>.jsonl
Arms (records that are not solver steps keep their recorded verdict):
  D      VLM detections matched one to one to detector people (xy <= 0.5 m, nearest first; ties lower number, then
         detector order); matched take the detector xy and yaw (own yaw where the detector's is None), keeping
         number, label, pixel; unmatched dropped and their numbers stripped (repairs.strip); unmatched detector
         people added as M+1, M+2, ... (M = max number listed or in any constraint) in detector order.
  Dpix   as D, but candidates are detector people whose box contains the detection's recorded pixel (edges in),
         greedy by pixel distance to the box centre (ties lower number, then detector order).
  Dkeep  as D, but unmatched VLM detections stay as recorded and the recorded constraints are passed unstripped.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent                 # .../early-detector/phase1c/a2
I09 = HERE.parent                                      # .../early-detector/phase1c
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402

I03 = paths.STILL / "perception" / "depth-blob-detector"
I01 = paths.STILL / "replay-harness" / "repairs"
STAGEB = I03 / "stageB"
sys.path.insert(0, str(I01))
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H  # noqa: E402
import repairs as R  # noqa: E402  (unchanged)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SB = _load("i03_stageB_replay", STAGEB / "replay.py")           # unchanged
PS = _load("i03_people_scorer", I03 / "people_scorer.py")   # unchanged

MATCH_M = 0.5
CLOSE_M = 0.25
REAL = ("real", "staffed", "plain")
PH = ("ph", "dev_seed0", "dev_seed1", "dev_seed2")
PLAN = {
    "old": [("s19", "qwen38_low", REAL), ("s19", "glm53", REAL), ("s20", "dsv41", REAL)],
    "new": [("s21", "qwen38_low", REAL), ("s23", "glm53", REAL), ("s23", "dsv41", REAL), ("s23", "haiku55", REAL),
            ("s23", "sonnet55_low", REAL), ("s23", "gemini38_low", REAL)],
    "ph": [("s19", "qwen38_low", PH), ("s19", "glm53", PH), ("s20", "dsv41", PH)],
}
ARMS = ("D", "Dpix", "Dkeep")


def now():
    return time.strftime("%F %T %z")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_hashes():
    """FROZEN.md, a1/eval_people.json (via a1/EVAL.sha256) and a2/CLARIFICATIONS.md verify on use."""
    def check(listing: Path, base: Path, name: str):
        for line in listing.read_text().splitlines():
            h, f = line.split(None, 1)
            if f.strip() == name:
                got = sha256(base / name)
                assert got == h, f"hash mismatch {base / name}"
                return got
        raise AssertionError(f"{name} not in {listing}")
    out = {"FROZEN.md": check(I09 / "FROZEN.sha256", I09, "FROZEN.md"),
           "a1/eval_people.json": check(I09 / "a1" / "EVAL.sha256", I09 / "a1", "eval_people.json"),
           "a2/CLARIFICATIONS.md": check(HERE / "CLARIFICATIONS.sha256", HERE, "CLARIFICATIONS.md")}
    assert out["FROZEN.md"].startswith("f315e96c1145ad9f")
    return out


def load_detector_people():
    ep = json.loads((I09 / "a1" / "eval_people.json").read_text())
    assert ep["facing_variant"] == "F5" and ep["person_mark_height_m"] == 1.1
    of = {}
    for folder, v in ep["scenes"].items():
        for scenes_dir, scene_id in v["listed"]:
            assert (scenes_dir, scene_id) not in of
            of[(scenes_dir, scene_id)] = (folder, v["people"])
    return of


# ------------------------------------------------------------------------------------------------ matching
def match_xy(det, dets) -> dict[int, int]:
    """{VLM number: detector index}: sgi.run._matched_people arithmetic with detector people in place of truth."""
    pairs = sorted((math.dist(d.xy, (q["x"], q["y"])), d.number, j) for d in det for j, q in enumerate(dets))
    out, taken = {}, set()
    for gap, n, j in pairs:
        if gap <= MATCH_M and n not in out and j not in taken:
            out[n] = j
            taken.add(j)
    return out


def in_box(pix, box) -> bool:
    u, v = pix
    x0, y0, x1, y1 = box
    return x0 <= u <= x1 and y0 <= v <= y1


def match_pix(det, dets) -> dict[int, int]:
    """{VLM number: detector index}: candidates are boxes containing the detection's recorded pixel (edges in);
    greedy one to one by pixel distance to the box centre; ties lower number, then detector order."""
    pairs = []
    for d in det:
        for j, q in enumerate(dets):
            if in_box(d.pixel, q["box"]):
                c = q["box_centre"]
                pairs.append((math.hypot(d.pixel[0] - c[0], d.pixel[1] - c[1]), d.number, j))
    pairs.sort()
    out, taken = {}, set()
    for gap, n, j in pairs:
        if n not in out and j not in taken:
            out[n] = j
            taken.add(j)
    return out


def build(det, cons, dets, pairs, keep_unmatched, M, obs, th):
    """(people, constraints passed to the solver (None = recorded), dropped numbers, added numbers, info)."""
    from sgi.geometry import in_image, project
    from sgi.types import DetectedPerson
    people, dropped, moved_m = [], set(), {}
    for d in det:
        if d.number in pairs:
            q = dets[pairs[d.number]]
            yaw = q["yaw"] if q["yaw"] is not None else d.yaw
            people.append(d.model_copy(update={"xy": (q["x"], q["y"]), "yaw": yaw}))
            moved_m[d.number] = math.dist(d.xy, (q["x"], q["y"]))
        elif keep_unmatched:
            people.append(d)
        else:
            dropped.add(d.number)
    used = set(pairs.values())
    added, nxt, off_img, box_centre_px, yaw_none_added = [], M + 1, 0, 0, 0
    for j, q in enumerate(dets):
        if j in used:
            continue
        pr = project(obs.camera, (q["x"], q["y"], th.person_mark_height_m))
        if pr is None:
            pix = tuple(q["box_centre"])
            box_centre_px += 1
        else:
            pix = (pr[0], pr[1])
            off_img += not in_image(obs.camera, pr[0], pr[1])
        yaw_none_added += q["yaw"] is None
        added.append(DetectedPerson(number=nxt, xy=(q["x"], q["y"]), yaw=q["yaw"], pixel=pix, label=None))
        nxt += 1
    out_cons = None if keep_unmatched else R.strip(cons, dropped)[0]
    add_nums = {a.number for a in added}
    for cs in (cons, out_cons or []):
        for c in cs:
            assert not add_nums & set(c.get("people") or []), "an added number appears in a constraint"
    nums = [p.number for p in people + added]
    assert len(set(nums)) == len(nums)
    info = dict(added_off_image=off_img, added_box_centre_pixel=box_centre_px, added_yaw_none=yaw_none_added,
                moved_m={str(k): round(v, 4) for k, v in moved_m.items()},
                matched_yaw_from_vlm=sum(dets[pairs[d.number]]["yaw"] is None for d in det if d.number in pairs))
    return people + added, out_cons, dropped, add_nums, info


# ------------------------------------------------------------------------------------------------ line class
def line_class(rec, res, people, sc, th, vis):
    """CLARIFICATIONS.md point 23 (descriptive): why a line step still fails under an arm."""
    from sgi.run import _matched_people
    from sgi.solver import target_person
    from sgi.types import GroundedTarget
    numbers = {p.number for p in people}
    tnum = target_person(sc.observation, GroundedTarget.model_validate(rec.target), people, th)
    used_ca = [u for u in res.uses if u.status == "used" and u.constraint.kind == "come_after"]
    waiting = set()
    for u in used_ca:
        waiting |= {n for n in u.constraint.people if n in numbers and n != tnum}
    joined = set(res.note.joined_waiting) if res.note is not None else set()
    gt = sc.truths[rec.episode_id].gt
    tid = rec.score.get("matched_target_id") if rec.score.get("target_correct") else gt.target_ids[0]
    group = sc.truth.group_for(rec.episode_id, tid)
    vis_ids = {p.id for p in vis}
    members = [] if group is None else [p.id for p in sc.truth.group(group) if p.id in vis_ids]
    matched = _matched_people(people, vis, th)
    in_line = waiting | joined
    out = [m for m in members if not any(n in in_line for n, t in matched.items() if t == m)]
    out_unmatched = [m for m in out if m not in matched.values()]
    info = {"waiting": sorted(waiting), "joined": sorted(joined), "members": len(members),
            "members_out": len(out), "members_out_unmatched": len(out_unmatched)}
    if not used_ca:
        return "1 no come_after used", info
    if out:
        return "2 a visible line member is not in the line", info
    return "3 line complete but still fails", info


# ------------------------------------------------------------------------------------------------ one solver step
def arms_step(rec, row, dets):
    from sgi.run import _matched_people
    from sgi.solver import target_person
    from sgi.types import DetectedPerson, GroundedTarget
    sc, th, vis, _ = R.scene_info(rec)
    assert th.person_radius_m == 0.25 and 2 * th.person_radius_m == MATCH_M and th.person_mark_height_m == 1.1
    obs = sc.observation
    det = [DetectedPerson.model_validate(p) for p in rec.people]
    nums = [d.number for d in det]
    assert len(set(nums)) == len(nums)
    cons = [u["constraint"] for u in rec.constraints]
    cnums = [n for c in cons for n in (c.get("people") or [])]
    M = max(nums + cnums, default=0)
    roles = PS.role_classes(sc.truth)
    m_rec = _matched_people(det, vis, th)
    rec_ids = set(m_rec.values())
    named = set(cnums)
    named_ca = {n for c in cons if c["kind"] == "come_after" for n in (c.get("people") or [])}
    gtarget = GroundedTarget.model_validate(rec.target)
    row.update(n_det=len(det), n_dets=len(dets), n_vis=len(vis), n_rec_matched=len(m_rec), M=M,
               empty_list=len(det) == 0, kinds=[c["kind"] for c in cons],
               has_come_after=any(c["kind"] == "come_after" for c in cons),
               tnum_C=target_person(obs, gtarget, det, th))
    rr = H.resolve(rec)
    row.update(C_again=bool(rr.score.success), C_again_moved_m=rr.moved_m)
    pairs_xy = match_xy(det, dets)
    pairs_px = match_pix(det, dets)
    plan = {"D": (pairs_xy, False), "Dpix": (pairs_px, False), "Dkeep": (pairs_xy, True)}
    for arm, (pairs, keep) in plan.items():
        people, out_cons, dropped, add_nums, info = build(det, cons, dets, pairs, keep, M, obs, th)
        res = H.resolve(rec, people=[p.model_dump() for p in people], constraints=out_cons)
        f = SB.score_fields(res)
        m_arm = _matched_people(people, vis, th)
        unmatched_vlm = [d.number for d in det if d.number not in pairs]     # dropped (D, Dpix) or kept (Dkeep)
        fate = {}
        for n in unmatched_vlm:
            k = ("true_" + roles[m_rec[n]]) if n in m_rec else "none"
            fate[k] = fate.get(k, 0) + 1
        ident = {"same": 0, "changed": 0, "none_to_someone": 0, "someone_to_none": 0, "none_to_none": 0}
        for n in pairs:
            a, b = m_rec.get(n), m_arm.get(n)
            if a is None and b is None:
                ident["none_to_none"] += 1
            elif a is None:
                ident["none_to_someone"] += 1
            elif b is None:
                ident["someone_to_none"] += 1
            else:
                ident["same" if a == b else "changed"] += 1
        all_pairs = sum(math.dist(people[i].xy, people[j].xy) < CLOSE_M
                        for i in range(len(people)) for j in range(i + 1, len(people)))
        d = {**f, "n_people": len(people), "n_pairs": len(pairs), "n_unmatched_vlm": len(unmatched_vlm),
             "dropped": sorted(dropped), "added": len(add_nums),
             "unmatched_vlm_fate": fate,
             "unmatched_vlm_named": sum(n in named for n in unmatched_vlm),
             "unmatched_vlm_named_come_after": sum(n in named_ca for n in unmatched_vlm),
             "identity": ident,
             "added_matched_missed": sum(n in m_arm and m_arm[n] not in rec_ids for n in add_nums),
             "added_matched_found": sum(n in m_arm and m_arm[n] in rec_ids for n in add_nums),
             "added_false": sum(n not in m_arm for n in add_nums),
             "added_roles": sorted(roles[m_arm[n]] for n in add_nums if n in m_arm),
             "close_pairs": all_pairs, "close_pairs_added": SB.close_pairs(people, add_nums),
             "n_arm_matched": len(m_arm),
             "tnum": target_person(obs, gtarget, people, th),
             "joined": list(res.note.joined_waiting) if res.note is not None else [],
             "joined_added": sum(n in add_nums for n in (res.note.joined_waiting if res.note is not None else [])),
             "uses": [u.status for u in res.uses],
             "moved_from_C_m": round(math.dist(res.pose_world[:2], rr.pose_world[:2]), 4), **info}
        if rec.situation == "waiting_line" and not f["success"]:
            d["class"], d["class_info"] = line_class(rec, res, people, sc, th, vis)
        row[arm] = f["success"]
        row[arm + "_d"] = d
    return row


# ------------------------------------------------------------------------------------------------ main
def reference_rows(wording, model):
    if wording in ("old", "new"):
        path = I01 / f"rows_{wording}_{model}.jsonl"
        keys = {"P": "P", "R3": "R3", "R4": "R4"}
    else:
        path = STAGEB / f"rows_ph_{model}.jsonl"
        keys = {"P": "P_i01", "R3": "R3_i01", "R4": "R4_i01"}
    rows = {(r["arm"], r["scene"], r["episode"], r["trial"]): r for r in map(json.loads, path.open())}
    return path, rows, keys


def mode_base(wording, only, log):
    for stage, model, arms in PLAN[wording]:
        if only and model not in only:
            continue
        path, ref_rows, keys = reference_rows(wording, model)
        rows, mism = [], []
        for arm in arms:
            ref = R.one_run(stage, arm, model)
            t0 = time.monotonic()
            recs = sorted(H.records(ref), key=lambda r: (r.scene_id, r.episode_id, r.trial))
            for rec in recs:
                row = SB.base_row(wording, stage, model, ref, rec)
                if rec.solver_step:
                    tmp = {"C_rec": row["C_rec"]}
                    R.solver_row(rec, tmp)
                    row.update(C=tmp["C"], C_moved_m=tmp["moved_m"], C_reproduced=tmp["reproduced"],
                               C_verdict_ok=tmp["C"] == row["C_rec"], P=tmp["P"], R3=tmp["R3"], R4=tmp["R4"],
                               P_tc=tmp["P_tc"], i01_diag={k: tmp[k] for k in (
                                   "n_det", "n_vis", "n_matched", "n_missed", "n_added", "n_spurious", "n_cons",
                                   "n_cons_stripped")})
                else:
                    row.update(C=row["C_rec"], P=row["C_rec"], R3=row["C_rec"], R4=row["C_rec"])
                o = ref_rows[(ref.arm, rec.scene_id, rec.episode_id, rec.trial)]
                checks = {k: o[v] for k, v in keys.items()}
                checks.update(C_rec=o["C_rec"], line_failure=o["line_failure"], solver_step=o["solver_step"])
                bad = [k for k, v in checks.items() if row[k] != v]
                row["ref_mismatch"] = bad
                if bad:
                    mism.append((ref.arm, rec.scene_id, rec.episode_id, rec.trial, bad))
                rows.append(row)
            st = [r for r in rows if r["arm"] == arm and r["solver_step"]]
            print(f"{now()} base {wording} {ref.folder}: {len(recs)} records, {len(st)} solver steps, reproduced "
                  f"{sum(r['C_reproduced'] for r in st)}, verdict ok {sum(r['C_verdict_ok'] for r in st)}, "
                  f"{time.monotonic() - t0:.0f} s", file=log, flush=True)
        print(f"{now()} base {wording} {model}: reference {path.name}: {len(mism)} records disagree {mism[:5]}",
              file=log, flush=True)
        if wording != "ph":
            assert not mism, f"realistic baseline disagrees with {path}: {mism[:5]}"
        with open(HERE / f"base_{wording}_{model}.jsonl", "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")


def mode_arms(wording, only, log):
    det_of = load_detector_people()
    for stage, model, arms in PLAN[wording]:
        if only and model not in only:
            continue
        base = {(r["arm"], r["scene"], r["episode"], r["trial"]): r
                for r in map(json.loads, open(HERE / f"base_{wording}_{model}.jsonl"))}
        steps = [r for r in base.values() if r["solver_step"]]
        assert all(r["C_reproduced"] and r["C_verdict_ok"] for r in steps), "step 0 failed: arms refused"
        if wording != "ph":
            assert not any(r["ref_mismatch"] for r in base.values()), "baseline mismatch: arms refused"
        rows = []
        for arm in arms:
            ref = R.one_run(stage, arm, model)
            t0 = time.monotonic()
            recs = sorted(H.records(ref), key=lambda r: (r.scene_id, r.episode_id, r.trial))
            for rec in recs:
                row = dict(base[(ref.arm, rec.scene_id, rec.episode_id, rec.trial)])
                folder, dets = det_of[(str(ref.scenes_dir), rec.scene_id)]
                row["det_folder"] = folder
                if rec.solver_step:
                    arms_step(rec, row, dets)
                else:
                    for k in ARMS:
                        row[k] = row["C_rec"]
                rows.append(row)
            st = [r for r in rows if r["arm"] == arm and r["solver_step"]]
            print(f"{now()} arms {wording} {ref.folder}: {len(recs)} records, {len(st)} solver steps, "
                  f"C again ok {sum(r['C_again'] == r['C_rec'] for r in st)}, "
                  f"D {sum(r['D'] for r in rows if r['arm'] == arm)}, {time.monotonic() - t0:.0f} s",
                  file=log, flush=True)
        with open(HERE / f"rows_{wording}_{model}.jsonl", "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")


def main(argv):
    mode, wording, only = argv[0], argv[1], set(argv[2:])
    assert mode in ("base", "arms") and wording in PLAN
    log = sys.stdout
    hashes = verify_hashes()
    print(f"{now()} start {mode} {wording} {sorted(only) or 'all'}; hashes verified {hashes}", file=log, flush=True)
    t = time.monotonic()
    (mode_base if mode == "base" else mode_arms)(wording, only, log)
    print(f"{now()} done {mode} {wording} in {time.monotonic() - t:.0f} s", file=log, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
