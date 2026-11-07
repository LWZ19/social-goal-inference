"""Independent verifier for I09 A2 (FROZEN.md section 11). Written from the note only; imports the package, the
harness and the frozen people scorer, never the compute agent's code.

Per record of each planned run: recorded C, step-0 replay, my own D / D-pix / D-keep, and my own re-implementation
of I01's P, R3, R4 (to compare with phase1/i01_repairs/rows_* and phase1b stageB rows_ph_*).

Usage: vreplay.py old|new|ph  -> vrows_<wording>_<model>.jsonl beside this file.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H  # noqa: E402

sys.path.insert(0, 'results/2026-10-17-beyond-vlm/phase1b/i03')
import people_scorer as PS  # noqa: E402

OUT = Path(__file__).resolve().parent
I09 = OUT.parent
FROZEN_SHA = "f315e96c1145ad9f2c86128d16711bb5305690d99f5d8513c0177290c474df15"
EVAL_SHA = "ffa595453c94b444c8dd030d053d37ac103441a7e74f07e3f172836b3d0c24b3"
REAL = ("real", "staffed", "plain")
PHA = ("ph", "dev_seed0", "dev_seed1", "dev_seed2")
PLAN = {
    "old": [("s19", "qwen38_low", REAL), ("s19", "glm53", REAL), ("s20", "dsv41", REAL)],
    "new": [("s21", "qwen38_low", REAL)] + [("s23", m, REAL) for m in
                                             ("glm53", "dsv41", "haiku55", "sonnet55_low", "gemini38_low")],
    "ph": [("s19", "qwen38_low", PHA), ("s19", "glm53", PHA), ("s20", "dsv41", PHA)],
}
BAD = ("keep/heldout", "s18", "s20_held", "eval_16", "eval_17", "eval_18", "eval_19", "group_photo", "lift_wait")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_det():
    assert sha(I09 / "FROZEN.md") == FROZEN_SHA
    assert sha(I09 / "a1/eval_people.json") == EVAL_SHA
    doc = json.loads((I09 / "a1/eval_people.json").read_text())
    assert doc["facing_variant"] == "F5", doc["facing_variant"]
    det = {}
    for folder, s in doc["scenes"].items():
        assert not any(b in folder for b in BAD), folder
        ppl = s["people"]
        sc = [p["score"] for p in ppl]
        assert sc == sorted(sc, reverse=True), folder           # detector order = descending score
        for p in ppl:
            assert p["yaw"] == p["yaws_all_variants"]["F5"]
        for d, sid in s["listed"]:
            key = (str(d), sid)
            assert key not in det, key
            det[key] = ppl
    return doc, det


def strip(cons, dropped):
    return [{**c, "people": [n for n in (c.get("people") or []) if n not in dropped]} for c in cons]


def greedy(pairs):
    """pairs (gap, number, index), already filtered; nearest first, one to one, lower number then earlier index."""
    out, taken = {}, set()
    for gap, number, index in sorted(pairs):
        if number not in out and index not in taken:
            out[number] = index
            taken.add(index)
    return out


def box_contains(box, u, v):
    x0, y0, x1, y1 = box
    return x0 <= u <= x1 and y0 <= v <= y1


def build_arm(arm, vlm, cons, dets, cam, th, project):
    """Returns (people list of DetectedPerson, constraints list of dicts, info)."""
    from sgi.types import DetectedPerson
    width = 2 * th.person_radius_m
    if arm in ("D", "Dkeep"):
        pairs = [(math.dist(v.xy, (d["x"], d["y"])), v.number, j) for v in vlm for j, d in enumerate(dets)]
        pairs = [p for p in pairs if p[0] <= width]
    elif arm == "Dpix":
        pairs = []
        for v in vlm:
            u, w = v.pixel
            for j, d in enumerate(dets):
                if box_contains(d["box"], u, w):
                    x0, y0, x1, y1 = d["box"]
                    pairs.append((math.dist((u, w), ((x0 + x1) / 2, (y0 + y1) / 2)), v.number, j))
    else:
        raise ValueError(arm)
    match = greedy(pairs)                                 # vlm number -> detector index
    M = max([v.number for v in vlm] + [n for c in cons for n in (c.get("people") or [])], default=0)
    out, dropped, moved = [], [], {}
    yaw_from_vlm = 0
    for v in vlm:
        if v.number in match:
            d = dets[match[v.number]]
            yaw = d["yaw"] if d["yaw"] is not None else v.yaw
            yaw_from_vlm += d["yaw"] is None
            moved[v.number] = math.dist(v.xy, (d["x"], d["y"]))
            out.append(v.model_copy(update={"xy": (d["x"], d["y"]), "yaw": yaw}))
        else:
            dropped.append(v.number)
            if arm == "Dkeep":
                out.append(v)
    used_idx = set(match.values())
    added = []
    k = 0
    for j, d in enumerate(dets):
        if j in used_idx:
            continue
        k += 1
        pr = project(cam, (d["x"], d["y"], th.person_mark_height_m))
        if pr is not None:
            pix = (float(pr[0]), float(pr[1]))
        else:
            x0, y0, x1, y1 = d["box"]
            pix = ((x0 + x1) / 2, (y0 + y1) / 2)
        added.append(DetectedPerson(number=M + k, xy=(d["x"], d["y"]), yaw=d["yaw"], pixel=pix, label=None))
    out = out + added
    c_arm = cons if arm == "Dkeep" else strip(cons, set(dropped))
    added_nums = {a.number for a in added}
    named = {n for c in cons for n in (c.get("people") or [])} | {n for c in c_arm for n in (c.get("people") or [])}
    assert not (added_nums & named), "number collision"
    nums = [p.number for p in out]
    assert len(nums) == len(set(nums)), "duplicate numbers"
    close = close_add = 0
    for a in range(len(out)):
        for b in range(a + 1, len(out)):
            if math.dist(out[a].xy, out[b].xy) < 0.25:
                close += 1
                close_add += out[a].number in added_nums or out[b].number in added_nums
    info = dict(match={str(k): v for k, v in match.items()}, dropped=dropped, added=sorted(added_nums), M=M,
                close=close, close_add=close_add, moved_max=max(moved.values(), default=0.0),
                yaw_from_vlm=yaw_from_vlm, n_final=len(out))
    return out, c_arm, info


def p_arms(vlm, cons, vis, th, cam, project):
    """My re-implementation of I01's R3, R4 and P (repairs.solver_row semantics, as described in its docstring)."""
    from sgi.run import _matched_people
    from sgi.types import DetectedPerson
    matched = _matched_people(vlm, vis, th)
    by_id = {p.id: p for p in vis}
    found = set(matched.values())
    dropped = {v.number for v in vlm if v.number not in matched}
    nxt = max((v.number for v in vlm), default=0) + 1
    added, i = [], 0
    for p in vis:
        if p.id in found:
            continue
        pr = project(cam, (p.x, p.y, th.person_mark_height_m))
        if pr is None:
            continue
        added.append(DetectedPerson(number=nxt + i, xy=(p.x, p.y), yaw=p.yaw, pixel=(pr[0], pr[1]), label=None))
        i += 1
    kept = [v for v in vlm if v.number in matched]
    pp = [v.model_copy(update={"xy": (by_id[matched[v.number]].x, by_id[matched[v.number]].y),
                               "yaw": by_id[matched[v.number]].yaw}) for v in kept] + added
    cs = strip(cons, dropped)
    return dict(R3=(vlm + added, cons), R4=(kept, cs), P=(pp, cs)), matched


def main(wording):
    from sgi.geometry import project
    from sgi.gt.derive import is_visible
    from sgi.run import _matched_people
    from sgi.types import DetectedPerson
    doc, DET = load_det()
    t_start = time.monotonic()
    log = open(OUT / f"vreplay_{wording}.log", "a")

    def say(msg):
        line = time.strftime("%Y-%m-%d %H:%M:%S %z") + " " + msg
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    say(f"start {wording}; FROZEN and eval_people hashes verified")
    for stage, model, arms in PLAN[wording]:
        rows = []
        for arm in arms:
            found = H.runs(stage=stage, arm=arm, model=model, method="con")
            assert len(found) == 1
            ref = found[0]
            assert not any(b in str(ref.scenes_dir) or b in ref.folder for b in BAD)
            th = H.recorded_config(ref)[1].thresholds
            assert th.person_radius_m == 0.25 and th.person_mark_height_m == 1.1
            recs = sorted(H.records(ref), key=lambda r: (r.scene_id, r.episode_id, r.trial))
            t0 = time.monotonic()
            scache = {}
            for rec in recs:
                s = rec.score
                row = dict(wording=wording, stage=stage, arm=arm, model=model, run=ref.folder, scene=rec.scene_id,
                           episode=rec.episode_id, trial=rec.trial, situation=rec.situation,
                           solver_step=rec.solver_step, C_rec=bool(s.get("success")),
                           target_correct=bool(s.get("target_correct")),
                           line_failure=bool(rec.situation == "waiting_line" and rec.solver_step
                                             and s.get("target_correct") and not s.get("success")
                                             and (not s.get("in_region") or not s.get("reachable"))),
                           has_people=rec.people is not None)
                if not rec.solver_step:
                    for k in ("step0", "D", "Dpix", "Dkeep", "P", "R3", "R4"):
                        row[k] = row["C_rec"]
                    rows.append(row)
                    continue
                key = (str(ref.scenes_dir), rec.scene_id)
                dets = DET[key]
                if rec.scene_id not in scache:
                    sc = H.scene(ref, rec.scene_id)
                    vis = [p for p in sc.truth.people if is_visible(p, sc.observation, th)]
                    scache[rec.scene_id] = (sc, vis, PS.role_classes(sc.truth))
                sc, vis, roles = scache[rec.scene_id]
                cam = sc.observation.camera
                vlm = [DetectedPerson.model_validate(p) for p in rec.people]
                cons = [u["constraint"] for u in rec.constraints]
                r0 = H.resolve(rec)
                row["step0_moved_m"] = r0.moved_m
                row["step0_reproduced"] = bool(r0.reproduced)
                row["step0"] = bool(r0.score.success)
                row["step0_verdict_ok"] = row["step0"] == row["C_rec"]
                rec_match = _matched_people(vlm, vis, th)              # recorded list vs visible truth
                rec_found = set(rec_match.values())
                row["n_vlm"] = len(vlm)
                row["n_det"] = len(dets)
                row["n_vis"] = len(vis)
                row["empty_list"] = len(vlm) == 0
                ca_named = {n for c in cons if c["kind"] == "come_after" for n in (c.get("people") or [])}
                all_named = {n for c in cons for n in (c.get("people") or [])}
                for armname in ("D", "Dpix", "Dkeep"):
                    ppl, c_arm, info = build_arm(armname, vlm, cons, dets, cam, th, project)
                    res = H.resolve(rec, people=[p.model_dump() for p in ppl], constraints=c_arm)
                    row[armname] = bool(res.score.success)
                    arm_match = _matched_people(ppl, vis, th)
                    added = set(info["added"])
                    a_cls = []
                    for p in ppl:
                        if p.number in added:
                            t = arm_match.get(p.number)
                            a_cls.append("false" if t is None else ("found" if t in rec_found else "missed:" + roles[t]))
                    fate = []
                    for n in info["dropped"]:
                        t = rec_match.get(n)
                        fate.append(dict(n=n, role=None if t is None else roles[t], named=n in all_named,
                                         ca=n in ca_named))
                    ident = dict(same=0, changed=0, none_to_someone=0, someone_to_none=0, none_to_none=0)
                    for n in map(int, info["match"].keys()):
                        a, b = rec_match.get(n), arm_match.get(n)
                        if a == b:
                            ident["same" if a is not None else "none_to_none"] += 1
                        elif a is None:
                            ident["none_to_someone"] += 1
                        elif b is None:
                            ident["someone_to_none"] += 1
                        else:
                            ident["changed"] += 1
                    uses = [(u.constraint.kind, u.status, list(u.constraint.people)) for u in res.uses]
                    ca_used = [u for u in uses if u[0] == "come_after" and u[1] == "used"]
                    line_cls = None
                    if rec.situation == "waiting_line" and not row[armname]:
                        tid = s.get("matched_target_id") if s.get("target_correct") else None
                        tid = tid or sc.truths[rec.episode_id].gt.target_ids[0]
                        grp = sc.truth.group_for(rec.episode_id, tid)
                        vis_ids = {p.id for p in vis}
                        members = [] if grp is None else [p.id for p in sc.truth.group(grp) if p.id in vis_ids]
                        waiting = {n for u in ca_used for n in u[2]} | set(res.note.joined_waiting or [])
                        in_line = {arm_match.get(n) for n in waiting}
                        if not ca_used:
                            line_cls = "no_come_after"
                        elif any(m not in in_line for m in members):
                            line_cls = "member_not_in_line"
                        else:
                            line_cls = "complete"
                    sres = res.score
                    row[armname + "_d"] = dict(
                        info, added_cls=a_cls, fate=fate, ident=ident, line_cls=line_cls,
                        ca_used=len(ca_used), joined=list(res.note.joined_waiting or []),
                        tc=bool(sres.target_correct), in_region=bool(getattr(sres, "in_region", False)),
                        reachable=bool(getattr(sres, "reachable", False)),
                        pose=list(res.pose_world))
                pa, _ = p_arms(vlm, cons, vis, th, cam, project)
                for k, (ppl, cc) in pa.items():
                    row[k] = bool(H.resolve(rec, people=[p.model_dump() for p in ppl], constraints=cc).score.success)
                rows.append(row)
            say(f"{wording} {ref.folder}: {len(recs)} records, {sum(r['solver_step'] for r in rows if r['arm'] == arm)}"
                f" solver steps, {time.monotonic() - t0:.0f} s")
        with open(OUT / f"vrows_{wording}_{model}.jsonl", "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
    say(f"done {wording} in {time.monotonic() - t_start:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1])
