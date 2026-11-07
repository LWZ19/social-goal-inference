"""I01 Steps 0-1: the counterfactual error budget (zero calls). Reads runs and scenes only through research_harness.

For every recorded C goal step (a record whose result went through the solver: research_harness Record.solver_step)
of the realistic arms (real, staffed, plain), the unchanged solver is run again (H.resolve) with ONE input repaired
at a time, and the place is scored by the package scorer (H.resolve's score). Arms:

  C     the recorded inputs, solved again (the Step 0 reproduction check: moved_m <= 1e-6).
  R0    constraints removed: H.resolve(rec, constraints=[]) (the I01 README's SA, recomputed as a check).
  R1    true positions: each detected person matched one-to-one to a visible true person (the package's matching,
        sgi.run._matched_people, centres within 2 x person_radius_m = 0.5 m, against the true people that pass
        sgi.gt.derive.is_visible under the run's thresholds, as RERUN R02 did) is moved to the true (x, y). Number,
        yaw, pixel and label are kept; unmatched detections stay as they are.
  R2    true facing: each matched detection gets the true person's yaw (position, number kept).
  R3    missed people added: every visible true person no detection is matched to is appended as an unlisted body,
        made as the package's oracle people source makes one (sgi.run.OraclePeople: true x, y, yaw; pixel = the
        projection of the point person_mark_height_m above them; a person whose mark does not project is left
        out, counted). They get new numbers max(model numbers) + 1, + 2, ... in ground-truth order, so no
        constraint references them. No merge or cut is applied (counted: added within person_radius_m of a kept
        detection, which the pipeline's merge would have dropped; max_people = 40 is never reached).
  R4    spurious people dropped: every unmatched detection is removed. Each constraint loses the dropped numbers
        from its `people` list (kind, target flag and range unchanged); a constraint left with no people is kept
        and the solver judges it as it judges any (no people and no target -> unusable). Sensitivity arm R4s: the
        same people, constraints left as written (the solver's own rule then ignores numbers it does not have, and
        a constraint all of whose numbers are gone becomes unusable).
  P     R1 + R2 + R3 + R4 together: matched detections at the true position and facing, unmatched dropped, missed
        added. The people list is then exactly the visible true people (projectable marks), carrying the model's
        numbers where matched. Ps = P with R4s's constraint rule.
  R5    reference constraints, with the recorded target and the recorded people: copied from
        tests/test_reference_constraints.py (reference_constraints): waiting_line -> come_after(members);
        conversation_group -> keep_out_of_shared_space(members), visible_to(needed), face(needed); audience ->
        keep_out_of_view(members, target=True), face(target=True); no_one -> none. members = the model's numbers
        matched to the visible members of the group at the target (truth.group_for(episode, target id)); needed =
        the model's number matched to the target person. Target id = score.matched_target_id when the recorded
        target is right, else the episode's first target (as the test). Where the target person was not matched
        (needed unknown) visible_to and face are omitted (counted). Circular: it reads the situation label and the
        true group.
  R6    the oracle target: sgi.gt.derive.oracle_target(observation, truth.targets[target id], th) (same target id
        rule), with the recorded people and constraints. A target whose lift fails keeps the recorded one (counted).
  P5    extra: P's people with reference constraints written against them (members all visible group members).
  ALL   extra: P5 with the oracle target (the test's configuration up to numbering; a ceiling check).
  P1m, R3_1m  sensitivity: P and R3 with the one-to-one matching radius widened to 1.0 m (person_radius_m 0.5 passed
        to _matched_people only), so a detection 0.5-1.0 m from a true person counts as that person, not as a
        spurious detection plus a missed person.

Records that are not solver steps keep their recorded verdict in every arm. Extra R6x (whole-record oracle target):
a non-solver record with status ok and a target gets sgi.stages.goal.standard_goal from the oracle target, scored
with H.score_pose(rec, pose, target=oracle) (assumes the pipeline would still have taken the standard path).

Usage: repairs.py old|new|all [MODEL ...]  -> rows_<wording>_<model>.jsonl beside this file.
"""
from __future__ import annotations

import collections
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H  # noqa: E402

OUT = Path(__file__).resolve().parent
ARMS = ("real", "staffed", "plain")
PLAN = {
    "old": [("s19", "qwen38_low"), ("s19", "glm53"), ("s20", "dsv41")],
    "new": [("s21", "qwen38_low"), ("s23", "glm53"), ("s23", "dsv41"), ("s23", "haiku55"), ("s23", "sonnet55_low"),
            ("s23", "gemini38_low")],
}
REPAIRS = ("C", "R0", "R1", "R2", "R3", "R4", "R4s", "P", "Ps", "R5", "R6", "P5", "ALL", "P1m", "R3_1m")


def one_run(stage, arm, model):
    found = H.runs(stage=stage, arm=arm, model=model, method="con")
    if len(found) != 1:
        raise SystemExit(f"error: {len(found)} runs for {stage} {arm} {model} con")
    return found[0]


_SCENE = {}


def scene_info(rec):
    """(scene, thresholds, visible true people, {true id: oracle DetectedPerson fields or None}) once per scene."""
    from sgi.geometry import project
    from sgi.gt.derive import is_visible
    th = H.recorded_config(rec.ref)[1].thresholds
    key = (str(rec.ref.scenes_dir), rec.scene_id, th.model_dump_json())
    if key not in _SCENE:
        sc = H.scene(rec.ref, rec.scene_id)
        vis = [p for p in sc.truth.people if is_visible(p, sc.observation, th)]
        oracle = {}
        for p in vis:          # as sgi.run.OraclePeople makes one
            marked = project(sc.observation.camera, (p.x, p.y, th.person_mark_height_m))
            oracle[p.id] = None if marked is None else dict(xy=(p.x, p.y), yaw=p.yaw, pixel=tuple(marked[:2]))
        _SCENE[key] = (sc, th, vis, oracle)
        while len(_SCENE) > 30:
            _SCENE.pop(next(iter(_SCENE)))
    return _SCENE[key]


def ang_deg(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi) * 180 / math.pi


def ask(kind, *people, target=False):
    # tests/solver_layouts.ask, copied: one constraint of an answer.
    return dict(because="reference constraint", kind=kind, people=list(people), range_m=None, target=target)


def reference_constraints(situation, members, needed):
    """tests/test_reference_constraints.reference_constraints, copied; `needed` None -> the two that name the
    needed person are omitted (the test always has them, its people being the true ones)."""
    if situation == "waiting_line":
        return [ask("come_after", *members)]
    if situation == "conversation_group":
        return [ask("keep_out_of_shared_space", *members)] + (
            [] if needed is None else [ask("visible_to", needed), ask("face", needed)])
    if situation == "audience":
        return [ask("keep_out_of_view", *members, target=True), ask("face", target=True)]
    assert situation == "no_one", situation
    return []


def strip(constraints, dropped):
    out, changed = [], 0
    for c in constraints:
        people = [n for n in (c.get("people") or []) if n not in dropped]
        changed += len(people) != len(c.get("people") or [])
        out.append({**c, "people": people})
    return out, changed


def solver_row(rec, row):
    from sgi.gt.derive import oracle_target
    from sgi.run import _matched_people
    from sgi.types import DetectedPerson
    sc, th, vis, oracle = scene_info(rec)
    gt = sc.truths[rec.episode_id].gt
    det = [DetectedPerson.model_validate(p) for p in rec.people]
    cons = [u["constraint"] for u in rec.constraints]
    matched = _matched_people(det, vis, th)                  # number -> true id
    by_id = {p.id: p for p in vis}
    number_of = {tid: n for n, tid in matched.items()}
    dropped = {d.number for d in det if d.number not in matched}
    missed = [p for p in vis if p.id not in number_of]
    added_ok = [p for p in missed if oracle[p.id] is not None]
    nxt = max((d.number for d in det), default=0) + 1
    added = [DetectedPerson(number=nxt + i, label=None, **oracle[p.id]) for i, p in enumerate(added_ok)]

    def move(d, pos=False, yaw=False):
        if d.number not in matched:
            return d
        tp = by_id[matched[d.number]]
        upd = {}
        if pos:
            upd["xy"] = (tp.x, tp.y)
        if yaw:
            upd["yaw"] = tp.yaw
        return d.model_copy(update=upd)

    dump = lambda people: [p.model_dump() for p in people]  # noqa: E731
    p_r1 = dump([move(d, pos=True) for d in det])
    p_r2 = dump([move(d, yaw=True) for d in det])
    p_r3 = dump(det + added)
    kept = [d for d in det if d.number in matched]
    p_r4 = dump(kept)
    p_p = dump([move(d, pos=True, yaw=True) for d in kept] + added)
    c_strip, n_changed = strip(cons, dropped)

    # target id for R5 / R6
    tid = rec.score.get("matched_target_id") if rec.score.get("target_correct") else None
    tid = tid or gt.target_ids[0]
    wanted = sc.truth.targets[tid]
    group = sc.truth.group_for(rec.episode_id, tid)
    member_ids = [] if group is None else [p.id for p in sc.truth.group(group) if p.id in by_id]
    members = [number_of[m] for m in member_ids if m in number_of]
    needed = number_of.get(wanted.person_id) if wanted.person_id else None
    ref5 = reference_constraints(rec.situation, members, needed)
    # against P's people: every visible member has a number
    number_p = dict(number_of)
    number_p.update({p.id: a.number for p, a in zip(added_ok, added)})
    members_p = [number_p[m] for m in member_ids if m in number_p]
    needed_p = number_p.get(wanted.person_id) if wanted.person_id else None
    ref5p = reference_constraints(rec.situation, members_p, needed_p)
    ot = oracle_target(sc.observation, wanted, th)

    res = {}
    rr = H.resolve(rec)
    res["C"] = rr
    res["R0"] = H.resolve(rec, constraints=[])
    res["R1"] = H.resolve(rec, people=p_r1)
    res["R2"] = H.resolve(rec, people=p_r2)
    res["R3"] = H.resolve(rec, people=p_r3)
    res["R4"] = H.resolve(rec, people=p_r4, constraints=c_strip)
    res["R4s"] = H.resolve(rec, people=p_r4, constraints=cons)
    res["P"] = H.resolve(rec, people=p_p, constraints=c_strip)
    res["Ps"] = H.resolve(rec, people=p_p, constraints=cons)
    res["R5"] = H.resolve(rec, constraints=ref5)
    res["R6"] = H.resolve(rec, target=ot) if ot is not None else rr
    res["P5"] = H.resolve(rec, people=p_p, constraints=ref5p)
    res["ALL"] = H.resolve(rec, target=ot, people=p_p, constraints=ref5p) if ot is not None else res["P5"]
    # sensitivity: matching within 1.0 m
    th1 = th.model_copy(update={"person_radius_m": 0.5})
    m1 = _matched_people(det, vis, th1)
    n1 = {t: n for n, t in m1.items()}
    drop1 = {d.number for d in det if d.number not in m1}
    add1_ok = [p for p in vis if p.id not in n1 and oracle[p.id] is not None]
    add1 = [DetectedPerson(number=nxt + i, label=None, **oracle[p.id]) for i, p in enumerate(add1_ok)]
    moved1 = []
    for d in det:
        if d.number in m1:
            tp = by_id[m1[d.number]]
            moved1.append(d.model_copy(update={"xy": (tp.x, tp.y), "yaw": tp.yaw}))
    c_strip1, _ = strip(cons, drop1)
    res["P1m"] = H.resolve(rec, people=dump(moved1 + add1), constraints=c_strip1)
    res["R3_1m"] = H.resolve(rec, people=dump(det + add1))
    row["n_matched_1m"] = len(m1)
    for k, v in res.items():
        row[k] = bool(v.score.success)
        row[k + "_tc"] = bool(v.score.target_correct)
    row["moved_m"] = rr.moved_m
    row["reproduced"] = rr.reproduced
    row["C_replay_ok"] = bool(rr.score.success)
    # diagnostics
    row.update(n_det=len(det), n_vis=len(vis), n_matched=len(matched), n_missed=len(missed),
               n_added=len(added), n_missed_unprojectable=len(missed) - len(added_ok), n_spurious=len(dropped),
               n_cons=len(cons), n_cons_stripped=n_changed,
               n_added_near_kept=sum(any(math.dist(a.xy, d.xy) <= th.person_radius_m for d in det) for a in added),
               kinds=[c["kind"] for c in cons], has_come_after=any(c["kind"] == "come_after" for c in cons),
               r5_members=len(members), r5_members_true=len(member_ids), r5_needed_missing=(
                   rec.situation == "conversation_group" and wanted.person_id is not None and needed is None),
               r6_target_id=tid, r6_lift_failed=ot is None,
               facing_err_deg=[round(ang_deg(d.yaw, by_id[matched[d.number]].yaw), 1) for d in det
                               if d.number in matched and d.yaw is not None],
               facing_none=sum(d.number in matched and d.yaw is None for d in det),
               uses_P=[u.status for u in res["P"].uses], uses_C=[u.status for u in rr.uses])
    return row


def other_row(rec, row):
    """R6x for a record that is not a solver step: the standard goal from the oracle target."""
    from sgi.gt.derive import oracle_target
    from sgi.stages.goal import standard_goal
    if rec.result.get("status") != "ok" or rec.target is None:
        row["R6x"] = row["C_rec"]
        row["R6x_note"] = "not ok or no target: kept"
        return row
    sc, th, vis, oracle = scene_info(rec)
    gt = sc.truths[rec.episode_id].gt
    tid = rec.score.get("matched_target_id") if rec.score.get("target_correct") else None
    tid = tid or gt.target_ids[0]
    ot = oracle_target(sc.observation, sc.truth.targets[tid], th)
    if ot is None:
        row["R6x"] = row["C_rec"]
        row["R6x_note"] = "lift failed: kept"
        return row
    sg = standard_goal(sc.observation, ot, th)
    if sg.pose_world is None:
        row["R6x"] = False
        row["R6x_note"] = "standard goal has no pose"
        return row
    row["R6x"] = bool(H.score_pose(rec, sg.pose_world, target=ot).success)
    row["R6x_note"] = "standard goal from oracle target"
    return row


def main(argv):
    wordings = ["old", "new"] if argv[0] == "all" else [argv[0]]
    only = set(argv[1:])
    started = time.monotonic()
    for wording in wordings:
        for stage, model in PLAN[wording]:
            if only and model not in only:
                continue
            rows = []
            for arm in ARMS:
                ref = one_run(stage, arm, model)
                t0 = time.monotonic()
                recs = sorted(H.records(ref), key=lambda r: (r.scene_id, r.episode_id, r.trial))
                for rec in recs:
                    s = rec.score
                    row = dict(wording=wording, stage=stage, arm=arm, model=model, run=ref.folder,
                               scene=rec.scene_id, episode=rec.episode_id, trial=rec.trial, situation=rec.situation,
                               source=rec.result.get("source"), status=rec.result.get("status"),
                               solver_step=rec.solver_step, C_rec=bool(s.get("success")),
                               target_correct=bool(s.get("target_correct")),
                               line_failure=bool(rec.situation == "waiting_line" and rec.solver_step
                                                 and s.get("target_correct") and not s.get("success")
                                                 and (not s.get("in_region") or not s.get("reachable"))))
                    if rec.solver_step:
                        solver_row(rec, row)
                        row["R6x"] = row["R6"]
                    else:
                        for k in REPAIRS:
                            row[k] = row["C_rec"]
                        other_row(rec, row)
                    rows.append(row)
                print(f"{wording} {ref.folder}: {len(recs)} records, "
                      f"{sum(r['solver_step'] for r in rows if r['arm'] == arm)} solver steps, "
                      f"{time.monotonic() - t0:.0f} s", flush=True)
            with open(OUT / f"rows_{wording}_{model}.jsonl", "w") as fh:
                for row in rows:
                    fh.write(json.dumps(row) + "\n")
    print(f"done in {time.monotonic() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["all"])
