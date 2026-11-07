"""I01 deliverable (1): solver-alone and people-blind standard-goal rows, six models (zero calls).

Reads runs and scenes only through research_harness. For every C record of the realistic arms (real, staffed,
plain) of the new-wording runs (s21 Qwen con; s23 glm53, dsv41, haiku55, sonnet55_low, gemini38_low con) and the
old-wording runs (s19 qwen38_low, glm53 con; s20 dsv41 con):

  C rec     the recorded verdict of the C record.
  C now     the recorded pose scored again now with score_pose (sanity check; must equal C rec).
  SA        solver alone: where the record has a recorded target and recorded people (people is not None; an empty
            list counts), H.resolve(rec, constraints=[]) -- the recorded target and people, no constraints -- scored
            by the package scorer. Every other record keeps its recorded verdict.
  SG        people-blind standard goal: where the record has a recorded target, sgi.stages.goal.standard_goal(
            observation, target, run thresholds), as replay_solver.solve_again calls it, scored with H.score_pose.
            A record with no target keeps its recorded verdict.
  CA        solver with only the come_after constraints kept (from the recorded constraint uses, whatever status the
            solver gave them): H.resolve(rec, constraints=[come_after ones]). A goal-step record that states no
            come_after gets SA (an empty list), a record with no goal step keeps its recorded verdict.
  R, E      recorded successes of the same stage's cot and fsc runs (new wording: s21 Qwen, s23 others; old wording:
            s16b Qwen and GLM, s20 DeepSeek), same realistic arms.

Writes per-record rows (records.jsonl) and the tables (tables.txt) beside this file.
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
SITS = ("no_one", "waiting_line", "conversation_group", "audience")
MODELS = ("qwen38_low", "glm53", "dsv41", "haiku55", "sonnet55_low", "gemini38_low")
NEW = {"qwen38_low": ("s21", "s21"), **{m: ("s23", "s23") for m in MODELS[1:]}}          # (C stage, R/E stage)
OLD = {"qwen38_low": ("s19", "s16b"), "glm53": ("s19", "s16b"), "dsv41": ("s20", "s20")}


def one_run(stage, arm, model, method):
    found = H.runs(stage=stage, arm=arm, model=model, method=method)
    if len(found) != 1:
        raise SystemExit(f"error: {len(found)} runs for {stage} {arm} {model} {method}")
    return found[0]


def kind_of(use) -> str:
    return (use.get("constraint") or {}).get("kind")


def c_rows(ref, wording):
    """One row per allowed record of a C run, with SA, SG and CA computed."""
    from sgi.stages.goal import standard_goal
    from sgi.types import GroundedTarget
    th = H.recorded_config(ref)[1].thresholds
    recs = sorted(H.records(ref), key=lambda r: (r.scene_id, r.episode_id, r.trial))
    rows = []
    for rec in recs:
        rec_ok = bool(rec.score.get("success"))
        has_target = rec.target is not None and rec.result.get("status") == "ok"
        goal_step = has_target and rec.people is not None
        kinds = [kind_of(u) for u in (rec.constraints or [])]
        row = dict(wording=wording, stage=ref.stage, arm=ref.arm, model=ref.model, scene=rec.scene_id,
                   episode=rec.episode_id, trial=rec.trial, situation=rec.situation, source=rec.result.get("source"),
                   has_target=has_target, goal_step=goal_step, solver_step=rec.solver_step,
                   n_people=None if rec.people is None else len(rec.people), kinds=kinds,
                   has_come_after="come_after" in kinds, target_correct=bool(rec.score.get("target_correct")),
                   C_rec=rec_ok)
        row["C_now"] = bool(H.score_pose(rec, rec.pose_world).success) if rec.pose_world is not None else False
        # solver alone: recorded target and people, no constraints
        if goal_step:
            sa = H.resolve(rec, constraints=[])
            row["SA"] = bool(sa.score.success)
            row["SA_tc"] = bool(sa.score.target_correct)
            if "come_after" in kinds:
                kept = [u["constraint"] for u in rec.constraints if kind_of(u) == "come_after"]
                ca = H.resolve(rec, constraints=kept)
                row["CA"] = bool(ca.score.success)
            else:
                row["CA"] = row["SA"]
        else:
            row["SA"] = rec_ok
            row["CA"] = rec_ok
        # people-blind standard goal for the recorded target
        if has_target:
            sc = H.scene(ref, rec.scene_id)
            target = GroundedTarget.model_validate(rec.target)
            sg = standard_goal(sc.observation, target, th)
            row["SG"] = bool(H.score_pose(rec, sg.pose_world).success)
            if row["source"] == "standard" and rec.pose_world is not None:
                row["SG_same_as_recorded_m"] = math.dist(sg.pose_world[:2], rec.pose_world[:2])
        else:
            row["SG"] = rec_ok
        rows.append(row)
    return rows


def re_rows(ref, wording, method_label):
    return [dict(wording=wording, stage=ref.stage, arm=ref.arm, model=ref.model, method=method_label,
                 situation=rec.situation, ok=bool(rec.score.get("success")), scene=rec.scene_id,
                 episode=rec.episode_id, trial=rec.trial)
            for rec in H.records(ref)]


def main():
    started = time.monotonic()
    out_rows, re_out = [], []
    for wording, plan in (("new", NEW), ("old", OLD)):
        for model, (c_stage, re_stage) in plan.items():
            for arm in ARMS:
                ref = one_run(c_stage, arm, model, "con")
                t0 = time.monotonic()
                rows = c_rows(ref, wording)
                out_rows += rows
                print(f"{wording} {ref.folder}: {len(rows)} records, {time.monotonic() - t0:.0f} s", flush=True)
                for method, label in (("cot", "R"), ("fsc", "E")):
                    re_out += re_rows(one_run(re_stage, arm, model, method), wording, label)
    with open(OUT / "records.jsonl", "w") as fh:
        for row in out_rows:
            fh.write(json.dumps(row) + "\n")
    with open(OUT / "re_records.jsonl", "w") as fh:
        for row in re_out:
            fh.write(json.dumps(row) + "\n")
    print(f"done in {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    main()
