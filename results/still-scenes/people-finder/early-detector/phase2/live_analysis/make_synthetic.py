"""Test inputs for analyse.py: synthetic "live" folders in the driver's output format, built from recorded data with
zero calls. NOT results: every config.yaml says `i09_live.synthetic`, and analyse.py prints a banner for them.

    make_synthetic.py [--root live_analysis/synthetic]

Each variant is `<root>/<variant>/runs/` with, per model, three run folders `<time>_s24_i09_<arm>_<model>_con/`
(config.yaml, results.jsonl), a check file `check_<model>_live_<time>.json` (the upstream comparison made with
driver.py's rule), a count-only report and one shared `ledger.jsonl` (one sent and one answered line per constraints
call in the records).

The records are the RECORDED t0 records of the six runs (s19 GLM, s20 DeepSeek; con; real, staffed, plain), read
through the step-0 harness, with `config.people_source` set and the people call taken out of `result.calls`; the
people list, constraints, score and everything else are the recorded ones. So c_copy reads as recorded C (GLM 64,
DeepSeek 65: "mixed"). d_replay sets each record's score.success to A2's D replay at t0 (69, 71: "consistent").
The other variants change one thing each to exercise a refusal, a validity failure or a boundary of the rule.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
RD = HERE.parents[2]
LIVE = HERE.parent / "live"
HARNESS_DIR = RD.parents[2] / "docs" / "model_notes" / "tools"
sys.path.insert(0, str(HARNESS_DIR))
import yaml                                         # noqa: E402

import research_harness as H                        # noqa: E402

MODELS = {"glm53": "s19", "dsv41": "s20"}
ARMS = ("real", "staffed", "plain")
PEOPLE_SOURCE = "detector_i09_a1_frozen"
EVAL_PEOPLE = RD / "phase1c" / "i09" / "a1" / "eval_people.json"
EVAL_PEOPLE_SHA256 = "ffa595453c94b444c8dd030d053d37ac103441a7e74f07e3f172836b3d0c24b3"
FIRST_BUILD_DRIVER = "e6e73181d47cfeb1d22ec9fcc91bcdbb6ec625e32dc002be4a877e75d2cf9470"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_driver_sha() -> str:
    return next(line.split()[0] for line in (LIVE / "BUILD.sha256").read_text().splitlines()
                if line.split()[1:] == ["driver.py"])


def compare_upstream(record: dict, recorded) -> dict:
    """driver.py's compare_upstream, same rule."""
    def plan_of(r):
        return None if r is None else {k: v for k, v in r.items() if k != "image_sha256"}
    mine, theirs = record["result"], recorded.result
    stages = lambda calls: [c["stage"] for c in calls or [] if c["stage"] != "people"]       # noqa: E731
    out = {
        "plan_equal": plan_of(record["plan"]) == plan_of(recorded.raw.get("plan")),
        "image_equal": (record["plan"] or {}).get("image_sha256") == (recorded.raw.get("plan") or {}).get("image_sha256"),
        "context_equal": mine.get("context_check") == theirs.get("context_check"),
        "location_equal": mine.get("location") == theirs.get("location"),
        "target_equal": mine.get("target") == theirs.get("target"),
        "calls_equal": stages(mine.get("calls")) == stages(theirs.get("calls")),
        "goal_step_live": any(c["stage"] == "constraints" for c in mine.get("calls") or []),
        "goal_step_recorded": any(c["stage"] == "people" for c in theirs.get("calls") or []),
    }
    out["all_equal"] = all(out[k] for k in ("plan_equal", "image_equal", "context_equal", "location_equal",
                                            "target_equal", "calls_equal"))
    return out


def load_recorded() -> dict:
    """Per (arm, model): the run, its config.yaml as written, and its t0 records (harness)."""
    out = {}
    for model, stage in MODELS.items():
        for arm in ARMS:
            ref, = H.runs(stage=stage, arm=arm, model=model, method="con")
            recs = [r for r in H.records(ref) if r.trial == 0]
            assert len(recs) == 28, (ref.folder, len(recs))
            out[(arm, model)] = {"ref": ref, "doc": H.recorded_config(ref)[0], "records": recs}
    return out


def a2_d(model: str) -> dict:
    rows = [json.loads(line) for line in (RD / "phase1c" / "i09" / "a2" / f"rows_old_{model}.jsonl")
            .read_text().splitlines() if line.strip()]
    return {(r["arm"], r["scene"], r["episode"]): bool(r["D"]) for r in rows if r["trial"] == 0}


def detector_counts() -> dict:
    data = EVAL_PEOPLE.read_bytes()
    assert hashlib.sha256(data).hexdigest() == EVAL_PEOPLE_SHA256
    doc = json.loads(data)
    return {e["scene_id"]: len(e["people"]) for e in doc["scenes"].values() if e["domain"] == "E-real"}


def to_live(rec, success=None, keep_people_call=False) -> dict:
    raw = copy.deepcopy(rec.raw)
    raw["config"]["people_source"] = PEOPLE_SOURCE
    if not keep_people_call:
        raw["result"]["calls"] = [c for c in raw["result"].get("calls") or [] if c["stage"] != "people"]
    if success is not None:
        raw["score"]["success"] = bool(success)
    return raw


def write_set(runs: Path, model: str, recorded: dict, records: dict, stamp: str, what: str, counts: dict,
              ledger: list, *, stopped_by=None, driver_sha=None, upstream_override=None,
              arms=ARMS, check=True) -> None:
    """Three run folders of one model (those of `arms`) and, if `check`, its check file."""
    entries = []
    for i, arm in enumerate(arms):
        rec = recorded[(arm, model)]
        name = f"s24_i09_{arm}_{model}_con"
        folder = runs / f"{stamp[:-2]}{int(stamp[-2:]) + i:02d}_{name}"
        folder.mkdir(parents=True)
        recs = records[arm]
        vlm_errors = sum(r["result"]["status"] == "vlm_error" for r in recs)
        stopped_here = stopped_by is not None and arm == arms[-1]
        inc = vlm_errors > 0 or stopped_here
        goal = [r for r in recs if any(c["stage"] == "constraints" for c in r["result"]["calls"])]
        cache = {"plan": {"hit": len(recs)},
                 "context": {"hit": sum(any(c["stage"] == "context" for c in r["result"]["calls"]) for r in recs)},
                 "constraints": {"sent": len(goal)}}
        doc = {"name": name, "scenes_dir": str(rec["ref"].scenes_dir), "config": rec["doc"]["config"],
               "example_ids": [], "examples_missing": [], "skipped_held_out": 0, "vlm_errors": vlm_errors,
               "incomplete": inc,
               "people_source": {"kind": "detector", "tag": PEOPLE_SOURCE, "file": str(EVAL_PEOPLE),
                                 "sha256": EVAL_PEOPLE_SHA256, "facing_variant": "F5"},
               "i09_live": {"mode": "live", "baseline_run": rec["ref"].folder, "trial": 0, "code_commit": "202c165",
                            "driver_sha256": (driver_sha or {}).get(arm, build_driver_sha()),
                            "frozen_sha256": sha256(LIVE / "FROZEN.md"), "cache": cache,
                            "synthetic": f"{what}: recorded t0 records of {rec['ref'].folder}, people call taken "
                                         f"out; NOT a live result"}}
        if stopped_here:
            doc["stopped_by"] = stopped_by
        (folder / "config.yaml").write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        with open(folder / "results.jsonl", "w", encoding="utf-8") as out:
            for r in recs:
                out.write(json.dumps(r) + "\n")
        by_key = {(r.scene_id, r.episode_id): r for r in rec["records"]}
        upstream = []
        for r in recs:
            cmp = compare_upstream(r, by_key[(r["scene_id"], r["episode_id"])])
            if upstream_override and (arm, r["scene_id"], r["episode_id"]) in upstream_override:
                cmp.update(upstream_override[(arm, r["scene_id"], r["episode_id"])])
            upstream.append({"scene": r["scene_id"], "episode": r["episode_id"], "status": r["result"]["status"],
                             **cmp})
        for r in goal:
            ledger.append({"event": "sent", "model": model, "arm": arm, "scene": r["scene_id"],
                           "episode": r["episode_id"], "trial": 0, "stage": "constraints"})
            ledger.append({"event": "answered", "model": model, "arm": arm, "scene": r["scene_id"],
                           "episode": r["episode_id"], "trial": 0, "stage": "constraints", "finish": "ok"})
        entries.append({"arm": arm, "baseline_run": rec["ref"].folder, "name": name, "mode": "live",
                        "scenes": {sid: {"fingerprint": "synthetic", "detector_people": counts.get(sid)}
                                   for sid in sorted({r["scene_id"] for r in recs})},
                        "records": len(recs), "upstream": upstream, "people_asked": len(goal),
                        "stopped_by": stopped_by if arm == arms[-1] else None, "run_folder": str(folder),
                        "cache": cache, "vlm_errors": vlm_errors,
                        "successes": sum(bool(r["score"]["success"]) for r in recs)})
    if check:
        report = {"model": model, "mode": "live", "synthetic": what, "driver_sha256": build_driver_sha(),
                  "frozen_sha256": sha256(LIVE / "FROZEN.md"), "arms": entries, "stopped_by": stopped_by}
        (runs / f"check_{model}_live_{stamp}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


def write_ledger(runs: Path, ledger: list, refused: list = ()) -> None:
    totals = {"all": 0}
    with open(runs / "ledger.jsonl", "w", encoding="utf-8") as out:
        for row in ledger:
            if row["event"] == "sent":
                totals["all"] += 1
                totals[row["model"]] = totals.get(row["model"], 0) + 1
                row = {**row, "total": totals["all"], "model_total": totals[row["model"]]}
            out.write(json.dumps({"time": "synthetic", **row}) + "\n")
        for row in refused:
            out.write(json.dumps({"time": "synthetic", "event": "refused", **row}) + "\n")


def write_count_only(runs: Path, model: str, records: dict, stamp: str) -> None:
    goal = sum(any(c["stage"] == "constraints" for c in r["result"]["calls"]) for arm in ARMS for r in records[arm])
    report = {"model": model, "mode": "count-only", "synthetic": True,
              "bounds": {"misses_by_stage": {"constraints": goal}, "lower": goal, "upper": goal, "upstream_misses": 0},
              "cache_total": {"plan": {"hit": 84}, "constraints": {"miss": goal}}}
    (runs / f"count_only_{model}_{stamp}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


def flip(records: dict, model: str, n: int, to: bool, arms=ARMS) -> list[str]:
    """Set score.success to `to` on the first n records (in arm and file order) that are not already `to`."""
    done = []
    for arm in arms:
        for r in records[model][arm]:
            if len(done) < n and bool(r["score"]["success"]) != to:
                r["score"]["success"] = to
                done.append(f"{r['scene_id']} {r['episode_id']}")
    assert len(done) == n, (model, n, done)
    return done


def main() -> int:
    parser = argparse.ArgumentParser(prog="make_synthetic.py")
    parser.add_argument("--root", type=Path, default=HERE / "synthetic")
    args = parser.parse_args()
    root = args.root.resolve()
    assert not root.is_relative_to(LIVE.resolve()), "never inside live/"
    if root.exists():
        shutil.rmtree(root)
    recorded = load_recorded()
    counts = detector_counts()
    d = {model: a2_d(model) for model in MODELS}
    made = {}

    def base_records(success_from=None):
        out = {}
        for model in MODELS:
            out[model] = {}
            for arm in ARMS:
                out[model][arm] = [to_live(r, None if success_from is None else
                                           success_from[model][(arm, r.scene_id, r.episode_id)])
                                   for r in recorded[(arm, model)]["records"]]
        return out

    def standard(variant, records, what, **per_model):
        runs = root / variant / "runs"
        runs.mkdir(parents=True)
        ledger = []
        for j, model in enumerate(MODELS):
            write_count_only(runs, model, records[model], f"20261019-0{j}0000")
            write_set(runs, model, recorded, records[model], f"20261019-0{j}1000", what, counts, ledger,
                      **per_model.get(model, {}))
        write_ledger(runs, ledger, per_model.get("refused", ()))
        return runs

    # c_copy: recorded C as it is -> GLM 64, DeepSeek 65, "mixed"
    standard("c_copy", base_records(), "c_copy")
    made["c_copy"] = "recorded t0 records, people call taken out; expect GLM 64, DeepSeek 65, mixed, exit 0"
    # d_replay: score.success = A2's D replay at t0 -> 69, 71, "consistent"
    standard("d_replay", base_records(d), "d_replay (score.success = A2 D at t0)")
    made["d_replay"] = "as c_copy with score.success = A2's D replay at t0; expect 69, 71, consistent, exit 0"
    # contradicts: one GLM and one DeepSeek success made a failure -> 63, 64
    recs = base_records()
    changed = flip(recs, "glm53", 1, False) + flip(recs, "dsv41", 1, False)
    standard("contradicts", recs, f"contradicts (successes made failures: {', '.join(changed)})")
    made["contradicts"] = "c_copy with 1 GLM and 1 DeepSeek success made failures; expect 63, 64, contradicts"
    # boundaries of the rule, from d_replay
    recs = base_records(d)
    changed = flip(recs, "glm53", 3, False) + flip(recs, "dsv41", 3, False)
    standard("edge_consistent", recs, f"edge_consistent ({', '.join(changed)} made failures)")
    made["edge_consistent"] = "d_replay with 3 + 3 successes made failures; expect 66, 68, consistent"
    recs = base_records(d)
    changed = flip(recs, "glm53", 3, False) + flip(recs, "dsv41", 4, False)
    standard("edge_mixed_ds67", recs, f"edge_mixed_ds67 ({', '.join(changed)} made failures)")
    made["edge_mixed_ds67"] = "d_replay with 3 + 4 successes made failures; expect 66, 67, mixed"
    recs = base_records()
    changed = flip(recs, "glm53", 1, False)
    standard("edge_mixed_glm63", recs, f"edge_mixed_glm63 ({', '.join(changed)} made a failure)")
    made["edge_mixed_glm63"] = "c_copy with 1 GLM success made a failure; expect 63, 65, mixed"
    # missing arm: DeepSeek staffed folder absent
    runs = standard("neg_missing_arm", base_records(), "neg_missing_arm",
                    dsv41={"arms": ("real", "plain")})
    made["neg_missing_arm"] = "no DeepSeek staffed folder; expect exit 2; with --allow-incomplete exit 3"
    # incomplete: GLM plain stopped by the cap, a refused ledger line
    standard("neg_cap_stop", base_records(), "neg_cap_stop",
             glm53={"stopped_by": "CapReached: synthetic"},
             refused=[{"model": "glm53", "arm": "plain", "stage": "constraints", "total": 48, "model_total": 48}])
    made["neg_cap_stop"] = "GLM plain incomplete (cap stop) and a refused ledger line; expect exit 2; " \
                           "with --allow-incomplete exit 3"
    # upstream: one GLM real record's context answer differs, and the check file says so
    recs = base_records()
    r0 = next(r for r in recs["glm53"]["real"] if r["result"].get("context_check"))
    r0["result"]["context_check"] = {**r0["result"]["context_check"], "reasoning": "synthetic change"}
    standard("neg_upstream", recs, "neg_upstream",
             glm53={"upstream_override": {("real", r0["scene_id"], r0["episode_id"]):
                                          {"context_equal": False, "all_equal": False}}})
    made["neg_upstream"] = f"{r0['scene_id']} {r0['episode_id']} context answer changed; expect exit 3 (confounded)"
    # driver sha: DeepSeek plain recorded with the first build's driver
    standard("neg_driver_sha", base_records(), "neg_driver_sha", dsv41={"driver_sha": {"plain": FIRST_BUILD_DRIVER}})
    made["neg_driver_sha"] = "DeepSeek plain config.yaml has the first build's driver sha256; expect exit 3"
    # people call left in one record
    recs = base_records()
    rp = next(r for r in recorded[("staffed", "dsv41")]["records"] if any(c["stage"] == "people"
                                                                          for c in r.result["calls"]))
    idx = [x["scene_id"] + x["episode_id"] for x in recs["dsv41"]["staffed"]].index(rp.scene_id + rp.episode_id)
    recs["dsv41"]["staffed"][idx] = to_live(rp, keep_people_call=True)
    standard("neg_people_call", recs, "neg_people_call")
    made["neg_people_call"] = f"{rp.scene_id} {rp.episode_id} keeps its people call; expect exit 3"
    # vlm_error after one re-run: GLM real has one vlm_error record in both invocations
    runs = root / "vlm_error_rerun" / "runs"
    runs.mkdir(parents=True)
    ledger = []
    recs = base_records()
    rv = next(r for r in recs["glm53"]["real"] if r["score"]["success"] and
              any(c["stage"] == "constraints" for c in r["result"]["calls"]))
    rv["result"]["status"] = "vlm_error"
    rv["result"]["calls"][-1]["status"] = "vlm_error"
    rv["score"]["success"] = False
    for j, model in enumerate(MODELS):
        write_count_only(runs, model, recs[model], f"20261019-0{j}0000")
    write_set(runs, "glm53", recorded, recs["glm53"], "20261019-001000", "vlm_error_rerun first", counts, ledger)
    write_set(runs, "glm53", recorded, recs["glm53"], "20261019-002000", "vlm_error_rerun re-run", counts, [])
    write_set(runs, "dsv41", recorded, recs["dsv41"], "20261019-011000", "vlm_error_rerun", counts, ledger)
    write_ledger(runs, ledger)
    made["vlm_error_rerun"] = (f"{rv['scene_id']} {rv['episode_id']} (a recorded success) is vlm_error in both GLM "
                               "invocations; expect exit 2; with --count-vlm-errors-as-failures GLM 63, DeepSeek 65, "
                               "mixed, exit 0")
    # a later incomplete folder after the complete one: the complete one is taken
    runs = standard("later_incomplete", base_records(), "later_incomplete")
    later = runs / "20261019-030000_s24_i09_real_glm53_con"
    later.mkdir()
    (later / "config.yaml").write_text(yaml.safe_dump({"name": "s24_i09_real_glm53_con", "incomplete": True,
                                                       "i09_live": {"synthetic": "later_incomplete"}}))
    made["later_incomplete"] = "c_copy plus a later incomplete GLM real folder; expect the earlier complete one " \
                               "taken, a note, exit 0"
    (root / "VARIANTS.json").write_text(json.dumps(made, indent=1), encoding="utf-8")
    for name, what in made.items():
        print(f"{name}: {what}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
