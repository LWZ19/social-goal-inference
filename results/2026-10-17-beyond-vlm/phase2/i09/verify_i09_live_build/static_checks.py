"""Verifier of i09_live_build, part 1 (no calls, nothing written outside this folder): recorded configs, prompts,
t0 records, goal steps, retries, baselines. Reads runs only through the step-0 harness. Uses my own git-archive copy
of 202c165 (archive_202c165/src), not the builder's."""
import collections
import hashlib
import json
import sys
from pathlib import Path

V = Path(__file__).resolve().parent
RD = V.parents[2]
sys.path.insert(0, str(V / "archive_202c165" / "src"))
sys.path.insert(1, "docs/model_notes/tools")
import sgi  # noqa: E402
assert str(V / "archive_202c165") in sgi.__file__, sgi.__file__
import research_harness as H  # noqa: E402

PROMPTS = V / "archive_202c165" / "prompts"
out = {}
RUNS = {"glm53": ("s19", "glm53"), "dsv41": ("s20", "dsv41")}
for model, (stage, mkey) in RUNS.items():
    out[model] = {}
    tot = collections.Counter()
    for arm in ("real", "staffed", "plain"):
        refs = H.runs(stage=stage, arm=arm, model=mkey, method="con")
        assert len(refs) == 1, refs
        ref = refs[0]
        raw_cfg, cfg = H.recorded_config(ref)
        stages = {s: (st.model, st.strategy) for s, st in cfg.stages.items()}
        entries = {}
        for s, (name, _) in stages.items():
            e = cfg.models[name]
            entries[name] = {"provider": e.provider, "model": e.model, "base_url": e.base_url,
                             "reasoning_effort": e.reasoning_effort, "rpm": e.max_requests_per_minute}
        pdir = H.guard_file(ref.path / "prompts")
        names = sorted(p.name for p in Path(pdir).iterdir())
        prompt_eq = {n: hashlib.sha256((Path(pdir) / n).read_bytes()).hexdigest()
                     == hashlib.sha256((PROMPTS / n).read_bytes()).hexdigest() for n in names}
        recs = [r for r in H.records(ref) if r.trial == 0]
        sit = collections.Counter(r.situation for r in recs)
        goal = [r for r in recs if any(c["stage"] == "people" for c in r.result.get("calls") or [])]
        con = [r for r in recs if any(c["stage"] == "constraints" for c in r.result.get("calls") or [])]
        retries = [(r.scene_id, r.episode_id, c["stage"], c.get("validation_retries"), c.get("transient_retries"))
                   for r in recs for c in r.result.get("calls") or []
                   if (c.get("validation_retries") or 0) or (c.get("transient_retries") or 0)]
        con_retries = [x for x in retries if x[2] == "constraints"]
        stages_seen = collections.Counter(c["stage"] for r in recs for c in r.result.get("calls") or [])
        succ = sum(bool(r.score.get("success")) for r in recs)
        line_succ = sum(bool(r.score.get("success")) for r in recs if r.situation == "waiting_line")
        statuses = collections.Counter(r.result.get("status") for r in recs)
        lat_con = sorted(c.get("latency_total_s") or 0 for r in recs for c in r.result.get("calls") or []
                         if c["stage"] == "constraints")
        lat_ppl = sorted(c.get("latency_total_s") or 0 for r in recs for c in r.result.get("calls") or []
                         if c["stage"] == "people")
        out[model][arm] = {"run": ref.folder, "scenes_dir": str(ref.scenes_dir), "stages": stages,
                           "entries": entries, "use_cache": cfg.run.use_cache, "trials": cfg.run.trials,
                           "people": cfg.run.people, "goal": cfg.run.goal, "grounding": cfg.run.grounding,
                           "grounding_source": cfg.run.grounding_source, "context_check": cfg.run.context_check,
                           "example_ids_in_cfg": raw_cfg.get("example_ids"),
                           "held_out": sorted(cfg.held_out_situations),
                           "prompts_equal": prompt_eq, "t0_records": len(recs), "situations": dict(sit),
                           "goal_steps_people_call": len(goal), "records_with_constraints_call": len(con),
                           "goal_eq_con": sorted((r.scene_id, r.episode_id) for r in goal)
                           == sorted((r.scene_id, r.episode_id) for r in con),
                           "calls_by_stage": dict(stages_seen), "retries_any": retries,
                           "constraints_retries": con_retries, "successes": succ, "line_successes": line_succ,
                           "statuses": dict(statuses), "n_con_lat": len(lat_con), "con_lat": lat_con,
                           "ppl_lat": lat_ppl}
        tot.update({"records": len(recs), "goal": len(goal), "succ": succ, "line": line_succ})
    out[model]["total"] = dict(tot)
    con_all = sorted(x for arm in ("real", "staffed", "plain") for x in out[model][arm]["con_lat"])
    ppl_all = sorted(x for arm in ("real", "staffed", "plain") for x in out[model][arm]["ppl_lat"])
    med = lambda xs: (xs[(len(xs) - 1) // 2] + xs[len(xs) // 2]) / 2 if xs else None  # noqa: E731
    out[model]["median_con_latency"] = (len(con_all), med(con_all))
    out[model]["median_people_latency"] = (len(ppl_all), med(ppl_all))

# Baselines from the A2 rows and I01 rows, at trial 0
base = {}
for model in ("glm53", "dsv41"):
    rows = [json.loads(l) for l in open(RD / "phase1c/i09/a2" / f"rows_old_{model}.jsonl")]
    t = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        for k in ("C_rec", "D", "P", "C"):
            t[r["trial"]][k] += bool(r[k])
            if r["situation"] == "waiting_line":
                t[r["trial"]][k + "_line"] += bool(r[k])
        t[r["trial"]]["n"] += 1
    i01 = [json.loads(l) for l in open(RD / "phase1/i01_repairs" / f"rows_old_{model}.jsonl")]
    i01_t0 = [r for r in i01 if r["trial"] == 0]
    # record-for-record P equality between A2 rows and I01 rows at t0
    a2p = {(r["run"], r["scene"], r["episode"]): bool(r["P"]) for r in rows if r["trial"] == 0}
    i01p = {(r["run"], r["scene"], r["episode"]): bool(r["P"]) for r in i01_t0}
    base[model] = {"by_trial": {k: dict(v) for k, v in sorted(t.items())},
                   "i01_P_t0": sum(i01p.values()), "i01_n_t0": len(i01p),
                   "P_records_equal": a2p == i01p}
out["baselines"] = base
for model in ("glm53", "dsv41"):
    b = base[model]["by_trial"][0]
    c, d = b["C_rec"], b["D"]
    out[model]["bars"] = {"consistent_min": max(c, d - 3), "contradicts_max": c - 1}

(V / "static_checks.json").write_text(json.dumps(out, indent=1, default=str))
for model in ("glm53", "dsv41"):
    print(model, out[model]["total"], "median con", out[model]["median_con_latency"], "median ppl",
          out[model]["median_people_latency"], "bars", out[model]["bars"])
    for arm in ("real", "staffed", "plain"):
        a = out[model][arm]
        print(" ", arm, a["run"], "t0", a["t0_records"], a["situations"], "goal", a["goal_steps_people_call"],
              "con", a["records_with_constraints_call"], "eq", a["goal_eq_con"], "calls", a["calls_by_stage"],
              "succ", a["successes"], "line", a["line_successes"], "con_retries", a["constraints_retries"],
              "prompts_eq", all(a["prompts_equal"].values()), len(a["prompts_equal"]))
        print("   stages", a["stages"])
        print("   entries", a["entries"], "people", a["people"], "goal", a["goal"], "ctx", a["context_check"],
              "ground", a["grounding"], a["grounding_source"], "examples", a["example_ids_in_cfg"],
              "statuses", a["statuses"], "retries_any", a["retries_any"])
    print(" baselines", base[model])
