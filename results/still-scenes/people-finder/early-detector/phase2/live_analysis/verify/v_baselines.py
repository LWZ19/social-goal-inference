"""Independent re-derivation of the I09 live-check baselines (verifier). Zero calls; recorded runs via the harness.

Writes verify/baselines.json and prints a table. Also prints the record schema bits the analysis relies on
(ConstraintUse fields, measure names, call latency field) so they can be checked against analyse.py.
"""
import collections
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
RD = HERE.parents[3]                      # .../2026-10-17-beyond-vlm
WT = RD.parents[2]
sys.path.insert(0, str(WT / "docs" / "model_notes" / "tools"))
import research_harness as H  # noqa: E402

RUNS = {"glm53": ("s19", {"real": "20261011-160500_s19_real_glm53_con",
                          "staffed": "20261011-160352_s19_staffed_glm53_con",
                          "plain": "20261011-160452_s19_plain_glm53_con"}),
        "dsv41": ("s20", {"real": "20261013-191337_s20_real_dsv41_con",
                          "staffed": "20261013-191345_s20_staffed_dsv41_con",
                          "plain": "20261013-191402_s20_plain_dsv41_con"})}


def rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


out = {}
schema = {"constraint_use_keys": collections.Counter(), "constraint_keys": collections.Counter(),
          "measure_keys": collections.Counter(), "call_keys": collections.Counter(), "people_keys": collections.Counter(),
          "flags_keys": collections.Counter(), "result_keys": collections.Counter(), "status": collections.Counter(),
          "timings_keys": collections.Counter()}
for model, (stage, names) in RUNS.items():
    rec_succ = {}
    sit = {}
    gs = collections.Counter()
    gs_with_constraints = collections.Counter()
    for arm, folder in names.items():
        found = H.runs(stage=stage, arm=arm, model=model, method="con")
        assert len(found) == 1 and found[0].folder == folder, (model, arm, [f.folder for f in found])
        n = 0
        for rec in H.records(found[0]):
            n += 1
            k = (arm, rec.scene_id, rec.episode_id, rec.trial)
            assert k not in rec_succ
            rec_succ[k] = bool(rec.score.get("success"))
            sit[k] = rec.situation
            stages = [c.get("stage") for c in rec.result.get("calls") or []]
            if rec.trial == 0 and "people" in stages:
                gs[arm] += 1
                if "constraints" in stages:
                    gs_with_constraints[arm] += 1
            if rec.trial == 0:
                r = rec.result
                schema["result_keys"].update(r.keys())
                schema["status"][r.get("status")] += 1
                schema["measure_keys"].update((rec.raw.get("measures") or {}).keys())
                schema["flags_keys"].update((r.get("flags") or {}).keys())
                schema["timings_keys"].update((r.get("timings_s") or {}).keys())
                for c in r.get("calls") or []:
                    schema["call_keys"].update(c.keys())
                for u in r.get("constraints") or []:
                    schema["constraint_use_keys"].update(u.keys())
                    schema["constraint_keys"].update((u.get("constraint") or {}).keys())
                for p in r.get("people") or []:
                    schema["people_keys"].update(p.keys())
        print(model, arm, folder, "records", n)
    a2 = {(r["arm"], r["scene"], r["episode"], r["trial"]): r for r in rows(RD / "phase1c/i09/a2" / f"rows_old_{model}.jsonl")}
    i01 = {(r["arm"], r["scene"], r["episode"], r["trial"]): r for r in rows(RD / "phase1/i01_repairs" / f"rows_old_{model}.jsonl")}
    assert set(a2) == set(rec_succ) == set(i01), (len(a2), len(rec_succ), len(i01))
    mism_crec = sum(a2[k]["C_rec"] != rec_succ[k] for k in a2)
    mism_p = sum(a2[k]["P"] != i01[k]["P"] for k in a2)
    mism_i01_crec = sum(i01[k]["C_rec"] != rec_succ[k] for k in a2)
    res = {"mismatch_A2_Crec_vs_record": mism_crec, "mismatch_A2P_vs_I01P": mism_p,
           "mismatch_I01_Crec_vs_record": mism_i01_crec,
           "goal_steps_t0": dict(gs), "goal_steps_t0_with_constraints_call": dict(gs_with_constraints)}
    for name, col in (("C", {k: rec_succ[k] for k in rec_succ}), ("D", {k: bool(a2[k]["D"]) for k in a2}),
                      ("P_I01", {k: bool(i01[k]["P"]) for k in i01}), ("P_A2", {k: bool(a2[k]["P"]) for k in a2})):
        per_trial = [sum(v for k, v in col.items() if k[3] == t) for t in (0, 1, 2)]
        by_sit = {s: sum(v for k, v in col.items() if k[3] == 0 and sit[k] == s) for s in ("no_one", "waiting_line", "conversation_group")}
        n_sit = collections.Counter(sit[k] for k in col if k[3] == 0)
        per_arm = {a: sum(v for k, v in col.items() if k[3] == 0 and k[0] == a) for a in names}
        res[name] = {"t0": per_trial[0], "per_trial": per_trial, "by_situation_t0": by_sit, "n_situation_t0": dict(n_sit),
                     "per_arm_t0": per_arm}
    out[model] = res
out["schema"] = {k: dict(v) for k, v in schema.items()}
(HERE / "baselines.json").write_text(json.dumps(out, indent=1))
for model in RUNS:
    r = out[model]
    print(model, {n: (r[n]["t0"], r[n]["by_situation_t0"]["waiting_line"], r[n]["per_trial"]) for n in ("C", "D", "P_I01", "P_A2")},
          "goal steps", r["goal_steps_t0"], sum(r["goal_steps_t0"].values()), "w/ constraints", r["goal_steps_t0_with_constraints_call"],
          "mismatches", r["mismatch_A2_Crec_vs_record"], r["mismatch_A2P_vs_I01P"], r["mismatch_I01_Crec_vs_record"])
    print("  by situation", {n: r[n]["by_situation_t0"] for n in ("C", "D", "P_I01")}, r["C"]["n_situation_t0"])
    print("  per arm", {n: r[n]["per_arm_t0"] for n in ("C", "D", "P_I01")})
print(json.dumps(out["schema"], indent=0)[:4000])
