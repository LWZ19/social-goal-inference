"""Verifier's independent computation of FROZEN 5.1-5.7 and the reading, compared with analyse.py's summary.json.

    v_measures.py     builds verify/measures_variant/runs (c_copy with deterministic changes so that D-live differs
                      from recorded C in every measure), runs analyse.py on it and on c_copy and d_replay, and
                      compares every number with this script's own computation. Writes verify/MEASURES.json.

Zero calls. Recorded C via the harness; A2 and I01 rows read as files; the synthetic live folders read as files.
"""
from __future__ import annotations

import collections
import copy
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
LA = HERE.parent
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402
RD = paths.STILL                          # results/still-scenes
WT = paths.REPO                           # the repository root
sys.path.insert(0, str(paths.HARNESS))
import research_harness as H  # noqa: E402

PY = str(WT / ".venv" / "bin" / "python")
ENV = {**os.environ, "PYTHONPATH": "", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1"}
RUNS = {"glm53": ("s19", ("real", "staffed", "plain")), "dsv41": ("s20", ("real", "staffed", "plain"))}
SIT = ("no_one", "waiting_line", "conversation_group")
CONS = {"glm53": 66, "dsv41": 68}
CONT = {"glm53": 63, "dsv41": 64}


def rows(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def baselines():
    out = {}
    for model, (stage, arms) in RUNS.items():
        t0, C = {}, {}
        for arm in arms:
            ref, = H.runs(stage=stage, arm=arm, model=model, method="con")
            for rec in H.records(ref):
                if rec.trial == 0:
                    k = (arm, rec.scene_id, rec.episode_id)
                    t0[k] = rec
                    C[k] = bool(rec.score.get("success"))
        a2 = {(r["arm"], r["scene"], r["episode"]): bool(r["D"]) for r in rows(RD / "people-finder/early-detector/phase1c/a2" / f"rows_old_{model}.jsonl") if r["trial"] == 0}
        p = {(r["arm"], r["scene"], r["episode"]): bool(r["P"]) for r in rows(RD / "replay-harness/repairs" / f"rows_old_{model}.jsonl") if r["trial"] == 0}
        out[model] = {"t0": t0, "C": C, "D": a2, "P": p}
    return out


def live_records(runs: Path):
    out = collections.defaultdict(dict)
    for folder in sorted(runs.iterdir()):
        if not folder.is_dir():
            continue
        parts = folder.name.split("_")          # <time>_s24_i09_<arm>_<model>_con
        arm, model = parts[3], parts[4]
        for r in rows(folder / "results.jsonl"):
            out[model][(arm, r["scene_id"], r["episode_id"])] = r
    return out


def q(xs, p):
    return None if not xs else float(np.quantile(np.array(xs, dtype=float), p))


def stages(r):
    return [c["stage"] for c in r["calls"] or []]


def lat(r, st, geo=True):
    return sum(c["latency_total_s"] or 0.0 for c in r["calls"] or [] if c["stage"] in st) + \
        ((r.get("timings_s") or {}).get("geometry", 0.0) if geo else 0.0)


def mine(runs: Path, base) -> dict:
    live = live_records(runs)
    res = {}
    for model in RUNS:
        b, L = base[model], live[model]
        sit = {k: b["t0"][k].situation for k in L}
        succ = {k: bool(r["score"]["success"]) for k, r in L.items()}
        m = {"m1": sum(succ.values()), "n": len(L),
             "vlm_error": sum(r["result"]["status"] == "vlm_error" for r in L.values())}
        m["m2"] = {s: [sum(succ[k] for k in L if sit[k] == s)] + [sum(b[x][k] for k in L if sit[k] == s) for x in "CDP"]
                   for s in SIT}
        m["m3"] = {}
        for x in "CDP":
            fl = [k for k in L if succ[k] and not b[x][k]]
            br = [k for k in L if not succ[k] and b[x][k]]
            m["m3"][x] = {"flips": len(fl), "breaks": len(br), "flip_layouts": len({k[1].split("_", 1)[1] for k in fl}),
                          "break_scenes": len({k[1] for k in br})}
        ls = {k: r["result"] for k, r in L.items() if "constraints" in stages(r["result"])}
        cs = {k: b["t0"][k].result for k in L if "people" in stages(b["t0"][k].result)}
        craw = {k: b["t0"][k].raw for k in cs}
        lraw = {k: L[k] for k in ls}

        def use(steps, raws):
            ans = {k: r for k, r in steps.items() if r.get("constraints") is not None and r.get("solver") is not None}
            st = collections.Counter(u["status"] for r in ans.values() for u in r["constraints"])
            nums = [(k, n) for k, r in ans.items() for u in r["constraints"] for n in (u["constraint"].get("people") or [])]
            shown = {k: {p["number"] for p in r.get("people") or []} for k, r in ans.items()}
            inl = sum(n in shown[k] for k, n in nums)
            outside = sum(any(n not in shown[k] for kk, n in nums if kk == k) for k in ans)
            cu = [raws[k]["measures"].get("constraints_unusable") for k in ans]
            eb = [raws[k]["measures"].get("exclusions_broken") for k in ans]
            per = [len(r["constraints"]) for r in ans.values()]
            return {"steps": len(ans), "used": st["used"], "unusable": st["unusable"], "superseded": st["superseded"],
                    "cut": st["cut"], "no_usable": sum(bool(r["flags"].get("no_usable_constraint")) for r in ans.values()),
                    "cu_sum": sum(x for x in cu if x is not None), "eb_sum": sum(x for x in eb if x is not None),
                    "per_mean": sum(per) / len(per) if per else None, "per_median": q(per, 0.5),
                    "named": len(nums), "in_list": inl, "steps_outside": outside}
        m["m4m5"] = {"live": use(ls, lraw), "C": use(cs, craw)}
        ll = {k: lat(r, ("constraints",)) for k, r in ls.items()}
        cl = {k: lat(r, ("people", "constraints")) for k, r in cs.items()}
        both = sorted(set(ll) & set(cl))
        m["m6"] = {"live_med": q(list(ll.values()), .5), "live_p90": q(list(ll.values()), .9),
                   "C_med": q(list(cl.values()), .5), "C_p90": q(list(cl.values()), .9), "paired_n": len(both),
                   "diff_med": q([ll[k] - cl[k] for k in both], .5), "diff_p90": q([ll[k] - cl[k] for k in both], .9),
                   "live_calls_med": q([lat(r, ("constraints",), False) for r in ls.values()], .5),
                   "live_det_med": q([v + 0.0556 for v in ll.values()], .5)}
        m["m7"] = {"live_people_mean": float(np.mean([len(r["people"]) for r in ls.values() if r.get("people") is not None])),
                   "C_people_mean": float(np.mean([len(r["people"]) for r in cs.values() if r.get("people") is not None]))}
        res[model] = m
    ok = all(res[m]["n"] == 84 for m in RUNS)
    s = {m: res[m]["m1"] for m in RUNS}
    res["reading"] = ("consistent with A2" if all(s[m] >= CONS[m] for m in RUNS) else
                      "contradicts A2" if all(s[m] <= CONT[m] for m in RUNS) else "mixed") if ok else None
    return res


def theirs(summary: dict) -> dict:
    res = {}
    for model in RUNS:
        m = summary["measures"][model]
        u = {}
        for arm in ("live", "C"):
            a, b = m["m4"][arm], m["m5"][arm]
            u[arm] = {"steps": a["goal_steps_with_answer"], **{k: a["status"][k] for k in ("used", "unusable", "superseded", "cut")},
                      "no_usable": a["no_usable_constraint_steps"], "cu_sum": a["constraints_unusable_sum"],
                      "eb_sum": a["exclusions_broken_sum"], "per_mean": a["per_goal_step_mean"],
                      "per_median": a["per_goal_step_median"], "named": b["numbers_named"], "in_list": b["in_list"],
                      "steps_outside": b["steps_naming_one_not_in_list"]}
        l6 = m["m6"]
        res[model] = {
            "m1": m["m1"]["successes"], "n": m["m1"]["n"], "vlm_error": m["m1"]["vlm_error"],
            "m2": {s: [m["m2"][s]["live"], m["m2"][s]["C"], m["m2"][s]["D"], m["m2"][s]["P"]] for s in SIT},
            "m3": {x: {"flips": m["m3"][x]["flips"], "breaks": m["m3"][x]["breaks"],
                       "flip_layouts": len(m["m3"][x]["flip"]["layouts"]), "break_scenes": len(m["m3"][x]["break"]["scenes"])}
                   for x in "CDP"},
            "m4m5": u,
            "m6": {"live_med": l6["all"]["live"]["median"], "live_p90": l6["all"]["live"]["p90"],
                   "C_med": l6["all"]["C"]["median"], "C_p90": l6["all"]["C"]["p90"], "paired_n": l6["paired"]["n"],
                   "diff_med": l6["paired"]["live_minus_C"]["median"], "diff_p90": l6["paired"]["live_minus_C"]["p90"],
                   "live_calls_med": l6["calls_alone"]["live"]["median"],
                   "live_det_med": l6["all"]["live_plus_detector_55_6ms"]["median"]},
            "m7": {"live_people_mean": m["m7"]["people_per_goal_step"]["live_detector"]["mean"],
                   "C_people_mean": m["m7"]["people_per_goal_step"]["C_vlm"]["mean"]}}
    res["reading"] = summary["reading"]["reading"] if summary["reading"]["reading"] != "not applicable" else \
        summary["reading"]["rule_on_these_numbers"]
    return res


def diff(a, b, path=""):
    out = []
    if isinstance(a, dict):
        for k in set(a) | set(b):
            out += diff(a.get(k), b.get(k), f"{path}.{k}")
    elif isinstance(a, list):
        if len(a) != len(b):
            return [f"{path}: {a} vs {b}"]
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        if a is None or b is None or abs(a - b) > 1e-9:
            out.append(f"{path}: {a} vs {b}")
    elif a != b:
        out.append(f"{path}: {a} vs {b}")
    return out


def build_variant(src: Path, dst: Path):
    """D-live made to differ from recorded C: per model, in its goal steps, numbers outside the list, statuses,
    latencies, flags, measures, a dropped person, and success flips/breaks."""
    if dst.parent.exists():
        shutil.rmtree(dst.parent)
    shutil.copytree(src, dst)
    rnd = random.Random(6244)
    for folder in sorted(dst.iterdir()):
        if not folder.is_dir():
            continue
        p = folder / "results.jsonl"
        rs = rows(p)
        for i, r in enumerate(rs):
            res = r["result"]
            if any(c["stage"] == "constraints" for c in res["calls"]):
                for c in res["calls"]:
                    if c["stage"] == "constraints":
                        c["latency_total_s"] = round(rnd.uniform(1.0, 12.0), 3)
                res["timings_s"]["geometry"] = round(rnd.uniform(0.01, 0.5), 4)
                if res.get("constraints"):
                    u = res["constraints"][0]
                    if i % 3 == 0:
                        u["constraint"]["people"] = list(u["constraint"].get("people") or []) + [99]
                    if i % 4 == 1:
                        u["status"] = "cut"
                    if i % 5 == 2:
                        u["status"] = "superseded"
                    if i % 7 == 3:
                        res["constraints"] = res["constraints"] + [copy.deepcopy(u)]
                if i % 6 == 4:
                    res["flags"]["no_usable_constraint"] = not res["flags"].get("no_usable_constraint")
                r["measures"]["constraints_unusable"] = rnd.randint(0, 3)
                r["measures"]["exclusions_broken"] = rnd.randint(0, 2)
                if res.get("people") and len(res["people"]) > 1 and i % 2 == 0:
                    res["people"] = res["people"][:-1]
            if i % 9 == 0:
                r["score"]["success"] = not r["score"]["success"]
        p.write_text("".join(json.dumps(r) + "\n" for r in rs))


def main():
    base = baselines()
    report = {}
    var = HERE / "measures_variant" / "runs"
    build_variant(LA / "synthetic" / "c_copy" / "runs", var)
    for name, runs in (("c_copy", LA / "synthetic" / "c_copy" / "runs"), ("d_replay", LA / "synthetic" / "d_replay" / "runs"),
                       ("measures_variant", var)):
        out = HERE / "measures_out" / name
        proc = subprocess.run(["nice", "-n", "10", PY, str(LA / "analyse.py"), "--runs", str(runs), "--out", str(out)],
                              capture_output=True, text=True, env=ENV)
        summary = json.loads((out / "summary.json").read_text())
        a, b = mine(runs, base), theirs(summary)
        d = diff(a, b)
        report[name] = {"exit": proc.returncode, "reading_analysis": summary["reading"]["reading"], "mine": a,
                        "differences": d}
        print(name, "exit", proc.returncode, "reading", summary["reading"]["reading"], "| mine:", a["reading"],
              "| GLM", a["glm53"]["m1"], "DeepSeek", a["dsv41"]["m1"], "| differences:", len(d), d[:10])
    (HERE / "MEASURES.json").write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
