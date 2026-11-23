"""Crowd test with the AI models: the analyses were fixed before the runs.

From the repository root:
    PYTHONPATH=src python results/still-scenes/crowds/vlm-run/crowd_score.py runs/*_crowd_vlm_*
Writes nothing; prints the tables. With --json PATH also writes the numbers as JSON.
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE.parent / "build" / "manifest.jsonl"
METHOD = {"cot": "R", "fsc": "E", "con": "C"}
LABEL = {"R": "pointing", "E": "pointing + examples", "C": "our method"}
FAMILY = {"base": "no one added", "A": "A behind the line", "B": "B cluster of 3", "C": "C second group",
          "D": "D bystanders", "E": "E near the stop"}
FLIPS, SEED = 200_000, 0


def manifest():
    out = {}
    for line in open(MANIFEST):
        x = json.loads(line)
        if x["accepted"]:
            out[(x["seed"], x["scene_id"])] = x
    return out


def situation(base_id):
    return re.sub(r"^eval_\d+_", "", base_id).rsplit("_", 1)[0]


def records(paths, man):
    latest = {}
    for p in sorted(paths):
        m = re.match(r"^\d{8}-\d{6}_crowd_vlm_seed(\d)_(.+)_(cot|fsc|con)$", Path(p).name)
        if m and (Path(p) / "results.jsonl").exists():
            latest[(m.group(2), int(m.group(1)), METHOD[m.group(3)])] = Path(p)  # later run wins
    recs, runs = [], {}
    for (model, seed, meth), p in latest.items():
        rows = []
        for line in open(p / "results.jsonl"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # a line cut off when a run was stopped
                pass
        runs[(model, seed, meth)] = (p.name, len(rows))
        for r in rows:
            x = man[(seed, r["scene_id"])]
            score, res = r.get("score") or {}, r.get("result") or {}
            recs.append(dict(model=model, seed=seed, meth=meth, scene=r["scene_id"], ep=r["episode_id"],
                             ok=1.0 if score.get("success") else 0.0, status=res.get("status", "none"),
                             fam=x["family"], level=x["level"], layout=(seed, x["base_scene_id"]),
                             sit=situation(x["base_scene_id"]), added=len(x["added"])))
    return recs, runs


def finished_models(recs, man):
    expected = defaultdict(int)
    for (seed, _), x in man.items():
        expected[seed] += 2  # two requests per scene
    got = defaultdict(int)
    for r in recs:
        got[(r["model"], r["seed"], r["meth"])] += 1
    models = sorted({r["model"] for r in recs})
    done = [m for m in models if all(got[(m, s, k)] == expected[s] for s in expected for k in "REC")]
    return done, got, expected


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def layout_scores(recs, model, meth):
    base, var = defaultdict(list), defaultdict(list)
    for r in recs:
        if r["model"] == model and r["meth"] == meth:
            (base if r["fam"] == "base" else var)[r["layout"]].append(r["ok"])
    return {k: mean(v) for k, v in base.items()}, {k: mean(v) for k, v in var.items()}


def signflip(d):
    d = [x for x in d if x == x]
    obs = abs(sum(d))
    rng = random.Random(SEED)
    hits = sum(abs(sum(x if rng.random() < 0.5 else -x for x in d)) >= obs - 1e-12 for _ in range(FLIPS))
    return (hits + 1) / (FLIPS + 1)


def pooled(per_model):
    """per_model: {model: {layout: value}} -> {layout: mean over the models that have it}."""
    acc = defaultdict(list)
    for vals in per_model.values():
        for k, v in vals.items():
            acc[k].append(v)
    return {k: mean(v) for k, v in acc.items()}


def wlt(d):
    return sum(x > 0 for x in d), sum(x < 0 for x in d), sum(x == 0 for x in d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    man = manifest()
    recs, runs = records(a.runs, man)
    done, got, expected = finished_models(recs, man)
    print("Runs used (latest per model, seed and method):")
    for k in sorted(runs):
        print(f"  {k[0]:20s} seed{k[1]} {k[2]}  {runs[k][1]:4d} records  {runs[k][0]}")
    print(f"Finished models (every seed and method complete): {', '.join(done) or 'none'}")
    for m in sorted({r['model'] for r in recs}):
        if m not in done:
            print(f"  unfinished: {m} " + " ".join(f"seed{s}{k}={got[(m, s, k)]}/{expected[s]}"
                                                  for s in sorted(expected) for k in "REC"))
    bad = defaultdict(int)
    for r in recs:
        if r["status"] == "vlm_error":
            bad[(r["model"], r["meth"])] += 1
    print("Records that ended in a failed call (counted as failures):",
          ", ".join(f"{m} {k} {n}" for (m, k), n in sorted(bad.items())) or "none")
    out = {"models": done, "by_family": {}, "by_added": {}, "tests": {}}
    if not done:
        return

    # 1-3: the fixed tests, per model and pooled
    for scope in done + ["pooled"]:
        models = done if scope == "pooled" else [scope]
        S = {k: {m: layout_scores(recs, m, k) for m in models} for k in "REC"}
        base = {k: pooled({m: S[k][m][0] for m in models}) for k in "REC"}
        var = {k: pooled({m: S[k][m][1] for m in models}) for k in "REC"}
        lays = sorted(set(var["R"]) & set(base["R"]))
        print(f"\n== {scope}: {len(lays)} layouts with variants ==")
        for k in "REC":
            d = [var[k][l] - base[k][l] for l in lays]
            p = signflip(d)
            print(f"  {LABEL[k]:20s} variants minus bases: mean {100*mean(d):+6.1f} points, "
                  f"layouts up/down/level {wlt(d)}, p {p:.4f}")
            out["tests"][f"{scope} H1 {k}"] = [mean(d), p]
        vl = sorted(var["C"])
        for k in "RE":
            d = [var["C"][l] - var[k][l] for l in vl]
            p = signflip(d)
            print(f"  our method minus {LABEL[k]:20s} on variants: mean {100*mean(d):+6.1f} points, "
                  f"layouts C better/worse/level {wlt(d)}, p {p:.4f}")
            out["tests"][f"{scope} H2 C-{k}"] = [mean(d), p]
        d = [(var["C"][l] - base["C"][l]) - (var["R"][l] - base["R"][l]) for l in lays]
        p = signflip(d)
        print(f"  (our drop) minus (pointing drop): mean {100*mean(d):+6.1f} points, {wlt(d)}, p {p:.4f}")
        out["tests"][f"{scope} H3"] = [mean(d), p]

    # descriptive, pooled over finished models (record-level shares)
    def table(key, title):
        cells = defaultdict(lambda: defaultdict(list))
        for r in recs:
            if r["model"] in done:
                cells[key(r)][r["meth"]].append(r["ok"])
        print(f"\n{title} (records pooled over {len(done)} models)")
        print(f"  {'':24s} {'n':>5s} " + " ".join(f"{LABEL[k]:>20s}" for k in "REC"))
        res = {}
        for c in sorted(cells, key=str):
            n = len(cells[c]["C"])
            print(f"  {str(c):24s} {n:5d} " + " ".join(f"{100*mean(cells[c][k]):19.1f}%" for k in "REC"))
            res[str(c)] = {k: [sum(cells[c][k]), len(cells[c][k])] for k in "REC"}
        return res

    out["by_family"] = table(lambda r: f"{r['fam']}{r['level'] or ''}", "Success by family and level")
    table(lambda r: (FAMILY[r["fam"]], r["sit"]), "Success by family and situation")
    out["by_added"] = table(lambda r: r["added"], "Success by number of people added")
    for m in done:
        cells = defaultdict(lambda: defaultdict(list))
        for r in recs:
            if r["model"] == m:
                cells["base" if r["fam"] == "base" else "variant"][r["meth"]].append(r["ok"])
        print(f"  {m:20s} " + "  ".join(f"{g}: " + " ".join(f"{k} {int(sum(cells[g][k]))}/{len(cells[g][k])}"
                                                            for k in "REC") for g in ("base", "variant")))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
