"""Tour test: per model, the right tour frame, the right target and good spots, next to the same
model's view-grounded run of our method on the same 70 scenes (the fresh-scene runs). Zero calls.
    PYTHONPATH=src python results/still-scenes/tour-grounding/tour_score.py"""
import json
import re
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir() and (p / "src").is_dir())
RUNS = REPO / "runs"
MODELS = {"soclaas_dsv41": "DeepSeek V4.1", "claude_haiku55": "Claude Haiku 5.5", "gemini_fast": "Gemini 3.8 Flash"}


def latest(pattern):
    """{seed: run folder}, the latest run per seed folder for this pattern."""
    out = {}
    for p in sorted(RUNS.glob(pattern)):
        m = re.search(r"seed(5\d\d)", p.name)
        if m and (p / "results.jsonl").exists():
            out[m.group(1)] = p
    return out


def records(folders):
    out = {}
    for p in folders.values():
        for line in open(p / "results.jsonl"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[(r["scene_id"], r["episode_id"], r["trial"])] = r
    return out


def sign_p(w, l):
    n = w + l
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(w, l) + 1)) / 2 ** n)


def share(rs, get):
    vals = [get(r) for r in rs]
    vals = [v for v in vals if v is not None]
    return (sum(bool(v) for v in vals), len(vals))


for model, name in MODELS.items():
    tour = records(latest(f"*_tour_seed*_{model}"))
    view = records(latest(f"*_s27_i12_seed*_{model}_con"))
    if len(tour) < 280:
        print(f"## {name}: tour run not complete ({len(tour)} of 280 records)\n")
        continue
    keys = sorted(set(tour) & set(view))
    print(f"## {name}: {len(tour)} tour records, {len(keys)} paired with the view-grounded run")
    t = [tour[k] for k in keys]
    v = [view[k] for k in keys]
    for label, get in (("right tour frame", lambda r: r["measures"].get("tour_frame_correct")),
                       ("right target", lambda r: (r.get("score") or {}).get("target_correct")),
                       ("good spot", lambda r: (r.get("score") or {}).get("success"))):
        ok, n = share(t, get)
        line = f"- {label}: tour {ok}/{n} ({100 * ok / max(n, 1):.1f}%)"
        if label != "right tour frame":
            ok2, n2 = share(v, get)
            line += f", view {ok2}/{n2} ({100 * ok2 / max(n2, 1):.1f}%)"
        print(line)
    by_scene = defaultdict(float)
    for k in keys:
        by_scene[k[0]] += float(bool((tour[k].get("score") or {}).get("success"))) - \
                          float(bool((view[k].get("score") or {}).get("success")))
    w = sum(d > 0 for d in by_scene.values())
    l = sum(d < 0 for d in by_scene.values())
    print(f"- good spots, tour minus view by scene: better {w}, worse {l}, level {len(by_scene) - w - l}, "
          f"sign p {sign_p(w, l):.4f}")
    chance = [len(r["tour"]["reference_frames"]) / len(r["tour"]["shown"]) for r in t if r.get("tour")]
    every = sum(1 for r in t if r.get("tour") and len(r["tour"]["reference_frames"]) == len(r["tour"]["shown"]))
    print(f"- chance of a right frame by a random pick: {100 * sum(chance) / len(chance):.1f}% "
          f"(the target shows in every shown frame in {every} of {len(t)} records)")
    errors = sum(1 for r in t if (r.get("result") or {}).get("status") == "vlm_error")
    print(f"- records that ended in a failed call: {errors}\n")
