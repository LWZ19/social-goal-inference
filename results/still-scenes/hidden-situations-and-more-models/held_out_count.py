"""Good spots in the two situations held out of development (waiting at a lift, a group photo), per model and method.
Each model has 36 tries per situation and method: 2 layouts, 3 seeds, 2 requests, 3 tries. Uses the latest complete,
error-free run of each model, seed and method. Reads stored runs only and makes no model call.
    PYTHONPATH=src python results/still-scenes/hidden-situations-and-more-models/held_out_count.py"""
import collections
import glob
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
NAME = re.compile(r"_s(?:18|20_held|25_held)_seed(\d)_(.+)_(cot|fsc3?|con)$")
MODELS = {"qwen38_low": "Qwen 3.8", "dsv41": "DeepSeek V4.1", "glm53": "GLM-5.3", "claude_sonnet55_low": "Claude Sonnet 5.5",
          "claude_haiku55": "Claude Haiku 5.5", "gemini_fast": "Gemini 3.8 Flash"}
METHODS = {"cot": "pointing", "fsc": "with examples", "con": "ours"}
TRIES_PER_RUN = 24


def records(run):
    return [json.loads(line) for line in open(Path(run) / "results.jsonl") if line.strip()]


latest = {}
for pattern in ("*_s18_seed*", "*_s20_held_*", "*_s25_held_*"):
    for run in sorted(glob.glob(str(REPO / "runs" / pattern))):
        found = NAME.search(Path(run).name)
        if not found or not (Path(run) / "results.jsonl").exists():
            continue
        rows = records(run)
        if len(rows) == TRIES_PER_RUN and not any((r.get("result") or {}).get("status") == "vlm_error" for r in rows):
            latest[(found[2], int(found[1]), found[3][:3])] = rows

good = collections.Counter()
for (model, _, method), rows in latest.items():
    for r in rows:
        good[(model, r["labels"]["situation"], method)] += bool((r.get("score") or {}).get("success"))
print(f"{len(latest)} runs of {len(MODELS) * 3 * 3} expected\n")
print("| model | lift: pointing / with examples / ours | photo: pointing / with examples / ours |")
print("|---|---|---|")
for model, name in MODELS.items():
    cells = [" / ".join(str(good[(model, s, m)]) for m in METHODS) for s in ("lift_wait", "group_photo")]
    print(f"| {name} | {cells[0]} | {cells[1]} |")
for situation in ("lift_wait", "group_photo"):
    totals = {m: sum(good[(k, situation, m)] for k in MODELS) for m in METHODS}
    shown = ", ".join(f"{METHODS[m]} {n} of 216 ({100 * n / 216:.1f}%)" for m, n in totals.items())
    print(f"\n{situation}, six models: {shown}")
