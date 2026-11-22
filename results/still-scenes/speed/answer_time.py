"""Seconds per answer on the still development scenes: the median over 252 tries of the summed model-call time of each
try, for pointing and for our method, per model. Reads stored runs only and makes no model call.
    PYTHONPATH=src python results/still-scenes/speed/answer_time.py > results/still-scenes/speed/answer_time_out.txt"""
import glob
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MODELS = {"DeepSeek V4.1": ("s23", "dsv41"), "Claude Haiku 5.5": ("s23", "haiku55"),
          "Claude Sonnet 5.5": ("s23", "sonnet55_low"), "GLM-5.3": ("s23", "glm53"),
          "Gemini 3.8 Flash": ("s23", "gemini38_low"), "Qwen 3.8": ("s21", "qwen38_low")}
SETS = ("real", "staffed", "plain")


def tries(run_set, key, method):
    """The summed model-call seconds of each try, from the latest run of each of the three scene sets."""
    seconds = []
    for scene_set in SETS:
        run = sorted(glob.glob(str(REPO / "runs" / f"*_{run_set}_{scene_set}_{key}_{method}")))[-1]
        for line in open(Path(run) / "results.jsonl"):
            if line.strip():
                calls = (json.loads(line).get("result") or {}).get("calls") or []
                seconds.append(sum(call.get("latency_s") or 0.0 for call in calls))
    return seconds


print("| model | tries | pointing (s) | our method (s) | ratio |")
print("|---|---|---|---|---|")
for name, (run_set, key) in MODELS.items():
    point, ours = tries(run_set, key, "cot"), tries(run_set, key, "con")
    a, b = statistics.median(point), statistics.median(ours)
    print(f"| {name} | {len(point)} | {a:.2f} | {b:.2f} | {b / a:.2f} |")
