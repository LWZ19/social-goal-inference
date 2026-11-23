"""Finding the target: share of records with score.target_correct, per model, on the main
still-scene comparison (development scenes, current prompts: Qwen s21, the other five s23; real, staffed and plain;
252 records per method) and on the fresh scenes (s27, 280 records per method), split by request type (grounded,
high-level) and by situation. Zero model calls; reads stored runs only.

The target comes from the plan call, which the three methods share (E and C reuse R's plan answer from the cache), so
the share is counted on each method separately and the script reports whether the three agree record by record.
Records in vlm_error (no plan answer) count as not found, as the scorer has it; their number is printed.

    PYTHONPATH=src .venv/bin/python results/still-scenes/target-found/target_found.py > results/still-scenes/target-found/target_found_out.txt
"""
from __future__ import annotations

import glob
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEV = {"Qwen 3.8": ("s21", "qwen38_low"), "GLM-5.3": ("s23", "glm53"), "DeepSeek V4.1": ("s23", "dsv41"),
       "Claude Haiku 5.5": ("s23", "haiku55"), "Claude Sonnet 5.5": ("s23", "sonnet55_low"),
       "Gemini 3.8": ("s23", "gemini38_low")}
FRESH = {"Qwen 3.8": "soclaas_qwen38_low", "GLM-5.3": "soclaas_glm53", "DeepSeek V4.1": "soclaas_dsv41",
         "Claude Haiku 5.5": "claude_haiku55", "Claude Sonnet 5.5": "claude_sonnet55_low", "Gemini 3.8": "gemini_fast"}
METHODS = ("cot", "fsc", "con")


def records(folders):
    out = {}
    for d in folders:
        for line in open(Path(d) / "results.jsonl"):
            if line.strip():
                r = json.loads(line)
                out[(r["scene_id"], r["episode_id"], r["trial"])] = r
    return out


def found(r):
    return bool(r.get("score") and r["score"].get("target_correct"))


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.1f}%)" if b else "-"


def report(title, by_model):
    print(f"## {title}\n")
    print("| model | all | grounded requests | high-level requests | nobody near | waiting line | conversation group |"
          " vlm_error records | R/E/C agree |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, per_method in by_model.items():
        recs = per_method["cot"]
        c, n = Counter(), Counter()
        for r in recs.values():
            keys = ("all", r["labels"]["request_type"], r["labels"]["situation"])
            for k in keys:
                n[k] += 1
                c[k] += found(r)
        errs = sum(r["result"].get("status") == "vlm_error" for r in recs.values())
        agree = sum(all(k in per_method[m] and found(per_method[m][k]) == found(r) for m in METHODS)
                    for k, r in recs.items())
        print(f"| {name} | {pct(c['all'], n['all'])} | {pct(c['grounded'], n['grounded'])} | "
              f"{pct(c['high_level'], n['high_level'])} | {pct(c['no_one'], n['no_one'])} | "
              f"{pct(c['waiting_line'], n['waiting_line'])} | {pct(c['conversation_group'], n['conversation_group'])} | "
              f"{errs} | {agree}/{len(recs)} |")
    tot = Counter()
    for per_method in by_model.values():
        for r in per_method["cot"].values():
            tot["n"] += 1
            tot["c"] += found(r)
    print(f"\nPooled over the six models (counted on the pointing runs): {pct(tot['c'], tot['n'])}\n")


def main():
    dev = {}
    for name, (stage, short) in DEV.items():
        dev[name] = {}
        for m in METHODS:
            folders = [sorted(glob.glob(str(REPO / f"runs/*_{stage}_{v}_{short}_{m}")))[0]
                       for v in ("real", "staffed", "plain")]
            dev[name][m] = {(f.split("_" + stage + "_")[1].split("_")[0], *k): r
                            for f in folders for k, r in records([f]).items()}
    report("Main comparison (development scenes, 252 records per method; counted on the pointing runs)", dev)
    fresh = {}
    for name, short in FRESH.items():
        fresh[name] = {m: records(sorted(glob.glob(str(REPO / f"runs/*_s27_i12_seed*_{short}_{m}")))) for m in METHODS}
    report("Fresh scenes (70 layouts, 280 records per method; counted on the pointing runs)", fresh)


if __name__ == "__main__":
    main()
