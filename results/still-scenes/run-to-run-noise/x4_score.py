"""Extra X4 (X4_FROZEN.md): run-to-run noise for GLM-5.3 and DeepSeek V4.1 on the development still scenes. Per model and
method (R cot, E fsc, C con): the batch totals of successes (of 252) for s23 and the fresh batches noise1 and noise2,
their mean and SD, the paired differences C-R and C-E within each batch, and the record-level flip rate between batches.
Reported, no bar. Zero calls; reads results.jsonl of the runs named below.

    .venv/bin/python results/still-scenes/run-to-run-noise/x4_score.py
"""
import json
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import sys

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402  (results/_lib/paths.py)

RUNS = paths.RUNS
MODELS = {"GLM-5.3": ("glm53", "soclaas_glm53"), "DeepSeek V4.1": ("dsv41", "soclaas_dsv41")}
VARIANTS, METHODS = ("plain", "staffed", "real"), ("cot", "fsc", "con")


def run_dir(pattern):
    hits = sorted(RUNS.glob(pattern))
    return hits[-1] if hits else None


def records(d):
    out = {}
    for line in (d / "results.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[(r["scene_id"], r["episode_id"], r["trial"])] = bool(r["score"]["success"])
    return out


def main():
    print("# X4: run-to-run noise on the development still scenes (reported, no bar)\n")
    for name, (short, entry) in MODELS.items():
        batches = {"s23": lambda v, m: f"*_s23_{v}_{short}_{m}",
                   "noise1": lambda v, m: f"*_s28_noise1_{v}_{entry}_{m}",
                   "noise2": lambda v, m: f"*_s28_noise2_{v}_{entry}_{m}"}
        rec = defaultdict(dict)                    # batch -> method -> {key: success}
        for b, pat in batches.items():
            for m in METHODS:
                for v in VARIANTS:
                    d = run_dir(pat(v, m))
                    if d is not None:
                        rec[b][m] = {**rec[b].get(m, {}), **records(d)}
        print(f"## {name}\n")
        print("| batch | R | E | C | C-R | C-E | records per method |\n|---|---|---|---|---|---|---|")
        tot = defaultdict(dict)
        for b in batches:
            if not all(m in rec[b] for m in METHODS):
                print(f"| {b} | not run | | | | | |")
                continue
            keys = set.intersection(*(set(rec[b][m]) for m in METHODS))
            t = {m: sum(rec[b][m][k] for k in keys) for m in METHODS}
            tot[b] = t
            print(f"| {b} | {t['cot']} | {t['fsc']} | {t['con']} | {t['con'] - t['cot']:+d} | {t['con'] - t['fsc']:+d} | {len(keys)} |")
        done = [b for b in batches if b in tot]
        if len(done) >= 2:
            for m, label in zip(METHODS, ("R", "E", "C")):
                v = [tot[b][m] for b in done]
                print(f"- {label}: batch totals {v}, mean {statistics.mean(v):.1f}, SD {statistics.stdev(v):.2f} (df {len(v) - 1})")
            for b1, b2 in combinations(done, 2):
                flips = []
                for m, label in zip(METHODS, ("R", "E", "C")):
                    keys = set(rec[b1][m]) & set(rec[b2][m])
                    rw = sum(rec[b1][m][k] and not rec[b2][m][k] for k in keys)
                    wr = sum(not rec[b1][m][k] and rec[b2][m][k] for k in keys)
                    flips.append(f"{label} {rw}/{wr} of {len(keys)}")
                print(f"- flips {b1} -> {b2} (right->wrong / wrong->right): " + "; ".join(flips))
        print()


if __name__ == "__main__":
    main()
