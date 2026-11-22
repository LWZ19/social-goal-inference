"""Scorer for the new situations: per situation family, success of pointing (R, "cot"), pointing with examples
(E, "fsc") and the constraint method (C, "con") for each model, and the layout-level exact sign-flip of C-R and C-E,
pooled over the six models (per-layout net summed over models, episodes and trials) and per model. Zero calls; reads
sgi.run output folders (config.yaml + results.jsonl). machine_use is a held-out situation.

    PYTHONPATH= .venv/bin/python e4_score.py RUN_DIR [RUN_DIR ...]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

SIX = ("soclaas_qwen38_low", "soclaas_glm53", "soclaas_dsv41", "claude_sonnet55_low", "claude_haiku55", "gemini_fast")
FAMILY = {"waiting_line": "bent line", "machine_use": "machine use"}


def signflip_p(nets):
    xs = [abs(int(x)) for x in nets if x]
    obs = abs(sum(int(x) for x in nets))
    if not xs:
        return 1.0
    counts = {0: 1}
    for x in xs:
        nxt = defaultdict(int)
        for s, c in counts.items():
            nxt[s + x] += c
            nxt[s - x] += c
        counts = nxt
    return sum(c for s, c in counts.items() if abs(s) >= obs) / 2 ** len(xs)


def load(dirs):
    rec = {}          # (model, method, scene, episode, trial) -> (situation, success)
    for d in dirs:
        d = Path(d)
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        model = cfg["config"]["stages"]["plan"]["model"]
        method = cfg["name"].rsplit("_", 1)[-1]
        if method not in ("cot", "fsc", "con"):
            raise SystemExit(f"{d}: run name {cfg['name']!r} does not end in _cot, _fsc or _con")
        for line in (d / "results.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rec[(model, method, r["scene_id"], r["episode_id"], r["trial"])] = (
                    r["labels"]["situation"], bool(r["score"]["success"]), r["result"]["status"])
    return rec


def main(dirs):
    rec = load(dirs)
    models = sorted({k[0] for k in rec}, key=lambda m: SIX.index(m) if m in SIX else 99)
    sits = sorted({v[0] for v in rec.values()})
    print("# E4 (I13_FROZEN.md): success by situation, model and method; layout-level sign-flip\n")
    for sit in sits:
        print(f"\n## {FAMILY.get(sit, sit)} ({sit})\n")
        print("| model | records per method | R | E | C | C-R layout W/L/T, net, p | C-E layout W/L/T, net, p |")
        print("|---|---|---|---|---|---|---|")
        pooled = {"R": defaultdict(int), "E": defaultdict(int)}
        worst = {"R": defaultdict(int), "E": defaultdict(int)}
        left_out = []
        for m in models:
            keys = {k[2:] for k in rec if k[0] == m and rec[k][0] == sit}
            full = [k for k in keys if all((m, meth, *k) in rec for meth in ("cot", "fsc", "con"))]
            # Records still in vlm_error after the last pass, in any method, are left out
            # of every method for that model and counted; the worst case counts each as a C loss
            excl = [k for k in full if any(rec[(m, meth, *k)][2] == "vlm_error" for meth in ("cot", "fsc", "con"))]
            both = [k for k in full if k not in set(excl)]
            if excl:
                left_out.append(f"{m} {len(excl)} of {len(full)}{' PARTIAL (> 10%)' if len(excl) > 0.10 * len(full) else ''}")
            if m in SIX:
                for k in excl:
                    worst["R"][k[0]] -= 1
                    worst["E"][k[0]] -= 1
            ok = {meth: sum(rec[(m, meth, *k)][1] for k in both) for meth in ("cot", "fsc", "con")}
            cells = []
            for ref, meth in (("R", "cot"), ("E", "fsc")):
                net = defaultdict(int)
                for k in both:
                    d = int(rec[(m, "con", *k)][1]) - int(rec[(m, meth, *k)][1])
                    net[k[0]] += d
                    if m in SIX:
                        pooled[ref][k[0]] += d
                v = list(net.values())
                cells.append(f"{sum(x > 0 for x in v)}/{sum(x < 0 for x in v)}/{sum(x == 0 for x in v)}, {sum(v):+d}, "
                             f"{signflip_p(v):.4f}")
            print(f"| {m} | {len(both)} | {ok['cot']} | {ok['fsc']} | {ok['con']} | {cells[0]} | {cells[1]} |")
        lines = []
        if left_out:
            lines.append(f"- left out for vlm_error (amendment 1): {'; '.join(left_out)}")
            for ref in ("R", "E"):
                agg = defaultdict(int, pooled[ref])
                for lay, d in worst[ref].items():
                    agg[lay] += d
                v = list(agg.values())
                lines.append(f"- worst case (each left-out record a C loss): C vs {ref} net {sum(v):+d}, p {signflip_p(v):.4f}: "
                             f"{sum(v) > 0 and signflip_p(v) <= 0.05}")
        for ref in ("R", "E"):
            v = list(pooled[ref].values())
            net, p = sum(v), signflip_p(v)
            lines.append(f"- BAR (pooled over the six models): C beats {'pointing' if ref == 'R' else 'pointing with examples'}"
                         f" on {FAMILY.get(sit, sit)}: net {net:+d}, layouts {sum(x > 0 for x in v)}/{sum(x < 0 for x in v)}/"
                         f"{sum(x == 0 for x in v)}, p {p:.4f}: {net > 0 and p <= 0.05}")
        missing = [m for m in SIX if m not in models]
        print("\n" + "\n".join(lines) + (f"\n- models missing: {missing}" if missing else ""))


if __name__ == "__main__":
    main(sys.argv[1:])
