"""The still-scene headline on fresh seeds of the 14 development rows. Per model, success of
pointing (R, "cot"), pointing with examples (E, "fsc") and the constraint method (C, "con"), and the exact two-sided
sign-flip over layouts (a layout is one scene) of C-R, C-E and E-R, per model and pooled over models (per-layout net
summed over models, episodes and trials). Also success by situation. Zero calls; reads sgi.run output folders.

    PYTHONPATH= .venv/bin/python i12_score.py RUN_DIR [RUN_DIR ...]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

FREE = ("soclaas_qwen38_low", "soclaas_glm53", "soclaas_dsv41")
SIX = FREE + ("claude_haiku55", "gemini_fast", "claude_sonnet55_low")
PAIRS = (("C-R", "con", "cot"), ("C-E", "con", "fsc"), ("E-R", "fsc", "cot"))


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
    rec = {}
    for d in dirs:
        d = Path(d)
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        model, method = cfg["config"]["stages"]["plan"]["model"], cfg["name"].rsplit("_", 1)[-1]
        if method not in ("cot", "fsc", "con"):
            raise SystemExit(f"{d}: run name {cfg['name']!r} does not end in _cot, _fsc or _con")
        for line in (d / "results.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                key = (model, method, r["scene_id"], r["episode_id"], r["trial"])
                if key in rec:
                    raise SystemExit(f"duplicate record {key}: two runs cover the same scene for one model and method")
                rec[key] = (r["labels"]["situation"], bool(r["score"]["success"]), r["result"]["status"])
    return rec


def fmt(v):
    return f"{sum(x > 0 for x in v)}/{sum(x < 0 for x in v)}/{sum(x == 0 for x in v)}, {sum(v):+d}, p {signflip_p(v):.4f}"


def main(dirs):
    rec = load(dirs)
    models = sorted({k[0] for k in rec}, key=lambda m: SIX.index(m) if m in SIX else 99)
    print("# Item 12: R, E and C on fresh seeds of the development rows; layout-level sign-flip\n")
    print("| model | records per method | layouts | R | E | C | C-R W/L/T, net, p | C-E W/L/T, net, p | E-R W/L/T, net, p |")
    print("|---|---|---|---|---|---|---|---|---|")
    pooled = {name: defaultdict(int) for name, _, _ in PAIRS}
    pooled_free = {name: defaultdict(int) for name, _, _ in PAIRS}
    verdicts, by_sit = [], defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for m in models:
        keys = {k[2:] for k in rec if k[0] == m}
        full = sorted(k for k in keys if all((m, meth, *k) in rec for meth in ("cot", "fsc", "con")))
        # A record still in vlm_error after the last re-run pass, in any method, is left
        # out of every method for that model (the comparison stays paired) and counted
        excluded = [k for k in full if any(rec[(m, meth, *k)][2] == "vlm_error" for meth in ("cot", "fsc", "con"))]
        both = [k for k in full if k not in set(excluded)]
        ok = {meth: sum(rec[(m, meth, *k)][1] for k in both) for meth in ("cot", "fsc", "con")}
        for k in both:
            for meth in ("cot", "fsc", "con"):
                sit, s, _ = rec[(m, meth, *k)]
                by_sit[(m, sit)][meth][0] += s
                by_sit[(m, sit)][meth][1] += 1
        cells, res = [], {}
        for name, a, b in PAIRS:
            net = defaultdict(int)
            for k in both:
                d = int(rec[(m, a, *k)][1]) - int(rec[(m, b, *k)][1])
                net[k[0]] += d
                if m in SIX:
                    pooled[name][k[0]] += d
                if m in FREE:
                    pooled_free[name][k[0]] += d
            v = list(net.values())
            res[name] = (sum(v), signflip_p(v))
            cells.append(fmt(v))
        print(f"| {m} | {len(both)} | {len({k[0] for k in both})} | {ok['cot']} | {ok['fsc']} | {ok['con']} | "
              f"{' | '.join(cells)} |")
        line = (f"- {m}: BAR C beats R {res['C-R'][0] > 0 and res['C-R'][1] <= 0.05}; "
                f"BAR C beats E {res['C-E'][0] > 0 and res['C-E'][1] <= 0.05}")
        if excluded:
            worst = []
            for name, a, b in PAIRS[:2]:              # worst case for C: each left-out record a C loss
                net = defaultdict(int)
                for k in both:
                    net[k[0]] += int(rec[(m, a, *k)][1]) - int(rec[(m, b, *k)][1])
                for k in excluded:
                    net[k[0]] -= 1
                v = list(net.values())
                worst.append(f"{name} worst case net {sum(v):+d}, p {signflip_p(v):.4f}: {sum(v) > 0 and signflip_p(v) <= 0.05}")
            share = len(excluded) / max(len(full), 1)
            line += (f"; {len(excluded)} records left out for vlm_error ({share:.1%}"
                     f"{', PARTIAL (> 10%)' if share > 0.10 else ''}); " + "; ".join(worst))
        verdicts.append(line)
    print("\nPer model (I12 bars: net > 0 and p <= 0.05):\n" + "\n".join(verdicts))
    for label, pool in (("the six models", pooled), ("the three free models", pooled_free)):
        print(f"\nPooled over {label}:")
        for name, _, _ in PAIRS:
            v = list(pool[name].values())
            print(f"- {name}: {fmt(v)}" + (f": BAR {sum(v) > 0 and signflip_p(v) <= 0.05}" if name != "E-R" else " (reported)"))
    print("\n## Success by situation (R / E / C)\n")
    sits = sorted({s for _, s in by_sit})
    print("| model | " + " | ".join(sits) + " |\n|---|" + "---|" * len(sits))
    for m in models:
        row = []
        for s in sits:
            t = by_sit.get((m, s))
            row.append("-" if not t else f"{t['cot'][0]}/{t['fsc'][0]}/{t['con'][0]} of {t['cot'][1]}")
        print(f"| {m} | " + " | ".join(row) + " |")
    missing = [m for m in SIX if m not in models]
    if missing:
        print(f"\nmodels not (yet) in the runs given: {missing}")


if __name__ == "__main__":
    main(sys.argv[1:])
