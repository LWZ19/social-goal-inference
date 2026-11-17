"""Fresh seeds 10-19: the judged comparisons of CT2_FROZEN.md, CT2m_FROZEN.md, CTs_FROZEN.md and E1D_FROZEN.md,
under the three scorings (SCORING.md as frozen; CT2_FROZEN's first-within-0.6 m; SCORING_STRICT.md).

    python3 summarise_fresh.py results/data/moving/runs/sonnet55_low [results/data/moving/runs/dsv41 ...]   (from the repo root)

Only episodes with seed >= 10 are read; seeds 0-2 are development data for these arms.
"""
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ARMS = ["C1", "R1", "Rm", "CT", "CT2", "CT2m", "CT2-vac", "CT2-rat", "CT2-nora", "CT2-norv", "CT2-none", "CTs", "CTd", "CTe"]
PAIRS = [("CT2", "CT", "primary (CT2_FROZEN)"), ("CT2", "R1", ""), ("CT2m", "CT2", "latency (CT2m_FROZEN)"),
         ("Rm", "R1", "latency, R (CT2m_FROZEN)"), ("CT2m", "Rm", "headline across"), ("CT2", "CT2-vac", "ablation"),
         ("CT2", "CT2-rat", "ablation"), ("CTs", "CT", "CTs_FROZEN"), ("CTd", "CT2", "tracking (E1D_FROZEN)"),
         ("CTe", "CTd", "detector people at answer (E1C_FROZEN)"),
         ("CT2", "CT2-nora", "post hoc ablation: readmit (CT2_NORA_POSTHOC)"), ("CT2-nora", "CT", ""),
         ("CT2", "CT2-norv", "post hoc: readmit + vacate (CT2_NORV_POSTHOC)"), ("CT2-norv", "CT", ""),
         ("CT2-none", "CT", "sanity: every part off (CT2_NONE_RAT_POSTHOC N1)"), ("CT2-rat", "CT", "R2"),
         ("CT2", "CT2-rat", "ratchet, all cells")]
BREAK = {"cafe_line6_stepout", "cafe_line_bent_moving"}


def load(roots):
    eps = []
    for root in roots:
        for f in Path(root).glob("*/seed*/*/episode.json"):
            d = json.loads(f.read_text())
            if d["seed"] >= 10:
                d["_path"] = str(f.parent)
                eps.append(d)
    return eps


def in_line(x, scoring):
    ok = bool(x["in_line"])
    if scoring == "frozen" or not ok:
        return ok
    if x["ahead"] == 0:
        return (x.get("to_slot0_m") or 99) <= 0.6
    return scoring == "first06" or x["side"] <= 0.6


def measures(e):
    s = e["samples"]
    out = {f"share_{k}": sum(in_line(x, k) for x in s) / len(s) for k in ("frozen", "first06", "strict")}
    out["end_strict"] = in_line(s[-1], "strict")
    out["cut"] = sum(x["cut_in"] for x in s)
    first = next((x["t"] - s[0]["t"] for x in s if in_line(x, "strict")), None)
    out["first_strict"] = first
    events = back = 0                                  # per service event with someone ahead (IDEAS_social_eval B2)
    for i in range(1, len(s)):
        a, b = s[i - 1], s[i]
        if a["ahead"] >= 1 and b["ahead"] < a["ahead"] and b["joined"]:
            events += 1
            window = s[i:i + 11]
            back += any(in_line(x, "strict") for x in window) and not any(x["cut_in"] for x in window)
    out["events"], out["back10"] = events, back
    return out


def sign_p(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def table(eps, title):
    print(f"\n## {title}\n")
    print("| Arm | Episodes | Share (frozen) | Share (first 0.6 m) | Share (strict) | End in line (strict) | Median first in line, strict (s) | Back in line within 10 s of a service | Cut-in samples |")
    print("|---|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        m = [measures(e) for e in eps if e["arm"] == arm]
        if not m:
            continue
        firsts = [x["first_strict"] for x in m if x["first_strict"] is not None]
        print(f"| {arm} | {len(m)} | {statistics.mean(x['share_frozen'] for x in m):.2f} | "
              f"{statistics.mean(x['share_first06'] for x in m):.2f} | {statistics.mean(x['share_strict'] for x in m):.2f} | "
              f"{sum(x['end_strict'] for x in m)} | {statistics.median(firsts) if firsts else '-'} | "
              f"{sum(x['back10'] for x in m)}/{sum(x['events'] for x in m)} | {sum(x['cut'] for x in m)} |")


def paired(eps, title):
    by = defaultdict(dict)
    for e in eps:
        by[(e["label"], e["scenario"], e["seed"])][e["arm"]] = measures(e)
    print(f"\n## {title}: paired by (model, scenario, seed); share in line under the strict scoring\n")
    print("| Comparison | Which | Pairs | Wins / losses / ties | Mean difference | Sign p | Same, frozen scoring (W/L/T) |")
    print("|---|---|---|---|---|---|---|")
    for a, b, note in PAIRS:
        pairs = [(v[a], v[b]) for v in by.values() if a in v and b in v]
        if not pairs:
            continue
        def wlt(key):
            w = sum(x[key] > y[key] + 1e-9 for x, y in pairs)
            l = sum(x[key] < y[key] - 1e-9 for x, y in pairs)
            return w, l, len(pairs) - w - l
        w, l, t = wlt("share_strict")
        fw, fl, ft = wlt("share_frozen")
        diff = statistics.mean(x["share_strict"] - y["share_strict"] for x, y in pairs)
        print(f"| {a} vs {b} | {note} | {len(pairs)} | {w} / {l} / {t} | {diff:+.3f} | {sign_p(w, l):.3f} | {fw} / {fl} / {ft} |")


def tracking(eps):
    rows = [e for e in eps if e["arm"] == "CTd" and e.get("det")]
    if not rows:
        return
    print("\n## CTd: people tracked from the robot's own camera (E1D_FROZEN.md)\n")
    print("| Model | Episodes | Frames | Errors | Median ms | Live tracks matched to a true person (0.5 m) | Id switches / live track-frames | Stale share of tracks | First-match rate |")
    print("|---|---|---|---|---|---|---|---|---|")
    for label in sorted({e["label"] for e in rows}):
        sub = [e for e in rows if e["label"] == label]
        frames = [f for e in sub for f in e["det"]]
        good = [f for f in frames if "error" not in f]
        live = matched = switches = stale = total = 0
        for e in sub:
            last_owner = {}
            for f in e["det"]:
                if "error" in f:
                    continue
                truth = f["truth"]
                for tid, x, y, is_stale in f["tracks"]:
                    total += 1
                    if is_stale:
                        stale += 1
                        continue
                    live += 1
                    near = min(((math.hypot(x - tx, y - ty), i) for i, tx, ty, _ in truth), default=(99, None))
                    if near[0] <= 0.5:
                        matched += 1
                        if tid in last_owner and last_owner[tid] != near[1]:
                            switches += 1
                        last_owner[tid] = near[1]
        mr = [e["summary"].get("match_rate") for e in sub if e["summary"].get("match_rate") is not None]
        ms = statistics.median(f["ms"] for f in good) if good else 0
        print(f"| {label} | {len(sub)} | {len(frames)} | {len(frames) - len(good)} | {ms:.0f} | "
              f"{matched}/{live} = {matched / max(live, 1):.2f} | {switches}/{live} | {stale / max(total, 1):.2f} | "
              f"{statistics.mean(mr) if mr else '-'} |")


def main():
    eps = load(sys.argv[1:])
    print(f"# Fresh seeds: {len(eps)} episodes; models {sorted({e['label'] for e in eps})}")
    main_eps = [e for e in eps if e["scenario"] not in BREAK]
    for label in sorted({e["label"] for e in main_eps}):
        sub = [e for e in main_eps if e["label"] == label]
        table(sub, f"{label}: evaluation scenarios, seeds {sorted({e['seed'] for e in sub})}")
        paired(sub, label)
    tracking(main_eps)
    for b in sorted(BREAK):
        sub = [e for e in eps if e["scenario"] == b]
        if sub:
            table(sub, f"Break-it case {b}")


if __name__ == "__main__":
    main()
