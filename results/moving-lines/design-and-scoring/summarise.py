"""Summarise moving-people episodes: per arm, overall and by scenario, and CT paired against each other arm.

    python3 summarise.py runs/dsv41 [runs/other ...]
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ARMS = ["R1", "Rq", "Rq0", "RT", "RTw", "C1", "Cq", "CTv", "CT0", "CT", "CTo", "CTt"]
DEV = {"cafe_busy"}                                   # development data (DEVIATIONS.md): reported apart
BREAK = {"cafe_line2_fast", "cafe_line6_stepout", "cafe_line_bent_moving"}   # break-it cases (QUESTIONS.md Q7)


def load(roots):
    eps = []
    for root in roots:
        for f in Path(root).glob("*/seed*/*/episode.json"):
            d = json.loads(f.read_text())
            d["_path"] = str(f.parent)
            eps.append(d)
    return eps


def row(eps):
    s = [e["summary"] for e in eps]
    if not s:
        return None
    firsts = [x["first_in_line_s"] for x in s if x.get("first_in_line_s") is not None]
    return {"n": len(s), "end": sum(bool(x["in_line_at_end"]) for x in s),
            "share": statistics.mean(x["share_in_line"] for x in s),
            "first_med": statistics.median(firsts) if firsts else None, "never": len(s) - len(firsts),
            "cut": sum(x["cut_in_samples"] > 0 for x in s), "lost": sum(x["lost_place"] for x in s),
            "calls": statistics.mean(x["calls"] for x in s),
            "lat": statistics.median([l for x in s for l in x["latencies_s"]] or [0])}


def show(title, eps):
    print(f"\n## {title}\n")
    print("| Arm | Episodes | In line at end | Mean share in line | Median first in line (s) | Never | Any cut-in | Lost place | Calls / ep | Median latency (s) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        r = row([e for e in eps if e["arm"] == arm])
        if r:
            first = "-" if r["first_med"] is None else f"{r['first_med']:.0f}"
            print(f"| {arm} | {r['n']} | {r['end']} | {r['share']:.2f} | {first} | {r['never']} | {r['cut']} | "
                  f"{r['lost']} | {r['calls']:.1f} | {r['lat']:.1f} |")


def paired(eps):
    by = defaultdict(dict)
    for e in eps:
        by[(e["label"], e["scenario"], e["seed"])][e["arm"]] = e["summary"]
    print("\n## CT against each arm, paired by (model, scenario, seed)\n")
    print("| Other arm | Pairs | CT share wins / losses / ties | Mean share difference | End: CT only / other only |")
    print("|---|---|---|---|---|")
    for other in ["R1", "Rq", "Rq0", "RT", "RTw", "C1", "Cq", "CTv", "CT0", "CTo", "CTt"]:
        pairs = [(v["CT"], v[other]) for v in by.values() if "CT" in v and other in v]
        if not pairs:
            continue
        hi = sum(a["share_in_line"] > b["share_in_line"] + 1e-9 for a, b in pairs)
        lo = sum(a["share_in_line"] < b["share_in_line"] - 1e-9 for a, b in pairs)
        diff = statistics.mean(a["share_in_line"] - b["share_in_line"] for a, b in pairs)
        only_ct = sum(bool(a["in_line_at_end"]) and not b["in_line_at_end"] for a, b in pairs)
        only_o = sum(bool(b["in_line_at_end"]) and not a["in_line_at_end"] for a, b in pairs)
        print(f"| {other} | {len(pairs)} | {hi} / {lo} / {len(pairs) - hi - lo} | {diff:+.2f} | {only_ct} / {only_o} |")


def per_scenario(eps):
    print("\n## Share in line by scenario (mean over seeds)\n")
    scen = sorted({e["scenario"] for e in eps})
    print("| Scenario | " + " | ".join(ARMS) + " |")
    print("|---|" + "---|" * len(ARMS))
    for s in scen:
        cells = []
        for arm in ARMS:
            r = row([e for e in eps if e["scenario"] == s and e["arm"] == arm])
            cells.append("-" if r is None else f"{r['share']:.2f} ({r['end']}/{r['n']})")
        print(f"| {s}{' (dev)' if s in DEV else ''} | " + " | ".join(cells) + " |")


def first_in_line(eps):
    """Q5: episodes in which the robot was ever first (nobody ahead), and how they went from then on."""
    print("\n## Once the robot was first in line (Q5)\n")
    print("| Arm | Episodes ever first | In line while first (share of those samples) | In line at end |")
    print("|---|---|---|---|")
    for arm in ARMS:
        sub = [e for e in eps if e["arm"] == arm]
        first = [e for e in sub if any(x["ahead"] == 0 and x["members"] > 0 or x["ahead"] == 0 for x in e["samples"]
                                       if x["t"] > e["samples"][0]["t"])]
        if not sub:
            continue
        lone = [x for e in first for x in e["samples"] if x["ahead"] == 0]
        share = sum(x["in_line"] for x in lone) / len(lone) if lone else 0.0
        print(f"| {arm} | {len(first)} of {len(sub)} | {share:.2f} | {sum(bool(e['summary']['in_line_at_end']) for e in first)} |")


def staleness(eps):
    print("\n## Staleness of answers: how far the line's people moved between the frame sent and the answer\n")
    print("| Arm | Answers | Median mean move (m) | Median max move (m) | Median latency (s) |")
    print("|---|---|---|---|---|")
    for arm in ARMS:
        ans = [a for e in eps if e["arm"] == arm for a in e["answers"] if a.get("line_moved_mean_m") is not None]
        if ans:
            med = lambda k: statistics.median(a[k] for a in ans)
            print(f"| {arm} | {len(ans)} | {med('line_moved_mean_m'):.2f} | {med('line_moved_max_m'):.2f} | {med('latency_s'):.1f} |")


def matches(eps):
    rates = defaultdict(list)
    for e in eps:
        if e["arm"] in ("CT", "CTv") and e["summary"].get("match_rate") is not None:
            rates[e["label"]].append(e["summary"]["match_rate"])
    if rates:
        print("\n## First-match rate of the model's numbered people to tracks (CT, CTv)\n")
        for label, r in sorted(rates.items()):
            print(f"- {label}: mean {statistics.mean(r):.2f} over {len(r)} episodes, min {min(r):.2f}")


REPO = Path(".")


def _slot0(scenario: str):
    import yaml
    sc = yaml.safe_load((REPO / "sim" / "scenarios" / f"{scenario}.yaml").read_text())
    (q,) = sc["queues"].values()
    s0 = q["slots"][0]
    return (s0["x"], s0["y"]), tuple(s0.get("face") or (s0["x"], s0["y"] + 1.0))


def post_hoc(e) -> dict:
    """Q9 measures (post hoc, labelled): reached the front in turn; keep-up after services; share in line while
    someone is ahead."""
    import math
    (x0, y0), (fx, fy) = _slot0(e["scenario"])
    samples, t0 = e["samples"], e["samples"][0]["t"]
    reached, cut = None, False
    for x in samples:
        cut = cut or x["cut_in"]
        rx, ry, yaw = x["robot"]
        off = abs(math.degrees((yaw - math.atan2(fy - ry, fx - rx) + math.pi) % (2 * math.pi) - math.pi))
        if not cut and math.hypot(rx - x0, ry - y0) <= 0.6 and off <= 45.0:
            reached = x["t"] - t0
            break
    events = kept = 0
    for i in range(1, len(samples)):
        a, b = samples[i - 1], samples[i]
        if b["joined"] and a["ahead"] >= 1 and b["ahead"] < a["ahead"]:
            events += 1
            later = [x["along"] for x in samples[i:i + 9]]
            kept += min(later) <= b["along"] - 0.35
    ahead = [x for x in samples if x["ahead"] >= 1]
    return {"reached": reached, "events": events, "kept": kept,
            "share_ahead": sum(x["in_line"] for x in ahead) / len(ahead) if ahead else None}


def decomposition(e) -> dict:
    """Q10 (post hoc): in line at the first settle (the first sample the robot has arrived somewhere after the
    request), in line at the end, and keep-up over service events with the robot in line just before."""
    samples = e["samples"]
    settle = next((x for x in samples if x["state"] == "arrived"), None)
    events = kept = 0
    for i in range(1, len(samples)):
        a, b = samples[i - 1], samples[i]
        if a["in_line"] and a["ahead"] >= 1 and b["ahead"] < a["ahead"]:
            events += 1
            kept += min(x["along"] for x in samples[i:i + 9]) <= b["along"] - 0.35
    usable = None
    answer = Path(e["_path"]) / "001" / "answer.json"
    if e["arm"].startswith("C") and answer.exists():
        keys = ((json.loads(answer.read_text()).get("inference") or {}).get("solver") or {}).get("keys") or {}
        usable = keys.get("turn_m") is not None and keys["turn_m"] <= 0.3
    return {"settled": settle is not None, "first_ok": bool(settle and settle["in_line"]),
            "end_ok": bool(samples[-1]["in_line"]), "events": events, "kept": kept, "usable": usable}


def show_decomposition(title, eps):
    print(f"\n## {title}: initial correctness x maintenance (Q10; post hoc)\n")
    print("| Arm | Episodes | Settled | In line at first settle | ...and at the end | Keep-up from in line | First answer has a place in turn |")
    print("|---|---|---|---|---|---|---|")
    for arm in ARMS:
        sub = [decomposition(e) for e in eps if e["arm"] == arm]
        if not sub:
            continue
        first = [x for x in sub if x["first_ok"]]
        us = [x["usable"] for x in sub if x["usable"] is not None]
        print(f"| {arm} | {len(sub)} | {sum(x['settled'] for x in sub)} | {len(first)} | "
              f"{sum(x['end_ok'] for x in first)} of {len(first)} | {sum(x['kept'] for x in sub)}/{sum(x['events'] for x in sub)} | "
              f"{(str(sum(us)) + ' of ' + str(len(us))) if us else '-'} |")


def _side(pts, x, y):
    import math
    best = 1e9
    for i, ((ax, ay), (bx, by)) in enumerate(zip(pts, pts[1:])):
        dx, dy = bx - ax, by - ay
        t = ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)
        t = min(max(t, -1e9 if i == 0 else 0.0), 1e9 if i == len(pts) - 2 else 1.0)
        best = min(best, math.hypot(x - (ax + t * dx), y - (ay + t * dy)))
    return best


def goal_left_line(e) -> bool:
    """Q12: the robot's goal, once within 1.0 m of the line's slots, later more than 1.0 m from them."""
    import yaml
    sc = yaml.safe_load((REPO / "sim" / "scenarios" / f"{e['scenario']}.yaml").read_text())
    (q,) = sc["queues"].values()
    pts = [(s["x"], s["y"]) for s in q["slots"]]
    on = False
    for x in e["samples"]:
        if x["goal"] is None:
            continue
        if _side(pts, *x["goal"]) <= 1.0:
            on = True
        elif on:
            return True
    return False


def show_post_hoc(title, eps):
    print(f"\n## {title}: post hoc measures (Q9; not frozen)\n")
    print("| Arm | Episodes | Reached the front in turn | Median time to front (s) | Keep-up after services | Share in line while someone is ahead | Goal left the line |")
    print("|---|---|---|---|---|---|---|")
    for arm in ARMS:
        sub = [post_hoc(e) for e in eps if e["arm"] == arm]
        if not sub:
            continue
        times = [x["reached"] for x in sub if x["reached"] is not None]
        ev, kp = sum(x["events"] for x in sub), sum(x["kept"] for x in sub)
        sh = [x["share_ahead"] for x in sub if x["share_ahead"] is not None]
        print(f"| {arm} | {len(sub)} | {len(times)} | {statistics.median(times):.0f} | " if times else f"| {arm} | {len(sub)} | 0 | - | ", end="")
        left = sum(goal_left_line(e) for e in eps if e["arm"] == arm)
        print(f"{kp}/{ev} | {statistics.mean(sh):.2f} | {left} |" if sh else f"{kp}/{ev} | - | {left} |")


def main():
    eps = load(sys.argv[1:])
    labels = sorted({e["label"] for e in eps})
    print(f"# Moving people: {len(eps)} episodes, models {labels}")
    test = [e for e in eps if e["scenario"] not in DEV | BREAK]
    show("Evaluation scenarios (all but cafe_busy and the break-it cases)", test)
    paired(test)
    per_scenario(eps)
    first_in_line(test)
    staleness(test)
    matches(test)
    show_post_hoc("Evaluation scenarios", test)
    show_decomposition("Evaluation scenarios", test)
    show("Development scenario cafe_busy", [e for e in eps if e["scenario"] in DEV])
    for b in sorted(BREAK):
        sub = [e for e in eps if e["scenario"] == b]
        if sub:
            show(f"Break-it case {b}", sub)


if __name__ == "__main__":
    main()
