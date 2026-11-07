"""EXPLORATORY follow-ups to the scored confirmatory test (CONFIRM_RESULT.md section 4). Nothing here is a claim.
Questions written before any per-layout confirm data was seen. Zero calls; read-only.

    .venv/bin/python round4/q_confirm_explore.py SELECTION_JSON RUN_ROOT [RUN_ROOT ...]
    (from ideas2_1b/; paths relative to moving/, e.g. sweep/confirm_selection.json runs_confirm4/dsv41
    runs_confirm4/sonnet55_low.) Per-layout rows go to round4/q_confirm_layouts.csv.

E1  Where the CT3F - CT4PJ gap sits: layouts where CT4PJ's own join test fired at least once, against never.
E2  Holding the place, per arm, from the first service on: the sim's "left" events (the robot leaves the line's
    corridor) per minute joined (a robot that never joins cannot leave, so raw counts mislead), and goal jumps
    (consecutive ok re-solves more than 0.5 m apart) per minute. Paired: on layouts where CT4PJ is below CT3F,
    how often it also leaves faster, and jumps more. Rank correlation of d(strict) with d(leave rate) and with
    d(jumps), CT4PJ - CT3F.
E3  By stress flags (lag x restless): mean d for CT4PJ - CT3F and CT4PJ - CT4P.
E4  With two run roots: rank correlation of each layout's d across the two models, and sign agreement.
"""
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "round3"))
from i_followups import MOVING, first_service_t  # noqa: E402
from score_sweep import episode_scores  # noqa: E402

ARMS = ("CT3F", "CT4", "CT4P", "CT4PJ")
TIE = 0.01
JUMP_M = 0.5


def load(root, sel):
    eps = {}
    for lay in sel:
        for arm in ARMS:
            hits = sorted((MOVING / root).glob(f"**/{lay['name']}/seed{lay['seed']}/{arm}/episode.json"))
            if hits:
                eps[(lay["name"], arm)] = json.loads(hits[0].read_text())
    return eps


def strict(e):
    return None if e is None else episode_scores(e)["after_service_strict"]


def place(e):
    """(left events per minute joined after the first service, or None if never joined then; goal jumps per minute
    after it); (None, None) before any service. Samples are 1 Hz."""
    t0 = first_service_t(e)
    if t0 is None:
        return None, None
    left = sum(1 for ev in e.get("joins", []) if ev["event"] == "left" and ev["t"] >= t0 - 1e-9)
    joined_min = sum(1 for x in e["samples"] if x["t"] >= t0 - 1e-9 and x["joined"]) / 60
    rate = left / joined_min if joined_min > 0 else None
    ok = [r for r in e["resolves"] if r.get("status") == "ok" and r.get("pose_world")
          and (r.get("delivered_t") or 0) >= t0 - 1e-9]
    jumps = sum(1 for a, b in zip(ok, ok[1:])
                if math.dist(a["pose_world"][:2], b["pose_world"][:2]) > JUMP_M)
    t_end = max([s["t"] for s in e["samples"]] or [t0])
    return rate, jumps / max((t_end - t0) / 60, 1e-9)


def ever_joined(e):
    return any(r.get("joined") for r in e["resolves"] if r.get("status") == "ok")


def ranks(x):
    x = np.asarray(x, float)
    order = x.argsort(kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):  # average ties
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def wlt(ds):
    return (sum(d >= TIE for d in ds), sum(d <= -TIE for d in ds), sum(abs(d) < TIE for d in ds))


def fmt(ds):
    if not ds:
        return "n 0"
    w, l, t = wlt(ds)
    return f"mean {statistics.mean(ds):+.3f}, {w}/{l}/{t}, n {len(ds)}"


def rows_for(root, sel):
    eps = load(root, sel)
    out = []
    for lay in sel:
        r = {"model": root, "name": lay["name"], "stratum": lay["stratum"], "lag": lay["lag"],
             "restless": lay["restless"]}
        for arm in ARMS:
            e = eps.get((lay["name"], arm))
            r[f"strict_{arm}"] = strict(e)
            rate, jumps = place(e) if e is not None else (None, None)
            r[f"leave_rate_{arm}"], r[f"jumps_{arm}"] = rate, jumps
        e = eps.get((lay["name"], "CT4PJ"))
        r["ct4pj_joined"] = ever_joined(e) if e is not None else None
        out.append(r)
    return out


def d(r, a, b):
    x, y = r[f"strict_{a}"], r[f"strict_{b}"]
    return None if x is None or y is None else x - y


def report(root, rows):
    print(f"\n## {root}\n")
    ok = [r for r in rows if d(r, "CT4PJ", "CT3F") is not None and d(r, "CT4PJ", "CT4P") is not None]
    print(f"- layouts with all values: {len(ok)} of {len(rows)}")

    print("\n### E1: CT4PJ's join test fired at least once, against never\n")
    for label, grp in (("fired", [r for r in ok if r["ct4pj_joined"]]), ("never", [r for r in ok if not r["ct4pj_joined"]])):
        print(f"- {label}: CT4PJ - CT3F {fmt([d(r, 'CT4PJ', 'CT3F') for r in grp])}; "
              f"CT4PJ - CT4P {fmt([d(r, 'CT4PJ', 'CT4P') for r in grp])}")

    print("\n### E2: holding the place, from the first service on\n")
    for arm in ARMS:
        lv = [r[f"leave_rate_{arm}"] for r in ok if r[f"leave_rate_{arm}"] is not None]
        jv = [r[f"jumps_{arm}"] for r in ok if r[f"jumps_{arm}"] is not None]
        print(f"- {arm}: leaves per minute joined {(statistics.mean(lv) if lv else float('nan')):.2f} "
              f"(layouts joined {len(lv)}, any leave {sum(x > 0 for x in lv)}); goal jumps per minute "
              f"{(statistics.mean(jv) if jv else float('nan')):.2f}")
    for key, word in (("leave_rate", "leaves faster"), ("jumps", "jumps more")):
        worse = [r for r in ok if d(r, "CT4PJ", "CT3F") <= -TIE
                 and r[f"{key}_CT4PJ"] is not None and r[f"{key}_CT3F"] is not None]
        more = sum(r[f"{key}_CT4PJ"] > r[f"{key}_CT3F"] + 1e-9 for r in worse)
        same = sum(abs(r[f"{key}_CT4PJ"] - r[f"{key}_CT3F"]) <= 1e-9 for r in worse)
        print(f"- layouts where CT4PJ is below CT3F, both with a value: {len(worse)}; CT4PJ {word} on {more}, "
              f"the same on {same}, less on {len(worse) - more - same}")
        pair = [(d(r, "CT4PJ", "CT3F"), r[f"{key}_CT4PJ"] - r[f"{key}_CT3F"]) for r in ok
                if r[f"{key}_CT4PJ"] is not None and r[f"{key}_CT3F"] is not None]
        print(f"- rank correlation, d(strict) with d({key}), CT4PJ - CT3F: "
              f"{spearman([a for a, _ in pair], [b for _, b in pair]):+.2f} (n {len(pair)})")

    print("\n### E3: by stress flags\n")
    for lag in (False, True):
        for rest in (False, True):
            grp = [r for r in ok if r["lag"] == lag and r["restless"] == rest]
            print(f"- lag {lag!s:5} restless {rest!s:5}: CT4PJ - CT3F {fmt([d(r, 'CT4PJ', 'CT3F') for r in grp])}; "
                  f"CT4PJ - CT4P {fmt([d(r, 'CT4PJ', 'CT4P') for r in grp])}")


def across(roots, by_root):
    print("\n## E4: layouts across models\n")
    a, b = roots[0], roots[1]
    ia = {r["name"]: r for r in by_root[a]}
    ib = {r["name"]: r for r in by_root[b]}
    for x, y in (("CT4PJ", "CT4P"), ("CT4PJ", "CT3F")):
        pairs = [(d(ia[n], x, y), d(ib[n], x, y)) for n in ia if n in ib
                 and d(ia[n], x, y) is not None and d(ib[n], x, y) is not None]
        agree = sum(1 for p, q in pairs if (p >= TIE and q >= TIE) or (p <= -TIE and q <= -TIE))
        opp = sum(1 for p, q in pairs if (p >= TIE and q <= -TIE) or (p <= -TIE and q >= TIE))
        print(f"- {x} - {y}: rank correlation {spearman([p for p, _ in pairs], [q for _, q in pairs]):+.2f}; "
              f"same sign {agree}, opposite {opp}, a tie in either {len(pairs) - agree - opp} (n {len(pairs)})")


def main():
    sel = json.loads((MOVING / sys.argv[1]).read_text())["layouts"]
    roots = sys.argv[2:]
    print(f"# EXPLORATORY follow-ups to the confirmatory test ({sys.argv[1]}; not claims)")
    by_root = {}
    for root in roots:
        by_root[root] = rows_for(root, sel)
        report(root, by_root[root])
    if len(roots) >= 2:
        across(roots, by_root)
    out = HERE / "q_confirm_layouts.csv"
    rows = [r for root in roots for r in by_root[root]]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nPer-layout rows: {out.name} ({len(rows)})")


if __name__ == "__main__":
    main()
