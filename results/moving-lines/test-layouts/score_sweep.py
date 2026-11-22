"""Scores the fresh sweep runs with the analysis fixed before they were run. Zero calls; read-only on the runs.

    python3 score_sweep.py results/data/moving/runs_sweep/dsv41 [results/data/moving/runs_sweep/sonnet55_low]   (from the repo root)

Definitions (every number below comes from one of these):
- **Ticks**: the episode's logged re-solves. They all come after the first delivery, and each carries `truth`: the
  sim's queue at the time it was sent, with `order` (front first), `ahead` (those ahead of the robot in that order;
  everyone while the robot has not joined) and `last` (the last of `ahead`).
- **Right last (primary)**: over ticks where `truth.ahead` is not empty, the share where the tick is not vacated
  and the solver's `last` equals `truth.last`. A vacated tick (CT3F taking the place of the last person served)
  has no solver last, and counts as not right, as offline. An episode with no such ticks has no value, and its
  layout drops out of any comparison that needs it (counted and printed).
- **C1**: the 36 lag-flagged layouts (`lag` true: lag only, plus lag and excursion), DeepSeek, d = right last of
  CT4 minus CT3F. Ties are |d| < 0.01. Exact two-sided sign test over untied layouts (`summarise_fresh.sign_p`).
  It passes when wins > losses and p <= 0.05.
- **Reported, not claimed**:
  - the same comparison on excursion-only (18) and clean (18) layouts, predicted ties (an untied win rate in
    0.35-0.65), and on lag-flagged "ours" layouts (20), predicted CT4 ahead;
  - by flag cell: capture (seconds with a bystander as the solver's last, not vacated; 0.5 s a tick), recall
    (the summed |roster ∩ ahead| over the summed |ahead|, over ticks with someone ahead), and empty (the share of
    ticks with someone ahead where the solver's last is None and the tick is not vacated). Over `ahead`, not the
    whole queue: once the robot has joined, people behind it do not bear on its goal.
    Predicted: CT3F lower capture in every cell; CT4 higher recall and lower empty in every cell;
  - cut-in samples, and strict share in line (`summarise_fresh.measures`, unchanged): no prediction;
  - on "ours" layouts:
    - time to a right roster after delivery, from the first delivery to the first tick where the roster, less
      those behind the robot in the queue, equals `truth.ahead`; "never" when it doesn't happen;
    - share in line after the first service that follows delivery. A service is a tick where the front of the
      previous tick's `order` has left the queue; the share is over 1-s samples from that tick's time to the
      end, both primary and strict;
  - every arm against CT3 on right last, and Gl and CT2 alongside;
  - a held / not held line for every stated prediction;
  - sensitivity, reported and not the claim: C1 with vacated ticks left out of the denominator (only CT3F
    vacates);
  - every layout and arm without a value, with the reason, and any case where one arm of a compared pair has a
    value and the other has none (which would bias a test);
  - mechanical checks: re-solve errors, missing truth or history_t, and whether each replaying arm's first answer is
    CT2's (from the cache, and the same pose).
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402
sys.path.insert(0, str(paths.DESIGN))
from summarise_fresh import in_line, measures, sign_p  # noqa: E402

ARMS = ("CT2", "CT3", "Gl", "CT3F", "CT4", "R1")
TIE = 0.01


def load(root: Path) -> tuple[list[dict], dict]:
    sel = json.loads((HERE / "selection.json").read_text())["layouts"]
    eps = {}
    for lay in sel:
        for arm in ARMS:
            f = root / "sweep" / lay["name"] / f"seed{lay['seed']}" / arm / "episode.json"
            if f.exists():
                eps[(lay["name"], arm)] = json.loads(f.read_text())
    return sel, eps


def episode_scores(e: dict) -> dict:
    rs = [r for r in e.get("resolves", []) if r.get("truth") is not None]
    bys = set(e.get("bystanders", []))
    ticks = [r for r in rs if r["truth"]["ahead"]]
    right = [(not r.get("vacated")) and r.get("status") == "ok" and r.get("last") == r["truth"]["last"] for r in ticks]
    live = [r for r in ticks if not r.get("vacated")]
    right_live = [r.get("status") == "ok" and r.get("last") == r["truth"]["last"] for r in live]
    reason = ("no re-solves" if not e.get("resolves") else "no truth" if not rs else
              "no ticks with someone ahead" if not ticks else None)
    out = {"right_last": (sum(right) / len(right)) if right else None, "ticks": len(ticks), "reason": reason,
           "vacated": len(ticks) - len(live),
           "right_last_novac": (sum(right_live) / len(right_live)) if right_live else None,
           "capture_s": 0.5 * sum(1 for r in rs if not r.get("vacated") and r.get("last") in bys),
           "recall": (sum(len(set(r.get("roster") or []) & set(r["truth"]["ahead"])) for r in ticks)
                      / sum(len(r["truth"]["ahead"]) for r in ticks)) if ticks else None,
           "empty": (sum(1 for r in ticks if r.get("last") is None and not r.get("vacated")) / len(ticks))
           if ticks else None}
    m = measures(e) if e.get("samples") else {}
    out.update(cut=m.get("cut"), share_strict=m.get("share_strict"), share_frozen=m.get("share_frozen"))
    # time to a right roster after delivery
    answers = e.get("answers") or []
    delivered = answers[0]["delivered_t"] if answers else None
    out["to_right_roster_s"] = None
    for r in rs:
        behind = set(r["truth"]["order"]) - set(r["truth"]["ahead"])
        if set(r.get("roster") or []) - behind == set(r["truth"]["ahead"]) and r["truth"]["ahead"]:
            out["to_right_roster_s"] = round(r["sent_t"] - delivered, 2) if delivered is not None else None
            break
    # share in line after the first service following delivery
    out["after_service_frozen"] = out["after_service_strict"] = None
    for a, b in zip(rs, rs[1:]):
        if a["truth"]["order"] and a["truth"]["order"][0] not in b["truth"]["order"]:
            later = [x for x in e["samples"] if x["t"] >= b["sent_t"] - 1e-9]
            if later:
                out["after_service_frozen"] = sum(x["in_line"] for x in later) / len(later)
                out["after_service_strict"] = sum(in_line(x, "strict") for x in later) / len(later)
            break
    return out


def compare(scores, layouts, a, b, key="right_last"):
    """Wins / losses / ties of arm a over arm b on `key` (higher is better), over layouts that have both."""
    w = l = t = missing = 0
    ds, asym = [], []
    for lay in layouts:
        x, y = scores.get((lay["name"], a), {}).get(key), scores.get((lay["name"], b), {}).get(key)
        if x is None or y is None:
            missing += 1
            if (x is None) != (y is None):
                asym.append(lay["name"])
            continue
        d = x - y
        ds.append(d)
        if abs(d) < TIE:
            t += 1
        elif d > 0:
            w += 1
        else:
            l += 1
    return dict(w=w, l=l, t=t, missing=missing, asym=asym, p=sign_p(w, l), mean=statistics.mean(ds) if ds else None)


def fmt(c):
    rate = c["w"] / (c["w"] + c["l"]) if c["w"] + c["l"] else float("nan")
    mean = f"{c['mean']:+.3f}" if c["mean"] is not None else "-"
    asym = f", ASYMMETRIC missing: {c['asym']}" if c["asym"] else ""
    return (f"{c['w']}/{c['l']}/{c['t']} (wins/losses/ties), untied win rate {rate:.2f}, sign p {c['p']:.3f}, "
            f"mean d {mean}, missing {c['missing']}{asym}")


def report(root: Path, claim: bool) -> None:
    sel, eps = load(root)
    scores = {k: episode_scores(e) for k, e in eps.items()}
    model = root.name
    print(f"\n# {model}: {len(eps)} episodes over {len(sel)} layouts\n")
    lag = [x for x in sel if x["lag"]]
    exc_only = [x for x in sel if x["restless"] and not x["lag"]]
    clean = [x for x in sel if not x["restless"] and not x["lag"]]
    ours_lag = [x for x in sel if x["stratum"] == "ours" and x["lag"]]
    c1 = compare(scores, lag, "CT4", "CT3F")
    if claim:
        ok = c1["w"] > c1["l"] and c1["p"] <= 0.05
        print(f"**C1 (lag-flagged, n = {len(lag)}), CT4 vs CT3F, right last: {fmt(c1)} -> {'PASS' if ok else 'FAIL'}**\n")
    else:
        print(f"C1's comparison (reported for this model), lag-flagged: {fmt(c1)}\n")
    sens = compare(scores, lag, "CT4", "CT3F", key="right_last_novac")
    print(f"- sensitivity (reported, not the claim): C1 with vacated ticks left out of the denominator: {fmt(sens)}")
    exc_c, clean_c = compare(scores, exc_only, "CT4", "CT3F"), compare(scores, clean, "CT4", "CT3F")
    ours_c = compare(scores, ours_lag, "CT4", "CT3F")
    print(f"- excursion only (n = {len(exc_only)}; predicted tie): {fmt(exc_c)}")
    print(f"- clean (n = {len(clean)}; predicted tie): {fmt(clean_c)}")
    print(f"- ours, lag-flagged (n = {len(ours_lag)}; predicted CT4 ahead): {fmt(ours_c)}")
    print(f"- all layouts (n = {len(sel)}): {fmt(compare(scores, sel, 'CT4', 'CT3F'))}")
    print("\n## Against CT3 (right last, all layouts)\n")
    for arm in ("CT4", "CT3F", "Gl", "CT2"):
        print(f"- {arm} vs CT3: {fmt(compare(scores, sel, arm, 'CT3'))}")
    print("\n## By flag cell: means over layouts\n")
    print("| Cell | Arm | n | Right last | Capture (s) | Recall | Empty | Vacated ticks | Cut-ins | Strict share |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    cells = {"clean": clean, "lag only": [x for x in sel if x["lag"] and not x["restless"]],
             "excursion only": exc_only, "both": [x for x in sel if x["lag"] and x["restless"]]}
    cell_means = {}
    for name, lays in cells.items():
        for arm in ("CT3", "CT3F", "CT4", "Gl", "CT2", "R1"):
            vals = [scores[(x["name"], arm)] for x in lays if (x["name"], arm) in scores]
            if not vals:
                continue

            def mean(k, vals=vals):
                v = [s[k] for s in vals if s.get(k) is not None]
                return statistics.mean(v) if v else None
            m = {k: mean(k) for k in ("right_last", "capture_s", "recall", "empty", "cut", "share_strict")}
            m["vacated"] = sum(s["vacated"] for s in vals)
            cell_means[(name, arm)] = m
            f = lambda v: "-" if v is None else f"{v:.3f}"           # noqa: E731
            if arm == "R1":                                           # no re-solves: context only
                print(f"| {name} | R1 | {len(vals)} | - | - | - | - | - | {f(m['cut'])} | {f(m['share_strict'])} |")
            else:
                print(f"| {name} | {arm} | {len(vals)} | {f(m['right_last'])} | {f(m['capture_s'])} | {f(m['recall'])} | "
                      f"{f(m['empty'])} | {m['vacated']} | {f(m['cut'])} | {f(m['share_strict'])} |")
    print("\n## Stated predictions: held or not\n")

    def held(ok):
        return "held" if ok else "NOT held"

    def rate(c):
        return c["w"] / (c["w"] + c["l"]) if c["w"] + c["l"] else None
    for label, c in (("excursion only: a tie (untied win rate in 0.35-0.65)", exc_c),
                     ("clean: a tie (untied win rate in 0.35-0.65)", clean_c)):
        r = rate(c)
        print(f"- {label}: {'no untied layouts' if r is None else f'{r:.2f}'} -> "
              f"{held(r is None or 0.35 <= r <= 0.65)}")
    print(f"- ours, lag-flagged: CT4 ahead (w > l): {ours_c['w']}/{ours_c['l']} -> {held(ours_c['w'] > ours_c['l'])}")

    def every(key, better):
        rows = []
        for name in cells:
            a, b = cell_means.get((name, "CT4"), {}).get(key), cell_means.get((name, "CT3F"), {}).get(key)
            rows.append((name, a, b, a is not None and b is not None and better(a, b)))
        return rows
    for label, key, better in (("CT3F lower capture", "capture_s", lambda c4, c3f: c3f < c4),
                               ("CT4 higher recall", "recall", lambda c4, c3f: c4 > c3f),
                               ("CT4 lower empty", "empty", lambda c4, c3f: c4 < c3f)):
        rows = every(key, better)
        detail = "; ".join(f"{n}: CT4 {'-' if a is None else f'{a:.3f}'}, CT3F {'-' if b is None else f'{b:.3f}'}"
                           for n, a, b, _ in rows)
        print(f"- {label} in every flag cell: {held(all(ok for *_, ok in rows))} ({detail})")
    print("\n## \"ours\" layouts: time to a right roster after delivery, and share in line after the first service\n")
    ours = [x for x in sel if x["stratum"] == "ours"]
    for arm in ("CT3", "CT3F", "CT4", "Gl", "CT2"):
        vals = [scores[(x["name"], arm)] for x in ours if (x["name"], arm) in scores]
        if not vals:
            continue
        times = sorted(s["to_right_roster_s"] for s in vals if s["to_right_roster_s"] is not None)
        never = sum(1 for s in vals if s["to_right_roster_s"] is None)
        fz = [s["after_service_frozen"] for s in vals if s["after_service_frozen"] is not None]
        st = [s["after_service_strict"] for s in vals if s["after_service_strict"] is not None]
        med = f"{statistics.median(times):.1f}" if times else "-"
        print(f"- {arm} (n = {len(vals)}): right roster after a median {med} s ({len(times)} layouts; never in "
              f"{never}); share in line after the first service: frozen "
              f"{statistics.mean(fz) if fz else float('nan'):.3f}, strict {statistics.mean(st) if st else float('nan'):.3f}")
    print("\n## Layouts and arms without a right-last value\n")
    gaps = [(x["name"], arm, scores[(x["name"], arm)]["reason"]) for x in sel for arm in ("CT2", "CT3", "Gl", "CT3F", "CT4")
            if (x["name"], arm) in scores and scores[(x["name"], arm)]["right_last"] is None]
    gaps += [(x["name"], arm, "no episode") for x in sel for arm in ("CT2", "CT3", "Gl", "CT3F", "CT4", "R1")
             if (x["name"], arm) not in eps]
    for g in gaps:
        print(f"- {g[0]} {g[1]}: {g[2]}")
    if not gaps:
        print("- none")
    print("\n## Mechanical checks (RUN_NOTE: these, not scores, may stop the next run)\n")
    for arm in ("CT2", "CT3", "Gl", "CT3F", "CT4", "R1"):
        mine = {n: e for (n, a), e in eps.items() if a == arm}
        errs = sum(1 for e in mine.values() for r in e.get("resolves", []) if r.get("status") not in ("ok", None))
        no_truth = sum(1 for e in mine.values() for r in e.get("resolves", []) if r.get("truth") is None)
        no_hist = sum(1 for e in mine.values() if arm in ("CT3F", "CT4") and e.get("resolves") and "history_t" not in e)
        not_ct2 = []
        if arm in ("CT3", "Gl", "CT3F", "CT4"):
            for n, e in mine.items():
                ref = eps.get((n, "CT2"))
                a0 = (e.get("answers") or [{}])[0]
                r0 = ((ref or {}).get("answers") or [{}])[0]
                if ref is None or not all(a0.get("cached") or [False]) or a0.get("pose_world") != r0.get("pose_world"):
                    not_ct2.append(n)
        vac = sum(scores[(n, arm)]["vacated"] for n in mine) if arm != "R1" else 0
        print(f"- {arm}: {len(mine)} episodes; re-solve errors {errs}; re-solves without truth {no_truth}; "
              f"history_t missing {no_hist}; vacated ticks with someone ahead {vac}; first answer not CT2's "
              f"(uncached or another pose) {len(not_ct2)}{' ' + str(not_ct2) if not_ct2 else ''}")


def main(argv) -> int:
    for i, root in enumerate(argv):
        report(Path(root), claim=(Path(root).name == "dsv41"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
