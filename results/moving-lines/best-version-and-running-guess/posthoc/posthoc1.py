"""Post-hoc 1 (POSTHOC_1.md): labelled exploratory, DeepSeek sweep logs only. Zero calls.

    python3 results/moving-lines/best-version-and-running-guess/posthoc/posthoc1.py runs_sweep/dsv41   (RUN_ROOT relative to results/data/moving/)
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402
MOVING = paths.MOVING_RUNS                # run folders are named relative to it, e.g. "runs_sweep/dsv41"
ARMS = ("CT2", "CT3", "Gl", "CT3F", "CT4")
CLASSES = ("right", "vacated", "empty", "bystander", "ex-member", "other non-member", "short, last listed",
           "short, last dropped", "behind robot")


def load(root):
    sel = json.loads((paths.MOVING / "test-layouts" / "selection.json").read_text())["layouts"]
    eps = {}
    for lay in sel:
        for arm in ARMS:
            f = root / "sweep" / lay["name"] / f"seed{lay['seed']}" / arm / "episode.json"
            if f.exists():
                eps[(lay["name"], arm)] = json.loads(f.read_text())
    return sel, eps


def classify(e):
    """Per scored tick: (class, resolve)."""
    bys = set(e.get("bystanders", []))
    seen = set()
    out = []
    for r in e.get("resolves", []):
        t = r.get("truth")
        if t is None:
            continue
        if t["ahead"]:
            last = r.get("last")
            roster = set(r.get("roster") or [])
            if r.get("vacated"):
                c = "vacated"
            elif last == t["last"] and r.get("status") == "ok":
                c = "right"
            elif last is None:
                c = "empty"
            elif last in bys:
                c = "bystander"
            elif last not in t["order"]:
                c = "ex-member" if last in seen else "other non-member"
            elif last in t["ahead"]:
                c = "short, last listed" if t["last"] in roster else "short, last dropped"
            else:
                c = "behind robot"
            out.append((c, r))
        seen |= set(t["order"])
    return out


def flag(lay):
    return {(False, False): "clean", (True, False): "lag only", (False, True): "excursion only",
            (True, True): "both"}[(lay["lag"], lay["restless"])]


def q1(sel, eps):
    print("## Q1. Scored ticks by class (share of all scored ticks, pooled over layouts)\n")
    for cell in ("all", "clean", "lag only", "excursion only", "both", "ours"):
        print(f"### {cell}\n")
        print("| Arm | ticks | " + " | ".join(CLASSES) + " |")
        print("|---|---|" + "---|" * len(CLASSES))
        for arm in ARMS:
            cnt = Counter()
            for lay in sel:
                if cell == "ours" and lay["stratum"] != "ours":
                    continue
                if cell not in ("all", "ours") and flag(lay) != cell:
                    continue
                e = eps.get((lay["name"], arm))
                if e:
                    cnt.update(c for c, _ in classify(e))
            n = sum(cnt.values())
            print(f"| {arm} | {n} | " + " | ".join(f"{cnt[c] / n:.3f}" if n else "-" for c in CLASSES) + " |")
        print()


def along(xy, d):
    return float(np.dot(np.asarray(xy, float), np.asarray(d, float)))


def q2(sel, eps):
    print("## Q2. CT4's tail (scored ticks, not vacated)\n")
    p_last, beyond, mid, n = [], 0, 0, 0
    bins = defaultdict(lambda: [0, 0])
    for lay in sel:
        e = eps.get((lay["name"], "CT4"))
        if not e:
            continue
        for c, r in classify(e):
            if c == "vacated":
                continue
            m = {int(k): v for k, v in (r.get("marginals") or {}).items()}
            if not m:
                continue
            n += 1
            t = r["truth"]
            p_last.append(m.get(t["last"], 0.0))
            mid += sum(1 for v in m.values() if 0.35 <= v <= 0.65)
            for i, v in m.items():
                b = min(int(v * 10), 9)
                bins[b][0] += 1
                bins[b][1] += i in t["order"]
            xy = {int(i): (x, y) for i, x, y in r["tracks"]}
            d = r.get("line_direction")
            if d and t["last"] in xy:
                a_last = along(xy[t["last"]], d)
                # line_direction points from the back towards the front: farther back = smaller projection
                if any(i in xy and i != t["last"] and along(xy[i], d) < a_last - 0.2
                       for i in (r.get("roster") or [])):
                    beyond += 1
    p = np.array(p_last)
    print(f"- ticks with marginals: {n}")
    print(f"- P(true last): median {np.median(p):.3f}; share >= 0.5: {np.mean(p >= 0.5):.3f}; "
          f"quartiles {np.percentile(p, 25):.3f} / {np.percentile(p, 75):.3f}")
    print(f"- ticks with a roster member more than 0.2 m farther back than the true last (along line_direction): "
          f"{beyond / n:.3f}")
    print(f"- marginals in 0.35-0.65 per tick: {mid / n:.2f}")
    print("- calibration: marginal bin -> share of those people in the queue (truth.order)")
    for b in range(10):
        k, hit = bins[b]
        if k:
            print(f"  - {b / 10:.1f}-{(b + 1) / 10:.1f}: {hit / k:.3f} (n {k})")
    print()


def matched(e):
    """1-s samples after the first delivery, each with the latest re-solve sent at or before it (or None)."""
    answers = e.get("answers") or []
    if not answers or answers[0].get("delivered_t") is None:
        return []
    t0 = answers[0]["delivered_t"]
    rs = [r for r in e.get("resolves", []) if r.get("truth") is not None]
    out, j = [], -1
    for s in e["samples"]:
        if s["t"] < t0:
            continue
        while j + 1 < len(rs) and rs[j + 1]["sent_t"] <= s["t"] + 1e-9:
            j += 1
        out.append((s, rs[j] if j >= 0 else None))
    return out


def first_service_t(e):
    rs = [r for r in e.get("resolves", []) if r.get("truth") is not None]
    for a, b in zip(rs, rs[1:]):
        if a["truth"]["order"] and a["truth"]["order"][0] not in b["truth"]["order"]:
            return b["sent_t"]
    return None


def q34(sel, eps, ours_after_service):
    title = ("## Q4. \"ours\" layouts, samples after the first service" if ours_after_service
             else "## Q3. All layouts, samples after the first delivery")
    print(title + "\n")
    print("| Arm | samples | share right-last | in line, last right | in line, last wrong | cut-in, last right | "
          "cut-in, last wrong | right-last re-solves with in_turn False | samples by state |")
    print("|---|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        k = Counter()
        states = Counter()
        rr = rr_noturn = 0
        for lay in sel:
            if ours_after_service and lay["stratum"] != "ours":
                continue
            e = eps.get((lay["name"], arm))
            if not e:
                continue
            ts = first_service_t(e) if ours_after_service else None
            if ours_after_service and ts is None:
                continue
            for s, r in matched(e):
                if ts is not None and s["t"] < ts - 1e-9:
                    continue
                if r is None or not r["truth"]["ahead"]:
                    continue
                right = (not r.get("vacated")) and r.get("last") == r["truth"]["last"]
                key = "R" if right else "W"
                k[key] += 1
                k[key + "in"] += bool(s["in_line"])
                k[key + "cut"] += bool(s["cut_in"])
                states[s["state"]] += 1
            for c, r in classify(e):
                if c == "right" and (ts is None or r["sent_t"] >= ts - 1e-9):
                    rr += 1
                    rr_noturn += not r.get("in_turn")
        n = k["R"] + k["W"]
        if not n:
            continue

        def sh(a, b):
            return f"{k[a] / k[b]:.3f}" if k[b] else "-"
        st = ", ".join(f"{s} {v / n:.2f}" for s, v in states.most_common(4))
        print(f"| {arm} | {n} | {k['R'] / n:.3f} | {sh('Rin', 'R')} | {sh('Win', 'W')} | {sh('Rcut', 'R')} | "
              f"{sh('Wcut', 'W')} | {rr_noturn / rr if rr else 0:.3f} ({rr}) | {st} |")
    print()


def main():
    root = MOVING / sys.argv[1] if len(sys.argv) > 1 else MOVING / "runs_sweep" / "dsv41"
    sel, eps = load(root)
    print(f"# Post-hoc 1 (exploratory): {root.relative_to(MOVING)}, {len(eps)} episodes\n")
    q1(sel, eps)
    q2(sel, eps)
    q34(sel, eps, False)
    q34(sel, eps, True)


if __name__ == "__main__":
    main()
