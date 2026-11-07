"""Round 6 part C (ROUND6_C.md), EXPLORATORY: the hold-unless-margin family (C1, description) and CT5's decision after
commitment with ROUND3's costs unchanged (C2), on CT4PJ's logged joined stretches. Zero calls; read-only.

    .venv/bin/python round6/c_tradeoff.py SELECTION_JSON RUN_ROOT [RUN_ROOT ...]
    (from ideas2_1b/; paths relative to moving/.)
"""
import json
import sys
from bisect import bisect_right
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "round3"))
from b_pred import find, ok_resolves, stretches, switches, tally  # noqa: E402
from i_followups import MOVING  # noqa: E402

MARGINS = (0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
C_CUT, C_FAR, C_MOVE_PER_M = 1.0, 0.25, 0.125          # ROUND3.md section 2, unchanged
SENS = {"ratio 2": (1.0, 0.5, 0.125), "ratio 8": (1.0, 0.125, 0.125), "move 0": (1.0, 0.25, 0.0),
        "move x2": (1.0, 0.25, 0.25)}


def evidence(r, robot, tgt):
    """[(id, P, s)] for candidates, nearest the robot first, and q over ids plus None."""
    d = r.get("line_direction") or r.get("solve_direction")
    if not d or robot is None:
        return [], {None: 1.0}
    marg = {int(k): v for k, v in (r.get("marginals") or {}).items()}
    behind = set(r.get("behind") or [])

    def s(xy):
        return (xy[0] - tgt[0]) * d[0] + (xy[1] - tgt[1]) * d[1]
    sr = s(robot)
    cand = sorted(((int(i), marg[int(i)], s((x, y))) for i, x, y in r["tracks"]
                   if int(i) in marg and int(i) not in behind and s((x, y)) < sr), key=lambda c: -c[2])
    q, none_left = {}, 1.0
    for i, p, _ in cand:
        q[i] = p * none_left
        none_left *= 1 - p
    q[None] = none_left
    return cand, q


def costs(cand, cut, far):
    """Expected cost of coming after each candidate (and None), CT5's terms with independent marginals."""
    out = {}
    for idx, (j, _, _) in enumerate(cand):
        c = cut * sum(p for _, p, _ in cand[:idx])
        f, none_nearer = 0.0, 1.0
        for k, (i, p, _) in enumerate(cand):
            if k >= idx:
                f += (1 - p) * none_nearer
            none_nearer *= 1 - p
        out[j] = c + far * f
    out[None] = cut * sum(p for _, p, _ in cand)
    return out


def run_margin(steps, m):
    c, out = "unset", []
    for t, cand, q in steps:
        b = max(q, key=q.get)
        if c == "unset" or q[b] - q.get(c, 0.0) > m:
            c = b
        out.append((t, c))
    return out


def run_cost(steps, cut, far, move):
    c, out = "unset", []
    for t, cand, q in steps:
        cost = costs(cand, cut, far)
        best = min(cost, key=cost.get)
        if c != "unset" and c in cost:
            sp = {None: 0.0, **{i: s for i, _, s in cand}}
            if cost[c] <= cost[best] + move * abs(sp[best] - sp[c]):
                out.append((t, c))
                continue
        c = best
        out.append((t, c))
    return out


def report(root, sel):
    print(f"\n## {root}\n")
    rows = {k: [] for k in ["CT3F", "CT4PJ"] + [f"m {m}" for m in MARGINS] + ["C2"] + [f"C2 {k}" for k in SENS]}
    chs = {k: [] for k in rows}
    for lay in sel:
        f3 = find(root, lay, "CT3F")
        if f3 is not None:
            for s_ in stretches(ok_resolves(json.loads(f3.read_text()))):
                rows["CT3F"] += [(r["delivered_t"], r.get("last"), r["truth"]) for r in s_]
                chs["CT3F"].append([(r["delivered_t"], r.get("last")) for r in s_])
        f = find(root, lay, "CT4PJ")
        if f is None:
            continue
        e = json.loads(f.read_text())
        a = f.parent / "001" / "answer.json"
        ans = json.loads(a.read_text()) if a.exists() else {}
        if ans.get("target_world") is None:
            continue
        tgt = tuple(ans["target_world"][:2])
        samp = sorted((s["t"], s.get("robot")) for s in e["samples"] if s.get("robot") is not None)
        st_t = [t for t, _ in samp]
        for s_ in stretches(ok_resolves(e)):
            truths = [r["truth"] for r in s_]
            rows["CT4PJ"] += [(r["delivered_t"], r.get("last"), r["truth"]) for r in s_]
            chs["CT4PJ"].append([(r["delivered_t"], r.get("last")) for r in s_])
            steps = []
            for r in s_:
                i = bisect_right(st_t, r["delivered_t"] + 1e-9) - 1
                robot = tuple(samp[max(i, 0)][1][:2]) if samp else None
                cand, q = evidence(r, robot, tgt)
                steps.append((r["delivered_t"], cand, q))
            variants = {f"m {m}": run_margin(steps, m) for m in MARGINS}
            variants["C2"] = run_cost(steps, C_CUT, C_FAR, C_MOVE_PER_M)
            variants.update({f"C2 {k}": run_cost(steps, *v) for k, v in SENS.items()})
            for k, ch in variants.items():
                rows[k] += [(t, c, tr) for (t, c), tr in zip(ch, truths)]
                chs[k].append(ch)
    print("| choice | joined re-solves | right last, sim agrees | right last, all joined | switches per minute joined |")
    print("|---|---|---|---|---|")
    for k in rows:
        n, na, ra, rall = tally_none(rows[k])
        sw, mins = switches(chs[k])
        print(f"| {k} | {n} | {ra / max(na, 1):.3f} | {rall / max(n, 1):.3f} | {sw / max(mins, 1e-9):.2f} |")


def tally_none(rows):
    """As b_pred.tally, but a choice of None is right when the sim's last is None."""
    n = len(rows)
    agree = [(c, tr) for _, c, tr in rows if tr.get("joined")]
    ra = sum(c == tr.get("last") for c, tr in agree)
    rall = sum(c == tr.get("last") for _, c, tr in rows)
    return n, len(agree), ra, rall


def main():
    sel = json.loads((MOVING / sys.argv[1]).read_text())["layouts"]
    print(f"# Round 6 part C, EXPLORATORY: the decision's trade-off on CT4PJ's joined stretches ({sys.argv[1]})")
    for root in sys.argv[2:]:
        report(root, sel)


if __name__ == "__main__":
    main()
