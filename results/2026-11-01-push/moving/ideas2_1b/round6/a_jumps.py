"""Round 6 (ROUND6.md), EXPLORATORY: goal jumps by kind and by the sim's joined state, and leaves after a jump.
Zero calls; read-only.

    .venv/bin/python round6/a_jumps.py SELECTION_JSON RUN_ROOT [RUN_ROOT ...]
    (from ideas2_1b/; paths relative to moving/.)
"""
import json
import math
import statistics
import sys
from bisect import bisect_right
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "round3"))
from i_followups import MOVING, first_service_t  # noqa: E402
from score_sweep import episode_scores  # noqa: E402

ARMS = ("CT3F", "CT4", "CT4P", "CT4PJ")
JUMP_M = 0.5
BEFORE_S = 5.0
TIE = 0.01


def load(root, sel):
    eps = {}
    for lay in sel:
        for arm in ARMS:
            hits = sorted((MOVING / root).glob(f"**/{lay['name']}/seed{lay['seed']}/{arm}/episode.json"))
            if hits:
                eps[(lay["name"], arm)] = json.loads(hits[0].read_text())
    return eps


def analyse(e):
    """Per episode, from the first service on: jumps [(t, kind, joined)], joined and free minutes, leave times,
    joined sample times. None before any service."""
    t0 = first_service_t(e)
    if t0 is None:
        return None
    samples = sorted((s["t"], bool(s["joined"])) for s in e["samples"])
    st = [t for t, _ in samples]

    def joined_at(t):
        i = bisect_right(st, t + 1e-9) - 1
        return samples[max(i, 0)][1] if samples else False

    ok = sorted((r for r in e["resolves"] if r.get("status") == "ok" and r.get("pose_world")
                 and (r.get("delivered_t") or 0) >= t0 - 1e-9), key=lambda r: r["delivered_t"])
    jumps = []
    for a, b in zip(ok, ok[1:]):
        if math.dist(a["pose_world"][:2], b["pose_world"][:2]) > JUMP_M:
            t = b["delivered_t"]
            jumps.append((t, "switch" if a.get("last") != b.get("last") else "same", joined_at(t)))
    after = [(t, j) for t, j in samples if t >= t0 - 1e-9]
    leaves = [ev["t"] for ev in e.get("joins", []) if ev["event"] == "left" and ev["t"] >= t0 - 1e-9]
    return {"jumps": jumps, "min_joined": sum(j for _, j in after) / 60, "min_free": sum(not j for _, j in after) / 60,
            "leaves": leaves, "joined_t": [t for t, j in after if j],
            "jpm": len(jumps) / max(((after[-1][0] - t0) / 60) if after else 0, 1e-9)}


def preceded(t, jumps, kind=None):
    return any(t - BEFORE_S < tj <= t + 1e-9 and (kind is None or k == kind) for tj, k, _ in jumps)


def report(root, sel):
    eps = load(root, sel)
    res = {(n, a): analyse(e) for (n, a), e in eps.items()}
    print(f"\n## {root}\n")

    print("### J-a: layouts where CT4PJ jumps more than CT3F (jumps per minute from the first service)\n")
    pairs = [(res[(l["name"], "CT4PJ")]["jpm"], res[(l["name"], "CT3F")]["jpm"]) for l in sel
             if res.get((l["name"], "CT4PJ")) and res.get((l["name"], "CT3F"))]
    print(f"- more {sum(a > b for a, b in pairs)}, same {sum(a == b for a, b in pairs)}, "
          f"less {sum(a < b for a, b in pairs)} (n {len(pairs)})")

    print("\n### J-b, J-c: pooled jumps per minute, by kind and by the sim's joined state\n")
    print("| arm | min joined | min free | switch, joined | same, joined | switch, free | same, free |")
    print("|---|---|---|---|---|---|---|")
    for arm in ARMS:
        rs = [res[(l["name"], arm)] for l in sel if res.get((l["name"], arm))]
        mj = sum(r["min_joined"] for r in rs)
        mf = sum(r["min_free"] for r in rs)

        def rate(kind, joined, m):
            return sum(1 for r in rs for _, k, j in r["jumps"] if k == kind and j == joined) / max(m, 1e-9)
        print(f"| {arm} | {mj:.0f} | {mf:.0f} | {rate('switch', True, mj):.2f} | {rate('same', True, mj):.2f} | "
              f"{rate('switch', False, mf):.2f} | {rate('same', False, mf):.2f} |")

    print(f"\n### J-d: leaves after a jump in the {BEFORE_S:.0f} s before (lift = share before leaves / share of joined "
          "seconds)\n")
    for arm in ARMS:
        rs = [res[(l["name"], arm)] for l in sel if res.get((l["name"], arm))]
        lv = [(t, r["jumps"]) for r in rs for t in r["leaves"]]
        base = [(t, r["jumps"]) for r in rs for t in r["joined_t"]]
        out = []
        for kind in (None, "switch"):
            p = sum(preceded(t, j, kind) for t, j in lv) / max(len(lv), 1)
            b = sum(preceded(t, j, kind) for t, j in base) / max(len(base), 1)
            out.append(f"{'any' if kind is None else 'switch'} {p:.2f} vs {b:.2f} (lift "
                       f"{(p / b) if b > 0 else float('nan'):.1f})")
        print(f"- {arm}: leaves {len(lv)}; " + "; ".join(out))

    print("\n### J-e: share of the CT4PJ - CT3F gap by whether CT4PJ's join test fired\n")
    rows = []
    for l in sel:
        a, b = eps.get((l["name"], "CT4PJ")), eps.get((l["name"], "CT3F"))
        if a is None or b is None:
            continue
        x, y = episode_scores(a)["after_service_strict"], episode_scores(b)["after_service_strict"]
        if x is None or y is None:
            continue
        rows.append((x - y, any(r.get("joined") for r in a["resolves"] if r.get("status") == "ok")))
    tot = sum(dd for dd, _ in rows)
    for label, flag in (("fired", True), ("never", False)):
        g = [dd for dd, f in rows if f == flag]
        print(f"- {label}: n {len(g)}, mean {statistics.mean(g) if g else float('nan'):+.3f}, share of the total gap "
              f"{(sum(g) / tot) if tot else float('nan'):.2f}")


def main():
    sel = json.loads((MOVING / sys.argv[1]).read_text())["layouts"]
    print(f"# Round 6, EXPLORATORY: goal jumps and leaving ({sys.argv[1]}; not claims)")
    for root in sys.argv[2:]:
        report(root, sel)


if __name__ == "__main__":
    main()
