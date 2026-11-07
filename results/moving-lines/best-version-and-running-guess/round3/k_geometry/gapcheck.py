import sys, statistics
from collections import Counter
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from i_followups import load, first_service_t, line_of, pos, in_line, ARMS
sel, eps = load("runs_pilot3/dsv41")
for arm in ARMS:
    gaps, st, far_goal = [], Counter(), []
    for lay in sel:
        e = eps.get((lay["name"], arm)); t0 = first_service_t(e) if e else None
        if t0 is None: continue
        rs = [r for r in e["resolves"] if r.get("status") == "ok"]
        for s in (x for x in e["samples"] if x["t"] >= t0 - 1e-9):
            if in_line(s, "strict") or s["ahead"] == 0 or s["in_line"]: continue
            r = max((r for r in rs if r["sent_t"] <= s["t"] + 1e-9), key=lambda r: r["sent_t"], default=None)
            if r is None or r.get("last") != r["truth"]["last"] or s["behind_m"] < 0: continue
            if 0.5 <= s["gap_m"] <= 1.2: continue
            gaps.append(s["gap_m"]); st[s["state"]] += 1
            if s.get("goal"):
                import math; far_goal.append(math.dist(s["robot"][:2], s["goal"]))
    q = statistics.quantiles(gaps, n=10) if len(gaps) > 10 else gaps
    print(arm, len(gaps), "gap deciles", [round(v, 2) for v in q], "too close", sum(g < 0.5 for g in gaps),
          "states", dict(st), "robot-to-goal median", round(statistics.median(far_goal), 2) if far_goal else None)
