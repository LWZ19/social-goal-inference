import sys, math, statistics
from collections import Counter, defaultdict
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from i_followups import load, first_service_t, line_of, pos, ARMS
from j_decompose import goal_check
sel, eps = load("runs_pilot3/dsv41")
for arm in ARMS:
    rows = []
    for lay in sel:
        e = eps.get((lay["name"], arm)); t0 = first_service_t(e) if e else None
        if t0 is None: continue
        line = line_of(lay["name"]); S = e["samples"]
        for r in (r for r in e["resolves"] if r.get("status") == "ok" and r.get("pose_world") and r["sent_t"] >= t0):
            tr = r["truth"]
            if not tr["ahead"] or r.get("last") != tr["last"]: continue
            if any(i not in tr["order"] for i in (r.get("roster") or [])): continue
            s = min(S, key=lambda x: abs(x["t"] - r["sent_t"]))
            gside = line.project(*r["pose_world"][:2])[1]
            rows.append((s["side"], gside, goal_check(r, line), lay["name"], r.get("in_turn")))
    near = [g for rs, g, *_ in rows if rs <= 0.6]; far = [g for rs, g, *_ in rows if rs > 0.6]
    okn = sum(c == "ok" for rs, g, c, *_ in rows if rs <= 0.6) / max(len(near), 1)
    okf = sum(c == "ok" for rs, g, c, *_ in rows if rs > 0.6) / max(len(far), 1)
    per = Counter(n for rs, g, c, n, _ in rows if c == "side")
    print(f"{arm}: robot on axis n {len(near)} goal ok {okn:.2f} (goal side med {statistics.median(near) if near else float('nan'):.2f}); "
          f"robot off axis n {len(far)} goal ok {okf:.2f} (goal side med {statistics.median(far) if far else float('nan'):.2f}); "
          f"side fails by layout {per.most_common(5)}")
