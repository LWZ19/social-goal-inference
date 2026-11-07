import sys, math, statistics
from collections import Counter, defaultdict
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from i_followups import load, first_service_t, line_of, pos, in_line, ARMS
from j_decompose import goal_check
sel, eps = load("runs_pilot3/dsv41")
for arm in ARMS:
    by = defaultdict(Counter); sides = defaultdict(list); lsides = defaultdict(list)
    for lay in sel:
        e = eps.get((lay["name"], arm)); t0 = first_service_t(e) if e else None
        if t0 is None: continue
        line = line_of(lay["name"])
        for r in (r for r in e["resolves"] if r.get("status") == "ok" and r.get("pose_world") and r["sent_t"] >= t0):
            tr = r["truth"]
            if not tr["ahead"]: continue
            last = r.get("last"); p = pos(r)
            cls = "none" if last is None else "right" if last == tr["last"] else "waiter" if last in tr["order"] else "non-waiter"
            g = goal_check(r, line)
            by[cls][g] += 1
            gs = line.project(*r["pose_world"][:2])[1]
            sides[cls].append(gs)
            if last is not None and last in p: lsides[cls].append(line.project(*p[last])[1])
    print(arm)
    for cls in ("right", "waiter", "non-waiter", "none"):
        c = by[cls]; n = sum(c.values())
        if not n: continue
        print(f"  {cls:10} n {n:4}  goal ok {c['ok']/n:.2f}  side-fail {c['side']/n:.2f}  gap-far {c['gap far']/n:.2f}  "
              f"goal side median {statistics.median(sides[cls]):.2f}  last's side median "
              f"{statistics.median(lsides[cls]) if lsides[cls] else float('nan'):.2f}")
