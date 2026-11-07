import sys, math, statistics
from collections import Counter, defaultdict
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from i_followups import load, first_service_t, line_of, pos, ARMS
from j_decompose import goal_check
sel, eps = load("runs_pilot3/dsv41")
def robot_xy_at(e, t):
    s = min(e["samples"], key=lambda x: abs(x["t"] - t)); return s["robot"][:2], abs(s["t"] - t)
tot = defaultdict(Counter); per_layout = defaultdict(lambda: defaultdict(list))
for arm in ARMS:
    for lay in sel:
        e = eps.get((lay["name"], arm))
        if not e: continue
        line = line_of(lay["name"])
        rs = [r for r in e["resolves"] if r.get("status") == "ok" and r.get("pose_world") and r["truth"]["ahead"]]
        for a, b in zip(rs, rs[1:]):
            ra, rb = a["truth"], b["truth"]
            wrong_a = a.get("last") != ra["last"]
            right_b = b.get("last") == rb["last"] and not any(i not in rb["order"] for i in (b.get("roster") or []))
            if not (wrong_a and right_b): continue
            # robot position when b was sent: where the previous (wrong) goals left it
            xy, dt = robot_xy_at(e, b["sent_t"])
            if dt > 0.6: continue
            rside = line.project(*xy)[1]
            ok = goal_check(b, line) == "ok"
            key = "robot off" if rside > 0.6 else "robot on"
            tot[arm][(key, ok)] += 1
            per_layout[(arm, lay["name"])][key].append(ok)
for arm in ARMS:
    c = tot[arm]
    on = c[("robot on", True)] + c[("robot on", False)]; off = c[("robot off", True)] + c[("robot off", False)]
    print(f"{arm}: first right tick after a wrong one — robot on axis n {on}, goal ok {c[('robot on', True)]/max(on,1):.2f}; "
          f"robot off axis n {off}, goal ok {c[('robot off', True)]/max(off,1):.2f}")
# within-layout: layouts (arm x layout) with both kinds
w = l = t = 0
for k, d in per_layout.items():
    if d["robot on"] and d["robot off"]:
        x = sum(d["robot on"]) / len(d["robot on"]) - sum(d["robot off"]) / len(d["robot off"])
        if abs(x) < 1e-9: t += 1
        elif x > 0: w += 1
        else: l += 1
print(f"within arm x layout, goal ok higher with robot on axis: {w}, lower: {l}, tied: {t}")
