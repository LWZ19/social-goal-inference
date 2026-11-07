import sys, math, statistics
from collections import Counter, defaultdict
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from i_followups import load, first_service_t, line_of, pos, ARMS
from j_decompose import goal_check
sel, eps = load("runs_pilot3/dsv41")
def ang(u, v):
    return math.degrees(math.acos(max(-1, min(1, (u[0]*v[0]+u[1]*v[1]) / (math.hypot(*u)*math.hypot(*v))))))
for arm in ARMS:
    c = Counter(); tilt_ok = defaultdict(list); vac = Counter()
    for lay in sel:
        e = eps.get((lay["name"], arm)); t0 = first_service_t(e) if e else None
        if t0 is None: continue
        line = line_of(lay["name"])
        for r in (r for r in e["resolves"] if r.get("status") == "ok" and r.get("pose_world") and r["sent_t"] >= t0):
            tr = r["truth"]; p = pos(r)
            if not tr["ahead"]: continue
            v = bool(r.get("vacated"))
            vac[v] += 1
            if r.get("last") != tr["last"] and not v: continue
            if any(i not in tr["order"] for i in (r.get("roster") or [])): continue
            ok = goal_check(r, line) == "ok"
            c[(v, ok)] += 1
            # tilt between the solve direction and the true axis at the true last
            d = r.get("solve_direction") or r.get("line_direction")
            if d and tr["last"] in p and not v:
                al, sd = line.project(*p[tr["last"]])
                # axis direction at the last: numeric derivative along the polyline
                k = max(0, min(len(line.length) - 1, int(sum(al >= s for s in line.start) - 1)))
                tilt = ang(d, line.unit[k])
                tilt_ok["<10" if tilt < 10 else "10-25" if tilt < 25 else ">=25"].append(ok)
    nv = c[(False, True)] + c[(False, False)]; vv = c[(True, True)] + c[(True, False)]
    print(f"{arm}: vacated ticks {vac[True]}/{vac[True]+vac[False]}; right+clean not vacated n {nv} goal ok {c[(False, True)]/max(nv,1):.2f}; "
          f"vacated n {vv} goal ok {c[(True, True)]/max(vv,1):.2f}; by direction tilt: " +
          ", ".join(f"{k} n {len(v)} ok {sum(v)/len(v):.2f}" for k, v in sorted(tilt_ok.items())))
