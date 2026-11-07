import sys, math, statistics
from collections import defaultdict
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from i_followups import load, first_service_t, line_of, pos, ARMS
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3] / "sweep"))
from score_sweep import episode_scores
sel, eps = load("runs_pilot3/dsv41")
def ang(u, v):
    return math.degrees(math.acos(max(-1, min(1, (u[0]*v[0]+u[1]*v[1]) / (math.hypot(*u)*math.hypot(*v))))))
rows = []
for lay in sel:
    e4 = eps.get((lay["name"], "CT4"))
    if not e4: continue
    line = line_of(lay["name"])
    rs = [r for r in e4["resolves"] if r.get("status") == "ok" and r.get("line_direction")]
    d = rs[0]["line_direction"]
    # median tilt against the axis at the true last, over the episode's ticks with someone ahead
    tilts = []
    for r in rs:
        tr = r["truth"]; p = pos(r)
        if tr["last"] in p:
            al, _ = line.project(*p[tr["last"]])
            k = max(0, min(len(line.length) - 1, int(sum(al >= s for s in line.start) - 1)))
            tilts.append(ang(d, line.unit[k]))
    tilt = statistics.median(tilts) if tilts else float("nan")
    st = {arm: episode_scores(eps[(lay["name"], arm)])["after_service_strict"] for arm in ARMS if (lay["name"], arm) in eps}
    rows.append((lay["name"], lay["stratum"], round(tilt, 1), st))
rows.sort(key=lambda x: x[2])
for n, s, t, st in rows:
    print(f"{n:12} {s:6} tilt {t:5.1f}  " + "  ".join(f"{a} {('%.2f' % v) if v is not None else '-'}" for a, v in st.items()))
for lo, hi in ((0, 15), (15, 999)):
    g = [st for n, s, t, st in rows if lo <= t < hi]
    print(f"\ntilt {lo}-{hi}: {len(g)} layouts; mean strict from 1st service: " +
          ", ".join(f"{a} {statistics.mean([x[a] for x in g if x.get(a) is not None]):.3f}" for a in ARMS))
