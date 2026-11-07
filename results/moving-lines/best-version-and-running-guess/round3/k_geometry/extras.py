import sys
from collections import Counter, defaultdict
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from i_followups import load, first_service_t, line_of, pos, ARMS
from j_decompose import goal_check
sel, eps = load("runs_pilot3/dsv41")
for arm in ARMS:
    c = defaultdict(Counter)
    for lay in sel:
        e = eps.get((lay["name"], arm)); t0 = first_service_t(e) if e else None
        if t0 is None: continue
        line = line_of(lay["name"])
        for r in (r for r in e["resolves"] if r.get("status") == "ok" and r.get("pose_world") and r["sent_t"] >= t0):
            tr = r["truth"]
            if not tr["ahead"] or r.get("last") != tr["last"]: continue
            p = pos(r)
            # people the solver had to come after (roster for threshold arms; CT5: roster nearer than choice + choice)
            ros = set(r.get("roster") or [])
            extra = [i for i in ros if i not in tr["order"]]
            off_extra = [i for i in extra if i in p and line.project(*p[i])[1] > 0.6]
            key = "no extra" if not extra else ("extra, all on axis" if not off_extra else "extra off axis")
            c[key][goal_check(r, line) == "ok"] += 1
    print(arm, "; ".join(f"{k}: n {v[True]+v[False]}, goal ok {v[True]/max(v[True]+v[False],1):.2f}"
                          for k, v in sorted(c.items())))
