import sys, statistics
from collections import Counter
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from i_followups import load, first_service_t, line_of, pos, in_line
from j_decompose import goal_check
sel, eps = load("runs_pilot3/dsv41")
Z = ["cafe_1008", "sweet_1092", "cafe_1001", "cafe_1081", "sweet_1011", "cafe_1090", "sweet_1112", "sweet_1003"]
for name in Z:
    e = eps[(name, "CT3F")]; line = line_of(name); t0 = first_service_t(e)
    rs = [r for r in e["resolves"] if r.get("status") == "ok" and r.get("pose_world")]
    g = Counter(); lastc = Counter(); why = Counter(); joined = Counter()
    for s in (x for x in e["samples"] if t0 is not None and x["t"] >= t0 - 1e-9):
        r = max((r for r in rs if r["sent_t"] <= s["t"] + 1e-9), key=lambda r: r["sent_t"], default=None)
        if r is None: continue
        g[goal_check(r, line)] += 1
        tr = r["truth"]; L = r.get("last")
        lastc["none" if L is None else "right" if L == tr["last"] else "waiter" if L in tr["order"] else "non-waiter"] += 1
        joined[s["joined"]] += 1
        if s["ahead"] and not s["in_line"]:
            why["behind<0" if s["behind_m"] < 0 else f"gap {s['gap_m']:.1f}" if not 0.5 <= s["gap_m"] <= 1.2 else
                "heading" if s["heading_off_deg"] > 45 else "nearest" if s["nearest_m"] < 0.55 else "?"] += 1
    print(f"{name}: first service {t0}; goal {dict(g.most_common(3))}; last {dict(lastc)}; joined {dict(joined)}; "
          f"sample fails {dict(why.most_common(4))}")
