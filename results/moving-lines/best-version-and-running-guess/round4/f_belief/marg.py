"""Round 4, step F (groundwork, post hoc, pilot design data): CT4's and W0's logged P(waits) by who people really
are (waiter / bystander / walker), and for non-waiters with P >= 0.5: are they moving, where do they stand
relative to the true tail? Zero calls."""
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "round3"))
from i_followups import load, line_of, pos  # noqa: E402

for arm in ("CT4", "W0"):
    sel, eps = load("runs_pilot3/dsv41")
    P = defaultdict(list)
    hi = Counter()
    feats = defaultdict(list)
    for lay in sel:
        e = eps.get((lay["name"], arm))
        if not e:
            continue
        line = line_of(lay["name"])
        bys = set(e.get("bystanders") or [])
        rs = [r for r in e["resolves"] if r.get("status") == "ok" and isinstance(r.get("marginals"), dict)]
        prev = None
        for r in rs:
            tr, p = r["truth"], pos(r)
            order = set(tr["order"])
            tail = max((line.project(*p[i])[0] for i in order if i in p), default=None)
            for k, v in r["marginals"].items():
                i = int(k)
                if i not in p:
                    continue
                cls = "waiter" if i in order else "bystander" if i in bys else "walker"
                P[cls].append(v)
                if cls != "waiter" and v >= 0.5:
                    hi[cls] += 1
                    al, side = line.project(*p[i])
                    sp = None
                    if prev is not None and i in prev[1]:
                        sp = math.dist(p[i], prev[1][i]) / max(r["sent_t"] - prev[0], 1e-6)
                    feats[cls].append((side, None if tail is None else al - tail, sp))
            prev = (r["sent_t"], p)
    print(f"## {arm}\n")
    for cls in ("waiter", "bystander", "walker"):
        v = P[cls]
        print(f"- {cls}: n {len(v)}, mean P {statistics.mean(v):.2f}, share P >= 0.5 {sum(x >= 0.5 for x in v) / len(v):.2f}, "
              f"share 0.2 <= P < 0.5 {sum(0.2 <= x < 0.5 for x in v) / len(v):.2f}")
    for cls in ("bystander", "walker"):
        f = feats[cls]
        sides = [a for a, b, c in f]
        beh = [b for a, b, c in f if b is not None]
        sps = [c for a, b, c in f if c is not None]
        print(f"- {cls} at P >= 0.5 ({hi[cls]} ticks): side from axis median {statistics.median(sides):.2f} m; "
              f"along minus true tail median {statistics.median(beh):.2f} m (share behind the tail "
              f"{sum(b > 0 for b in beh) / max(len(beh), 1):.2f}); speed median {statistics.median(sps):.2f} m/s, "
              f"share moving > 0.3 m/s {sum(s > 0.3 for s in sps) / max(len(sps), 1):.2f}")
    print()
