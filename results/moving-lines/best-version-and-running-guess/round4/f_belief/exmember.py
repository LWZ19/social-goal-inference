"""Round 4, step F3 (groundwork, post hoc): non-waiters at P(waits) >= 0.5, split by history: ex-members (in the
sim's queue earlier in the episode, now out: served or left), never-members named by the answer, and never-members
not named. For ex-members: seconds from leaving the queue to P < 0.5. Zero calls."""
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "round3"))
from i_followups import load, pos  # noqa: E402

sel, eps = load("runs_pilot3/dsv41")
for arm in ("CT3F", "CT4", "W0", "CT5"):
    c = Counter()
    lag = []
    for lay in sel:
        e = eps.get((lay["name"], arm))
        if not e:
            continue
        bys = set(e.get("bystanders") or [])
        rs = [r for r in e["resolves"] if r.get("status") == "ok"]
        named = {row[1] for row in (rs[0].get("answer_come_after") or []) if rs and row[1] is not None
                 and row[2] is not None and row[2] <= 0.5} if rs else set()
        ever, left_at, dropped = set(), {}, set()
        for r in rs:
            order, p = set(r["truth"]["order"]), pos(r)
            for i in list(ever):
                if i not in order and i not in left_at:
                    left_at[i] = r["sent_t"]
            for i in order:
                ever.add(i)
                left_at.pop(i, None)
            if isinstance(r.get("marginals"), dict):
                hi = {int(k) for k, v in r["marginals"].items() if v >= 0.5}
            else:                                            # CT3F: its roster
                hi = set(r.get("roster") or [])
            for i in hi:
                if i not in p or i in order:
                    continue
                kind = ("ex-member" if i in ever else
                        ("bystander" if i in bys else "walker") + (", named" if i in named else ", not named"))
                c[kind] += 1
            for i, t in left_at.items():
                if i in p and i not in hi and (i, t) not in dropped:
                    dropped.add((i, t))
                    lag.append(r["sent_t"] - t)
    tot = sum(c.values())
    print(f"{arm}: non-waiter ticks counted as waiting {tot}; " +
          ", ".join(f"{k} {v} ({v / max(tot, 1):.2f})" for k, v in c.most_common()) +
          (f"; ex-member seconds to drop: median {statistics.median(lag):.1f} (n {len(lag)})" if lag else ""))
