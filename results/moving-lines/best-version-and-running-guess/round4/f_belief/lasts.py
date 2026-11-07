"""Round 4, step F4 (groundwork, post hoc): the solver's non-waiting lasts (ticks with someone ahead), split as in
exmember.py: ex-member / bystander named or not / walker named or not. Zero calls."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "round3"))
from i_followups import load  # noqa: E402

sel, eps = load("runs_pilot3/dsv41")
for arm in ("CT3F", "CT4", "W0", "CT5"):
    c = Counter()
    n = 0
    for lay in sel:
        e = eps.get((lay["name"], arm))
        if not e:
            continue
        bys = set(e.get("bystanders") or [])
        rs = [r for r in e["resolves"] if r.get("status") == "ok"]
        named = {row[1] for row in (rs[0].get("answer_come_after") or []) if rs and row[1] is not None
                 and row[2] is not None and row[2] <= 0.5} if rs else set()
        ever = set()
        for r in rs:
            order = set(r["truth"]["order"])
            ever |= order
            if not r["truth"]["ahead"]:
                continue
            n += 1
            last = r.get("last")
            if last is None or last in order:
                continue
            kind = ("ex-member" if last in ever else
                    ("bystander" if last in bys else "walker") + (", named" if last in named else ", not named"))
            c[kind] += 1
    tot = sum(c.values())
    print(f"{arm}: non-waiting last on {tot} of {n} ticks ({tot / max(n, 1):.3f}); " +
          ", ".join(f"{k} {v / max(n, 1):.3f}" for k, v in c.most_common()))
