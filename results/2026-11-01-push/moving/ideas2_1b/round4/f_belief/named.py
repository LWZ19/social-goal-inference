"""Round 4, step F2 (groundwork, post hoc): high-P non-waiters split by whether the answer named them as waiting
(`answer_come_after`), and how long a named walker stays at P >= 0.5. Zero calls."""
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "round3"))
from i_followups import load, pos  # noqa: E402

sel, eps = load("runs_pilot3/dsv41")
for arm in ("CT4", "W0"):
    c = Counter()
    named_people = Counter()
    hi_time = defaultdict(float)
    for lay in sel:
        e = eps.get((lay["name"], arm))
        if not e:
            continue
        bys = set(e.get("bystanders") or [])
        rs = [r for r in e["resolves"] if r.get("status") == "ok" and isinstance(r.get("marginals"), dict)]
        named = {row[1] for row in (rs[0].get("answer_come_after") or []) if row[1] is not None and row[2] is not None and row[2] <= 0.5} if rs else set()
        order0 = set(rs[0]["truth"]["order"]) if rs else set()
        for i in named:
            named_people["waiter" if i in order0 else "bystander" if i in bys else "walker"] += 1
        for r in rs:
            order, p = set(r["truth"]["order"]), pos(r)
            for k, v in r["marginals"].items():
                i = int(k)
                if i not in p or i in order or v < 0.5:
                    continue
                cls = "bystander" if i in bys else "walker"
                c[(cls, i in named)] += 1
                hi_time[(lay["name"], i, cls, i in named)] += 0.5
    print(f"## {arm}")
    print(f"- people the answer named as waiting, by truth at the first re-solve: {dict(named_people)}")
    for cls in ("bystander", "walker"):
        tot = c[(cls, True)] + c[(cls, False)]
        print(f"- {cls} ticks at P >= 0.5: {tot}; named by the answer {c[(cls, True)]} ({c[(cls, True)] / max(tot, 1):.2f}), "
              f"not named {c[(cls, False)]}")
        for nm in (True, False):
            v = [s for (l, i, k, n), s in hi_time.items() if k == cls and n == nm]
            if v:
                print(f"  - {'named' if nm else 'not named'}: {len(v)} people, seconds at P >= 0.5 per person: median "
                      f"{statistics.median(v):.1f}, max {max(v):.1f}")
    print()
