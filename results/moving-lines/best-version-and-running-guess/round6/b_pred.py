"""Round 6 part B (ROUND6_B.md), EXPLORATORY: committing in the wrong place, and a counterfactual frozen predecessor
on CT4PJ's logged joined stretches. Zero calls; read-only.

    .venv/bin/python round6/b_pred.py SELECTION_JSON RUN_ROOT [RUN_ROOT ...]
    (from ideas2_1b/; paths relative to moving/.)
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "round3"))
from i_followups import MOVING  # noqa: E402

WALK_MPS = 0.3
AWAY_M = 0.5
BACK_S = 1.0


def find(root, lay, arm):
    hits = sorted((MOVING / root).glob(f"**/{lay['name']}/seed{lay['seed']}/{arm}/episode.json"))
    return hits[0] if hits else None


def ok_resolves(e):
    return sorted((r for r in e["resolves"] if r.get("status") == "ok" and r.get("truth") is not None
                   and r.get("delivered_t") is not None), key=lambda r: r["delivered_t"])


def stretches(rs):
    out, cur = [], []
    for r in rs:
        if r.get("joined"):
            cur.append(r)
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def tally(rows):
    """rows: [(t, choice, truth)] over joined re-solves -> counts."""
    n = len(rows)
    agree = [(c, tr) for _, c, tr in rows if tr.get("joined")]
    right_agree = sum(c is not None and c == tr.get("last") for c, tr in agree)
    right_all = sum(c is not None and c == tr.get("last") for _, c, tr in rows)
    return n, len(agree), right_agree, right_all


def switches(stretch_choices):
    """[(t, choice)] per stretch -> (identity changes, minutes)."""
    sw, mins = 0, 0.0
    for ch in stretch_choices:
        sw += sum(1 for a, b in zip(ch, ch[1:]) if a[1] != b[1])
        if len(ch) > 1:
            mins += (ch[-1][0] - ch[0][0]) / 60
        mins += 0.5 / 60          # each re-solve stands for its 0.5 s
    return sw, mins


def frozen(stretch, tgt):
    """Counterfactual choices along one CT4PJ stretch."""
    first = stretch[0]
    pos0 = {int(i): (x, y) for i, x, y in first["tracks"]}
    p = first.get("last")
    order = sorted((i for i in first.get("roster") or [] if i in pos0), key=lambda i: math.dist(pos0[i], tgt))
    hist, bout, out = {}, {}, []
    for r in stretch:
        t = r["delivered_t"]
        pos = {int(i): (x, y) for i, x, y in r["tracks"]}
        for i, xy in pos.items():
            h = hist.setdefault(i, [])
            prev = h[-1] if h else None
            h.append((t, xy))
            walking = prev is not None and t > prev[0] and math.dist(prev[1], xy) / (t - prev[0]) > WALK_MPS
            if not walking:
                bout.pop(i, None)
                continue
            b = bout.get(i)
            if b is None:
                back = [q for q in h if q[0] <= t - BACK_S + 1e-9]
                b = bout[i] = {"a": (back[-1][1] if back else h[0][1]), "broken": False}
            if math.dist(xy, tgt) - math.dist(b["a"], tgt) > AWAY_M:
                b["broken"] = True
        while p is not None and (p not in pos or bout.get(p, {}).get("broken")):
            ahead = [i for i in order if i != p and pos0.get(i) is not None and p in pos0
                     and math.dist(pos0[i], tgt) < math.dist(pos0[p], tgt) and i in pos
                     and not bout.get(i, {}).get("broken")]
            order = [i for i in order if i != p]
            p = max(ahead, key=lambda i: math.dist(pos0[i], tgt)) if ahead else None
        out.append((t, p))
    return out


def report(root, sel):
    print(f"\n## {root}\n")
    res = {}
    for arm in ("CT3F", "CT4PJ", "frozen"):
        res[arm] = {"rows": [], "ch": [], "eps": 0, "eps_wrong": 0}
    for lay in sel:
        for arm in ("CT3F", "CT4PJ"):
            f = find(root, lay, arm)
            if f is None:
                continue
            e = json.loads(f.read_text())
            rs = ok_resolves(e)
            st = stretches(rs)
            res[arm]["eps"] += 1
            res[arm]["eps_wrong"] += any(not r["truth"].get("joined") for s in st for r in s)
            for s in st:
                res[arm]["rows"] += [(r["delivered_t"], r.get("last"), r["truth"]) for r in s]
                res[arm]["ch"].append([(r["delivered_t"], r.get("last")) for r in s])
            if arm == "CT4PJ":
                a = f.parent / "001" / "answer.json"
                ans = json.loads(a.read_text()) if a.exists() else {}
                if ans.get("target_world") is None:
                    continue
                tgt = tuple(ans["target_world"][:2])
                for s in st:
                    ch = frozen(s, tgt)
                    res["frozen"]["rows"] += [(t, c, r["truth"]) for (t, c), r in zip(ch, s)]
                    res["frozen"]["ch"].append(ch)
    print("| choice | joined re-solves | sim says not in line | episodes with any | right last, sim agrees | "
          "right last, all joined | switches per minute joined |")
    print("|---|---|---|---|---|---|---|")
    for arm, x in res.items():
        n, na, ra, rall = tally(x["rows"])
        sw, mins = switches(x["ch"])
        eps = f"{x['eps_wrong']}/{x['eps']}" if arm != "frozen" else "(CT4PJ's)"
        print(f"| {arm} | {n} | {(n - na) / max(n, 1):.3f} | {eps} | {ra / max(na, 1):.3f} | {rall / max(n, 1):.3f} | "
              f"{sw / max(mins, 1e-9):.2f} |")


def main():
    sel = json.loads((MOVING / sys.argv[1]).read_text())["layouts"]
    print(f"# Round 6 part B, EXPLORATORY: committing in the wrong place, and a frozen predecessor ({sys.argv[1]})")
    for root in sys.argv[2:]:
        report(root, sel)


if __name__ == "__main__":
    main()
