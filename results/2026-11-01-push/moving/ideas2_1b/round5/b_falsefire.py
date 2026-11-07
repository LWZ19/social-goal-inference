"""Round 5, check 1 (ROUND5.md 13eecdfd): share of walking bouts that break the step-up bounds at any point (running
path > 2.0 m, or running net change in distance to the target > +0.5 m), by who walks. Bar: at most 5% of
"member, stays" bouts. Same bout definition as a_bouts.py. Zero calls."""
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "round3"))
from r3lib import design_cells  # noqa: E402
from g_pilot_predict import pilot_cells  # noqa: E402

PATH_M, AWAY_M = 2.0, 0.5


def fires(cell):
    fr, dt, tgt = cell.dump["frames"], cell.dump["every_s"], cell.target
    pos, inq = defaultdict(list), defaultdict(list)
    for f in fr:
        for p in f["people"]:
            pos[p["id"]].append((f["t"], tuple(p["xy"])))
            inq[p["id"]].append(bool(p.get("in_queue")))
    lag = max(1, int(round(1.0 / dt)))
    out = []
    for i, h in pos.items():
        sp = [0.0] * lag + [math.dist(h[k][1], h[k - lag][1]) / (h[k][0] - h[k - lag][0]) for k in range(lag, len(h))]
        k, ever = 0, False
        while k < len(h):
            ever = ever or inq[i][k]
            if sp[k] <= 0.3:
                k += 1
                continue
            s0 = max(k - lag, 0)
            a = h[s0][1]
            path, fired, why = 0.0, False, None
            j = s0 + 1
            while k < len(h) and sp[k] > 0.3:
                while j <= k:
                    path += math.dist(h[j][1], h[j - 1][1])
                    j += 1
                net = math.dist(h[k][1], tgt) - math.dist(a, tgt)
                if not fired and (path > PATH_M or net > AWAY_M):
                    fired, why = True, "path" if path > PATH_M else "away"
                k += 1
            e = k - 1
            start_m, end_m = inq[i][s0], inq[i][e]
            kind = ("member, stays" if start_m and end_m else "member, leaves" if start_m else
                    "joins" if end_m else "ex-member" if ever else "never a member")
            out.append((kind, fired, why))
    return out


def main():
    for label, cs in (("design cells", design_cells()), ("pilot layouts", [c for _, c in pilot_cells("runs_pilot3/dsv41")[0]])):
        n, f, w = Counter(), Counter(), Counter()
        for c in cs:
            for kind, fired, why in fires(c):
                n[kind] += 1
                f[kind] += fired
                if fired:
                    w[(kind, why)] += 1
        print(f"## {label}")
        for kind in ("member, stays", "member, leaves", "joins", "ex-member", "never a member"):
            if n[kind]:
                print(f"- {kind}: {f[kind]}/{n[kind]} bouts break a bound ({f[kind] / n[kind]:.3f}); by path "
                      f"{w[(kind, 'path')]}, by away {w[(kind, 'away')]}")
        ok = f["member, stays"] / max(n["member, stays"], 1) <= 0.05
        print(f"- bar (member bouts at most 5%): {'PASS' if ok else 'FAIL'}\n")


if __name__ == "__main__":
    main()
