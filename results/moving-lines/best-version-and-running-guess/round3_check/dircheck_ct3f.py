"""CT4P's failed direction check: does CT3F's own path rule pass it? Applies l_ct4p_score.direction_check (unchanged)
to CT3F episodes that log solve_direction (runs_pilot3/dsv41_dircheck), and CT4P alongside, per layout too."""
import statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "round3"))
from l_ct4p_score import load, angle, direction_check, HIGH
from i_followups import line_of, pos

for root, arm in (("runs_pilot3/dsv41_dircheck", "CT3F"), ("runs_pilot3/dsv41", "CT4P")):
    sel, eps = load(root, (arm,))
    med, fb, n = direction_check(sel, eps, arm)
    print(f"{arm} ({root}): median {med:.1f} deg over high-error layouts; re-solves {n}")
    for lay in sel:
        if lay["name"] not in HIGH or (lay["name"], arm) not in eps:
            continue
        line, a = line_of(lay["name"]), []
        for r in eps[(lay["name"], arm)]["resolves"]:
            d, tr, p = r.get("solve_direction"), r["truth"], pos(r)
            if r.get("status") != "ok" or not d or tr["last"] not in p:
                continue
            al, _ = line.project(*p[tr["last"]])
            k = max(0, min(len(line.length) - 1, sum(al >= s for s in line.start) - 1))
            a.append(angle(d, line.unit[k]))
        print(f"  {lay['name']:12} {statistics.median(a) if a else float('nan'):6.1f}  (n {len(a)})")
