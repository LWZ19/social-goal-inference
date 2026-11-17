"""Round 4, step E: is the CT4P port exact? For every CT4P re-solve with direction_source "path", recompute the
remembered path's back tangent from the logged roster and tracks (Resolver with path_memory True, line_span_m 0.5,
built from the cached answer and the robot-idle dump's tracks at the photo) and compare it with the logged
solve_direction. Reads only the direction inputs and outputs, no outcome measure. Zero calls."""
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b_pl import Path_  # noqa: E402
from g_pilot_predict import pilot_cells  # noqa: E402


def main():
    cs, missing = pilot_cells("runs_pilot3/dsv41", "CT4P")
    print(f"# CT4P port check: {len(cs)} episodes ({len(missing)} missing: {missing})\n")
    tot = Counter()
    for lay, c in cs:
        path = Path_(c)
        k = Counter()
        for r in c.episode["resolves"]:
            if r.get("status") != "ok":
                continue
            src, d = r.get("direction_source"), r.get("solve_direction")
            pos = {int(i): (x, y) for i, x, y in r["tracks"]}
            pts = [pos[i] for i in (r.get("roster") or []) if i in pos]
            mine = None
            if path.r._path is not None and pts:
                mine = path.r._on_path(max(pts, key=path.r._along))[2]
            if src == "path":
                k["path"] += 1
                if mine is None:
                    k["mine none"] += 1
                elif math.dist(mine, d) < 1e-3:
                    k["same"] += 1
            else:
                k["belief"] += 1
                k["belief, mine none"] += mine is None
        tot.update(k)
        print(f"- {lay['name']:12} path {k['path']}, same {k['same']}, mine none {k['mine none']}; belief {k['belief']} "
              f"(of which I also have none {k['belief, mine none']})")
    print(f"\nTotal: path {tot['path']}, same {tot['same']} ({tot['same'] / max(tot['path'], 1):.3f}); belief "
          f"{tot['belief']}, of which I also have none {tot['belief, mine none']}")


if __name__ == "__main__":
    main()
