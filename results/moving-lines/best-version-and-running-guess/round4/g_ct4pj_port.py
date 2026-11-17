"""Round 4, step G: is the CT4PJ port right? Reads only the join and "behind" fields and the tracks; no outcome
measure. Zero calls.
- On re-solves where the arm logs joined: rebuild "behind" (tracked people whose arc length on the geometry
  Resolver's path exceeds the robot's, or distance to the target without a path). The robot's position is
  interpolated from the 1-s samples, so boundary cases are expected. Compare with the logged `behind`.
- Check that the solver's last is never a logged "behind" id.
- J1's violations (sim joined, last more than 0.25 m behind the robot along the sim's line), by the arm's flag.
"""
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "round3"))
from b_pl import Path_  # noqa: E402
from g_pilot_predict import pilot_cells  # noqa: E402
from i_followups import line_of  # noqa: E402
from replay_lib import robot_at  # noqa: E402


def main():
    cs, _ = pilot_cells("runs_pilot3/dsv41", "CT4PJ")
    k = Counter()
    viol = Counter()
    for lay, c in cs:
        path, line, S = Path_(c), line_of(lay["name"]), c.episode["samples"]
        for r in c.episode["resolves"]:
            if r.get("status") != "ok":
                continue
            pos = {int(i): (x, y) for i, x, y in r["tracks"]}
            beh = set(r.get("behind") or [])
            if r.get("last") is not None and r["last"] in beh:
                k["last is behind (should never happen)"] += 1
            rob = robot_at(c.episode, r["sent_t"])
            if r.get("joined"):
                k["joined"] += 1
                if path.r._path is not None:
                    mine = path.r._along(rob)
                    rebuilt = {i for i, xy in pos.items() if path.r._along(xy) > mine + 1e-9}
                else:
                    tg = c.target
                    mine = math.dist(rob, tg)
                    rebuilt = {i for i, xy in pos.items() if math.dist(xy, tg) > mine + 1e-9}
                if rebuilt == beh:
                    k["behind same"] += 1
                else:
                    k["behind differs"] += 1
                    k["differs by one person"] += len(rebuilt ^ beh) == 1
            s = min(S, key=lambda x: abs(x["t"] - r["sent_t"]))
            if s["joined"] and abs(s["t"] - r["sent_t"]) <= 0.6:
                last = r.get("last")
                if last is not None and last in pos and line.project(*pos[last])[0] > s["along"] + 0.25:
                    viol[(lay["name"], bool(r.get("joined")))] += 1
    print(f"# CT4PJ port check\n\n{dict(k)}\n")
    print("J1 violations by layout and the arm's own joined flag:", dict(viol))


if __name__ == "__main__":
    main()
