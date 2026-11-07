"""Round 5, step A (design data; descriptive, to check a first-principles definition, not to fit it): walking bouts
in the robot-idle dumps of the round 3 design cells and the 24 pilot layouts. A bout = consecutive 0.5 s frames
with speed (over 1 s) above 0.3 m/s. For each bout: who (a queue member at its start and end, by the dump's truth;
an ex-member once out; or never a member), duration, net change in distance to the target point, and the path
length. Zero calls."""
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "round3"))
from r3lib import design_cells  # noqa: E402
from g_pilot_predict import pilot_cells  # noqa: E402


def bouts(cell):
    fr = cell.dump["frames"]
    dt = cell.dump["every_s"]
    tgt = cell.target
    pos = defaultdict(list)
    inq = defaultdict(list)
    for f in fr:
        for p in f["people"]:
            pos[p["id"]].append((f["t"], tuple(p["xy"])))
            inq[p["id"]].append(bool(p.get("in_queue")))
    out = []
    for i, h in pos.items():
        lag = max(1, int(round(1.0 / dt)))
        sp = [0.0] * lag + [math.dist(h[k][1], h[k - lag][1]) / (h[k][0] - h[k - lag][0]) for k in range(lag, len(h))]
        k = 0
        ever = False
        while k < len(h):
            ever = ever or inq[i][k]
            if sp[k] <= 0.3:
                k += 1
                continue
            s = k
            while k < len(h) and sp[k] > 0.3:
                k += 1
            e = k - 1
            a, b = h[max(s - lag, 0)][1], h[e][1]
            d_net = math.dist(b, tgt) - math.dist(a, tgt)
            length = sum(math.dist(h[j][1], h[j - 1][1]) for j in range(max(s - lag, 0) + 1, e + 1))
            start_m, end_m = inq[i][max(s - lag, 0)], inq[i][e]
            kind = ("member, stays" if start_m and end_m else "member, leaves" if start_m else
                    "joins" if end_m else "ex-member" if ever else "never a member")
            out.append(dict(kind=kind, dur=(e - max(s - lag, 0)) * dt, d_net=d_net, length=length))
    return out


def main():
    rows = defaultdict(list)
    for c in design_cells():
        for b in bouts(c):
            rows[b["kind"]].append(b)
    cs, _ = pilot_cells("runs_pilot3/dsv41")
    for _, c in cs:
        for b in bouts(c):
            rows["pilot: " + b["kind"]].append(b)
    print("| who | bouts | duration median (p90) s | net change in distance to target, median (p10, p90) m | path median m |")
    print("|---|---|---|---|---|")
    for kind in sorted(rows):
        v = rows[kind]
        q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))]
        dur = [b["dur"] for b in v]
        dn = [b["d_net"] for b in v]
        ln = [b["length"] for b in v]
        print(f"| {kind} | {len(v)} | {statistics.median(dur):.1f} ({q(dur, 0.9):.1f}) | {statistics.median(dn):+.2f} "
              f"({q(dn, 0.1):+.2f}, {q(dn, 0.9):+.2f}) | {statistics.median(ln):.2f} |")


if __name__ == "__main__":
    main()
