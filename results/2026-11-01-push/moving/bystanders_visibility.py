"""How much of each person the robot sees from its start pose, from the scenario file alone (no sim).

    python3 bystanders_visibility.py ../../../../sim/scenarios/cafe_line2_bystanders.yaml [...]

People are discs of radius 0.22 m; a person is hidden where a nearer disc covers the same bearings. Queue members
stand in the first slots in file order; wanderers are shown at their start (they move). Also prints, for each
bystander, the distance to the line's last member and whether the geometry rule (nearest to the target, chain within
1 m) would take them into the line at the start.
"""
import math
import sys

import yaml

R, HALF_FOV = 0.22, 40.0


def bearing(o, p):
    return math.degrees(math.atan2(p[1] - o[1], p[0] - o[0]))


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def visible(o, people, i):
    """Share of person i's bearing interval inside the view and not behind a nearer person."""
    p, d = people[i][1], math.dist(o, people[i][1])
    b, w = bearing(o, p), math.degrees(math.asin(min(1.0, R / d)))
    lo, hi = b - w, b + w
    n, seen = 60, 0
    for k in range(n):
        a = lo + (hi - lo) * (k + 0.5) / n
        hidden = False
        for j, (_, q, _) in enumerate(people):
            if j == i or math.dist(o, q) >= d:
                continue
            bj, wj = bearing(o, q), math.degrees(math.asin(min(1.0, R / math.dist(o, q))))
            if abs(wrap(a - bj)) <= wj:
                hidden = True
                break
        seen += not hidden
    return seen / n


def main(paths):
    for path in paths:
        sc = yaml.safe_load(open(path))
        rob = (sc["robot"]["x"], sc["robot"]["y"])
        facing = bearing(rob, sc["robot"]["face"])
        people, line = [], []
        used = {name: 0 for name in sc.get("queues", {})}
        for h in sc["humans"]:
            if h.get("queue"):
                s = sc["queues"][h["queue"]]["slots"][used[h["queue"]]]
                used[h["queue"]] += 1
                people.append((h["avatar"] + " (line)", (s["x"], s["y"]), "line"))
                line.append((s["x"], s["y"]))
            elif h.get("stand") is not None:
                kind = "bystander" if h.get("bystander") else "stand"
                people.append((h["avatar"] + f" ({kind})", tuple(h["stand"]), kind))
            else:
                people.append((h["avatar"] + " (wanders)", tuple(h["start"]), "wander"))
        print(f"\n{path.split('/')[-1]}: robot {rob}, facing {facing:.0f} deg")
        for i, (name, p, kind) in enumerate(people):
            off = wrap(bearing(rob, p) - facing)
            in_view = abs(off) <= HALF_FOV
            v = visible(rob, people, i) if in_view else 0.0
            extra = ""
            if kind in ("bystander", "stand") and line:
                extra = f"  {math.dist(p, line[-1]):.2f} m from the last in line"
            print(f"  {name:24s} at ({p[0]:5.2f}, {p[1]:5.2f})  {off:+6.1f} deg  {math.dist(rob, p):4.1f} m  "
                  f"seen {v:4.0%}{extra}")


if __name__ == "__main__":
    main(sys.argv[1:])
