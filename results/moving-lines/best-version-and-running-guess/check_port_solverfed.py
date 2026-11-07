"""Diagnostic for check_port.py's CT3F result: the same offline role logic (frozen variants.CT3FixGated), but at live
re-solves `record` gets the run's own solver output (logged pose_world and in_turn) instead of the place-in-turn
stand-in. If agreement rises to the bar, the port is right and the gap was the offline loop's approximation of the
full solve (which decides `joined`). Zero calls.

    .venv/bin/python check_port_solverfed.py runs_ideas2_check/dsv41 runs_ideas2_check/sonnet55_low
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_port as C  # noqa: E402
from replay_lib import CT3, cells, robot_at, solve_roles  # noqa: E402
from variants import CT3FixGated  # noqa: E402

agg = defaultdict(int)
for c in cells(roots=sys.argv[1:], arm="CT3F", seeds=range(0, 100),
               scenes=("cafe_line2_bystanders", "room_coffee_cart_bystanders", "sweet_spot_line2_bystanders")):
    wait, src = C.wait_ticks(c)
    r = CT3FixGated(c.people, c.constraints, c.target, wait[0][1], **CT3)
    start = tuple(c.episode["samples"][0]["robot"][:2])
    n = same = 0
    for t, tracks, lg in [(t, tr, None) for t, tr in wait] + C.live_ticks(c):
        robot = robot_at(c.episode, t) if lg is not None else start
        people, cons = r.inputs(tracks, robot, t)
        if r.vacated_pose is not None:
            x, y, _ = r.vacated_pose
            r.record((x, y), True)
            continue
        ln, jn, place = solve_roles(r, people, cons, robot, c.target, r.exempt())
        last = None if place is None else r.track_of(ln)
        if lg is None:
            r.record(robot if place is None else tuple(place), place is not None, jn)
            continue
        pose = lg.get("pose_world")
        r.record(tuple(pose[:2]) if pose else robot, bool(lg.get("in_turn")), jn)
        L = lg.get("last", "absent")
        if L == "absent":
            continue
        n += 1
        same += (last == L)
    agg["n"] += n
    agg["same"] += same
    print(f"{c.model:13} {c.scenario:28} s{c.seed:<3} re-solves {n}, same last {same}")
print(f"\nCT3F fed the run's own solver output: same last {agg['same']}/{agg['n']} = {agg['same'] / max(agg['n'], 1):.3f}")
