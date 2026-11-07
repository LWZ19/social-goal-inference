"""CT4_FROZEN.md section 5: does each ported arm reproduce the frozen offline code on its own tracks? Zero calls.

For every CT3F / CT4 run under the given roots: the ticks are the wait (the run's logged `history` frames if present,
else the people dump at the photo + k x 0.5 s before delivery: exact while the robot is idle), then the run's own
logged re-solve tracks at their logged send times. They are fed to the frozen offline code (variants.CT3FixGated
through replay_lib's loop; ct4.CT4 through replay_ct4's decision), and the per-re-solve `last` is compared with the
log. Bar: agreement >= 0.95 of re-solves for each arm. For CT4 also: roster agreement and the largest marginal
difference. Disagreeing re-solves are listed (first 10 per arm).

    .venv/bin/python check_port.py runs_ideas2_check/dsv41 runs_ideas2_check/sonnet55_low
"""
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ct4 import CT4  # noqa: E402
from replay_lib import CT3, TH, cells, dump_tracks, robot_at, solve_roles  # noqa: E402
from sgi import solver as S  # noqa: E402
from sgi.resolve import Track  # noqa: E402
from variants import CT3FixGated  # noqa: E402


def wait_ticks(c):
    """The wait's frames as the sim sends them (controller, 3 Nov): frame 0 one sim step after the photo
    (asked_t + 1/30), then one at each asked_t + 0.5 k; times rounded to 3 decimals; only frames before the first
    re-solve's send time, strictly. The episode's `history_t` is used when logged. Tracks come from the people dump
    (exact while the robot is idle), interpolated to the frame time."""
    rs = [r for r in c.episode.get("resolves", []) if r.get("delivered_t") is not None and r.get("tracks")]
    first_sent = rs[0]["sent_t"] if rs else c.delivered_t
    ts = c.episode.get("history_t")
    src = "logged history_t"
    if not ts:
        src = "reconstructed schedule"
        ts, k = [], 0
        while True:
            t = round(c.asked_t + (1.0 / 30.0 if k == 0 else 0.5 * k), 3)
            if t >= first_sent - 1e-9:
                break
            ts.append(t)
            k += 1
    ts = [t for t in ts if t < first_sent - 1e-9]
    return [(t, dump_tracks(c.dump, t)) for t in ts], src


def live_ticks(c):
    rs = [r for r in c.episode["resolves"] if r.get("delivered_t") is not None and r.get("tracks")]
    return [(r["sent_t"], [Track(id=int(i), xy=(x, y)) for i, x, y in r["tracks"]], r) for r in rs]


def check_ct3f(c):
    wait, src = wait_ticks(c)
    at_answer = wait[0][1] if wait else dump_tracks(c.dump, c.asked_t)
    r = CT3FixGated(c.people, c.constraints, c.target, at_answer, **CT3)
    start = tuple(c.episode["samples"][0]["robot"][:2])
    rows = []
    for t, tracks, logged in [(t, tr, None) for t, tr in wait] + live_ticks(c):
        robot = robot_at(c.episode, t) if logged is not None else start
        people, cons = r.inputs(tracks, robot, t)
        if r.vacated_pose is not None:
            x, y, _ = r.vacated_pose
            r.record((x, y), True)
            last = "vacated"
        else:
            last_n, joined_n, place = solve_roles(r, people, cons, robot, c.target, r.exempt())
            if place is None:
                r.record(robot, False, [])
                last = None
            else:
                r.record(tuple(place), True, joined_n)
                last = r.track_of(last_n)
        if logged is not None:
            lg = logged.get("last", "absent")
            if last == "vacated" or lg == "absent":
                continue
            rows.append((t, last, lg, None, None))
    return rows, src


def check_ct4(c):
    wait, src = wait_ticks(c)
    est = CT4(c)
    rows = []
    prev = None
    for t, tracks, logged in [(t, tr, None) for t, tr in wait] + live_ticks(c):
        dt = 0.5 if prev is None else max(t - prev, 1e-6)
        est.step(t, tracks, dt)
        prev = t
        if logged is None:
            continue
        m = est.marginals()
        ids = {tr.id for tr in tracks}
        roster = {i for i, v in m.items() if v >= 0.5 and i in ids}
        last = None
        if roster:
            bodies = {tr.id + 1: S._Body(tr.id + 1, np.array(tr.xy, dtype=np.float64), tr.yaw) for tr in tracks}
            waiting = sorted(i + 1 for i in roster if i + 1 in bodies)
            if waiting:
                apart = set(bodies) - set(waiting)
                turn = S._turn(waiting, apart, bodies, None, est.tgt,
                               np.array(robot_at(c.episode, t), dtype=np.float64), TH, est.dir)
                last = turn.last - 1
        lg_m = logged.get("marginals")
        dm = None
        if isinstance(lg_m, dict):
            dm = max((abs(m.get(int(k), 0.0) - float(v)) for k, v in lg_m.items()), default=0.0)
        lg_roster = logged.get("roster")
        same_roster = None if lg_roster is None else (set(int(i) for i in lg_roster) == roster)
        rows.append((t, last, logged.get("last", "absent"), same_roster, dm))
    return rows, src


def main():
    roots = sys.argv[1:]
    agg = defaultdict(lambda: [0, 0, 0, 0, 0.0])
    bad = defaultdict(list)
    for arm, fn in (("CT3F", check_ct3f), ("CT4", check_ct4)):
        for c in cells(roots=roots, arm=arm, scenes=tuple({
                "cafe_line2_bystanders", "room_coffee_cart_bystanders", "sweet_spot_line2_bystanders",
                "cafe_line2_moving", "room_coffee_cart_slots", "sweet_spot_line2_moving"}), seeds=range(0, 100)):
            rows, src = fn(c)
            a = agg[arm]
            for t, mine, lg, same_roster, dm in rows:
                a[0] += 1
                a[1] += (mine == lg)
                if same_roster is not None:
                    a[2] += 1
                    a[3] += same_roster
                if dm is not None:
                    a[4] = max(a[4], dm)
                if mine != lg and len(bad[arm]) < 10:
                    bad[arm].append((c.model, c.scenario, c.seed, round(t, 2), mine, lg))
            print(f"{arm:5} {c.model:13} {c.scenario:28} s{c.seed:<3} wait from {src}: re-solves {len(rows)}, "
                  f"same last {sum(1 for r in rows if r[1] == r[2])}")
    for arm, (n, same, nr, sr, dm) in agg.items():
        verdict = "PASS" if n and same / n >= 0.95 else "FAIL"
        extra = f"; roster same {sr}/{nr}; largest marginal difference {dm:.4f}" if arm == "CT4" else ""
        print(f"\n{arm}: same last in {same}/{n} = {same / max(n, 1):.3f} (bar >= 0.95): {verdict}{extra}")
        for b in bad[arm]:
            print(f"  differs: {b}")


if __name__ == "__main__":
    main()
