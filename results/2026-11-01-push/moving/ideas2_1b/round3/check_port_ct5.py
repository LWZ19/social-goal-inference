"""Closed-loop reproduction check for the CT5 and W0 ports (ROUND3.md section 8, item 4). Zero calls.

For every CT5 / W0 pilot episode: the wait from the robot-idle dump at the episode's `history_t` (before the first
re-solve's send time, as CT4_FROZEN section 5's check), then each re-solve's logged tracks at its send time, fed to
ct5.CT5 with W4's options. Compared with the log per re-solve:
- CT5: `choice` (ct5.decide with the per-session choice state), the roster, the largest marginal difference, and
  the solver's last (sgi.solver._turn given come_after = roster members nearer the target than the choice, plus the
  choice, everyone else exempt);
- W0: the roster, the largest marginal difference, and the last (the threshold roster through _turn).
Bar: agreement >= 0.95 on the choice (CT5) and on the last (both). The first 10 disagreements per arm are listed.

    .venv/bin/python round3/check_port_ct5.py runs_pilot3/dsv41      (from ideas2_1b/)
"""
import sys
from collections import defaultdict

import numpy as np

from r3lib import UNSET, decide, robot_at, turn_last  # first: it sets the import path
from ct5 import CT5  # noqa: E402
from g_pilot_predict import WIN, pilot_cells  # noqa: E402
from replay_lib import dump_tracks  # noqa: E402
from sgi.resolve import Track  # noqa: E402


def wait_ticks(c):
    rs = [r for r in c.episode.get("resolves", []) if r.get("delivered_t") is not None and r.get("tracks")]
    first_sent = rs[0]["sent_t"] if rs else c.delivered_t
    ts = c.episode.get("history_t")
    if not ts:
        ts, k = [], 0
        while True:
            t = round(c.asked_t + (1.0 / 30.0 if k == 0 else 0.5 * k), 3)
            if t >= first_sent - 1e-9:
                break
            ts.append(t)
            k += 1
    return [(t, dump_tracks(c.dump, t)) for t in ts if t < first_sent - 1e-9]


def check(c, arm):
    est = CT5(c, WIN)
    rs = [r for r in c.episode["resolves"] if r.get("delivered_t") is not None and r.get("tracks")]
    ticks = [(t, tr, None) for t, tr in wait_ticks(c)] + \
            [(r["sent_t"], [Track(id=int(i), xy=(x, y)) for i, x, y in r["tracks"]], r) for r in rs]
    rows, prev, cur = [], None, UNSET
    for t, tracks, logged in ticks:
        dt = 0.5 if prev is None else max(t - prev, 1e-6)
        est.step(t, tracks, dt)
        prev = t
        if logged is None:
            continue
        m = est.marginals()
        present = {tr.id for tr in tracks}
        roster = {i for i, v in m.items() if v >= 0.5 and i in present}
        robot = robot_at(c.episode, t)
        if arm == "CT5":
            cur, _ = decide(est, present, cur)
            if cur is None:
                last = None
            else:
                dj = est.dist_now[cur]
                last = turn_last({i for i in roster if est.dist_now.get(i, np.inf) < dj} | {cur}, tracks, est, robot)
            lg_choice = logged.get("choice", "absent")
            same_choice = (cur == lg_choice) if lg_choice != "absent" else None
        else:
            last = turn_last(roster, tracks, est, robot)
            same_choice = None
        lg_m = logged.get("marginals")
        dm = max((abs(m.get(int(k), 0.0) - float(v)) for k, v in lg_m.items()), default=0.0) \
            if isinstance(lg_m, dict) else None
        lg_roster = logged.get("roster")
        same_roster = None if lg_roster is None else (set(int(i) for i in lg_roster) == roster)
        rows.append(dict(t=t, same_choice=same_choice, same_roster=same_roster, dm=dm,
                         same_last=(last == logged.get("last", "absent")), mine=(cur if arm == "CT5" else last),
                         logged=(logged.get("choice") if arm == "CT5" else logged.get("last"))))
    return rows


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "runs_pilot3/dsv41"
    for arm in ("CT5", "W0"):
        cs, missing = pilot_cells(root, arm)
        agg = defaultdict(float)
        bad = []
        for lay, c in cs:
            rows = check(c, arm)
            for r in rows:
                agg["n"] += 1
                agg["last"] += r["same_last"]
                if r["same_choice"] is not None:
                    agg["nc"] += 1
                    agg["choice"] += r["same_choice"]
                if r["same_roster"] is not None:
                    agg["nr"] += 1
                    agg["roster"] += r["same_roster"]
                if r["dm"] is not None:
                    agg["dm"] = max(agg["dm"], r["dm"])
                key = r["same_choice"] if arm == "CT5" else r["same_last"]
                if key is False and len(bad) < 10:
                    bad.append((c.scenario, c.seed, round(r["t"], 2), r["mine"], r["logged"]))
            print(f"{arm:4} {c.scenario:12} s{c.seed:<6} re-solves {len(rows)}, same last "
                  f"{sum(r['same_last'] for r in rows)}" +
                  (f", same choice {sum(bool(r['same_choice']) for r in rows)}" if arm == "CT5" else ""))
        n = max(agg["n"], 1)
        verdict_key = ("choice", "nc") if arm == "CT5" else ("last", "n")
        share = agg[verdict_key[0]] / max(agg[verdict_key[1]], 1)
        print(f"\n{arm}: {len(cs)} episodes ({len(missing)} missing); same {verdict_key[0]} "
              f"{int(agg[verdict_key[0]])}/{int(agg[verdict_key[1]])} = {share:.3f} (bar >= 0.95): "
              f"{'PASS' if agg[verdict_key[1]] and share >= 0.95 else 'FAIL'}; same last {int(agg['last'])}/{int(n)}; "
              f"same roster {int(agg['roster'])}/{int(agg['nr'])}; largest marginal difference {agg['dm']:.4f}")
        for b in bad:
            print(f"  differs: {b}")
        print()


if __name__ == "__main__":
    main()
