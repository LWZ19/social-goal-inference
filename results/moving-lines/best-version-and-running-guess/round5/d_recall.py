"""Round 5, exploratory diagnosis: true queue members (present) at P >= 0.5 under W0 but < 0.5 under QB v2, robot
idle, pilot layouts. For each such member-tick: did QB mark a broken bout for that person within the last 10 s?
And how long had they been a member (since entering the dump's queue)? Zero calls."""
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent / "round4", HERE.parent / "round3", HERE.parent):
    sys.path.insert(0, str(p))
from g_pilot_predict import WIN, pilot_cells  # noqa: E402
from qb import QB  # noqa: E402
from ct5 import CT5  # noqa: E402
from r3lib import ticks_of  # noqa: E402
from replay_lib import dump_truth  # noqa: E402

cs, _ = pilot_cells("runs_pilot3/dsv41")
k = Counter()
since = []
for lay, c in cs:
    a, b = CT5(c, WIN), QB(c, {"BOUT_PATH_M": float("inf")})
    joined_at = {}
    for t, tracks, _ in ticks_of(c, None):
        a.step(t, tracks, 0.5)
        b.step(t, tracks, 0.5)
        q, _o = dump_truth(c.dump, t)
        for i in q:
            joined_at.setdefault(i, t)
        for i in list(joined_at):
            if i not in q:
                joined_at.pop(i)
        if t + 1e-9 < c.delivered_t:
            continue
        ma, mb = a.marginals(), b.marginals()
        present = {tr.id for tr in tracks}
        for i in q & present:
            if ma.get(i, 0) >= 0.5 and mb.get(i, 0) < 0.5:
                k["lost"] += 1
                recent = any(i == j and t - 10.0 <= tb <= t for tb, j in b.log_broken)
                k["with a broken bout in the last 10 s"] += recent
                since.append(t - joined_at[i])
                k["member for under 10 s"] += (t - joined_at[i]) < 10.0
                k["named" if i in a.named else "not named"] += 1
            elif ma.get(i, 0) >= 0.5:
                k["kept"] += 1
print(dict(k))
if since:
    since.sort()
    print("seconds since joining the queue, median", since[len(since) // 2], "p90", since[int(0.9 * len(since))])
