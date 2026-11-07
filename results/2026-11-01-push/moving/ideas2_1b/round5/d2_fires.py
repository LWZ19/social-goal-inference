"""Round 5, exploratory diagnosis: QB v2's broken bouts on the pilot layouts (robot idle), by the person's status in
the dump's queue at the moment the bound broke: member, not yet a member but in the queue within the next 15 s (a
joiner on the way), ex-member, never a member. Zero calls."""
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent / "round4", HERE.parent / "round3", HERE.parent):
    sys.path.insert(0, str(p))
from g_pilot_predict import pilot_cells  # noqa: E402
from qb import QB  # noqa: E402
from r3lib import ticks_of  # noqa: E402
from replay_lib import dump_truth  # noqa: E402

cs, _ = pilot_cells("runs_pilot3/dsv41")
k = Counter()
for lay, c in cs:
    b = QB(c, {"BOUT_PATH_M": float("inf")})
    for t, tracks, _ in ticks_of(c, None):
        b.step(t, tracks, 0.5)
    ever = set()
    for tb, i in b.log_broken:
        q, _ = dump_truth(c.dump, tb)
        past = set()
        for f in c.dump["frames"]:
            if f["t"] <= tb:
                past |= {p["id"] for p in f["people"] if p.get("in_queue")}
        soon = any(i in dump_truth(c.dump, tb + dt)[0] for dt in (2, 4, 6, 8, 10, 12, 15))
        kind = ("member" if i in q else "joiner on the way" if soon else "ex-member" if i in past else "never a member")
        k[kind] += 1
print(dict(k))
