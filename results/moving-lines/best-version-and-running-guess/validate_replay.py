"""Does the offline replay reproduce the runs? Per re-solve, the replayed solver `last` (and roster drops) against the
logged ones, on each run's own logged tracks (CT3 and CT2 cells). Zero calls.

    .venv/bin/python validate_replay.py
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_lib import CT2, CT3, cells, replay  # noqa: E402


def main():
    for arm, opts in (("CT3", CT3), ("CT2", CT2)):
        agree = Counter()
        worst = []
        for c in cells(arm=arm):
            ticks = replay(c, opts, source="logged")
            same = n = 0
            for k in ticks:
                lg = k["logged"]
                if k["vacated"] or lg.get("last", "absent") == "absent":
                    continue
                n += 1
                same += (k["last"] == lg.get("last"))
            agree[(c.model, "same")] += same
            agree[(c.model, "n")] += n
            if n and same / n < 0.9:
                worst.append((round(same / n, 2), c.model, c.scenario, c.seed))
        for m in sorted({k[0] for k in agree}):
            print(f"{arm} {m:13} re-solves compared {agree[(m, 'n')]}, same last {agree[(m, 'same')]} "
                  f"({agree[(m, 'same')] / max(agree[(m, 'n')], 1):.3f})")
        worst.sort()
        print(f"  cells under 0.9 agreement: {len(worst)}; lowest: {worst[:8]}")


if __name__ == "__main__":
    main()
