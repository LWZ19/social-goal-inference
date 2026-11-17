"""Offline check of the src port of CT5 (ROUND3.md section 8): does `sgi.roles` (RoleBelief with STATIC_WIN_S 5,
`decide`) give the same marginals and the same choices as the research code (`round3/ct5.py` a27fb000, variant W4)
on the same ticks? Zero calls.

The research object's starting state (ids, named, direction, target, sets, prior) is copied into the src object,
whose construction from an answer was checked for CT4 already; the check is of `step` with the window, and `decide`.
Every design cell (r3lib.design_cells) under each condition (clean, S2m, S3, S23m).

    .venv/bin/python results/moving-lines/best-version-and-running-guess/round3_check/check_port_ct5.py   (from the repo root)
"""
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "round3"))
import r3lib as L  # noqa: E402
from ct5 import CT5, UNSET as UNSET_R, decide as decide_r  # noqa: E402
from sgi.resolve import Track  # noqa: E402
from sgi.roles import UNSET as UNSET_S, RoleBelief, decide as decide_s  # noqa: E402

W4P = dict(WALKERS_STATIC=True, LR_EXIT=1.0, STATIC_WIN_S=5.0)       # c_eval.py: CT4_AS_CT5 + WIN, as W4


def src_twin(est):
    at = [Track(id=i, xy=(0.0, 0.0), yaw=None) for i in est.ids]
    mine = RoleBelief([], [], tuple(est.tgt), at, (0.0, 0.0), {"STATIC_WIN_S": 5.0})
    for k in ("ids", "k", "n", "named", "dir", "tgt", "bits"):
        setattr(mine, k, getattr(est, k))
    mine.b = est.b.copy()
    return mine


def main() -> int:
    cells = L.design_cells()
    print(f"{len(cells)} design cells; research params {W4P}")
    total = Counter()
    worst = 0.0
    bad = []
    for cond, stress in L.CONDITIONS.items():
        n = agree = 0
        for cell in cells:
            est = CT5(cell, W4P)
            mine = src_twin(est)
            cur_r, cur_s = UNSET_R, UNSET_S
            for t, tracks, _ in L.ticks_of(cell, stress):
                est.step(t, tracks, 0.5)
                mine.step(t, tracks, 0.5)
                if t + 1e-9 < cell.delivered_t:
                    continue
                present = {tr.id for tr in tracks}
                cur_r, _ = decide_r(est, present, cur_r)
                cur_s, _ = decide_s(mine, present, cur_s)
                mr, ms = est.marginals(), mine.marginals()
                worst = max(worst, max((abs(mr[i] - ms[i]) for i in mr), default=0.0))
                n += 1
                agree += cur_r == cur_s
                if cur_r != cur_s and len(bad) < 10:
                    bad.append((cond, cell.scenario, cell.seed, t, cur_r, cur_s))
        total["n"] += n
        total["agree"] += agree
        print(f"  {cond}: {agree}/{n} choices agree ({agree / max(n, 1):.4f})")
    print(f"all: {total['agree']}/{total['n']} ({total['agree'] / max(total['n'], 1):.4f}); "
          f"worst marginal difference {worst:.2e}")
    for b in bad:
        print("  differs:", b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
