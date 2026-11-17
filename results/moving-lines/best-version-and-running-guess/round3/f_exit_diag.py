"""Round 3, step F: why does exit evidence add empty ticks? Fit seeds, clean. For ticks where W0 has a last and W1
is empty: the exits in the 30 s before (who, member or not at the exit, distance to the target), and W1's marginals
of the true queue."""
from collections import Counter

from r3lib import CONDITIONS, design_cells, dump_truth, ticks_of, turn_last, robot_at  # noqa: F401
from ct5 import CT5  # noqa: E402

WIN = dict(STATIC_WIN_S=5.0, WALKERS_STATIC=True)
k = Counter()
shown = 0
for c in [c for c in design_cells() if c.seed < 35]:
    a = CT5(c, dict(WIN, LR_EXIT=1.0))
    b = CT5(c, dict(WIN, LR_EXIT=65.228))
    for t, tracks, _ in ticks_of(c, None):
        a.step(t, tracks, 0.5)
        b.step(t, tracks, 0.5)
        if t + 1e-9 < c.delivered_t:
            continue
        q, order = dump_truth(c.dump, t)
        if not q:
            continue
        ma, mb = a.marginals(), b.marginals()
        ra = {i for i, v in ma.items() if v >= 0.5}
        rb = {i for i, v in mb.items() if v >= 0.5}
        if ra and not rb:
            ex = [(te, i) for te, i in b.log_exit if t - 30 <= te <= t]
            k["ticks"] += 1
            k["with an exit of a current member in 30 s"] += any(i in q for _, i in ex)
            k["with any exit in 30 s"] += bool(ex)
            if shown < 6:
                shown += 1
                print(f"{c.model} {c.scenario} s{c.seed} t {t:.1f}: queue {sorted(q, key=lambda i: order[i])}; "
                      f"exits {[(round(te, 1), i, i in q) for te, i in ex]}; "
                      f"P(queue) W1 {[round(mb[i], 2) for i in sorted(q, key=lambda i: order[i]) if i in mb]}; "
                      f"W0 {[round(ma[i], 2) for i in sorted(q, key=lambda i: order[i]) if i in ma]}; "
                      f"named {sorted(b.named)}")
print(dict(k))
