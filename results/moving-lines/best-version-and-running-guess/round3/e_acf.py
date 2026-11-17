"""Round 3, step E: the decorrelation time of the static evidence, measured (sets STATIC_WIN_S by counting, not by
search). Fit seeds 10-34, clean + S2m. For each person and episode, the residual under the true set (as in
b_calibrate.py), every 0.5 s from the photo; the pooled autocorrelation of the hand-set log ratio by lag; the
1/e time.

    .venv/bin/python round3/e_acf.py      (from best-version-and-running-guess/)
"""
import math
from collections import defaultdict

import numpy as np

from r3lib import CONDITIONS, design_cells, dump_truth, run  # first: it sets the import path
from ct5 import CT5  # noqa: E402


def main():
    cs = [c for c in design_cells() if c.seed < 35]
    series = []
    for c in cs:
        for cond in ("clean", "S2m"):
            per = defaultdict(list)

            def hook(when, est, t, tracks, per=per, c=c):
                if when == "before":
                    est.keep_static = True
                    return
                q, _ = dump_truth(c.dump, t)
                W = sum(1 << est.k[j] for j in est.ids if j in q)
                for t_, i, r in est.log_static:
                    rr = float(r[W] if i in q else r[W | (1 << est.k[i])])
                    lr = -0.5 * (rr / 0.35) ** 2 + 0.5 * (0.8 / 0.35) ** 2
                    per[i].append(max(-math.log(5), min(math.log(5), lr)))
                est.log_static.clear()
            run(c, CONDITIONS[cond], est_cls=CT5, params=dict(WALKERS_STATIC=True), hook=hook)
            series += [np.array(v) for v in per.values() if len(v) > 40]
    print(f"# Decorrelation of the static log ratio: {len(series)} person-episodes\n")
    lags = range(0, 31)
    acf = []
    for L in lags:
        num = den = 0.0
        for s in series:
            x = s - s.mean()
            num += float((x[:len(x) - L] * x[L:]).sum()) if L else float((x * x).sum())
            den += float((x * x).sum())
        acf.append(num / den if den else 0.0)
    for L, a in zip(lags, acf):
        if L % 2 == 0:
            print(f"- lag {L * 0.5:4.1f} s: {a:.3f}")
    e = next((L * 0.5 for L, a in zip(lags, acf) if a < 1 / math.e), None)
    print(f"\n1/e time: {e} s")


if __name__ == "__main__":
    main()
