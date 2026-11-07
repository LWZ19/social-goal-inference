"""Round 3, step B (ROUND3.md section 3, E3): calibration by counting. Design data only, zero calls.

Each likelihood term is its own frequency against the dump's truth, counted over the FIT set: design seeds 10-34,
conditions clean and S2m. The counts don't depend on the belief (follow checks and exits are triggered by motion,
residuals by geometry), so one pass with CT5's defaults suffices. Output: fitted values beside the hand-set ones,
written to fitted.json.

    .venv/bin/python round3/b_calibrate.py      (from best-version-and-running-guess/)
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from r3lib import CONDITIONS, design_cells, dump_truth, run
from ct5 import DEFAULTS
from ct4 import _named
from replay_lib import dump_tracks

HERE = Path(__file__).resolve().parent
FIT_SEEDS = range(10, 35)
WALKERS = "walkers" in sys.argv          # count residuals of walkers too (for WALKERS_STATIC True)
PARAMS = dict(WALKERS_STATIC=True) if WALKERS else None
EDGES = [0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.5, np.inf]


def main():
    cs = [c for c in design_cells() if c.seed in FIT_SEEDS]
    ans = Counter()
    fol = Counter()
    ex = Counter()
    res_w, res_n = [], []
    for c in cs:
        at = dump_tracks(c.dump, c.asked_t)
        named = _named(c, at)
        q0, _ = dump_truth(c.dump, c.asked_t)
        for tr in at:
            ans[(tr.id in q0, tr.id in named)] += 1
        for cond in ("clean", "S2m"):
            state = {}

            def hook(when, est, t, tracks, state=state):
                if when == "before":
                    state["elig"] = set(est.near_min)
                    state["nexit"] = len(est.log_exit)
                    est.keep_static = True
                    return
                q, _ = dump_truth(c.dump, t)
                for i in state["elig"]:
                    ex[("time", i in q)] += 0.5
                for _, i in est.log_exit[state["nexit"]:]:
                    ex[("event", i in q)] += 1
                for t_, i, r in est.log_static:
                    W = sum(1 << est.k[j] for j in est.ids if j in q)
                    if i in q:
                        res_w.append(float(r[W]))
                    else:
                        res_n.append(float(r[W | (1 << est.k[i])]))
                est.log_static.clear()

            from ct5 import CT5
            est_rows = run(c, CONDITIONS[cond], est_cls=CT5, params=PARAMS, hook=hook)
            del est_rows
        # follow checks need the estimator; rerun cheaply with a capturing class
    # follow checks: a second pass keeping the estimators
    from ct5 import CT5
    for c in cs:
        for cond in ("clean", "S2m"):
            keep = {}

            def hook2(when, est, t, tracks, keep=keep):
                keep["est"] = est
            run(c, CONDITIONS[cond], est_cls=CT5, params=PARAMS, hook=hook2)
            est = keep["est"]
            for t, i, j, followed in est.log_follow:
                q, _ = dump_truth(c.dump, t - est.p["T_FOLLOW"] - est.p["PERSIST_S"])
                fol[(i in q and j in q, followed)] += 1

    out = {}
    # answer reliability and base rate
    w_named, w_not = ans[(True, True)], ans[(True, False)]
    n_named, n_not = ans[(False, True)], ans[(False, False)]
    out["P_NAMED_W"] = w_named / max(w_named + w_not, 1)
    out["P_NAMED_N"] = n_named / max(n_named + n_not, 1)
    out["P0"] = (w_named + w_not) / max(sum(ans.values()), 1)
    # follow
    out["Q"] = fol[(True, True)] / max(fol[(True, True)] + fol[(True, False)], 1)
    out["B"] = fol[(False, True)] / max(fol[(False, True)] + fol[(False, False)], 1)
    # exit: event-rate ratio, not-waiting over waiting, +0.5 smoothing
    rate_w = (ex[("event", True)] + 0.5) / max(ex[("time", True)], 1e-9)
    rate_n = (ex[("event", False)] + 0.5) / max(ex[("time", False)], 1e-9)
    out["LR_EXIT"] = rate_n / rate_w
    # static residual table: density ratio per bin, +0.5 smoothing
    rw, rn = np.array(res_w), np.array(res_n)
    tab = []
    lo = -np.inf
    for hi in EDGES:
        nw = ((rw > lo) & (rw <= hi)).sum() + 0.5
        nn = ((rn > lo) & (rn <= hi)).sum() + 0.5
        tab.append([float(hi) if np.isfinite(hi) else 1e9, float((nw / (len(rw) + 0.5 * len(EDGES))) /
                                                                 (nn / (len(rn) + 0.5 * len(EDGES))))])
        lo = hi
    out["STATIC_TABLE"] = tab

    print(f"# Calibration by counting: {len(cs)} design cells (seeds 10-34), clean + S2m"
          f"{'; static residuals of walkers included' if WALKERS else ''}\n")
    print(f"- answer at the photo (waits, named) counts: {dict(ans)}")
    print(f"- follow checks (both wait, followed) counts: {dict(fol)}")
    print(f"- exits: events while waiting {ex[('event', True)]}, while not {ex[('event', False)]}; eligible "
          f"person-seconds waiting {ex[('time', True)]:.0f}, not {ex[('time', False)]:.0f}")
    print(f"- static residuals: waiters {len(rw)}, non-waiters {len(rn)}\n")
    print("| Term | hand-set | fitted |")
    print("|---|---|---|")
    for k in ("P0", "P_NAMED_W", "P_NAMED_N", "Q", "B", "LR_EXIT"):
        print(f"| {k} | {DEFAULTS[k]} | {out[k]:.3f} |")
    print("\nStatic likelihood ratio by residual (fitted table; hand-set: Gaussian sigma 0.35, neutral at 0.8 m, "
          "bounded to [1/5, 5]):\n")
    lo = 0.0
    for hi, lr in tab:
        g = np.exp(np.clip(-0.5 * (min(hi, 2.0) / 0.35) ** 2 + 0.5 * (0.8 / 0.35) ** 2, -np.log(5), np.log(5)))
        print(f"- r in ({lo:.2f}, {hi if hi < 1e8 else 'inf'}]: fitted {lr:.3f}; hand-set at the bin's top {g:.3f}")
        lo = hi
    if WALKERS:
        out["WALKERS_STATIC"] = True
    (HERE / ("fitted_walkers.json" if WALKERS else "fitted.json")).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
