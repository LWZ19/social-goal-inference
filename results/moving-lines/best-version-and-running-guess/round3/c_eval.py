"""Round 3, step C (ROUND3.md section 5): offline comparison on design cells, zero calls.

Variants:
- V0 CT4-min frozen, threshold (the sweep's arm);
- V4 CT5 set to behave as CT4 (walkers in the static snapshot, no exit evidence), cost decision: the decision alone;
- V1 CT5 hand-set (E1, E2 with LR_EXIT 10), threshold;
- V2 CT5 calibrated by counting (fitted.json), threshold;
- V3 CT5 calibrated, cost decision (ROUND3 costs).
Per condition (clean, S2m, S3, S23m) and seed split (fit 10-34, check 35-44): right / cut / far / empty shares of
ticks with a queue; switches of the chosen last per minute; roster changes per tick; calibration error (ECE, 10 bins)
and the share in the queue for marginals in 0.5-0.7.

    .venv/bin/python round3/c_eval.py [sens]     (from best-version-and-running-guess/)
"""
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from r3lib import CONDITIONS, classify, design_cells, run  # first: it sets the import path
from ct4 import CT4  # noqa: E402
from ct5 import CT5  # noqa: E402

HERE = Path(__file__).resolve().parent
FITTED = json.loads((HERE / "fitted.json").read_text())
CT4_AS_CT5 = dict(WALKERS_STATIC=True, LR_EXIT=1.0)
VARIANTS = {
    "V0 CT4 threshold": (CT4, None, "threshold", None),
    "V4 CT4-like + cost": (CT5, CT4_AS_CT5, "cost", None),
    "V1 CT5 hand-set": (CT5, None, "threshold", None),
    "V2 CT5 fitted": (CT5, FITTED, "threshold", None),
    "V3 CT5 fitted + cost": (CT5, FITTED, "cost", None),
}
FITTED_W = json.loads((HERE / "fitted_walkers.json").read_text()) if (HERE / "fitted_walkers.json").exists() else {}
WIN = dict(STATIC_WIN_S=5.0)                           # e_acf_out.txt: the static log ratio's 1/e time
WINDOW = {
    "V0 CT4 threshold": (CT4, None, "threshold", None),
    "W0 CT4-like + window": (CT5, dict(CT4_AS_CT5, **WIN), "threshold", None),
    "W1 W0 + exit (fitted)": (CT5, dict(CT4_AS_CT5, **WIN, LR_EXIT=FITTED.get("LR_EXIT", 10.0)), "threshold", None),
    "W2 fitted (walkers) + window": (CT5, dict(FITTED_W, **WIN), "threshold", None),
    "W3 W2 + cost": (CT5, dict(FITTED_W, **WIN), "cost", None),
    "W4 W0 + cost": (CT5, dict(CT4_AS_CT5, **WIN), "cost", None),
}
W4P = dict(CT4_AS_CT5, **WIN)
SENS_W4 = {
    "V0 CT4 threshold": (CT4, None, "threshold", None),
    "W4 (frozen costs)": (CT5, W4P, "cost", None),
    "W4 ratio 2": (CT5, W4P, "cost", dict(c_far=0.5)),
    "W4 ratio 8": (CT5, W4P, "cost", dict(c_far=0.125)),
    "W4 move 0": (CT5, W4P, "cost", dict(c_move=0.0)),
    "W4 move x2": (CT5, W4P, "cost", dict(c_move=0.25)),
}
SENS = {
    "V3 ratio 2": (CT5, FITTED, "cost", dict(c_far=0.5)),
    "V3 ratio 8": (CT5, FITTED, "cost", dict(c_far=0.125)),
    "V3 move 0": (CT5, FITTED, "cost", dict(c_move=0.0)),
    "V3 move x2": (CT5, FITTED, "cost", dict(c_move=0.25)),
}


def one(args):
    ci, cond, name, spec = args
    c = design_cells()[ci]
    cls, params, decision, costs = spec
    rows = run(c, CONDITIONS[cond], est_cls=cls, params=params, decision=decision, costs=costs)
    k = Counter(classify(r) for r in rows)
    sw = sum(1 for a, b in zip(rows, rows[1:]) if a["last"] != b["last"])
    rf = sum(1 for a, b in zip(rows, rows[1:]) if a["roster"] != b["roster"])
    minutes = max((rows[-1]["t"] - rows[0]["t"]) / 60, 1e-9) if rows else 1e-9
    cal = defaultdict(lambda: [0, 0, 0.0])
    for r in rows:
        for i, v in r["m"].items():
            b = min(int(v * 10), 9)
            cal[b][0] += 1
            cal[b][1] += i in r["q"]
            cal[b][2] += v
    return (c.seed, cond, name, dict(k), sw, rf, len(rows), minutes, {b: tuple(v) for b, v in cal.items()})


def main():
    variants = SENS_W4 if "sensw4" in sys.argv else WINDOW if "window" in sys.argv else dict(VARIANTS, **SENS) if "sens" in sys.argv else VARIANTS
    cs = design_cells()
    jobs = [(ci, cond, name, spec) for ci in range(len(cs)) for cond in CONDITIONS for name, spec in variants.items()]
    with ProcessPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(one, jobs, chunksize=4))
    print(f"# Round 3, step C: {len(cs)} design cells x {len(CONDITIONS)} conditions x {len(variants)} variants\n")
    for split, seeds in (("fit seeds 10-34", range(10, 35)), ("check seeds 35-44", range(35, 45))):
        print(f"## {split}\n")
        for cond in CONDITIONS:
            print(f"### {cond}{' (stress unseen by the fit)' if cond in ('S3', 'S23m') else ''}\n")
            print("| Variant | right | cut | far | empty | last switches / min | roster changes / tick | ECE | "
                  "in queue at 0.5-0.7 |")
            print("|---|---|---|---|---|---|---|---|---|")
            for name in variants:
                k, sw, rf, nt, mins = Counter(), 0, 0, 0, 0.0
                cal = defaultdict(lambda: [0, 0, 0.0])
                for seed, cd, nm, kk, s, r, n, m, cb in res:
                    if cd != cond or nm != name or seed not in seeds:
                        continue
                    k.update(kk)
                    sw += s
                    rf += r
                    nt += n
                    mins += m
                    for b, v in cb.items():
                        for z in range(3):
                            cal[b][z] += v[z]
                scored = sum(v for key, v in k.items() if key is not None)
                tot = sum(v[0] for v in cal.values())
                ece = sum(abs(v[1] - v[2]) for v in cal.values()) / max(tot, 1)
                mid = (cal[5][1] + cal[6][1]) / max(cal[5][0] + cal[6][0], 1)
                print(f"| {name} | " + " | ".join(f"{k[x] / scored:.3f}" for x in ("right", "cut", "far", "empty")) +
                      f" | {sw / mins:.1f} | {rf / max(nt, 1):.3f} | {ece:.3f} | {mid:.3f} |")
            print()


if __name__ == "__main__":
    main()
