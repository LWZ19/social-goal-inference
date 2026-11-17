"""Audit item 6 (zero calls): cut-ins normalised by exposure, from the logged 1-Hz samples.

    python item6_cutin_exposure.py > item6_out.txt            # tables
    python item6_cutin_exposure.py --cells > item6_cells_out.txt   # plus every cell's counts

Definitions (README.md):
- exposure  = samples with side <= 1.0 m (the robot within 1 m of the line's axis; SCORING.md's cut-in needs it);
- cut-ins   = samples with cut_in true (= summarise_fresh.measures(e)["cut"], asserted);
- rate      = cut-ins / exposure, undefined when exposure is 0.
Paired by cell (model, scenario, seed). Sign tests use summarise_fresh.sign_p (exact, two-sided, untied pairs).
Counts are integers, so a tie is an equal count. Rate ties: |d| < 0.01 (the audit's rule), exact ties given beside it.
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402

M = paths.MOVING_RUNS                    # run folders: M / "runs" / <model> / ...
sys.path.insert(0, str(paths.DESIGN))
from summarise_fresh import measures, sign_p  # noqa: E402

EXPO_M = 1.0
FRESH = ["cafe_slots", "room_coffee_cart_slots", "cafe_line2_moving", "cafe_line6_moving", "sweet_spot_line2_moving"]
FRESH_SEEDS = {"sonnet55_low": range(10, 20), "dsv41": range(10, 15), "claude_haiku55": range(10, 15),
               "gemini_fast": range(10, 15), "x3": range(10, 13)}
BYS3 = ["cafe_line2_bystanders", "room_coffee_cart_bystanders", "sweet_spot_line2_bystanders"]
BYS2 = ["cafe_line2_bystanders", "sweet_spot_line2_bystanders"]

# (block title, model folder, planned cells as [(scenes, seeds)], [(A, B)], note)
BLOCKS = [("(a) Fresh headline cells", m, [(FRESH, FRESH_SEEDS[m])], [("CT2", "R1"), ("CT2", "CT")], "")
          for m in FRESH_SEEDS]
BLOCKS += [
    ("(b) Bystander confirmation (BYSTANDERS_CONFIRM_FROZEN.md)", "dsv41", [(BYS2, range(15, 45))], [("CT3", "CT2")],
     "the frozen confirmation's 60 cells"),
    ("(b') Bystanders, DeepSeek pilot + confirmation (SUMMARY section 7's cell set)", "dsv41",
     [(BYS3, range(10, 15)), (BYS2, range(15, 45))], [("CT3", "CT2")],
     "pilot: 3 scenes x seeds 10-14 (BYSTANDERS_FROZEN.md); confirmation: cafe + sweet_spot x seeds 15-44"),
    ("(b) Sonnet bystander addendum (BYSTANDERS_SONNET_ADDENDUM.md)", "sonnet55_low", [(BYS2, range(10, 20))],
     [("CT3", "CT2")], "the addendum's 20 cells"),
    ("(b') Sonnet bystanders, addendum + room optional arm (SUMMARY section 7's cell set)", "sonnet55_low",
     [(BYS3, range(10, 20))], [("CT3", "CT2")], "room's 10 cells are BYSTANDERS_FROZEN.md's optional arm"),
]


def stats(e):
    s = e["samples"]
    cut = sum(bool(x["cut_in"]) for x in s)
    assert cut == measures(e)["cut"]
    assert all(x["side"] <= EXPO_M for x in s if x["cut_in"]), "a cut-in outside the exposure"
    expo = sum(x["side"] <= EXPO_M for x in s)
    return dict(cut=cut, expo=expo, n=len(s), rate=(cut / expo if expo else None))


def load(model, cellset, arms):
    out, missing = defaultdict(dict), []
    for sc, sd in cellset:
        for arm in arms:
            f = M / "runs" / model / sc / f"seed{sd}" / arm / "episode.json"
            if not f.exists():
                missing.append((sc, sd, arm))
                continue
            d = json.loads(f.read_text())
            assert (d["scenario"], d["seed"], d["arm"]) == (sc, sd, arm), f
            out[(sc, sd)][arm] = stats(d) | {"label": d["label"]}
    return out, missing


def wlt(rows, tie):
    w = sum(a - b >= tie if tie else a > b + 1e-9 for a, b in rows)
    l = sum(b - a >= tie if tie else a < b - 1e-9 for a, b in rows)
    return w, l, len(rows) - w - l


def compare(by, a, b, show_cells):
    keys = sorted(k for k, v in by.items() if a in v and b in v)
    A = [by[k][a] for k in keys]
    B = [by[k][b] for k in keys]
    n = sum(x["n"] for x in A)
    ca, cb = sum(x["cut"] for x in A), sum(x["cut"] for x in B)
    ea, eb = sum(x["expo"] for x in A), sum(x["expo"] for x in B)
    w, l, t = wlt([(x["cut"], y["cut"]) for x, y in zip(A, B)], 0)
    both = [(x["rate"], y["rate"]) for x, y in zip(A, B) if x["expo"] and y["expo"]]
    za = sum(1 for x, y in zip(A, B) if not x["expo"] and y["expo"])
    zb = sum(1 for x, y in zip(A, B) if x["expo"] and not y["expo"])
    z2 = sum(1 for x, y in zip(A, B) if not x["expo"] and not y["expo"])
    rw, rl, rt = wlt(both, 0.01)
    ew, el, et = wlt(both, 0)
    md = statistics.mean(x - y for x, y in both) if both else float("nan")
    ma = statistics.mean(x for x, _ in both) if both else float("nan")
    mb = statistics.mean(y for _, y in both) if both else float("nan")
    print(f"\n**{a} vs {b}**, {len(keys)} paired cells ({n} samples per arm)\n")
    print("| Arm | Cut-in samples | Exposure (samples, side <= 1 m) | Exposure share of samples | Pooled rate (cut-ins / exposure) |")
    print("|---|---|---|---|---|")
    for arm, c, e in ((a, ca, ea), (b, cb, eb)):
        print(f"| {arm} | {c} | {e} | {e / n:.3f} | {c / e if e else float('nan'):.3f} |")
    print(f"\n- Raw cut-in counts per cell: {a} more / fewer / equal = {w} / {l} / {t}, sign p {sign_p(w, l):.3f}")
    print(f"- Rate per cell, over the {len(both)} cells where both arms have exposure > 0: {a} higher / lower / tie "
          f"(|d| < 0.01) = {rw} / {rl} / {rt}, sign p {sign_p(rw, rl):.3f}; exact ties {ew} / {el} / {et}, sign p "
          f"{sign_p(ew, el):.3f}; mean rate {a} {ma:.3f}, {b} {mb:.3f}, mean difference {md:+.3f}")
    print(f"- Cells dropped from the rate test for zero exposure: {za + zb + z2} ({a} only {za}, {b} only {zb}, both {z2})")
    if show_cells:
        print(f"\n| Scenario | Seed | {a} cut | {a} exposure | {a} rate | {b} cut | {b} exposure | {b} rate |")
        print("|---|---|---|---|---|---|---|---|")
        f = lambda r: "-" if r is None else f"{r:.3f}"  # noqa: E731
        for k, x, y in zip(keys, A, B):
            print(f"| {k[0]} | {k[1]} | {x['cut']} | {x['expo']} | {f(x['rate'])} | {y['cut']} | {y['expo']} | {f(y['rate'])} |")
    by_sc = defaultdict(lambda: [0, 0, 0, 0, 0])
    for k, x, y in zip(keys, A, B):
        r = by_sc[k[0]]
        r[0] += 1; r[1] += x["cut"]; r[2] += y["cut"]; r[3] += x["expo"]; r[4] += y["expo"]
    print(f"\n| Scenario | Cells | Cut-ins {a} / {b} | Exposure {a} / {b} | Pooled rate {a} / {b} |")
    print("|---|---|---|---|---|")
    for sc, (c, x1, y1, x2, y2) in by_sc.items():
        ra = f"{x1 / x2:.3f}" if x2 else "-"
        rb = f"{y1 / y2:.3f}" if y2 else "-"
        print(f"| {sc} | {c} | {x1} / {y1} | {x2} / {y2} | {ra} / {rb} |")
    return dict(cells=len(keys), ca=ca, cb=cb, w=w, l=l)


def main():
    show = "--cells" in sys.argv
    print("# Audit item 6: cut-ins normalised by exposure (zero calls; logs only)\n")
    print(f"exposure = samples with side <= {EXPO_M} m; cut-ins = samples with cut_in; rate = cut-ins / exposure "
          "(undefined at zero exposure). Paired by (scenario, seed) within a model. Every episode has 91 samples.")
    res = {}
    for title, model, groups, pairs, note in BLOCKS:
        arms = sorted({x for p in pairs for x in p})
        cellset = [(sc, sd) for scenes, seeds in groups for sc in scenes for sd in seeds]
        by, missing = load(model, cellset, arms)
        if not by:
            continue
        labels = sorted({v["label"] for c in by.values() for v in c.values()})
        print(f"\n## {title}: {model} ({', '.join(labels)})\n")
        print("Cells: " + " + ".join(f"{scenes} x seeds {min(seeds)}-{max(seeds)}" for scenes, seeds in groups)
              + f" = {len(cellset)} planned" + (f"; {note}" if note else ""))
        miss = defaultdict(list)
        for sc, sd, arm in missing:
            miss[arm].append(f"{sc} s{sd}")
        txt = []
        for arm, v in miss.items():
            planned = len(cellset)
            txt.append(f"{arm} {len(v)} of {planned}" + ("" if len(v) > 12 else f" ({', '.join(v)})"))
        print("Missing episodes: " + ("; ".join(txt) if txt else "none"))
        for a, b in pairs:
            if not any(a in v and b in v for v in by.values()):
                print(f"\n**{a} vs {b}**: no paired cells")
                continue
            res[(title, model, a, b)] = compare(by, a, b, show)

    print("\n## Check against SUMMARY.md section 7 (and results/overview/ideas-round-2/SHORTLIST.md section 0)\n")
    checks = [("(a) Fresh headline cells", "sonnet55_low", "CT2", "R1", 50, 481, 162, 25, 4),
              ("(a) Fresh headline cells", "dsv41", "CT2", "R1", 25, 226, 504, 9, 10),
              ("(b') Bystanders, DeepSeek pilot + confirmation (SUMMARY section 7's cell set)", "dsv41", "CT3", "CT2",
               75, 1036, 881, 21, 14),
              ("(b') Sonnet bystanders, addendum + room optional arm (SUMMARY section 7's cell set)", "sonnet55_low",
               "CT3", "CT2", 30, 138, 89, 5, 3)]
    print("| Quoted | Cells | Recount A vs B | More / fewer | Match |")
    print("|---|---|---|---|---|")
    for title, model, a, b, n, qa, qb, qw, ql in checks:
        r = res[(title, model, a, b)]
        ok = (r["cells"], r["ca"], r["cb"], r["w"], r["l"]) == (n, qa, qb, qw, ql)
        print(f"| {model} {a} {qa} vs {b} {qb}, {qw}/{ql}, {n} pairs | {r['cells']} | {r['ca']} vs {r['cb']} | "
              f"{r['w']} / {r['l']} | {'yes' if ok else 'NO'} |")


if __name__ == "__main__":
    main()
