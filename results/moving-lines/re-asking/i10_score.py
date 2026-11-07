"""Item 10 (I10_FROZEN.md): asking once and re-solving (CT2) against re-asking (Rq0: the marked point asked again as
soon as an answer arrives; Cq: the constraint method asked again every 10 s) on fresh headline cells. Zero calls.

    .venv/bin/python results/moving-lines/re-asking/i10_score.py           (the frozen reading)
    .venv/bin/python results/moving-lines/re-asking/i10_score.py --test    (old data: seeds 0-2, CT)
"""
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import mcompare as C  # noqa: E402

MARGIN = -0.05
# Amendment 1 (I10_FROZEN.md): CT2, Rq0 and Cq all run in one invocation into runs_i10/<model> (same machine load);
# the earlier CT2 in runs/<model> is read only for a repeat check.
MODELS = {"DeepSeek V4.1": ("runs_i10/dsv41", range(10, 15)), "Qwen 3.8": ("runs_i10/soclaas_qwen38_low", range(10, 15)),
          "Sonnet 5.5": ("runs_i10/sonnet55_low", range(10, 15))}
OLD = ("cafe", "room_coffee_cart", "cafe_line2_moving", "cafe_line6_moving", "sweet_spot_line2_moving")


def t95(df):
    z = 1.6448536269514722
    return z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)


def lb(ps, key):
    d = [x[key] - y[key] for _, x, y in ps]
    if len(d) < 3:
        return float("nan"), float("nan")
    m = statistics.mean(d)
    return m, m - t95(len(d) - 1) * statistics.stdev(d) / math.sqrt(len(d))


RATIO = 1.5
# Each model's normal re-ask/first-answer latency ratio, measured on the old seeds 0-2 runs under normal gateway speed
# (i10_score_test_out.txt): DeepSeek Rq0 0.49, Cq 0.51; Sonnet Cq 0.79 (no Rq0 run: Cq's ratio stands in); Qwen 3.8 has
# no re-asking runs, so x-test-3's (Qwen 3.8 Flash Next on the same gateway) stands in: Rq0 0.91, Cq 0.93.
NORMAL = {"DeepSeek V4.1": {"Rq0": 0.49, "Cq": 0.51}, "Sonnet 5.5": {"Rq0": 0.79, "Cq": 0.79},
          "Qwen 3.8": {"Rq0": 0.91, "Cq": 0.93}}


def reask_ratio(eps, arm):
    """Amendment 3 (I10_FROZEN.md): median latency of the arm's live re-asks (answers after the first) over the median
    latency of its first answers (cache replays, stored when first made). None without both."""
    first = [e["answers"][0]["latency_s"] for (_, _, a), e in eps.items() if a == arm and e is not None
             and e.get("answers") and e["answers"][0].get("status") == "ok" and e["answers"][0].get("latency_s")]
    re = [x["latency_s"] for (_, _, a), e in eps.items() if a == arm and e is not None
          for x in e.get("answers", [])[1:] if x.get("status") == "ok" and x.get("latency_s")]
    if not first or not re:
        return None, None, None
    f, r = statistics.median(first), statistics.median(re)
    return f, r, r / f


def calls(eps, arm):
    v = [e["summary"].get("calls") for (_, _, a), e in eps.items() if a == arm and e is not None and e.get("summary")]
    v = [x for x in v if x is not None]
    return (statistics.mean(v), max(v)) if v else (float("nan"), float("nan"))


def main():
    test = "--test" in sys.argv
    ref = "CT" if test else "CT2"
    groups = ({"DeepSeek (seeds 0-2)": ("runs/dsv41", range(0, 3)), "x-test-3 (seeds 0-2)": ("runs/x3", range(0, 3)),
               "Sonnet (seeds 0-2)": ("runs/sonnet55_low", range(0, 3))} if test else MODELS)
    scen = OLD if test else C.EVAL5
    print(f"# Item 10: {ref} against re-asking (Rq0, Cq){' TEST on old data' if test else ''}\n")
    print("| model | comparison | pairs | strict mean d | strict 95% lower bound | primary mean d | primary lower bound | strict W/L/T, p |")
    print("|---|---|---|---|---|---|---|---|")
    verdicts, pooled = [], {"Rq0": [], "Cq": []}
    for name, (root, seeds) in groups.items():
        eps = C.load(root, scen, seeds, (ref, "Rq0", "Cq", "R1"))
        if not any(e is not None for (_, _, a), e in eps.items() if a in ("Rq0", "Cq")):
            print(f"| {name} | not run | | | | | | |")
            continue
        res = {}
        for a, b in ((ref, "Rq0"), (ref, "Cq"), ("Rq0", "R1")):
            ps = C.pairs(eps, a, b)
            if not ps:
                continue
            ms, ls = lb(ps, "share_strict")
            mf, lf = lb(ps, "share_frozen")
            r = C.row("", ps)
            res[(a, b)] = (ls, lf, r, len(ps))
            print(f"| {name} | {a} vs {b} | {len(ps)} | {ms:+.3f} | {ls:+.3f} | {mf:+.3f} | {lf:+.3f} | "
                  f"{'/'.join(map(str, r['strict']))}, {r['p_strict']:.4f} |")
        bound = False
        if not test:
            ct = [e for (_, _, a), e in eps.items() if a == ref and e is not None]
            late = sum(1 for e in ct if not e.get("answers") or e["answers"][0].get("status") != "ok"
                       or (e["answers"][0].get("latency_s") or 0) + 2.0 > 92.0)
            bound = bool(ct) and late > len(ct) / 2          # item 9's latency-bound rule (I9_FROZEN.md)
            if bound:
                verdicts.append(f"- {name}: LATENCY-BOUND ({late} of {len(ct)} {ref} episodes without a first answer in "
                                f"time): reported only, left out of the pooled bar")
        confounded = False
        for arm in ("Rq0", "Cq"):
            f, r, q = reask_ratio(eps, arm)
            if q is not None:
                normal = NORMAL.get(name, {}).get(arm)
                flag = normal is not None and q > RATIO * normal
                confounded = confounded or flag
                verdicts.append(f"- {name}: {arm} re-ask latency median {r:.1f} s vs its first answers' {f:.1f} s: ratio "
                                f"{q:.2f} (normal {normal}){' > 1.5 x normal: GATEWAY-CONFOUNDED' if flag else ''}")
        if confounded and not test:
            verdicts.append(f"- {name}: GATEWAY-CONFOUNDED (amendment 3): reported only, left out of the pooled bar")
        for b in ("Rq0", "Cq"):
            if (ref, b) in res and (test or not (bound or confounded)):
                pooled[b] += C.pairs(eps, ref, b)
        if not test:
            old = C.load(root.replace("runs_i10/", "runs/"), scen, seeds, (ref,))
            both = {(*k[:2], "new"): e for k, e in eps.items() if k[2] == ref}
            both.update({(*k[:2], "old"): e for k, e in old.items()})
            rp = C.pairs(both, "new", "old")
            if rp:
                dd = [x["share_strict"] - y["share_strict"] for _, x, y in rp]
                verdicts.append(f"- {name}: repeat check, {ref} this run vs the earlier run, {len(rp)} cells: mean d "
                                f"{statistics.mean(dd):+.3f}, mean |d| {statistics.mean(abs(v) for v in dd):.3f}")
        cs = {a: calls(eps, a) for a in (ref, "Rq0", "Cq")}
        print(f"| {name} | calls per episode, mean (max) | | {ref} {cs[ref][0]:.1f} ({cs[ref][1]}) | Rq0 {cs['Rq0'][0]:.1f} "
              f"({cs['Rq0'][1]}) | Cq {cs['Cq'][0]:.1f} ({cs['Cq'][1]}) | | |")
        for b in ("Rq0", "Cq"):
            if (ref, b) in res:
                ls, lf, r, n = res[(ref, b)]
                expected = len(seeds) * 5
                verdicts.append(f"- {name}: BAR H10 {ref} non-inferior to {b} (strict lower bound > {MARGIN}): {ls > MARGIN}; "
                                f"primary (reported) {lf > MARGIN}; superior (reported, sign test): {C.better(r, 'strict')}"
                                f"{'' if n == expected else f'  [pairs {n}, expected {expected}: check mechanics]'}")
    for b in ("Rq0", "Cq"):
        ps = pooled[b]
        ms, ls = lb(ps, "share_strict")
        mf, lf = lb(ps, "share_frozen")
        r = C.row("", ps) if ps else None
        verdicts.append(f"- POOLED{' (test groups)' if test else ''} over {len(ps)} cells: BAR H10-{b} ({ref} non-inferior to {b}, "
                        f"strict lower bound > {MARGIN}): {ls > MARGIN} (mean {ms:+.3f}, bound {ls:+.3f}); primary (reported) "
                        f"{lf > MARGIN}; superior (reported): {C.better(r, 'strict') if r else False}")
    print("\n" + "\n".join(verdicts))


if __name__ == "__main__":
    main()
