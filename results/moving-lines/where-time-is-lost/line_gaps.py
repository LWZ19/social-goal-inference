"""Descriptive: the predictions and readings fixed before the runs, applied mechanically to the ceiling runs, at band
tops 1.2 m (the decisions) and 1.5 m (reported beside them). Zero calls; read-only.

    .venv/bin/python results/moving-lines/where-time-is-lost/line_gaps.py SELECTION_JSON CEILING_ROOT CONFIRM_ROOT
    (SELECTION_JSON relative to results/moving-lines/, the roots relative to results/data/moving/. OW, CT4OT, CT4O and
    CT4PJT come from CEILING_ROOT; CT4PJ and CT3F
    from CONFIRM_ROOT, which used the same cached answers.)

Per episode, the strict share from the first service splits exactly into a front part (seconds with nobody ahead
that pass, over all window seconds) and a behind part (seconds with someone ahead that pass, over all window
seconds), so every gap splits into front + behind.

Gaps are means of paired per-layout differences over layouts where both arms have a window:
- place and follow: OW - CT4OT;
- target: CT4OT - CT4O;
- who, model's target: CT4O - CT4PJ (the WA reading uses this one);
- who, true target: CT4OT - CT4PJT.

If the two "who" gaps give different WA decisions, both are reported, as for the two band tops.
"""
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402
sys.path.insert(0, str(HERE))
from time_lost import MOVING, window  # noqa: E402
from band_sensitivity import TOPS, strict_at, t95  # noqa: E402
from summarise_fresh import sign_p  # noqa: E402

CEILING, CONFIRM = ("OW", "CT4OT", "CT4O", "CT4PJT"), ("CT4PJ", "CT3F")
GAPS = (("place and follow", "OW", "CT4OT"), ("target", "CT4OT", "CT4O"), ("who, model's target", "CT4O", "CT4PJ"),
        ("who, true target", "CT4OT", "CT4PJT"))
MIN_EFFECT = 0.05          # smallest difference read as an effect
TIE = 0.01


def episode(e):
    _, win = window(e)
    if not win:
        return None
    n = len(win)
    out = {}
    for top in TOPS:
        ok = [strict_at(x, top) for x in win]
        front = sum(v for x, v in zip(win, ok) if x["ahead"] == 0)
        out[top] = {"all": sum(ok) / n, "front": front / n, "behind": (sum(ok) - front) / n,
                    "front_s": sum(1 for x in win if x["ahead"] == 0), "front_pass": front,
                    "behind_s": sum(1 for x in win if x["ahead"] >= 1), "behind_pass": sum(ok) - front}
    return out


def load(sel, root, arms):
    sc = {}
    for lay in sel:
        for arm in arms:
            hits = sorted((MOVING / root).glob(f"**/{lay['name']}/seed{lay['seed']}/{arm}/episode.json"))
            sc[(lay["name"], arm)] = episode(json.loads(hits[0].read_text())) if hits else None
    return sc


def gap(sc, names, a, b, top):
    rows = [(sc[(n, a)][top], sc[(n, b)][top]) for n in names if sc.get((n, a)) and sc.get((n, b))]
    if len(rows) < 3:
        return None
    d = [x["all"] - y["all"] for x, y in rows]
    w, lo = sum(v >= TIE for v in d), sum(v <= -TIE for v in d)
    m = statistics.mean(d)
    return {"n": len(d), "mean": m, "front": statistics.mean(x["front"] - y["front"] for x, y in rows),
            "behind": statistics.mean(x["behind"] - y["behind"] for x, y in rows), "w": w, "l": lo,
            "t": len(d) - w - lo, "p": sign_p(w, lo), "lb": m - t95(len(d) - 1) * statistics.stdev(d) / math.sqrt(len(d))}


def pooled(sc, names, arm, top, part):
    v = [sc[(n, arm)][top] for n in names if sc.get((n, arm))]
    s, p = sum(x[f"{part}_s"] for x in v), sum(x[f"{part}_pass"] for x in v)
    return (p / s if s else None), s


def fmt(v, spec="+.3f"):
    return "-" if v is None else format(v, spec)


def readings(sc, names, top):
    ow = [sc[(n, "OW")][top]["all"] for n in names if sc.get((n, "OW"))]
    g = {k: gap(sc, names, a, b, top) for k, a, b in GAPS}
    out = []
    if ow:
        m = statistics.mean(ow)
        low = [n for n in names if sc.get((n, "OW")) and sc[(n, "OW")][top]["all"] < 0.5]
        t = ("T: reachable (OW >= 0.7)" if m >= 0.7 else
             "T: reachable with a following cost (OW 0.5-0.7): report methods also as a share of OW" if m >= 0.5 else
             "T: capped (OW < 0.5): run time_lost.py on OW first")
        out.append(f"{t}. OW mean {m:.3f} over {len(ow)}; layouts with OW < 0.5: {len(low)} {low}")
    wa = {}
    for k in ("who, model's target", "who, true target"):
        if g[k]:
            wa[k] = g[k]["mean"] >= MIN_EFFECT
    if "who, model's target" in wa:
        dec = ("WA: walking away meets the 0.05 mark" if wa["who, model's target"]
               else "WA: walking away does not meet the 0.05 mark (who < 0.05)")
        who = g["who, model's target"]["mean"]
        out.append(f"{dec}; who (model's target) {who:+.3f}")
        if len(set(wa.values())) > 1:
            out.append("WA: the two 'who' gaps disagree on the 0.05 mark: both are reported")
    if g["target"]:
        tg = g["target"]
        front_mostly = tg["mean"] > 0 and tg["front"] > tg["mean"] / 2
        out.append(f"TG: target {tg['mean']:+.3f} (front {tg['front']:+.3f}, behind {tg['behind']:+.3f}): "
                   + ("the counter point is the first answer's job (>= 0.05 and mostly at the front)"
                      if tg["mean"] >= MIN_EFFECT and front_mostly else "not met"))
    if g["place and follow"]:
        pf = g["place and follow"]["mean"]
        out.append(f"PF: place and follow {pf:+.3f}: " + ("the largest open part is where to stand and getting there"
                                                         " (>= 0.10)" if pf >= 0.10 else "below 0.10"))
    big = {k: v["mean"] for k, v in g.items() if v and k != "who, true target"}
    if big:
        k = max(big, key=big.get)
        out.append(f"Largest of the three fixed gaps: {k} {big[k]:+.3f}" + ("" if big[k] >= MIN_EFFECT else
                                                                           " (below 0.05: none qualifies)"))
    return g, out


def main():
    sel_path, croot, froot = sys.argv[1:4]
    sel = json.loads((paths.MOVING / sel_path).read_text())["layouts"]
    sc = {**load(sel, croot, CEILING), **load(sel, froot, CONFIRM)}
    names = [lay["name"] for lay in sel]
    print(f"# The ceiling runs against the predictions and readings ({sel_path}; {croot}; {froot})\n")
    for top in TOPS:
        tag = "as scored, the decisions" if top == 1.2 else "sensitivity"
        print(f"\n## Band top {top} m ({tag})\n")
        print("| arm | episodes | strict mean | front part | behind part | strict at the front (pooled s) |"
              " strict with someone ahead (pooled s) |")
        print("|---|---|---|---|---|---|---|")
        for arm in CEILING + CONFIRM:
            v = [sc[(n, arm)][top] for n in names if sc.get((n, arm))]
            if not v:
                print(f"| {arm} | 0 | - | - | - | - | - |")
                continue
            fr, fs = pooled(sc, names, arm, top, "front")
            be, bs = pooled(sc, names, arm, top, "behind")
            print(f"| {arm} | {len(v)} | {statistics.mean(x['all'] for x in v):.3f} | "
                  f"{statistics.mean(x['front'] for x in v):.3f} | {statistics.mean(x['behind'] for x in v):.3f} | "
                  f"{fmt(fr, '.3f')} ({fs}) | {fmt(be, '.3f')} ({bs}) |")
        g, out = readings(sc, names, top)
        print("\n| gap (a - b) | n | mean | front part | behind part | wins/losses/ties | sign p | 95% lower bound |")
        print("|---|---|---|---|---|---|---|---|")
        for k, a, b in GAPS:
            x = g[k]
            if x is None:
                print(f"| {k} ({a} - {b}) | - | - | - | - | - | - | - |")
                continue
            print(f"| {k} ({a} - {b}) | {x['n']} | {x['mean']:+.3f} | {x['front']:+.3f} | {x['behind']:+.3f} | "
                  f"{x['w']}/{x['l']}/{x['t']} | {x['p']:.3f} | {x['lb']:+.3f} |")
        print("\nPredictions fixed before the runs:")
        ow = [sc[(n, "OW")][top]["all"] for n in names if sc.get((n, "OW"))]
        p1 = statistics.mean(ow) >= 0.7 if ow else None
        f_ot, _ = pooled(sc, names, "CT4OT", top, "front")
        f_o, _ = pooled(sc, names, "CT4O", top, "front")
        b_ot, _ = pooled(sc, names, "CT4OT", top, "behind")
        p2 = None if f_ot is None or f_o is None else (f_ot >= 0.5 and f_o < f_ot)
        three = {k: g[k]["mean"] for k in ("place and follow", "target", "who, model's target") if g[k]}
        p3 = (None if b_ot is None or len(three) < 3 else
              (0.25 <= b_ot <= 0.45 and max(three, key=three.get) == "place and follow"))
        for name, v in (("P1 OW mean >= 0.7", p1), ("P2 CT4OT front >= 0.5 and CT4O lower there", p2),
                        ("P3 CT4OT with someone ahead 0.25-0.45 and place-and-follow the largest gap", p3)):
            print(f"- {name}: {'-' if v is None else ('held' if v else 'not held')}")
        print("\nReadings:")
        for line in out:
            print(f"- {line}")


if __name__ == "__main__":
    main()
