"""Item 11 (I11_FROZEN.md): CT3F against CT2 (and CT3, reported) on a fresh layout range, DeepSeek. Zero calls;
read-only.

    .venv/bin/python results/moving-lines/best-version-new-layouts/i11_score.py SELECTION_JSON RUN_ROOT [RUN_ROOT ...]
    (SELECTION_JSON relative to results/moving-lines/, RUN_ROOT relative to results/data/moving/;
     e.g. test-layouts/i11_selection.json runs_i11/dsv41)

Measure (as the confirmatory test, CONFIRM_CT4PJ.md): strict share in line from the first service after delivery,
test-layouts/score_sweep.py's episode_scores()["after_service_strict"]. Ties |d| < 0.01. Exact two-sided sign test over
layouts with both values; paired-t one-sided 95% lower bound reported. Whole-episode strict share and cut-ins are
reported.
"""
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402

MOVING = paths.MOVING_RUNS                 # run folders, e.g. runs_i11/dsv41
sys.path.insert(0, str(paths.DESIGN))
sys.path.insert(0, str(paths.MOVING / "test-layouts"))
from score_sweep import episode_scores  # noqa: E402
from summarise_fresh import measures, sign_p  # noqa: E402

TIE = 0.01
ARMS = ("CT3F", "CT2", "CT3")


def t95(df):
    z = 1.6448536269514722
    return z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)


def score(root, sel):
    out = {}
    for lay in sel:
        for arm in ARMS:
            hits = sorted((MOVING / root).glob(f"**/{lay['name']}/seed{lay['seed']}/{arm}/episode.json"))
            if hits:
                e = json.loads(hits[0].read_text())
                m = measures(e)
                out[(lay["name"], arm)] = dict(after=episode_scores(e)["after_service_strict"], whole=m["share_strict"],
                                               cut=m["cut"], err=sum(1 for a in e.get("answers", [])[:1] if a.get("status") == "vlm_error"),
                                               none=not e.get("answers"))
    return out


def compare(sc, sel, a, b, key):
    # amendment 2 (I11_FROZEN.md): a layout whose first answer ended in vlm_error in either arm (a replay's cache miss)
    # is left out of the comparison and counted
    d = [sc[(l["name"], a)][key] - sc[(l["name"], b)][key] for l in sel
         if (l["name"], a) in sc and (l["name"], b) in sc
         and not sc[(l["name"], a)]["err"] and not sc[(l["name"], b)]["err"]
         and sc[(l["name"], a)][key] is not None and sc[(l["name"], b)][key] is not None]
    w, lo = sum(x >= TIE for x in d), sum(x <= -TIE for x in d)
    m = statistics.mean(d) if d else float("nan")
    lb = m - t95(len(d) - 1) * statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 2 else float("nan")
    return dict(n=len(d), w=w, l=lo, t=len(d) - w - lo, p=sign_p(w, lo), mean=m, lb=lb)


def main():
    sel = json.loads((paths.MOVING / sys.argv[1]).read_text())["layouts"]
    print(f"# Item 11: CT3F against CT2 ({sys.argv[1]})")
    for root in sys.argv[2:]:
        sc = score(root, sel)
        print(f"\n## {root}\n")
        for arm in ARMS:
            v = [sc[(l["name"], arm)] for l in sel if (l["name"], arm) in sc]
            if v:
                af = [x["after"] for x in v if x["after"] is not None]
                print(f"- {arm}: no first answer within the episode in {sum(1 for x in v if x['none'])} of {len(v)} "
                      f"(the answer came after the episode ended); first answer in vlm_error {sum(1 for x in v if x['err'])}")
                print(f"- {arm}: {len(v)} episodes ({len(af)} with a service after delivery); strict from the first "
                      f"service {statistics.mean(af):.3f}; whole-episode strict {statistics.mean(x['whole'] for x in v):.3f}; "
                      f"cut-in samples {sum(x['cut'] for x in v)}; vlm_error episodes {sum(1 for x in v if x['err'])}")
        print("\n| comparison | measure | n | W/L/T | sign p | mean d | 95% lower bound |\n|---|---|---|---|---|---|---|")
        res = {}
        for a, b in (("CT3F", "CT2"), ("CT3F", "CT3"), ("CT3", "CT2")):
            for key, label in (("after", "strict from the first service"), ("whole", "whole-episode strict")):
                r = compare(sc, sel, a, b, key)
                res[(a, b, key)] = r
                if r["n"]:
                    print(f"| {a} - {b} | {label} | {r['n']} | {r['w']}/{r['l']}/{r['t']} | {r['p']:.4f} | {r['mean']:+.3f} | {r['lb']:+.3f} |")
        h = res[("CT3F", "CT2", "after")]
        print(f"\nBAR H11 (CT3F - CT2, strict from the first service: wins > losses and sign p <= 0.05): "
              f"{h['w'] > h['l'] and h['p'] <= 0.05}  (n {h['n']})")
        hw = res[("CT3F", "CT2", "whole")]
        print(f"Reported: the same on whole-episode strict share: {hw['w'] > hw['l'] and hw['p'] <= 0.05}")


if __name__ == "__main__":
    main()
