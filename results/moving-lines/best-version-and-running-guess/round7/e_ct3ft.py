"""Round 7 part D (ROUND7_D.md), DESCRIPTIVE: CT3FT read by its fixed rule. Zero calls; read-only.

    .venv/bin/python round7/e_ct3ft.py SELECTION_JSON CEILING_ROOT CONFIRM_ROOT
    (from ideas2_1b/; paths relative to moving/. CT3FT, CT4OT, CT4PJT and OW from CEILING_ROOT; CT3F from CONFIRM_ROOT.)
"""
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from a_loss import MOVING  # noqa: E402
from b_band import TOPS  # noqa: E402
from c_gaps import MIN_EFFECT, gap, load, pooled  # noqa: E402

CEILING, CONFIRM = ("CT3FT", "CT4OT", "CT4PJT", "OW"), ("CT3F",)


def outcome(g1, g2):
    if g1["mean"] > 0.121 - MIN_EFFECT and g2["mean"] < MIN_EFFECT:      # see ROUND7_D: G2 = (CT4OT - CT3F) - G1
        return "A: lead 1 first; lead 2 closed (future work)"
    if g1["mean"] >= MIN_EFFECT:
        return "B: lead 1 first; lead 2 second, only after lead 1"
    return "C: target alone does not lift CT3F testably; who and target together go to the project as one question"


def main():
    sel_path, croot, froot = sys.argv[1:4]
    sel = json.loads((MOVING / sel_path).read_text())["layouts"]
    names = [lay["name"] for lay in sel]
    sc = {**load(sel, croot, CEILING), **load(sel, froot, CONFIRM)}
    print(f"# Round 7 part D, DESCRIPTIVE: CT3FT by ROUND7_D.md ({sel_path}; {croot}; {froot})")
    results = {}
    for top in TOPS:
        print(f"\n## Band top {top} m ({'the decision' if top == 1.2 else 'reported beside it'})\n")
        print("| arm | episodes | strict mean | front part | behind part | strict at the front (pooled s) | / OW |")
        print("|---|---|---|---|---|---|---|")
        ow = [sc[(n, "OW")][top]["all"] for n in names if sc.get((n, "OW"))]
        for arm in ("CT3FT", "CT3F", "CT4OT", "CT4PJT", "OW"):
            v = [sc[(n, arm)][top] for n in names if sc.get((n, arm))]
            if not v:
                print(f"| {arm} | 0 | - | - | - | - | - |")
                continue
            fr, fs = pooled(sc, names, arm, top, "front")
            m = statistics.mean(x["all"] for x in v)
            print(f"| {arm} | {len(v)} | {m:.3f} | {statistics.mean(x['front'] for x in v):.3f} | "
                  f"{statistics.mean(x['behind'] for x in v):.3f} | {'-' if fr is None else f'{fr:.3f}'} ({fs}) | "
                  f"{m / statistics.mean(ow):.2f} |" if ow else "")
        rows = (("G1 target for the best arm", "CT3FT", "CT3F"), ("G2 who once the target is right", "CT4OT", "CT3FT"),
                ("check: CT4OT - CT3F", "CT4OT", "CT3F"), ("reported: CT3FT - CT4PJT", "CT3FT", "CT4PJT"))
        g = {}
        print("\n| gap (a - b) | n | mean | front part | behind part | wins/losses/ties | sign p | 95% lower bound |")
        print("|---|---|---|---|---|---|---|---|")
        for k, a, b in rows:
            x = gap(sc, names, a, b, top)
            g[k] = x
            if x is None:
                print(f"| {k} ({a} - {b}) | - | - | - | - | - | - | - |")
                continue
            print(f"| {k} ({a} - {b}) | {x['n']} | {x['mean']:+.3f} | {x['front']:+.3f} | {x['behind']:+.3f} | "
                  f"{x['w']}/{x['l']}/{x['t']} | {x['p']:.3f} | {x['lb']:+.3f} |")
        g1, g2, chk = g["G1 target for the best arm"], g["G2 who once the target is right"], g["check: CT4OT - CT3F"]
        if g1 is None or g2 is None:
            print("\n(no CT3FT values: no reading)")
            continue
        print("\nReading (ROUND7_D.md):")
        if top == 1.2 and chk is not None and abs(chk["mean"] - 0.121) > 0.0015:
            print(f"- NOTE: CT4OT - CT3F is {chk['mean']:+.3f}, not the 0.121 the cuts assume; read G2 directly")
        out = outcome(g1, g2)
        results[top] = out[0]
        print(f"- outcome {out}")
        if out[0] in "AB" and g1["lb"] < 0:
            print("- G1's lower bound is below 0: point estimate only")
        print(f"- G1 front part {g1['front']:+.3f} of {g1['mean']:+.3f}: "
              + ("the gain is at the front" if g1["mean"] > 0 and g1["front"] > g1["mean"] / 2 else
                 "the gain is not mostly at the front"))
    if len(set(results.values())) > 1:
        print(f"\nThe outcome differs between band tops ({results}): both go to the project.")


if __name__ == "__main__":
    main()
