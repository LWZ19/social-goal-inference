"""Front-of-line fix (exploratory, 7 Nov): CT2f (CT2 + no step back once in line + vacate_rejoined) against CT2 on the
same 25 cells per model (EVAL5, seeds 10-14), CT2 from runs_i10 (current code), CT2f from runs_front. Zero calls.
    python3 front_score.py   (run from moving/; uses fill/mcompare.py and summarise_fresh.in_line)"""
import statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mcompare as C
from summarise_fresh import in_line

def front(e):                                   # share of the seconds with nobody ahead that are in line (strict)
    s = [x for x in e["samples"] if x["ahead"] == 0 and x["joined"]]
    return (sum(in_line(x, "strict") for x in s) / len(s)) if s else None

def back_steps(e):                              # seconds in which a joined robot moved more than 0.2 m back along the line
    s = e["samples"]
    return sum(1 for a, b in zip(s, s[1:]) if a["joined"] and b["joined"] and b["along"] - a["along"] > 0.2)

for model in ("dsv41", "sonnet55_low"):
    eps = {**C.load(f"runs_i10/{model}", C.EVAL5, range(10, 15), ("CT2",)),
           **C.load(f"runs_front/{model}", C.EVAL5, range(10, 15), ("CT2f",))}
    ps = C.pairs(eps, "CT2f", "CT2")
    r = C.row(model, ps)
    a = [x["share_strict"] for _, x, _ in ps]; b = [y["share_strict"] for _, _, y in ps]
    print(f"## {model}: {len(ps)} paired cells")
    print(f"- time in the right spot (strict): CT2f {statistics.mean(a):.3f} vs CT2 {statistics.mean(b):.3f}; "
          f"W/L/T {'/'.join(map(str, r['strict']))}, sign p {r['p_strict']:.4f}")
    fa = [front(eps[(*k, 'CT2f')]) for k, _, _ in ps]; fb = [front(eps[(*k, 'CT2')]) for k, _, _ in ps]
    both = [(x, y) for x, y in zip(fa, fb) if x is not None and y is not None]
    if both:
        print(f"- at the front (seconds with nobody ahead, {len(both)} cells with such seconds in both): "
              f"CT2f {statistics.mean(x for x, _ in both):.3f} vs CT2 {statistics.mean(y for _, y in both):.3f}")
    print(f"- seconds stepping back > 0.2 m after joining: CT2f {sum(back_steps(eps[(*k, 'CT2f')]) for k, _, _ in ps)} "
          f"vs CT2 {sum(back_steps(eps[(*k, 'CT2')]) for k, _, _ in ps)}")
    if model == "dsv41":
        k = ("cafe_slots", 10)
        if eps.get((*k, "CT2f")) and eps.get((*k, "CT2")):
            print(f"- the video's cell (cafe_slots seed 10): CT2f {C.measures(eps[(*k, 'CT2f')])['share_strict']:.3f} "
                  f"vs CT2 {C.measures(eps[(*k, 'CT2')])['share_strict']:.3f}")
    print()
