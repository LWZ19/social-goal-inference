"""Round 5, check 2 (EXPLORATORY for QB v2; RESULTS_1.md): QB v2 against W0 and CT4, robot idle, threshold decision,
chord direction. Pilot layouts (with the proxy goal) and design cells × 4 conditions. Per tick from delivery:
false-member ticks (P >= 0.5, present, not in the dump's queue) split as round4/f_belief/exmember.py; recall (queue
members present with P >= 0.5); right / cut / far / empty; last switches per minute. Zero calls.

    .venv/bin/python round5/c_eval.py [pilot|design|both]
"""
import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "round4"))
sys.path.insert(0, str(HERE.parent / "round3"))
sys.path.insert(0, str(HERE.parent))
from b_pl import CT4, CT5, classify, line_of, pilot_cells, proxy_ok, run  # noqa: E402
from ct4 import _named  # noqa: E402
from g_pilot_predict import WIN  # noqa: E402
from qb import QB, QB3  # noqa: E402
from r3lib import CONDITIONS, design_cells  # noqa: E402
from replay_lib import dump_tracks  # noqa: E402

INF = float("inf")
VARIANTS = {
    "CT4": (CT4, None),
    "W0": (CT5, WIN),
    "QB v2 (away only)": (QB, {"BOUT_PATH_M": INF}),
    "QB v3 (v2 + tail joins)": (QB3, None),
    "QB v4 (v3, tail joins need arrival)": (QB3, {"TAIL_NEEDS_ARRIVAL": True}),
}


def ever_in_queue(cell):
    """t -> set of ids in the dump's queue at any frame up to t."""
    seen, out = set(), []
    for f in cell.dump["frames"]:
        seen |= {p["id"] for p in f["people"] if p.get("in_queue")}
        out.append((f["t"], set(seen)))
    return out


def ever_at(table, t):
    best = set()
    for ts, s in table:
        if ts <= t + 1e-9:
            best = s
        else:
            break
    return best


def evaluate(cells_stress, lines=None):
    k, fm, sw, mins = Counter(), Counter(), 0, 0.0
    rec_n = rec_d = 0
    ok = n_ok = 0
    for cell, stress, cls, params, key in cells_stress:
        rows = run(cell, stress, cls, params, "threshold", "chord")
        named = _named(cell, dump_tracks(cell.dump, cell.asked_t))
        table = ever_in_queue(cell)
        k.update(classify(r) for r in rows)
        s = sum(1 for a, b in zip(rows, rows[1:]) if a["last"] != b["last"])
        mins += max((rows[-1]["t"] - rows[0]["t"]) / 60, 1e-9) if rows else 0.0
        sw += s
        for r in rows:
            present = set(r["pos"])
            q = r["q"] & present
            rec_d += len(q)
            rec_n += len(q & r["roster"])
            ever = ever_at(table, r["t"])
            for i in r["roster"] - r["q"]:
                kind = ("ex-member" if i in ever else
                        ("bystander" if i in cell.bystanders else "walker") + (", named" if i in named else ", not named"))
                fm[kind] += 1
            if lines is not None and r["q"]:
                n_ok += 1
                ok += proxy_ok(r, lines[key])
    scored = sum(v for kk, v in k.items() if kk is not None)
    out = {x: k[x] / max(scored, 1) for x in ("right", "cut", "far", "empty")}
    out.update(sw=sw / max(mins, 1e-9), recall=rec_n / max(rec_d, 1), fm=fm, ticks=scored,
               proxy=(ok / max(n_ok, 1)) if lines is not None else None)
    return out


def show(name, s):
    fm = s["fm"]
    walkers = fm["walker, named"] + fm["walker, not named"]
    print(f"| {name} | {s['recall']:.3f} | {fm['ex-member']} | {walkers} | {fm['bystander, named']} | "
          f"{fm['bystander, not named']} | {s['right']:.3f} | {s['cut']:.3f} | {s['far']:.3f} | {s['empty']:.3f} | "
          f"{s['sw']:.1f} | " + ("-" if s["proxy"] is None else f"{s['proxy']:.3f}") + " |")


HEAD = ("| Variant | recall | false: ex-member | false: walker | false: named bystander | false: other bystander | "
        "right | cut | far | empty | switches / min | proxy goal in line |\n|---|---|---|---|---|---|---|---|---|---|---|---|")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    print("# Round 5, check 2 (EXPLORATORY: QB v2's bound was chosen after check 1)\n")
    if which in ("pilot", "both"):
        cs, _ = pilot_cells("runs_pilot3/dsv41")
        lines = {lay["name"]: line_of(lay["name"]) for lay, _ in cs}
        print(f"## Pilot layouts, robot idle ({len(cs)})\n\n{HEAD}")
        for name, (cls, params) in VARIANTS.items():
            show(name, evaluate([(c, None, cls, params, lay["name"]) for lay, c in cs], lines))
    if which in ("design", "both"):
        cs = design_cells()
        print(f"\n## Design cells × 4 conditions, pooled ({len(cs)} cells)\n\n{HEAD}")
        for name, (cls, params) in VARIANTS.items():
            show(name, evaluate([(c, st, cls, params, None) for c in cs for st in CONDITIONS.values()]))


if __name__ == "__main__":
    main()
