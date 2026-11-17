"""Zero calls. SNAP_AMENDMENT_1's scene set, picked mechanically from layouts_v2.csv (scan_v2.py), written to
selected_v2.csv for render_subset_v2.py.

    PYTHONPATH= .venv/bin/python results/still-scenes/group-of-four-case/snapset/select_v2.py [--bearings rule|literal]

--bearings rule (default): the allowed bearings the amendment's rule gives on bearing_sweep_out.txt as printed
  (median in the bin, chest covered in <= 10 of 20). literal: the list the amendment's text gives, with "70 and 75
  (default)" read as the two numbers it names, 70 and 75; the true default, 72.5, is left out, since its 15-40 layouts
  are the development case's near-copies that the amendment's "Not taken" rules out. Writes selected_v2.csv or
  selected_v2_literal.csv.

Order everywhere: seed ascending, then configuration in scan_v2.CONFIGS order (P0, then c<size>b<bearing>), eval
before examples, row index. Each configuration patches one group size; the others stay at the default.
- Occluded (12): cells in the fixed order (40-55, cg2), (40-55, cg3), (40-55, cg4), (15-40, cg2), (15-40, cg3),
  (15-40, cg4); 2 per cell. A layout qualifies for a bin when its bearing is allowed for that bin and its cylinder
  share is in it (15-40: [0.15, 0.40); 40-55: [0.40, 0.55]). At most 1 layout per seed in the set, at most 2 per
  (situation, bearing). A cell left short gives its places to the other situations of its bin, in the order cg2, cg3,
  cg4, recorded in `note`. Spares: then, cell by cell in the same order, the next 2 qualifying layouts from seeds not
  yet in the set (no pair cap on spares; the assembler keeps the cap on the final 12).
- One set per (seed, split, row): the same row at another bearing shows the same group again, so A and B skip the rows
  the occluded set took, and B those A took.
- A (6): default bearings (P0), cg2, cg3 and no_one_person, 2 each, target < 5% hidden and >= 90% in the image:
  selected.csv's picks that qualify (its cg4 pick does not: the amendment's situations exclude it), then the first in
  order to fill each situation; 1 spare each.
- B (6): target < 5% hidden and covering >= 15% of another person's in-image cylinder samples, at any allowed (either
  bin) or default bearing, or no_one_person under any configuration; 1 per seed, at most 2 per (situation, bearing);
  2 spares. If fewer than 4 qualify, B is made up to 6 with selected.csv's visitor-place layouts (a visitor 0.75-0.85 m
  from the anchor), labelled weak, their spares after them.
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from scan_v2 import CONFIGS                       # noqa: E402

ALLOWED = {
    "rule": {"15-40": {2: (20.0,), 3: (15.0, 50.0), 4: (15.0, 70.0)},
             "40-55": {2: (15.0,), 3: (), 4: (10.0, 35.0)}},
    "literal": {"15-40": {2: (20.0,), 3: (15.0, 55.0), 4: (30.0, 70.0, 75.0)},
                "40-55": {2: (15.0,), 3: (50.0,), 4: (35.0, 10.0)}},
}
CELLS = [("40-55", 2), ("40-55", 3), ("40-55", 4), ("15-40", 2), ("15-40", 3), ("15-40", 4)]
PER_CELL, SPARES, B_WANT, B_SPARES, B_MIN = 2, 2, 6, 2, 4
ORDER = list(CONFIGS)
FIELDS = ["set", "bin", "situation", "n_group", "bearing", "config", "b2", "b3", "b4", "seed", "split", "index",
          "twin_id", "scene_name", "hidden_real", "occluders_real", "b_person", "b_hidden_by_target",
          "b_person_in_image", "spare", "note"]


def key(r):
    return int(r["seed"]), ORDER.index(r["config"]), r["split"] != "eval", int(r["index"])


def in_bin(share: float, b: str) -> bool:
    return 0.15 <= share < 0.40 if b == "15-40" else 0.40 <= share <= 0.55


def kind(r) -> str:
    """The row's recipe: conversation_group_2/3/4 or no_one_person."""
    return r["twin_id"].split("_", 2)[-1]


def out_row(r, set_name, b="", spare=False, note=""):
    c = CONFIGS[r["config"]]
    return {**{k: r.get(k, "") for k in FIELDS if k in r}, "set": set_name, "bin": b,
            "situation": f"cg{r['n_group']}" if r.get("n_group") else ("no_one_person" if r["target_kind"] == "person"
                                                                       else r["twin_id"].split("_", 2)[-1]),
            "b2": c[2], "b3": c[3], "b4": c[4], "spare": spare, "note": note,
            "scene_name": f"s{r['seed']}{r['config'].lower()}_real_{r['twin_id']}"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bearings", choices=tuple(ALLOWED), default="rule")
    args = ap.parse_args(argv)
    allowed = ALLOWED[args.bearings]
    rows = [r for r in csv.DictReader(open(HERE / "layouts_v2.csv")) if not r["error"]]
    person = sorted((r for r in rows if r["target_kind"] == "person"), key=key)

    def qualifies(r, b, n):
        return (r["n_group"] == str(n) and float(r["bearing"]) in allowed[b][n]
                and in_bin(float(r["hidden_real"]), b))

    picked, used, pair = [], set(), defaultdict(int)

    def take(b, n, want, note=""):
        got = 0
        for r in person:
            if got == want:
                break
            if qualifies(r, b, n) and r["seed"] not in used and pair[(n, r["bearing"])] < 2:
                picked.append(out_row(r, "occluded", b, note=note))
                used.add(r["seed"])
                pair[(n, r["bearing"])] += 1
                got += 1
        return got

    short = defaultdict(int)
    for b, n in CELLS:
        short[b] += PER_CELL - take(b, n, PER_CELL)
    for b in ("40-55", "15-40"):
        for n in (2, 3, 4):
            if short[b]:
                short[b] -= take(b, n, short[b], note=f"place moved in from a short cell of bin {b}")
        if short[b]:
            print(f"bin {b}: {short[b]} places left unfilled")
    for b, n in CELLS:
        got = 0
        for r in person:
            if got == SPARES:
                break
            if qualifies(r, b, n) and r["seed"] not in used:
                picked.append(out_row(r, "occluded", b, spare=True))
                used.add(r["seed"])
                got += 1
    # A (seed, split, row) goes to one set only: the same row at another bearing is the same group seen again.
    taken = {(p["seed"], p["split"], p["index"]) for p in picked}
    # Set A: selected.csv's unhidden-person picks that the amendment's situations admit, re-measured here at the
    # default bearings (P0), then the first in order to fill 2 per situation, then 1 spare each.
    p0 = {(r["seed"], r["twin_id"]): r for r in person if r["config"] == "P0"}

    def a_ok(r):
        return (kind(r) in ("conversation_group_2", "conversation_group_3", "no_one_person")
                and float(r["hidden_real"]) < 0.05 and float(r["target_in_image"]) >= 0.9
                and (r["seed"], r["split"], r["index"]) not in taken)
    old_a = [p0[(r["seed"], r["twin_id"])] for r in csv.DictReader(open(HERE / "selected.csv"))
             if r["role_in_set"] == "control_person" and r["spare"] == "False" and (r["seed"], r["twin_id"]) in p0]
    by_kind = defaultdict(list)
    for r in [*(r for r in old_a if a_ok(r)), *(r for r in p0.values() if a_ok(r))]:
        if r not in by_kind[kind(r)]:
            by_kind[kind(r)].append(r)
    for k in ("conversation_group_2", "conversation_group_3", "no_one_person"):
        got = by_kind[k][:3]
        picked += [out_row(r, "A", spare=i >= 2, note="" if r in old_a else "not in selected.csv")
                   for i, r in enumerate(got)]
        taken |= {(r["seed"], r["split"], r["index"]) for r in got}
    # Set B: target in front.
    any_allowed = {n: {*allowed["15-40"][n], *allowed["40-55"][n], CONFIGS["P0"][n]} for n in (2, 3, 4)}
    b_cands = [r for r in person if float(r["hidden_real"]) < 0.05 and float(r["b_hidden_by_target"]) >= 0.15
               and (not r["n_group"] or float(r["bearing"]) in any_allowed[int(r["n_group"])])
               and (r["seed"], r["split"], r["index"]) not in taken]
    b_seen, b_pair, b_got = set(), defaultdict(int), []
    for r in b_cands:
        situation_bearing = (kind(r), r["bearing"])
        if r["seed"] in b_seen or (b_pair[situation_bearing] >= 2 and len(b_got) < B_WANT):
            continue
        b_got.append(r)
        b_seen.add(r["seed"])
        b_pair[situation_bearing] += 1
        if len(b_got) == B_WANT + B_SPARES:
            break
    print(f"B: {len(b_cands)} qualifying layouts in {len({r['seed'] for r in b_cands})} seeds "
          f"(hidden person >= 50% in the image: {sum(float(r['b_person_in_image']) >= 0.5 for r in b_cands)})")
    picked += [out_row(r, "B", spare=i >= B_WANT) for i, r in enumerate(b_got)]
    if len(b_got) < B_MIN:
        old = [r for r in csv.DictReader(open(HERE / "selected.csv")) if r["role_in_set"] == "control_object"]
        need = B_WANT - min(len(b_got), B_WANT)
        for i, r in enumerate(old):
            r = {**r, "config": "P0", "n_group": "", "bearing": ""}
            picked.append(out_row(r, "B", spare=i >= need, note="fallback: place with a visitor 0.75-0.85 m; weak"))
    name = "selected_v2.csv" if args.bearings == "rule" else "selected_v2_literal.csv"
    with open(HERE / name, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(picked)
    summary = defaultdict(int)
    for p in picked:
        summary[(p["set"], p["bin"], p["situation"], p["bearing"], p["spare"])] += 1
    for k in sorted(summary, key=str):
        print(f"  {summary[k]:2d}  {k}")
    print(f"{len(picked)} rows -> {name}; renders in {len({(p['seed'], p['config']) for p in picked})} (seed, config) runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
