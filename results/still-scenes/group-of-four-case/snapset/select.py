"""Zero calls. A fixed, mechanical pick from candidates.csv (scan.py) of the SNAP_FROZEN.md scene set, plus spares,
written to selected.csv for render_subset.py. Rules (set before any picture or call):
  - order: seed ascending, then eval before examples, then row index;
  - occluded: per bin (15-40, 40-70), the first WANT + SPARE layouts, round-robin over rows (situation variety where
    the scan offers it);
  - control_person (< 5% hidden, >= 90% of the silhouette in the image): WANT + SPARE, round-robin over rows,
    seeds already picked first (one replay per seed serves several scenes);
  - control_object: target kinds "object" first; then places, a visitor (not staff) within 1 m first, and among
    those, layouts whose anchor pixel falls inside a person's projected box first (the snap has a candidate there).
A spare is rendered too; render_subset.py's assemble keeps the first WANT per set that render and keep their bin.
    PYTHONPATH= .venv/bin/python results/still-scenes/group-of-four-case/snapset/select.py
"""
import csv
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

HERE = Path(__file__).resolve().parent
WANT = {"15-40": 6, "40-70": 6, "control_person": 6, "control_object": 6}
SPARE = 2


def order(r):
    return int(r["seed"]), r["split"] != "eval", int(r["index"])


def row_kind(r):
    return r["twin_id"].split("_", 2)[-1]


def round_robin(rows, n, first_seeds=()):
    by = defaultdict(list)
    for r in sorted(rows, key=lambda r: (int(r["seed"]) not in first_seeds, *order(r))):
        by[row_kind(r)].append(r)
    out = [r for group in zip_longest(*by.values()) for r in group if r is not None]
    return out[:n]


rows = list(csv.DictReader(open(HERE / "candidates.csv")))
picked = []
for b in ("15-40", "40-70"):
    got = round_robin([r for r in rows if r["set"] == "occluded" and r["bin"] == b], WANT[b] + SPARE)
    picked += [{**r, "role_in_set": f"occluded {b}", "spare": i >= WANT[b]} for i, r in enumerate(got)]
    print(f"occluded {b}: {len(got)} picked of {WANT[b]} wanted + {SPARE} spares")
seeds = {int(r["seed"]) for r in picked}
got = round_robin([r for r in rows if r["set"] == "control_person"], WANT["control_person"] + SPARE, seeds)
picked += [{**r, "role_in_set": "control_person", "spare": i >= WANT["control_person"]} for i, r in enumerate(got)]
print(f"control_person: {len(got)} picked")
seeds |= {int(r["seed"]) for r in got}
objects = [r for r in rows if r["set"] in ("control_object", "control_place")]
objects.sort(key=lambda r: (r["target_kind"] != "object", r["near_role"] != "", not r["anchor_in_box"],
                            int(r["seed"]) not in seeds, *order(r)))
got = objects[:WANT["control_object"] + SPARE]
picked += [{**r, "role_in_set": "control_object", "spare": i >= WANT["control_object"]} for i, r in enumerate(got)]
print(f"control_object: {len(got)} picked ({sum(r['target_kind'] == 'object' for r in got)} of kind object, "
      f"{sum(r['near_role'] == '' for r in got)} with a visitor within 1 m, {sum(bool(r['anchor_in_box']) for r in got)} "
      "with the anchor inside a person's box)")
fields = list(picked[0])
with open(HERE / "selected.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fields)
    w.writeheader()
    w.writerows(picked)
print(f"{len(picked)} layouts in {len({r['seed'] for r in picked})} seeds -> selected.csv")
