"""Zero calls, no Blender, no GPU. For each placeholder seed, the layouts the realistic generator would replay
(sgi.synth.generate.accepted_layouts: the first pass, eval_00-13 and examples_00-06, with the placeholder's own CPU
depth render for the scene check), and for each its target's hidden share by group4/occlusion.py's cylinder model
(imported unchanged), on two truths:
  - the placeholder twin's (the layout's people), and
  - the realistic `real` dressing's at attempt 0 (dress(): the same visitors in the same places, plus staff), which
    is what occlusion.py would read off the rendered scene.json if attempt 0 is the one kept.
For a target that is not a person: the nearest person, their distance to the anchor and footprint, how much of
them the picture shows (cylinder samples in the image and not hidden), and whether the anchor pixel lies inside a
person's projected box (when the snap could have a candidate on an object click).

    PYTHONPATH= nice -n 10 .venv/bin/python results/2026-11-01-push/group4/snapset/scan.py \
        --seeds 400-449 --workers 4

Writes layouts.csv (every layout) and candidates.csv (the layouts that fall in a SNAP_FROZEN.md set) beside this file.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("occlusion", HERE.parent / "occlusion.py")
occ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(occ)                      # the frozen measure, unchanged

NEAR_M = 1.0                                       # SNAP_FROZEN: a person within 1 m of an object target
SHOWN_MIN = 0.5                                    # "visible in the picture": half of their silhouette samples shown
SAMPLES = 15 * 18                                  # occlusion.silhouettes samples per person
SPLIT_NO = {"eval": 0, "examples": 1}
FIELDS = ["seed", "split", "index", "twin_id", "real_id", "situation", "n_visitors", "n_staff_real", "target_kind",
          "target_label", "hidden_twin", "occluders_twin", "hidden_real", "occluders_real", "target_in_image",
          "bin", "near_person", "near_role", "near_anchor_m", "near_footprint_m", "near_shown", "anchor_in_box",
          "set"]


def _bin(share):
    if share is None:
        return ""
    return ("unhidden" if share < 0.05 else "15-40" if 0.15 <= share < 0.40 else "40-70" if 0.40 <= share <= 0.70
            else "5-15" if share < 0.15 else ">70")


def _shown(scene: dict, pid: str) -> tuple[float, float]:
    """Share of the person's 270 cylinder samples that lie in the image, and that are in it and not hidden."""
    s, Tinv, K = occ.silhouettes(scene)
    pts = s[pid]["pts"]
    hidden = sum(any(occ.covers(Tinv, K, s[o], u, v, d) for o in s if o != pid) for u, v, d in pts)
    return len(pts) / SAMPLES, (len(pts) - hidden) / SAMPLES


def _anchor_in_box(scene: dict, anchor) -> list[str]:
    """People whose projected cylinder box contains the target anchor's pixel, at any depth."""
    s, Tinv, K = occ.silhouettes(scene)
    q = occ.project(Tinv, K, anchor)
    if q is None:
        return []
    return [pid for pid in s if occ.covers(Tinv, K, s[pid], q[0], q[1], math.inf)]


def _dist_polygon(xy, polygon) -> float:
    from sgi.polygons import distance_to_polygon
    return float(distance_to_polygon(xy, polygon))


def _rows_for_seed(seed: int) -> list[dict]:
    from sgi.config import load_config
    from sgi.synth.generate import accepted_layouts
    from sgi.synth.realistic.dress import VARIANTS, dress
    from sgi.synth.realistic.library import load_library
    cfg, library, rows = load_config(), load_library(), []
    with tempfile.TemporaryDirectory(prefix=f"snapscan_{seed}_") as work:
        for split, index, layout, twin in accepted_layouts(Path(work), cfg, seed):
            twin_d = twin.model_dump(mode="json")
            dressed = dress(layout, VARIANTS["real"], library, (seed, SPLIT_NO[split], index, 0))
            real_d = {**twin_d, "ground_truth": dressed.scene_gt.model_dump(mode="json")}
            target = twin.ground_truth.targets[0]
            situation = twin.episodes[0].ground_truth.situation
            row = dict(seed=seed, split=split, index=index, twin_id=twin.scene_id, real_id=f"real_{twin.scene_id}",
                       situation=situation, n_visitors=len(twin.ground_truth.people),
                       n_staff_real=sum(p.role is not None for p in dressed.scene_gt.people),
                       target_kind=target.kind, target_label=target.label)
            if target.kind == "person":
                h_t, by_t = occ.hidden_share(twin_d, target.person_id)
                h_r, by_r = occ.hidden_share(real_d, target.person_id)
                in_image, _ = _shown(twin_d, target.person_id)
                row.update(hidden_twin=None if h_t is None else round(h_t, 3), occluders_twin=" ".join(by_t or []),
                           hidden_real=None if h_r is None else round(h_r, 3), occluders_real=" ".join(by_r or []),
                           target_in_image=round(in_image, 3), bin=_bin(h_r))
                row["set"] = ("occluded" if row["bin"] in ("15-40", "40-70") else
                              "control_person" if row["bin"] == "unhidden" and in_image >= 0.9 else "")
            else:
                people = dressed.scene_gt.people                    # staff included: they stand in the picture
                ax, ay = target.anchor[0], target.anchor[1]
                best = None
                for p in people:
                    d_a = math.hypot(p.x - ax, p.y - ay)
                    d_f = _dist_polygon((p.x, p.y), target.footprint) if target.footprint else d_a
                    in_image, shown = _shown(real_d, p.id)
                    key = (not (min(d_a, d_f) <= NEAR_M and shown >= SHOWN_MIN), d_a)
                    if best is None or key < best[0]:
                        best = (key, p, d_a, d_f, shown)
                if best is not None:
                    _, p, d_a, d_f, shown = best
                    row.update(near_person=p.id, near_role=p.role or "", near_anchor_m=round(d_a, 2),
                               near_footprint_m=round(d_f, 2), near_shown=round(shown, 3))
                row["anchor_in_box"] = " ".join(_anchor_in_box(real_d, target.anchor))
                near = best is not None and min(best[2], best[3]) <= NEAR_M and best[4] >= SHOWN_MIN
                row["set"] = ("control_object" if near and target.kind == "object" else
                              "control_place" if near else "")
            rows.append(row)
    return rows


def _seeds(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True, help="e.g. 400-449 or 400,402-405")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="", help="suffix for the CSV names")
    args = ap.parse_args(argv)
    seeds, begun, rows = _seeds(args.seeds), time.time(), []
    with Pool(args.workers) as pool:
        for got in pool.imap_unordered(_rows_for_seed, seeds):
            rows += got
            print(f"seed {got[0]['seed']} done at {time.time() - begun:.0f} s", flush=True)
    rows.sort(key=lambda r: (r["seed"], r["split"], r["index"]))
    for name, keep in ((f"layouts{args.tag}.csv", rows), (f"candidates{args.tag}.csv", [r for r in rows if r["set"]])):
        with open(HERE / name, "w", newline="") as f:
            w = csv.DictWriter(f, FIELDS)
            w.writeheader()
            w.writerows(keep)
    # Summary.
    from collections import Counter
    c = Counter((r["set"], r["bin"] if r["target_kind"] == "person" else "", r["situation"],
                 r["twin_id"].split("_", 2)[-1]) for r in rows if r["set"])
    print(f"{len(seeds)} seeds, {len(rows)} layouts, {time.time() - begun:.0f} s")
    for k, n in sorted(c.items()):
        print(f"  {n:4d}  {k}")
    person = [r for r in rows if r["target_kind"] == "person"]
    print("person-target hidden share (real truth) by row:")
    by_row = {}
    for r in person:
        by_row.setdefault(r["twin_id"].split("_", 2)[-1], []).append(r["hidden_real"])
    for k, v in sorted(by_row.items()):
        v = np.array([x for x in v if x is not None])
        print(f"  {k:32} n={len(v):3d} min={v.min():.2f} median={np.median(v):.2f} max={v.max():.2f} "
              f">=0.15: {(v >= 0.15).sum()}  >=0.40: {(v >= 0.40).sum()}")
    diff = [r for r in person if r["hidden_twin"] != r["hidden_real"]]
    print(f"person targets whose hidden share differs twin vs real (staff): {len(diff)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
