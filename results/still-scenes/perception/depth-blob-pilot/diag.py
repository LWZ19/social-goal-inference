"""Diagnostics for the blob pilot rerun: recall over people in the image, the "42 placeholder people" subsets, the
missed staff (are they behind a counter?), false blobs, and facing on the placeholder subsets.

Reads the per-scene cache of blobpilot.py and, for the staff check only, the scenes through research_harness.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blobpilot as B  # noqa: E402
from blobpilot import H  # noqa: E402
from sgi.geometry import lift, project  # noqa: E402

OUT = Path(__file__).resolve().parent
lines: list[str] = []


def say(s=""):
    print(s)
    lines.append(s)


recs = {}
refs = {}
for sref in H.distinct_scenes():
    r = B.scene_record(sref)
    recs[(r["set"], r["scene_id"])] = r
    refs[(r["set"], r["scene_id"])] = sref


def subset(pred):
    return [r for k, r in recs.items() if pred(k)]


def recall(rs, split, rad, people_pred=lambda p: True):
    hit = n = 0
    for r in rs:
        kept = [b for b in r["blobs"][split] if b["kept"]]
        people = [p for p in r["people"] if people_pred(p)]
        m = B.match(people, kept, rad)
        hit += len(m)
        n += len(people)
    return hit, n


def false_blobs(rs, split, rad=0.4):
    """Kept blobs left unmatched when matching against all truth people."""
    f = 0
    for r in rs:
        kept = [b for b in r["blobs"][split] if b["kept"]]
        m = B.match(r["people"], kept, rad)
        f += len(kept) - len(m)
    return f


SETS = {
    "plain": lambda k: k[0] == "plain",
    "real": lambda k: k[0] == "real",
    "staffed": lambda k: k[0] == "staffed",
    "ph eval_00-13 + dev seed0 (eval_14,15,20,21)": lambda k: k[0] in ("ph", "dev_seed0"),
    "ph eval_00-13 + dev seed1": lambda k: k[0] in ("ph", "dev_seed1"),
    "ph eval_00-13 + dev seed2": lambda k: k[0] in ("ph", "dev_seed2"),
    "ph eval_00-13 + all 3 dev seeds": lambda k: k[0] in ("ph", "dev_seed0", "dev_seed1", "dev_seed2"),
}
in_img = lambda p: p["in_image"]

say("== Recall over people whose body projects into the image (the pilot's denominators: 34, 42, 42, 42) ==")
say("split half-width | set | in-image people (all truth people) | recall@0.25 | @0.40 | @0.60 | false blobs / image @0.40")
for split in ("0.15", "0.25", "0.35"):
    for name, pred in SETS.items():
        rs = subset(pred)
        cells = []
        for rad in B.RADII:
            h, n = recall(rs, split, rad, in_img)
            cells.append(f"{h}/{n}")
        n_all = sum(len(r["people"]) for r in rs)
        fb = false_blobs(rs, split)
        say(f"{split} | {name:46s} | {n} ({n_all}) | " + " | ".join(cells) + f" | {fb}/{len(rs)} = {fb / len(rs):.2f}")
    say("")

# ---------------------------------------------------------------------------------------------- staff check
say("== In-image staff on real and staffed: found? lower body hidden? by what? ==")
say("per staff: matched at 0.4 m for split 0.15/0.25/0.35; samples on the vertical axis at 0.1-0.8 m: seen / hidden"
    " by a nearer surface; median height and camera-Z gap of the hiding surface; the sub-blob (split 0.25) holding"
    " the staff's xy: top, footprint, cells")
LOW = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
for key in sorted(recs):
    if key[0] not in ("real", "staffed"):
        continue
    r = recs[key]
    staff = [p for p in r["people"] if p["role"] and p["in_image"]]
    if not staff:
        continue
    sc = H.scene(refs[key].runs[0], key[1])
    obs = sc.observation
    cam = obs.camera
    for p in staff:
        found = []
        for split in ("0.15", "0.25", "0.35"):
            kept = [b for b in r["blobs"][split] if b["kept"]]
            found.append("Y" if p["id"] in B.match(r["people"], kept, 0.4) else "n")
        seen = hidden = 0
        occ_h, occ_gap = [], []
        for z in LOW:
            pr = project(cam, (p["x"], p["y"], z))
            if pr is None or not (0 <= pr[0] < cam.width and 0 <= pr[1] < cam.height):
                continue
            u, v, zc = pr
            d = float(obs.depth[math.floor(v), math.floor(u)])
            if math.isfinite(d) and d > 0 and abs(d - zc) <= 0.3:
                seen += 1
            elif math.isfinite(d) and d > 0 and d < zc - 0.3:
                hidden += 1
                w = lift(obs.depth, cam, u, v, window=1)
                if w is not None:
                    occ_h.append(float(w[2]))
                    occ_gap.append(zc - d)
        # sub-blob holding the staff's cell (nearest cell centre among blobs' centroids is not enough: use centroid
        # within 0.6 m and report all of them)
        near = [b for b in r["blobs"]["0.25"] if math.hypot(b["x"] - p["x"], b["y"] - p["y"]) <= 0.6]
        near_s = "; ".join(f"top {b['top']:.2f} foot {b['footprint']:.2f} cells {b['cells']} kept {b['kept']}"
                           f" d {math.hypot(b['x'] - p['x'], b['y'] - p['y']):.2f}" for b in near) or "none"
        oh = f"{np.median(occ_h):.2f}" if occ_h else "-"
        og = f"{np.median(occ_gap):.2f}" if occ_gap else "-"
        say(f"{key[0]:8s} {key[1]:36s} {p['id']} found {'/'.join(found)} vis(0.9-1.7) {p['visible_samples']}/9 "
            f"low seen {seen}/8 hidden {hidden}/8 hider z {oh} m gap {og} m | blobs within 0.6 m: {near_s}")
    H._SCENES.clear()

# ---------------------------------------------------------------------------------------------- false blobs
say("")
say("== False blobs (kept, unmatched at 0.4 m), split 0.25: per set, top / footprint / distance to the nearest target anchor ==")
for set_name in ("plain", "real", "staffed", "ph", "dev_seed0", "dev_seed1", "dev_seed2"):
    rows = []
    for key, r in sorted(recs.items()):
        if key[0] != set_name:
            continue
        kept = [b for b in r["blobs"]["0.25"] if b["kept"]]
        m = B.match(r["people"], kept, 0.4)
        used = set(m.values())
        for b in kept:
            if b["id"] in used:
                continue
            dt = min((math.hypot(b["x"] - t["x"], b["y"] - t["y"]) for t in r["targets"]), default=float("nan"))
            dp = min((math.hypot(b["x"] - p["x"], b["y"] - p["y"]) for p in r["people"]), default=float("nan"))
            rows.append((key[1], b, dt, dp))
    say(f"-- {set_name}: {len(rows)} false blobs")
    for sid, b, dt, dp in rows:
        say(f"   {sid:36s} top {b['top']:.2f} foot {b['footprint']:.2f} cells {b['cells']:3d} xy ({b['x']:.2f},{b['y']:.2f})"
            f" nearest target {dt:.2f} m nearest person {dp:.2f} m")

# ---------------------------------------------------------------------------------------------- facing on subsets
say("")
say("== Facing axis (minor axis of PCA, points / cells), within 45 deg mod 180, matched at 0.4 m; camera-ray null ==")
for split in ("0.15", "0.25", "0.35"):
    for name, pred in SETS.items():
        rs = subset(pred)
        pts = cel = cam = agree = 0
        n = 0
        for r in rs:
            kept = {b["id"]: b for b in r["blobs"][split] if b["kept"]}
            m = B.match(r["people"], list(kept.values()), 0.4)
            for p in r["people"]:
                if p["id"] not in m:
                    continue
                b = kept[m[p["id"]]]
                n += 1
                pts += B.axis_err(b["axis_minor"], p["yaw"]) <= B.FACING_TOL
                cel += B.axis_err(b["axis_minor_cells"], p["yaw"]) <= B.FACING_TOL
                cam += B.axis_err(b["cam_bearing"], p["yaw"]) <= B.FACING_TOL
                agree += B.axis_err(b["axis_minor"], b["cam_bearing"]) <= B.FACING_TOL
        say(f"{split} | {name:46s} | n {n:3d} | points {pts}/{n} = {pts / n:.3f} | cells {cel}/{n} = {cel / n:.3f}"
            f" | camera-ray axis {cam}/{n} = {cam / n:.3f} | PCA minor axis within 45 deg of camera ray {agree}/{n} = {agree / n:.3f}")
    say("")

say("Facing by situation (split 0.25, points minor axis; realistic = plain+real+staffed, placeholder = ph + 3 dev seeds)")
for fam, members in (("realistic", ("plain", "real", "staffed")), ("placeholder", ("ph", "dev_seed0", "dev_seed1", "dev_seed2"))):
    by = {}
    for key, r in recs.items():
        if key[0] not in members:
            continue
        sit = "line" if "waiting_line" in key[1] else "group" if "group" in key[1] else "audience" if "audience" in key[1] else "other"
        kept = {b["id"]: b for b in r["blobs"]["0.25"] if b["kept"]}
        m = B.match(r["people"], list(kept.values()), 0.4)
        for p in r["people"]:
            if p["id"] in m:
                b = kept[m[p["id"]]]
                t = by.setdefault(sit, [0, 0, 0])
                t[0] += 1
                t[1] += B.axis_err(b["axis_minor"], p["yaw"]) <= B.FACING_TOL
                t[2] += B.axis_err(b["cam_bearing"], p["yaw"]) <= B.FACING_TOL
    for sit, (n, h, c) in sorted(by.items()):
        say(f"  {fam:11s} {sit:9s} n {n:3d} PCA {h}/{n} = {h / n:.3f}  camera-ray {c}/{n} = {c / n:.3f}")

(OUT / "blobpilot_diag.txt").write_text("\n".join(lines) + "\n")
