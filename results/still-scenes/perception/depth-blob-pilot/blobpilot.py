"""Depth-blob pilot (I03 / I33), rewritten from the report's description and run on development scenes only.

Reads scenes only through research_harness (H.distinct_scenes, H.scene). No model call, no network, numpy only.

Method (as described in REPORT I03; the thresholds are the pilot's, used only to check its numbers):
  1. lift the depth image to world points with the scene's camera (sgi.geometry.lift_all; the floor is z = 0);
  2. keep points 0.15 m < z < 2.1 m;
  3. height map on a 5 cm grid (thresholds.grid_resolution_m): per cell the highest kept point;
  4. connected components of occupied cells (8-connected);
  5. split each component at local maxima: a seed is a cell whose height equals the maximum of the component's
     heights within a square window of half-width SPLIT_R (default 0.25 m; the pilot's value is not stated, so
     0.15 and 0.35 m are reported as variants, not chosen between); seed cells that touch are one seed; every
     cell of the component goes to its nearest seed (Euclidean, in the plane);
  6. keep body-sized blobs: top height (highest point) in [1.2, 2.0] m and footprint in [0.2, 0.8] m, where the
     footprint is the extent of the blob's cell centres along their first principal axis plus one cell.
  Blob position: the mean of its cell centres.

Matching to truth.people: one-to-one, greedy by increasing distance, a pair kept when within R (0.4 m primary;
0.25 and 0.6 m reported). A kept blob left unmatched at 0.4 m is a false blob.

Facing axis (I33): PCA of the xy of the matched blob's points (the kept points whose cell is in the blob); the
facing axis is the minor axis (perpendicular to the shoulder line), modulo 180 deg; a hit when within 45 deg of the
true yaw modulo 180. The major axis is also reported. Nulls: see summarise().

Per-scene outputs go to per_scene/<set>/<scene_id>.json, and are used as a cache (--force recomputes).
"""
from __future__ import annotations

import argparse
import collections
import gc
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H  # noqa: E402
from sgi.config import Thresholds  # noqa: E402
from sgi.geometry import lift_all, project  # noqa: E402
from sgi.gt.derive import _VISIBILITY_HEIGHTS_M, _sample_seen, is_visible  # noqa: E402

OUT = Path(__file__).resolve().parent
PER_SCENE = OUT / "per_scene"
Z_LO, Z_HI = 0.15, 2.1
TOP_LO, TOP_HI = 1.2, 2.0
FOOT_LO, FOOT_HI = 0.2, 0.8
SPLIT_RS = (0.15, 0.25, 0.35)
SPLIT_PRIMARY = 0.25
RADII = (0.25, 0.4, 0.6)
R_PRIMARY = 0.4
FACING_TOL = math.radians(45)
TH = Thresholds()
RES = TH.grid_resolution_m                          # 0.05 m
NEIGH8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def set_of(scene_ref) -> str:
    d = str(scene_ref.scenes_dir)
    for name in ("plain", "real", "staffed"):
        if f"realistic_final/{name}/" in d + "/":
            return name
    if "dev_new/seed" in d:
        return "dev_" + d.split("dev_new/")[1].split("/")[0]
    if d.endswith("placeholder/eval"):
        return "ph"
    raise SystemExit(f"unknown set: {d}")


# ------------------------------------------------------------------------------------------------ grid and blobs
def components(occ: np.ndarray) -> tuple[np.ndarray, int]:
    """8-connected labels of a boolean grid (0 = empty), by breadth-first search."""
    lab = np.zeros(occ.shape, dtype=np.int32)
    n = 0
    rows, cols = occ.shape
    for r0, c0 in zip(*np.nonzero(occ)):
        if lab[r0, c0]:
            continue
        n += 1
        lab[r0, c0] = n
        q = collections.deque([(r0, c0)])
        while q:
            r, c = q.popleft()
            for dr, dc in NEIGH8:
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols and occ[rr, cc] and not lab[rr, cc]:
                    lab[rr, cc] = n
                    q.append((rr, cc))
    return lab, n


def max_filter(a: np.ndarray, half: int) -> np.ndarray:
    """Square-window maximum (window 2*half+1), separable, padding with -inf."""
    w = 2 * half + 1
    p = np.pad(a, ((half, half), (0, 0)), constant_values=-np.inf)
    a1 = np.lib.stride_tricks.sliding_window_view(p, w, axis=0).max(axis=-1)
    p = np.pad(a1, ((0, 0), (half, half)), constant_values=-np.inf)
    return np.lib.stride_tricks.sliding_window_view(p, w, axis=1).max(axis=-1)


def split_component(hsub: np.ndarray, mask: np.ndarray, half: int) -> np.ndarray:
    """Sub-labels (1..k, 0 outside) of one component's cells, split at its local maxima (nearest seed)."""
    h = np.where(mask, hsub, -np.inf)
    seeds = mask & (h >= max_filter(h, half))
    seed_lab, k = components(seeds)
    cells = np.argwhere(mask)
    seed_cells = np.argwhere(seed_lab > 0)
    seed_ids = seed_lab[seed_lab > 0]
    out = np.zeros(mask.shape, dtype=np.int32)
    if k == 1:
        out[mask] = 1
        return out
    best = np.full(len(cells), np.inf)
    who = np.zeros(len(cells), dtype=np.int32)
    for start in range(0, len(seed_cells), 512):  # chunks keep memory modest
        sc = seed_cells[start:start + 512]
        d = ((cells[:, None, :] - sc[None, :, :]) ** 2).sum(-1)
        j = d.argmin(1)
        dj = d[np.arange(len(cells)), j]
        better = dj < best
        best[better] = dj[better]
        who[better] = seed_ids[start:start + 512][j[better]]
    out[cells[:, 0], cells[:, 1]] = who
    return out


def pca_axes(xy: np.ndarray):
    """(major unit vector, minor unit vector, eigenvalues descending) of 2-D points."""
    c = xy - xy.mean(0)
    cov = c.T @ c / max(len(xy), 1)
    w, v = np.linalg.eigh(cov)                       # ascending
    return v[:, 1], v[:, 0], (float(w[1]), float(w[0]))


def blobs_of(obs, split_rs=SPLIT_RS) -> dict:
    """Every sub-blob of the scene's height map, per split radius, with its features and whether it is kept."""
    pts = lift_all(obs.depth, obs.camera)
    pts = pts[(pts[:, 2] > Z_LO) & (pts[:, 2] < Z_HI)]
    ij = np.floor(pts[:, :2] / RES).astype(np.int64)
    lo = ij.min(0)
    ij -= lo
    shape = tuple(ij.max(0) + 1)
    hmap = np.full(shape, -np.inf)
    np.maximum.at(hmap, (ij[:, 0], ij[:, 1]), pts[:, 2])
    occ = np.isfinite(hmap)
    lab, n = components(occ)
    cam_xy = obs.camera.T_world_camera[:2, 3]
    out = {}
    for split_r in split_rs:
        half = int(round(split_r / RES))
        blob_lab = np.zeros(shape, dtype=np.int32)
        nb = 0
        # the cells of each component, grouped by label
        idx = np.argwhere(lab > 0)
        comp_ids = lab[idx[:, 0], idx[:, 1]]
        order = np.argsort(comp_ids, kind="stable")
        idx, comp_ids = idx[order], comp_ids[order]
        bounds = np.flatnonzero(np.diff(comp_ids)) + 1
        for cells in np.split(idx, bounds):
            comp = lab[cells[0, 0], cells[0, 1]]
            (r0, c0), (r1, c1) = cells.min(0), cells.max(0) + 1
            mask = lab[r0:r1, c0:c1] == comp
            sub = split_component(hmap[r0:r1, c0:c1], mask, half)
            k = sub.max()
            region = blob_lab[r0:r1, c0:c1]
            region[mask] = sub[mask] + nb
            nb += k
        point_blob = blob_lab[ij[:, 0], ij[:, 1]]
        blobs = []
        cell_idx = np.argwhere(blob_lab > 0)
        cell_ids = blob_lab[cell_idx[:, 0], cell_idx[:, 1]]
        order = np.argsort(cell_ids, kind="stable")
        cell_idx, cell_ids = cell_idx[order], cell_ids[order]
        starts = np.concatenate([[0], np.flatnonzero(np.diff(cell_ids)) + 1, [len(cell_ids)]])
        porder = np.argsort(point_blob, kind="stable")
        pstarts = np.searchsorted(point_blob[porder], np.arange(1, nb + 2))
        for a, b in zip(starts[:-1], starts[1:]):
            bid = int(cell_ids[a])
            cells = cell_idx[a:b]
            centres = (cells + lo + 0.5) * RES
            top = float(hmap[cells[:, 0], cells[:, 1]].max())
            if len(cells) >= 2:
                major, minor, _ = pca_axes(centres)
                proj = centres @ major
                foot = float(proj.max() - proj.min() + RES)
                proj2 = centres @ minor
                width = float(proj2.max() - proj2.min() + RES)
            else:
                foot = width = RES
            keep = TOP_LO <= top <= TOP_HI and FOOT_LO <= foot <= FOOT_HI
            blob = {"id": bid, "x": float(centres[:, 0].mean()), "y": float(centres[:, 1].mean()), "top": top,
                    "footprint": foot, "width": width, "cells": int(len(cells)), "kept": bool(keep)}
            if keep:
                bp = pts[porder[pstarts[bid - 1]:pstarts[bid]], :2]
                major, minor, ev = pca_axes(bp)
                blob["points"] = int(len(bp))
                blob["axis_minor"] = float(math.atan2(minor[1], minor[0]))
                blob["axis_major"] = float(math.atan2(major[1], major[0]))
                majc, minc, _ = pca_axes(centres)
                blob["axis_minor_cells"] = float(math.atan2(minc[1], minc[0]))
                blob["eig"] = ev
                blob["cam_bearing"] = float(math.atan2(blob["y"] - cam_xy[1], blob["x"] - cam_xy[0]))
            blobs.append(blob)
        out[f"{split_r:.2f}"] = blobs
    return out


# ------------------------------------------------------------------------------------------------ truth and matching
def visible_samples(person, obs) -> int:
    return int(sum(_sample_seen(obs, (person.x, person.y, z), TH) for z in _VISIBILITY_HEIGHTS_M))


def in_image(person, obs) -> bool:
    cam = obs.camera
    for z in (0.2, 1.0, 1.7):
        p = project(cam, (person.x, person.y, z))
        if p is not None and 0 <= p[0] < cam.width and 0 <= p[1] < cam.height:
            return True
    return False


def match(people: list[dict], blobs: list[dict], radius: float) -> dict[str, int]:
    """person id -> blob id, one-to-one, greedy by increasing distance within radius."""
    pairs = []
    for p in people:
        for b in blobs:
            d = math.hypot(p["x"] - b["x"], p["y"] - b["y"])
            if d <= radius:
                pairs.append((d, p["id"], b["id"]))
    pairs.sort()
    used_p, used_b, out = set(), set(), {}
    for d, pid, bid in pairs:
        if pid in used_p or bid in used_b:
            continue
        used_p.add(pid)
        used_b.add(bid)
        out[pid] = bid
    return out


def axis_err(a: float, b: float, period: float = math.pi) -> float:
    d = (a - b) % period
    return min(d, period - d)


def scene_record(sref, force=False) -> dict:
    sset = set_of(sref)
    path = PER_SCENE / sset / f"{sref.scene_id}.json"
    if path.exists() and not force:
        return json.loads(path.read_text())
    sc = H.scene(sref.runs[0], sref.scene_id)
    obs, truth = sc.observation, sc.truth
    people = [{"id": p.id, "x": p.x, "y": p.y, "yaw": p.yaw, "group": p.group, "role": p.role,
               "visible_samples": visible_samples(p, obs), "is_visible": bool(is_visible(p, obs, TH)),
               "in_image": in_image(p, obs),
               "is_target": any(t.person_id == p.id for t in truth.targets.values())} for p in truth.people]
    targets = [{"id": t.id, "kind": t.kind, "x": t.anchor[0], "y": t.anchor[1], "z": t.anchor[2]}
               for t in truth.targets.values()]
    blobs = blobs_of(obs)
    rec = {"set": sset, "scene_id": sref.scene_id, "run_used": sref.runs[0].folder, "people": people,
           "targets": targets, "blobs": blobs}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1))
    H._SCENES.clear()
    return rec


# ------------------------------------------------------------------------------------------------ summary
def summarise(recs: list[dict], out_lines: list[str]):
    def say(s=""):
        print(s)
        out_lines.append(s)

    rng = np.random.default_rng(0)
    sets = ["plain", "real", "staffed", "ph", "dev_seed0", "dev_seed1", "dev_seed2"]
    groups = {"plain": ["plain"], "real": ["real"], "staffed": ["staffed"], "ph (eval_00-13)": ["ph"],
              "dev_new seeds 0-2": ["dev_seed0", "dev_seed1", "dev_seed2"],
              "ph + dev_new": ["ph", "dev_seed0", "dev_seed1", "dev_seed2"]}
    say("== People per set (truth.people), scenes per set ==")
    for s in sets:
        rs = [r for r in recs if r["set"] == s]
        n = sum(len(r["people"]) for r in rs)
        staff = sum(1 for r in rs for p in r["people"] if p["role"])
        notvis = sum(1 for r in rs for p in r["people"] if not p["is_visible"])
        outimg = sum(1 for r in rs for p in r["people"] if not p["in_image"])
        say(f"{s:10s} scenes {len(rs):2d} people {n:3d} (with a role: {staff}; not visible by spec-4 rule: {notvis};"
            f" outside the image: {outimg})")
    say("people per scene, placeholder and dev_new:")
    for r in recs:
        if r["set"].startswith("dev") or r["set"] == "ph":
            say(f"  {r['set']:10s} {r['scene_id']:32s} {len(r['people'])}")

    for split in [f"{x:.2f}" for x in SPLIT_RS]:
        tag = " (PRIMARY)" if split == f"{SPLIT_PRIMARY:.2f}" else " (variant)"
        say("")
        say(f"================ split half-width {split} m{tag} ================")
        say("set                  | images | people | recall@0.25 | recall@0.40 | recall@0.60 | kept blobs | false@0.40 | false/img | false/img excl. <0.5 m of a target anchor | med pos err (m)")
        for g, members in groups.items():
            rs = [r for r in recs if r["set"] in members]
            n = sum(len(r["people"]) for r in rs)
            rec_at, false_at, kept_tot, false_far, errs = {}, 0, 0, 0, []
            for rad in RADII:
                hit = 0
                for r in rs:
                    kept = [b for b in r["blobs"][split] if b["kept"]]
                    m = match(r["people"], kept, rad)
                    hit += len(m)
                    if rad == R_PRIMARY:
                        kept_tot += len(kept)
                        unmatched = [b for b in kept if b["id"] not in set(m.values())]
                        false_at += len(unmatched)
                        false_far += sum(1 for b in unmatched if all(
                            math.hypot(b["x"] - t["x"], b["y"] - t["y"]) >= 0.5 for t in r["targets"]))
                        bid = {b["id"]: b for b in kept}
                        for p in r["people"]:
                            if p["id"] in m:
                                b = bid[m[p["id"]]]
                                errs.append(math.hypot(p["x"] - b["x"], p["y"] - b["y"]))
                rec_at[rad] = hit
            say(f"{g:20s} | {len(rs):6d} | {n:6d} | " + " | ".join(f"{rec_at[r]:3d}/{n:<3d} {rec_at[r] / max(n, 1):.3f}" for r in RADII)
                + f" | {kept_tot:10d} | {false_at:10d} | {false_at / max(len(rs), 1):9.2f} | {false_far / max(len(rs), 1):9.2f}"
                + f" | {np.median(errs) if errs else float('nan'):.3f}")

        say("")
        say(f"-- misses at 0.4 m (split {split}): person, role, visible samples of 9, in image, nearest sub-blob (any) within 0.6 m: top, footprint, why rejected")
        for r in recs:
            kept = [b for b in r["blobs"][split] if b["kept"]]
            m = match(r["people"], kept, R_PRIMARY)
            for p in r["people"]:
                if p["id"] in m:
                    continue
                near = sorted(r["blobs"][split], key=lambda b: math.hypot(p["x"] - b["x"], p["y"] - b["y"]))
                desc = "none"
                if near:
                    b = near[0]
                    d = math.hypot(p["x"] - b["x"], p["y"] - b["y"])
                    why = []
                    if not TOP_LO <= b["top"] <= TOP_HI:
                        why.append(f"top {b['top']:.2f}")
                    if not FOOT_LO <= b["footprint"] <= FOOT_HI:
                        why.append(f"footprint {b['footprint']:.2f}")
                    if b["kept"]:
                        why.append("kept but taken by another person or too far")
                    desc = f"d={d:.2f} top={b['top']:.2f} foot={b['footprint']:.2f} cells={b['cells']} [{', '.join(why)}]"
                say(f"  {r['set']:10s} {r['scene_id']:34s} {p['id']:4s} role={p['role']} vis={p['visible_samples']}/9 "
                    f"in_image={p['in_image']} target={p['is_target']} | nearest: {desc}")

        # facing
        say("")
        say(f"-- facing axis within 45 deg, matched people at 0.4 m (split {split})")
        say("set                  | matched | minor-axis mod180 | major-axis mod180 | minor (cells) mod180 | NULLS: uniform axis mod180 (analytic 0.5) | random dir mod360 (analytic 0.25; mean of 2000 draws) | random axis mod180 at 22.5 deg (0.25) | camera-ray axis mod180 | shuffled true yaws mod180 | one random-dir draw mod360")
        for g, members in groups.items():
            rs = [r for r in recs if r["set"] in members]
            rows = []
            for r in rs:
                kept = {b["id"]: b for b in r["blobs"][split] if b["kept"]}
                m = match(r["people"], list(kept.values()), R_PRIMARY)
                for p in r["people"]:
                    if p["id"] in m:
                        rows.append((p["yaw"], kept[m[p["id"]]]))
            if not rows:
                continue
            yaw = np.array([y for y, _ in rows])
            minor = np.array([b["axis_minor"] for _, b in rows])
            major = np.array([b["axis_major"] for _, b in rows])
            minc = np.array([b["axis_minor_cells"] for _, b in rows])
            camb = np.array([b["cam_bearing"] for _, b in rows])
            f = lambda a, b, per=math.pi, tol=FACING_TOL: float(np.mean([axis_err(x, y, per) <= tol for x, y in zip(a, b)]))
            rand360 = np.mean([f(rng.uniform(-math.pi, math.pi, len(yaw)), yaw, 2 * math.pi) for _ in range(2000)])
            rand180_22 = np.mean([f(rng.uniform(-math.pi, math.pi, len(yaw)), yaw, math.pi, math.radians(22.5)) for _ in range(2000)])
            unif180 = np.mean([f(rng.uniform(-math.pi, math.pi, len(yaw)), yaw) for _ in range(2000)])
            shuffled = np.mean([f(rng.permutation(yaw), yaw) for _ in range(2000)])
            one360 = f(np.random.default_rng(1).uniform(-math.pi, math.pi, len(yaw)), yaw, 2 * math.pi)
            say(f"{g:20s} | {len(rows):7d} | {f(minor, yaw):.3f} | {f(major, yaw):.3f} | {f(minc, yaw):.3f} | {unif180:.3f} | {rand360:.3f} | {rand180_22:.3f} | {f(camb, yaw):.3f} | {shuffled:.3f} | {one360:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    recs = []
    for sref in H.distinct_scenes():
        rec = scene_record(sref, force=args.force)
        n_kept = sum(b["kept"] for b in rec["blobs"][f"{SPLIT_PRIMARY:.2f}"])
        print(f"{rec['set']:10s} {rec['scene_id']:34s} people {len(rec['people'])} kept blobs {n_kept}", flush=True)
        recs.append(rec)
        gc.collect()
    lines: list[str] = []
    summarise(recs, lines)
    (OUT / "blobpilot_tables.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
