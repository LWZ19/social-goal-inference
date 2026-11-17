"""Independent verifier for I03 Stage B: my own re-implementation of the Stage A blob detector (FROZEN.md section 2,
DEVIATIONS.md clarifications 2-3), with the frozen setting of stageA/chosen.json (original detector, no fix).

Written from the note, not from blobs.py: components and seed groups by iterative min-label propagation on the
grid; the split window maximum by explicit shifts; nearest seed cell by brute force with a row-major tie rule.
Scenes are read only through the research harness (H.distinct_scenes, H.scene).

Writes v_blobs.json (per distinct scene folder: kept blobs in blob order) and prints a comparison with
stageB/blob_cache.json and stageA/rows.jsonl kept counts. No model, no network.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H  # noqa: E402


def here(stored) -> str:
    """A scene folder as a stored file names it (the recorded absolute path without its local root), as it lies
    here: through the harness's relocated() and from the repository root."""
    return str((H.main_checkout() / H.relocated(stored)).resolve())

HERE = Path(__file__).resolve().parent
I03 = HERE.parent
RES = 0.05


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def propagate_labels(mask: np.ndarray) -> np.ndarray:
    """8-connected component labels by min-label propagation. Label = 1 + row-major linear index of the
    component's first cell; 0 outside."""
    big = np.iinfo(np.int64).max
    lin = np.arange(mask.size, dtype=np.int64).reshape(mask.shape)
    lab = np.where(mask, lin, big)
    while True:
        p = np.pad(lab, 1, constant_values=big)
        best = lab.copy()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                sh = p[1 + dr:1 + dr + mask.shape[0], 1 + dc:1 + dc + mask.shape[1]]
                best = np.minimum(best, sh)
        best = np.where(mask, best, big)
        if np.array_equal(best, lab):
            break
        lab = best
    return np.where(mask, lab + 1, 0)


def window_max(h: np.ndarray, half: int) -> np.ndarray:
    """Max over the square window of half-width `half` (in cells), -inf outside."""
    p = np.pad(h, half, constant_values=-np.inf)
    out = np.full(h.shape, -np.inf)
    for dr in range(-half, half + 1):
        for dc in range(-half, half + 1):
            out = np.maximum(out, p[half + dr:half + dr + h.shape[0], half + dc:half + dc + h.shape[1]])
    return out


def detect(pts: np.ndarray, s: dict):
    z_lo, z_hi = s["z_lo"], s["z_hi"]
    p = pts[(pts[:, 2] > z_lo) & (pts[:, 2] < z_hi)]
    if len(p) == 0:
        return [], []
    ij = np.floor(p[:, :2] / RES).astype(np.int64)
    lo = ij.min(0)
    rel = ij - lo
    shape = tuple(rel.max(0) + 1)
    h = np.full(shape, -np.inf)
    # highest kept point per cell
    order = np.lexsort((p[:, 2], rel[:, 1], rel[:, 0]))
    r_s, z_s = rel[order], p[order, 2]
    last = np.ones(len(order), dtype=bool)
    last[:-1] = np.any(r_s[1:] != r_s[:-1], axis=1)
    h[r_s[last, 0], r_s[last, 1]] = z_s[last]
    occ = np.isfinite(h)
    comp = propagate_labels(occ)
    comp_ids = np.unique(comp[comp > 0])                  # ascending = row-major order of first cell
    half = int(round(s["split_r"] / RES))
    blobs = []
    for cid in comp_ids:
        cells = np.argwhere(comp == cid)                   # row-major
        r0, c0 = cells.min(0)
        r1, c1 = cells.max(0) + 1
        m = comp[r0:r1, c0:c1] == cid
        hs = np.where(m, h[r0:r1, c0:c1], -np.inf)
        wm = window_max(hs, half)
        seeds = m & (hs >= wm)
        slab = propagate_labels(seeds)
        sids = np.unique(slab[slab > 0])
        local = cells - [r0, c0]
        if len(sids) <= 1:
            assign = np.zeros(len(local), dtype=int)
        else:
            seed_cells = np.argwhere(slab > 0)             # row-major
            seed_group = np.searchsorted(sids, slab[seed_cells[:, 0], seed_cells[:, 1]])
            assign = np.empty(len(local), dtype=int)
            for a in range(0, len(local), 2000):
                d = ((local[a:a + 2000, None, :] - seed_cells[None, :, :]) ** 2).sum(-1)
                assign[a:a + 2000] = seed_group[np.argmin(d, axis=1)]   # argmin: first minimum = row-major tie
        for g in range(max(1, len(sids))):
            bc = cells[assign == g]
            centres = (bc + lo + 0.5) * RES
            top = float(h[bc[:, 0], bc[:, 1]].max())
            if len(bc) >= 2:
                c = centres - centres.mean(0)
                w, v = np.linalg.eigh(c.T @ c / len(c))
                major = v[:, np.argmax(w)]
                pr = centres @ major
                foot = float(pr.max() - pr.min() + RES)
            else:
                foot = RES
            blobs.append({"x": float(centres[:, 0].mean()), "y": float(centres[:, 1].mean()), "top": top,
                          "footprint": foot, "cells": int(len(bc))})
    top_hi = z_hi - 0.05
    kept = [b for b in blobs if s["top_lo"] <= b["top"] <= top_hi + 1e-9
            and s["foot_lo"] - 1e-9 <= b["footprint"] <= s["foot_hi"] + 1e-9]
    return kept, blobs


def main():
    from sgi.geometry import lift_all
    chosen = json.loads((I03 / "stageA" / "chosen.json").read_text())
    assert sha(I03 / "stageA" / "chosen.json") == "0ae61bae1893ad9681783b36b245c8a429f2bc0d8df8efad0ef569cd879125ae"
    assert chosen["detector"] == "original" and chosen["drop_clip"] is False
    s = chosen["setting"]
    cache = {here(k): v for k, v in json.loads((I03 / "stageB" / "blob_cache.json").read_text())["scenes"].items()}
    rowsA = {here(json.loads(l)["folder"]): json.loads(l) for l in (I03 / "stageA" / "rows.jsonl").open()}
    out, diffs = {}, []
    ds = H.distinct_scenes()
    print("distinct scenes", len(ds))
    maxdx = 0.0
    for sref in ds:
        sc = H.scene(sref.runs[0], sref.scene_id)
        kept, allb = detect(lift_all(sc.observation.depth, sc.observation.camera), s)
        key = str(sref.folder)
        out[key] = {"scene_id": sref.scene_id, "listed": [[str(d), i] for d, i in sref.listed], "blobs": kept}
        cb = cache[key]["blobs"]
        same = len(cb) == len(kept)
        if same:
            for a, b in zip(kept, cb):
                dx = max(abs(a["x"] - b["x"]), abs(a["y"] - b["y"]), abs(a["top"] - b["top"]),
                         abs(a["footprint"] - b["footprint"]))
                maxdx = max(maxdx, dx)
                same &= dx < 1e-9 and a["cells"] == b["cells"]
        nA = rowsA[key]["kept"]
        if not same or nA != len(kept):
            diffs.append((sref.scene_id, len(kept), len(cb), nA))
        print(f"{sref.scene_id:38s} mine {len(kept):2d} cache {len(cb):2d} stageA {nA:2d} {'OK' if same else 'DIFF'}",
              flush=True)
    (HERE / "v_blobs.json").write_text(json.dumps(out, indent=1))
    print("scenes", len(out), "blobs", sum(len(v["blobs"]) for v in out.values()), "diffs", diffs,
          "max abs diff", maxdx)


if __name__ == "__main__":
    main()
