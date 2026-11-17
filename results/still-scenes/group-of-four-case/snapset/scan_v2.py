"""Zero calls, no Blender, no GPU (SNAP_AMENDMENT_1). For each seed, the layouts the realistic generator would
replay with `recipes._GROUP_BEARING_DEG` patched in-process to each bearing configuration below, measured on the
`real` dressing at attempt 0 with group4/occlusion.py (imported unchanged):
  - the person target's hidden share (cylinder), and its occluders;
  - set B's measure: for every other person q, the share of q's in-image cylinder samples that the target covers
    (target nearer), and the largest of them with its q and q's in-image fraction.

The replay is the generator's own accept step (sgi.synth.generate._accept, the scene check included), with one
random generator per seed as accepted_layouts draws it. Eval rows 0-9 (no conversation group; nothing there reads
the bearing) are accepted once; the generator's state after them is saved and restored for each configuration, and
the rest of both splits is accepted again under that configuration, from the first row of the patched size (the
rows before it are P0's, drawn alike). Configuration P0 is the generator's default and
must reproduce layouts.csv (checked by check_v2.py), as must a patched one against a plain accepted_layouts run.

    PYTHONPATH= nice -n 10 .venv/bin/python results/still-scenes/group-of-four-case/snapset/scan_v2.py \
        --seeds 400-449 --workers 6
Writes layouts_v2.csv (person-target layouts of every configuration; P0 with every layout) beside this file.
"""
from __future__ import annotations

import argparse
import copy
import csv
import shutil
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scan as S                                   # noqa: E402  (scan.py: occlusion.py loaded unchanged, measures)

occ = S.occ
# One configuration per (group size, bearing) that the amendment's list or its rule (applied to
# bearing_sweep_out.txt) allows, each patching that size only, the others at the generator's default; P0 is the default
# ({2: 90, 3: 60, 4: 72.5}). One size at a time: a size whose layouts the scene check keeps rejecting (check 8 hides a
# group member at some bearings) then voids only its own configuration of a seed, as the realistic replay would.
_DEFAULT = {2: 90.0, 3: 60.0, 4: 72.5}
_BEARINGS = {2: (15.0, 20.0), 3: (15.0, 50.0, 55.0), 4: (10.0, 15.0, 30.0, 35.0, 70.0, 75.0)}
CONFIGS = {"P0": dict(_DEFAULT)}
PATCHED = {"P0": None}
for _n, _bs in _BEARINGS.items():
    for _b in _bs:
        CONFIGS[f"c{_n}b{int(_b)}"] = {**_DEFAULT, _n: _b}
        PATCHED[f"c{_n}b{int(_b)}"] = _n
FIELDS = ["seed", "config", "split", "index", "twin_id", "situation", "n_group", "bearing", "target_kind",
          "hidden_real", "occluders_real", "target_in_image", "b_hidden_by_target", "b_person", "b_person_in_image",
          "b_person_hidden_total", "error"]


def hides(scene: dict, target: str) -> tuple[float, str, float, float]:
    """The other person of whom the target covers the largest share of in-image samples: (share, id, their in-image
    fraction, their hidden share by anyone)."""
    s, Tinv, K = occ.silhouettes(scene)
    best = (0.0, "", 0.0, 0.0)
    for q, entry in s.items():
        if q == target or not entry["pts"]:
            continue
        pts = entry["pts"]
        by_target = sum(occ.covers(Tinv, K, s[target], u, v, d) for u, v, d in pts) / len(pts)
        if by_target > best[0]:
            best = (by_target, q, len(pts) / S.SAMPLES, occ.hidden_share(scene, q)[0])
    return best


def _measure(seed, config, split, index, layout, twin, library, everything) -> dict | None:
    from sgi.synth.realistic.dress import VARIANTS, dress
    target = twin.ground_truth.targets[0]
    if target.kind != "person" and not everything:
        return None
    row = dict(seed=seed, config=config, split=split, index=index, twin_id=twin.scene_id,
               situation=twin.episodes[0].ground_truth.situation, target_kind=target.kind)
    name = layout.name
    if name.startswith("conversation_group_"):
        n = int(name.rsplit("_", 1)[1])
        row.update(n_group=n, bearing=CONFIGS[config][n])
    if target.kind != "person":
        return row
    real = dress(layout, VARIANTS["real"], library, (seed, S.SPLIT_NO[split], index, 0))
    scene = {**twin.model_dump(mode="json"), "ground_truth": real.scene_gt.model_dump(mode="json")}
    share, by = occ.hidden_share(scene, target.person_id)
    in_image, _ = S._shown(scene, target.person_id)
    b = hides(scene, target.person_id)
    row.update(hidden_real=round(share, 3), occluders_real=" ".join(by), target_in_image=round(in_image, 3),
               b_hidden_by_target=round(b[0], 3), b_person=b[1], b_person_in_image=round(b[2], 3),
               b_person_hidden_total=round(b[3], 3) if b[1] else "")
    return row


def _rows_for_seed(seed: int) -> list[dict]:
    import numpy as np
    import sgi.synth.generate as G
    import sgi.synth.recipes as R
    from sgi.config import load_config
    from sgi.synth.realistic.library import load_library
    cfg, library, rows = load_config(), load_library(), []
    default = dict(R._GROUP_BEARING_DEG)
    assert default == CONFIGS["P0"], default
    eval_rows = dict(G._SPLITS)["eval"]
    first = next(i for i, row in enumerate(eval_rows) if row[0] == "conversation_group")
    counts = dict.fromkeys(G._REASONS, 0)
    starts = {n: next(i for i, row in enumerate(eval_rows) if row[0] == "conversation_group" and row[1] == n)
              for n in (2, 3, 4)}              # the eval row of each group size: nothing before it reads its bearing
    with tempfile.TemporaryDirectory(prefix=f"snapscan2_{seed}_") as work:
        rng = np.random.default_rng(seed)
        prefix = []
        for index in range(first):
            layout, twin, _, _ = G._accept(Path(work) / "pre" / "eval", index, eval_rows[index], rng, cfg, counts)
            shutil.rmtree(Path(work) / "pre" / "eval" / twin.scene_id)
            prefix.append(("eval", index, layout, twin))
        saved = {}                                 # the generator's state before eval rows first.. under P0
        try:
            for config, bearings in CONFIGS.items():
                R._GROUP_BEARING_DEG.clear()
                R._GROUP_BEARING_DEG.update(bearings)
                start = first if PATCHED[config] is None else starts[PATCHED[config]]
                if config != "P0":
                    rng.bit_generator.state = copy.deepcopy(saved[start])
                items = list(prefix) if config == "P0" else []
                try:
                    for split, split_rows in G._SPLITS:
                        for index, row in enumerate(split_rows):
                            if split == "eval" and index < start:
                                continue
                            if config == "P0" and split == "eval" and index in starts.values():
                                saved[index] = copy.deepcopy(rng.bit_generator.state)
                            split_dir = Path(work) / config / split
                            layout, twin, _, _ = G._accept(split_dir, index, row, rng, cfg, counts)
                            shutil.rmtree(split_dir / twin.scene_id)
                            items.append((split, index, layout, twin))
                except RuntimeError as error:      # a row with no accepted layout in 20 attempts: the seed's
                    rows.append(dict(seed=seed, config=config, error=str(error)[:200]))   # replay would stop too
                    if config == "P0":
                        raise
                    continue
                for split, index, layout, twin in items:
                    got = _measure(seed, config, split, index, layout, twin, library, config == "P0")
                    if got is not None:
                        rows.append(got)
        finally:
            R._GROUP_BEARING_DEG.clear()
            R._GROUP_BEARING_DEG.update(default)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)
    seeds, begun, rows = S._seeds(args.seeds), time.time(), []
    with Pool(args.workers) as pool:
        for got in pool.imap_unordered(_rows_for_seed, seeds):
            rows += got
            print(f"seed {got[0]['seed']} done at {time.time() - begun:.0f} s", flush=True)
    rows.sort(key=lambda r: (r["seed"], r["config"], r.get("split", ""), r.get("index", -1)))
    with open(HERE / "layouts_v2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(seeds)} seeds, {len(rows)} rows, {time.time() - begun:.0f} s; "
          f"config failures: {sum(bool(r.get('error')) for r in rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
