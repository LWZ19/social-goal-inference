"""Independent verifier, I06 build: reproduce the 30 base layouts with the package only (no builder code imported).

    v_bases.py SEED

For one seed: draws the first pass with default_rng(seed) through generate._accept for rows eval_00..eval_13 (the
first six only advance the stream) and the second pass with default_rng([seed, generate._MORE_STREAM]) for rows
eval_14 and eval_15, then stops (eval_16 is group_photo and is never instantiated). Each base row is written exactly
as generate._write_scene writes it (regions and tour) into out/bases/seed<SEED>/<scene id>, and its Layout is pickled
into out/bases/seed<SEED>.pkl for the replay. Temporary folders under work/ are removed.
"""
from __future__ import annotations

import json
import pickle
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from sgi.config import load_config
from sgi.scene import save_scene
from sgi.synth import generate as G

HERE = Path(__file__).resolve().parent
BASE_ROWS = range(6, 16)


def main(seed: int) -> None:
    assert seed in (0, 1, 2), seed
    cfg = load_config()
    out = HERE / "out" / "bases" / f"seed{seed}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    work = Path(tempfile.mkdtemp(prefix=f"bases{seed}_", dir=HERE / "work"))
    layouts, log = {}, []
    t0 = time.perf_counter()
    try:
        counts = dict.fromkeys(G._REASONS, 0)
        rng = np.random.default_rng(seed)
        split = work / "eval"
        rows = list(enumerate(G._EVAL))[:14]
        assert [r[1][0] for r in rows].count("group_photo") == 0
        for index, row in rows:
            before = sum(counts.values())
            layout, scene, rgb, depth = G._accept(split, index, row, rng, cfg, counts)
            folder = split / scene.scene_id
            if index in BASE_ROWS:
                toured, frames = G._build_tour(layout, G._with_regions(folder, scene, cfg), cfg.thresholds)
                save_scene(out / scene.scene_id, toured, rgb, depth, tour=frames)
                layouts[index] = (row, scene.scene_id, layout)
            log.append({"index": index, "scene_id": scene.scene_id, "rejected_before": sum(counts.values()) - before})
            shutil.rmtree(folder)
            print(f"seed {seed} row {index} {scene.scene_id} ({time.perf_counter() - t0:.0f} s)", flush=True)
        more = np.random.default_rng([seed, G._MORE_STREAM])
        for index, row in list(enumerate(G._EVAL_MORE, start=len(G._EVAL)))[:2]:
            assert row[0] == "audience", row
            before = sum(counts.values())
            layout, scene, rgb, depth = G._accept(split, index, row, more, cfg, counts)
            folder = split / scene.scene_id
            toured, frames = G._build_tour(layout, G._with_regions(folder, scene, cfg), cfg.thresholds)
            save_scene(out / scene.scene_id, toured, rgb, depth, tour=frames)
            layouts[index] = (row, scene.scene_id, layout)
            log.append({"index": index, "scene_id": scene.scene_id, "rejected_before": sum(counts.values()) - before})
            shutil.rmtree(folder)
            print(f"seed {seed} row {index} {scene.scene_id} ({time.perf_counter() - t0:.0f} s)", flush=True)
        # nothing more is drawn from `more`
    finally:
        shutil.rmtree(work, ignore_errors=True)
    with open(HERE / "out" / "bases" / f"seed{seed}.pkl", "wb") as f:
        pickle.dump(layouts, f)
    (HERE / "out" / "bases" / f"seed{seed}_log.json").write_text(json.dumps({"rows": log, "counts": counts}, indent=1))
    print(f"seed {seed}: done in {time.perf_counter() - t0:.0f} s; rejections {G._summary(counts)}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))
