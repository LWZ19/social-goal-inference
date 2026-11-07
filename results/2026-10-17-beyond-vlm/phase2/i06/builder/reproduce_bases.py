"""I06 step 2a: reproduce the 10 base layouts of one seed and write them as level-0 scenes (FROZEN.md section 2).

    reproduce_bases.py SEED

Writes scenes/seed<SEED>/<scene id> (regions filled, no tour), build/base_layouts/seed<SEED>.pkl (the BaseLayout
objects, for the variant generator) and build/bases_seed<SEED>.jsonl (one manifest line per base). Temporary folders
go under builder/tmp and are removed.
"""
from __future__ import annotations

import json
import pickle
import sys
import time

from sgi.config import load_config
from sgi.scene import save_scene

import crowd as C


def main(seed: int) -> None:
    cfg = load_config()
    out = C.SCENES / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    (C.BUILD / "base_layouts").mkdir(parents=True, exist_ok=True)
    bases, lines = [], []
    started = time.perf_counter()
    for base in C.reproduce_bases(seed, cfg, C.I06 / "builder" / "tmp"):
        folder = out / base.scene.scene_id
        save_scene(folder, base.scene, base.rgb, base.depth)
        bases.append(base)
        entry = {"kind": "base", "seed": seed, "row": base.index, "row_spec": list(base.row),
                 "base_scene_id": base.scene.scene_id, "family": "base", "level": 0, "stream": base.stream,
                 "scene_id": base.scene.scene_id, "attempts": 1 + sum(base.rejected_before.values()),
                 "reasons": [*(f"{k}x{v}" for k, v in base.rejected_before.items()), "accepted"], "accepted": True,
                 "rejected_reason": None, "added": [], "folder": str(folder.relative_to(C.I06)),
                 "sha256": C.folder_hashes(folder)}
        lines.append(entry)
        print(f"seed {seed} {base.scene.scene_id}: people {len(base.layout.scene_gt.people)}, rejected before "
              f"{base.rejected_before or 'none'} ({time.perf_counter() - started:.0f} s)", flush=True)
    assert [b.index for b in bases] == list(C.BASE_ROWS), [b.index for b in bases]
    with open(C.BUILD / "base_layouts" / f"seed{seed}.pkl", "wb") as f:
        pickle.dump(bases, f)
    with open(C.BUILD / f"bases_seed{seed}.jsonl", "w", encoding="utf-8") as f:
        for entry in lines:
            f.write(json.dumps(entry) + "\n")
    print(f"seed {seed}: {len(bases)} bases written in {time.perf_counter() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))
