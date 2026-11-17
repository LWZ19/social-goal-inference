"""I06 step 3: generate every variant of families A-E for one seed (FROZEN.md sections 3-4).

    generate_variants.py SEED

Reads build/base_layouts/seed<SEED>.pkl (written by reproduce_bases.py, checked by base_check.py), writes accepted
variants to scenes/seed<SEED>/<scene id> and one manifest line per planned variant to build/variants_seed<SEED>.jsonl
(appended and flushed one by one; a variant already recorded there is skipped, so a stopped run resumes). The base
person tables (every relevant person's pixel share included) go to build/base_tables_seed<SEED>.json.
"""
from __future__ import annotations

import json
import pickle
import sys
import time

from sgi.config import load_config
from sgi.gt.derive import prepare
from sgi.scene import load_scene

import crowd as C


def main(seed: int) -> None:
    cfg = load_config()
    with open(C.BUILD / "base_layouts" / f"seed{seed}.pkl", "rb") as f:
        bases = pickle.load(f)
    out = C.SCENES / f"seed{seed}"
    manifest = C.BUILD / f"variants_seed{seed}.jsonl"
    done = set()
    if manifest.exists():
        done = {json.loads(line)["scene_id"] for line in manifest.read_text(encoding="utf-8").splitlines() if line}
    tables_path = C.BUILD / f"base_tables_seed{seed}.json"
    base_tables = json.loads(tables_path.read_text()) if tables_path.exists() else {}
    started = time.perf_counter()
    n = 0
    for base in bases:
        sid = base.scene.scene_id
        if sid not in base_tables:
            folder = out / sid
            inputs, truth = load_scene(folder)
            ets = prepare(inputs, truth, cfg)
            relevant = {*ets["e0"].relevant_people, *ets["e1"].relevant_people}
            base_tables[sid] = C.person_table(folder, base.layout, cfg, share_for=relevant)
            tables_path.write_text(json.dumps(base_tables, indent=1), encoding="utf-8")
        for family, levels in C.LEVELS.items():
            if not C.applies(family, base.row):
                continue
            for level in levels:
                scene_id = C.scene_id_of(seed, family, level, base.index, base.layout.name)
                if scene_id in done:
                    continue
                entry = C.build_variant(base, family, level, cfg, out)
                with open(manifest, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
                n += 1
                status = "kept" if entry["accepted"] else f"rejected ({entry['rejected_reason']})"
                print(f"seed {seed} {scene_id}: {status} after {entry['attempts']} attempt(s) {entry['reasons']} "
                      f"[{entry['seconds']:.1f} s; total {time.perf_counter() - started:.0f} s]", flush=True)
    print(f"seed {seed}: {n} variants generated this run in {time.perf_counter() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))
