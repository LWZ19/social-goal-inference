"""E2 (E2_FROZEN.md section 3) scene assembly, after the renders. No GPU, zero calls.

    PYTHONPATH= .venv/bin/python e2_assemble.py --renders <E2>/renders \
        --out <E2>/run_scenes

Seeds 500-509 are taken in order. A seed is used only if all 8 people rows are under renders/seed<N>/eval/ and each
passes the scene check (sgi.check_scene, errors only). The first 5 such seeds are used. Each used scene is copied to
out/s<N>_real_<twin id>, with scene_id rewritten to that name. Written beside out/: manifest.csv (seed, row, scene_id,
sha256 of rgb.png and depth.npy) and assemble_log.txt (every seed looked at, and why one was passed over).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

SEEDS = range(500, 510)
WANT_SEEDS = 5
ROWS = ("eval_06_waiting_line_1", "eval_07_waiting_line_2", "eval_08_waiting_line_4", "eval_09_waiting_line_6",
        "eval_10_conversation_group_2", "eval_11_conversation_group_3", "eval_12_conversation_group_4",
        "eval_13_waiting_line_3")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_ok(renders: Path, seed: int, cfg) -> tuple[bool, str]:
    from sgi.check_scene import check_scene
    base = renders / f"seed{seed}" / "eval"
    if not base.is_dir():
        return False, "not rendered"
    for row in ROWS:
        folder = base / f"real_{row}"
        if not (folder / "scene.json").exists():
            return False, f"{row}: missing"
        errors = [r for r in check_scene(folder, cfg, write_png=False) if r.level == "error"]
        if errors:
            return False, f"{row}: scene check: " + "; ".join(f"{r.number} {r.name}" for r in errors)
    return True, "ok"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    from sgi.config import load_config
    cfg = load_config()
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit(f"{a.out} is not empty: refusing")
    a.out.mkdir(parents=True, exist_ok=True)
    used, log = [], []
    for seed in SEEDS:
        if len(used) == WANT_SEEDS:
            break
        ok, why = seed_ok(a.renders, seed, cfg)
        log.append(f"seed {seed}: {why}")
        if ok:
            used.append(seed)
    rows = []
    for seed in used:
        for row in ROWS:
            src = a.renders / f"seed{seed}" / "eval" / f"real_{row}"
            name = f"s{seed}_real_{row}"
            dst = a.out / name
            shutil.copytree(src, dst)
            doc = json.loads((dst / "scene.json").read_text())
            doc["scene_id"] = name
            (dst / "scene.json").write_text(json.dumps(doc, indent=1))
            rows.append({"seed": seed, "row": row, "scene_id": name, "rgb_sha256": sha(dst / "rgb.png"),
                         "depth_sha256": sha(dst / "depth.npy")})
    with open(a.out.parent / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["seed"])
        w.writeheader()
        w.writerows(rows)
    log.append(f"used seeds {used}: {len(rows)} scenes" + ("" if len(used) == WANT_SEEDS else
                                                             f" (only {len(used)} of {WANT_SEEDS}: log a deviation)"))
    (a.out.parent / "assemble_log.txt").write_text("\n".join(log) + "\n")
    print("\n".join(log))
    return 0 if len(used) == WANT_SEEDS else 2


if __name__ == "__main__":
    raise SystemExit(main())
