"""Independent verifier, I06 build: compare MY base reproduction (v_bases.py, package only, written as
generate._write_scene writes a scene) with (1) the build's level-0 bases in scenes/seed<N>/, (2) the whitelisted
development scenes through the step-0 harness (fingerprint and fields), and (3) the build's byte-identity folders.
Writes out/compare_bases.txt."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H  # noqa: E402

from sgi.scene import load_scene  # noqa: E402

HERE = Path(__file__).resolve().parent
I06 = HERE.parent
LINES = []


def say(t=""):
    print(t, flush=True)
    LINES.append(t)


def hashes(folder: Path) -> dict:
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def stripped(js: dict) -> dict:
    js = json.loads(json.dumps(js))
    js["tour"] = []
    for t in js["ground_truth"]["targets"]:
        t["tour_frames"] = None
    return js


def fields(inputs, truth):
    out = {"people": [(p.id, p.x, p.y, p.yaw, p.group, p.role) for p in truth.people],
           "targets": {k: v.model_dump(exclude={"tour_frames"}) for k, v in truth.targets.items()},
           "T": np.asarray(inputs.observation.camera.T_world_camera).tolist(),
           "robot": inputs.observation.robot,
           "bounds": list(map(tuple, truth.bounds)), "obstacles": [list(map(tuple, o)) for o in truth.obstacles]}
    for eid, ep in truth.episodes.items():
        out[eid] = ([(list(map(tuple, r.polygon)), tuple(r.representative_point), r.heading.model_dump())
                     for r in ep.acceptable_regions or []],
                    [(f.kind, list(map(tuple, f.polygon))) for f in ep.forbidden_regions or []],
                    ep.situation, ep.group_at_target, tuple(ep.target_ids))
    return out


def main():
    bad = 0
    say("1. My reproduced bases (package only, regions + tour) vs the build's level-0 bases (scenes/seed<N>/):")
    for seed in (0, 1, 2):
        mine_dir = HERE / "out" / "bases" / f"seed{seed}"
        names = sorted(p.name for p in mine_dir.iterdir())
        theirs_names = sorted(p.name for p in (I06 / "scenes" / f"seed{seed}").iterdir() if not p.name.startswith("i06_"))
        say(f"  seed {seed}: mine {len(names)} bases, build {len(theirs_names)} bases, same names: {names == theirs_names}")
        bad += names != theirs_names
        log = json.loads((HERE / "out" / "bases" / f"seed{seed}_log.json").read_text())
        say(f"    rejected layouts on the way (all 16 rows drawn): {sum(log['counts'].values())}")
        for n in names:
            a, b = mine_dir / n, I06 / "scenes" / f"seed{seed}" / n
            ha, hb = hashes(a), hashes(b)
            arrays = ha["rgb.png"] == hb["rgb.png"] and ha["depth.npy"] == hb["depth.npy"]
            js = stripped(json.loads((a / "scene.json").read_text())) == json.loads((b / "scene.json").read_text())
            extra = sorted(set(hb) - {"rgb.png", "depth.npy", "scene.json"})
            ok = arrays and js and not extra
            bad += not ok
            say(f"    {n}: rgb/depth bytes equal {arrays}; scene.json equal apart from tour {js}; extra files in build {extra or 'none'}")

    say("\n2. My reproduced bases vs whitelisted development scenes (harness: distinct_scenes, fingerprint, scene):")
    ds = H.distinct_scenes()
    want = {}
    for s in ds:
        sd = str(s.scenes_dir)
        if sd.endswith("data/placeholder/eval"):
            want[(0, s.scene_id)] = s
        for n in (0, 1, 2):
            if sd.endswith(f"dev_scenes/dev_new/seed{n}/scenes"):
                want[(n, s.scene_id)] = s
    checked = 0
    for seed in (0, 1, 2):
        for n in sorted(p.name for p in (HERE / "out" / "bases" / f"seed{seed}").iterdir()):
            s = want.get((seed, n))
            if s is None:
                say(f"  seed {seed} {n}: no whitelisted placeholder copy")
                continue
            checked += 1
            fa = H.fingerprint(HERE / "out" / "bases" / f"seed{seed}" / n)
            fb = H.fingerprint(s.folder)
            sc = H.scene(s.runs[0], n)
            mi, mt = load_scene(HERE / "out" / "bases" / f"seed{seed}" / n)
            same_fields = fields(mi, mt) == fields(sc.inputs, sc.truth)
            arrays = np.array_equal(np.nan_to_num(mi.observation.depth, nan=-1), np.nan_to_num(sc.observation.depth, nan=-1)) \
                and np.array_equal(mi.observation.rgb, sc.observation.rgb)
            ok = fa == fb and same_fields and arrays
            bad += not ok
            say(f"  seed {seed} {n}: fingerprint {'IDENTICAL' if fa == fb else 'DIFFERENT'} ({fa[:12]} vs {fb[:12]}; "
                f"{s.scenes_dir.parent.name}/{s.scenes_dir.name}); fields equal {same_fields}; arrays equal {arrays}")
    say(f"  whitelisted bases checked: {checked}")
    # cross-seed sanity: seeds 1 and 2 eval_06..13 are NOT seed 0's
    diffs = []
    for seed in (1, 2):
        for n in sorted(p.name for p in (HERE / "out" / "bases" / f"seed{seed}").iterdir()):
            if (0, n) in want and (seed, n) not in want:
                diffs.append(H.fingerprint(HERE / "out" / "bases" / f"seed{seed}" / n) != H.fingerprint(want[(0, n)].folder))
    say(f"  sanity: seeds 1-2 eval_06..13 differ from seed 0's whitelisted copies: {sum(diffs)}/{len(diffs)}")

    say("\n3. The build's byte-identity folders (build/byte_identity), re-hashed here:")
    bi = I06 / "build" / "byte_identity"
    for seed in (0, 1):
        pa, pb = bi / "package" / f"seed{seed}" / "eval", bi / "builder" / f"seed{seed}" / "eval"
        na, nb = sorted(p.name for p in pa.iterdir()), sorted(p.name for p in pb.iterdir())
        say(f"  seed {seed}: package {na}; builder same folders {na == nb}")
        for n in na:
            same = hashes(pa / n) == hashes(pb / n)
            bad += not same
            mine = HERE / "out" / "bases" / f"seed{seed}" / n
            vs_mine = (H.fingerprint(mine) == H.fingerprint(pa / n)) if mine.exists() else None
            say(f"    {n}: package == builder (every file sha256) {same}; package == my reproduction (fingerprint) {vs_mine}")
    vdir = bi / "builder_variants"
    say(f"  variants built inside the builder run: {[p.name for s in sorted(vdir.iterdir()) for p in sorted(s.iterdir())]}")
    say("\nRESULT: " + ("PASS" if bad == 0 else f"FAIL ({bad})"))
    (HERE / "out" / "compare_bases.txt").write_text("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
