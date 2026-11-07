"""I06: proof that the base scenes stay byte-identical with the builder loaded (FROZEN.md section 3, REPORT's
'[seed, 2] stream').

    byte_identity.py package   # a fresh process that imports only the package
    byte_identity.py builder   # a fresh process that imports the builder and builds variants between base rows
    byte_identity.py compare

Both runs write, with the package's own writer generate._write_scene (regions and tour, exactly as generate() writes a
scene), seed 0's first-pass rows eval_00..eval_07 (main stream default_rng(0)) and seed 1's second-pass rows eval_14..15
(stream default_rng([1, 1]); drawing stops before eval_16). The builder run also loads crowd.py and, right after eval_06
(seed 0) and eval_14 (seed 1) are accepted, builds variants of them (A6 and D2 of eval_06; D5 and E1 of eval_14) from
their [seed, 2, ...] streams, then goes on with the next base row. The two scenes compared are the next rows, seed 0
eval_07_waiting_line_2 and seed 1 eval_15_audience_5 (every other scene written is compared too). Equal file hashes
mean that loading the builder changes nothing in the package and that building variants draws nothing from a base
stream. compare also fingerprints those scenes against the whitelisted copies through the harness (H.fingerprint),
and checks that the level-0 base scenes in scenes/ have the same rgb.png and depth.npy bytes and the same scene.json
apart from the tour.

Output folders build/byte_identity/{package,builder}; result build/byte_identity.txt.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

I06 = Path(__file__).resolve().parents[1]
OUT = I06 / "build" / "byte_identity"


def write_rows(mode: str) -> None:
    from sgi.config import load_config
    from sgi.synth import generate as G
    cfg = load_config()
    out = OUT / mode
    if out.exists():
        shutil.rmtree(out)
    counts = dict.fromkeys(G._REASONS, 0)
    C = None
    if mode == "builder":
        sys.path.insert(0, str(I06 / "builder"))
        import crowd as C                           # noqa: N812
    variants_dir = OUT / f"{mode}_variants"

    def with_variants(split_dir, index, row, rng, seed, stream, plan):
        """generate._write_scene's body, step by step (the same calls, so the same draws), keeping the layout; then
        the variants of `plan` are built from it."""
        from sgi.scene import save_scene
        layout, scene, rgb, depth = G._accept(split_dir, index, row, rng, cfg, counts)
        folder = split_dir / scene.scene_id
        with_regions = G._with_regions(folder, scene, cfg)
        toured, frames = G._build_tour(layout, with_regions, cfg.thresholds)
        save_scene(folder, toured, rgb, depth, tour=frames)
        base = C.BaseLayout(seed, index, row, stream, layout, with_regions, rgb, depth, {})
        for family, level in plan:
            entry = C.build_variant(base, family, level, cfg, variants_dir / f"seed{seed}")
            print(f"  built variant {entry['scene_id']}: accepted {entry['accepted']} {entry['reasons']}", flush=True)

    # seed 0, first pass, rows eval_00..eval_07
    rng = np.random.default_rng(0)
    split = out / "seed0" / "eval"
    split.mkdir(parents=True)
    for index, row in enumerate(G._EVAL[:8]):
        if mode == "builder" and index == 6:
            with_variants(split, index, row, rng, 0, "default_rng(seed)", [("A", 6), ("D", 2)])
        else:
            G._write_scene(split, index, row, rng, cfg, counts)
        print(f"{mode}: seed 0 row {index} written", flush=True)
    # seed 1, second pass, rows eval_14..eval_15; nothing more is drawn (eval_16 is group_photo)
    more = np.random.default_rng([1, G._MORE_STREAM])
    split = out / "seed1" / "eval"
    split.mkdir(parents=True)
    for index, row in enumerate(G._EVAL_MORE[:2], start=len(G._EVAL)):
        assert row[0] == "audience"
        if mode == "builder" and index == 14:
            with_variants(split, index, row, more, 1, "default_rng([seed, 1])", [("D", 5), ("E", 1)])
        else:
            G._write_scene(split, index, row, more, cfg, counts)
        print(f"{mode}: seed 1 row {index} written", flush=True)
    print(f"{mode}: rejected layouts {G._summary(counts)}; builder module loaded: {'crowd' in sys.modules}", flush=True)


def hashes(folder: Path) -> dict[str, str]:
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def compare() -> int:
    sys.path.insert(0, 'docs/model_notes/tools')
    import research_harness as H
    lines, bad = [], 0

    def say(text=""):
        print(text, flush=True)
        lines.append(text)

    say("I06 byte-identity proof: base scenes written by generate._write_scene in a process with only the package "
        "loaded (package) and in a process with the builder loaded that builds variants between base rows (builder).")
    for seed in (0, 1):
        a_dir, b_dir = OUT / "package" / f"seed{seed}" / "eval", OUT / "builder" / f"seed{seed}" / "eval"
        names = sorted(p.name for p in a_dir.iterdir())
        if names != sorted(p.name for p in b_dir.iterdir()):
            bad += 1
            say(f"seed {seed}: different scene folders")
        for name in names:
            ha, hb = hashes(a_dir / name), hashes(b_dir / name)
            same = ha == hb
            bad += not same
            key = " (one of the two scenes named for the proof)" if name in ("eval_07_waiting_line_2", "eval_15_audience_5") else ""
            say(f"seed {seed} {name}: {len(ha)} files; package vs builder sha256 {'ALL EQUAL' if same else 'DIFFERENT'}{key}")
            if not same:
                for f in sorted(set(ha) | set(hb)):
                    if ha.get(f) != hb.get(f):
                        say(f"    {f}: {ha.get(f)} vs {hb.get(f)}")
    for seed in (0, 1):
        vdir = OUT / f"builder_variants" / f"seed{seed}"
        made = sorted(p.name for p in vdir.iterdir()) if vdir.exists() else []
        say(f"variants built in the builder run between seed {seed}'s rows (kept ones on disk): {made or 'none kept'}")

    say("\nAgainst the whitelisted copies (H.fingerprint, the sha256 of a folder's file paths and bytes):")
    ph_run = H.runs(stage="s10", model="qwen38_low", method="con")[0]
    dev1 = H.runs(arm="dev_seed1")[0]
    for seed, name, run in ((0, "eval_06_waiting_line_1", ph_run), (0, "eval_07_waiting_line_2", ph_run),
                            (1, "eval_14_audience_3", dev1), (1, "eval_15_audience_5", dev1)):
        mine = OUT / "package" / f"seed{seed}" / "eval" / name
        theirs = run.scenes_dir / name
        fa, fb = H.fingerprint(mine), H.fingerprint(theirs)
        say(f"  seed {seed} {name}: package {fa[:16]}..., whitelisted ({theirs.parent.name}/{name}) {fb[:16]}...: "
            f"{'IDENTICAL' if fa == fb else 'differ'}")

    say("\nLevel-0 base scenes in scenes/ (written without a tour) against the package-written scene:")
    for seed, name in ((0, "eval_06_waiting_line_1"), (0, "eval_07_waiting_line_2"), (1, "eval_14_audience_3"),
                       (1, "eval_15_audience_5")):
        pkg = OUT / "package" / f"seed{seed}" / "eval" / name
        lvl0 = I06 / "scenes" / f"seed{seed}" / name
        hp, hl = hashes(pkg), hashes(lvl0)
        arrays = all(hp[f] == hl[f] for f in ("rgb.png", "depth.npy"))
        jp = json.loads((pkg / "scene.json").read_text())
        jl = json.loads((lvl0 / "scene.json").read_text())
        jp["tour"] = []
        for t in jp["ground_truth"]["targets"]:
            t["tour_frames"] = None
        same_json = jp == jl
        bad += not (arrays and same_json)
        say(f"  seed {seed} {name}: rgb.png and depth.npy bytes {'equal' if arrays else 'DIFFERENT'}; scene.json equal "
            f"apart from the tour entries and tour_frames: {same_json}")
    say("\nRESULT: " + ("PASS" if bad == 0 else f"FAIL ({bad})"))
    (I06 / "build" / "byte_identity.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    what = sys.argv[1]
    if what in ("package", "builder"):
        write_rows(what)
    else:
        raise SystemExit(compare())
