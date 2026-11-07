"""Render only the chosen layouts (realistic `real` variant), one Blender at a time, then gather them into one flat
folder that `sgi.run --scenes` takes. WRITTEN, NOT RUN. Uses the GPU: announce it before starting.

Run from the scenes worktree, which has Blender (.tools/) and the prepared assets (assets/realistic/_cache):
    cd .
    G=results/still-scenes/group-of-four-case/snapset
    PYTHONPATH= .venv/bin/python $G/render_subset.py render   --selected $G/selected.csv --out $G/real
    PYTHONPATH= .venv/bin/python $G/render_subset.py assemble --selected $G/selected.csv --out $G/real \
        --scenes $G/run_scenes

render: per seed, sgi.synth.realistic.generate.generate(out/seed<N>, cfg, N, "real", only=<that seed's ids>).
  generate() still replays the whole seed's accepted layouts on the CPU (~80 s a seed, no GPU), and renders only the
  named scenes: per scene 3 depth-only judge calls (one per variant) and 1 call for the observation, the tour and
  the target-alone depths. Seeds run one after another, so at most one Blender runs from here; before each seed it
  waits while any other Blender process is running (the GPU is shared with the simulators). It records, for each
  rendered scene, the target's hidden share from Blender's own depth (target drawn alone vs the whole scene, the
  observation view): zero extra renders, it reads frames generate() draws anyway.
assemble: copies each chosen scene to <scenes>/s<seed>_<id> with scene_id rewritten to that name (ids repeat across
  seeds, and sgi.run refuses two folders with one scene_id), re-measures the target with occlusion.py on the final
  scene.json, and writes assembled.csv with the selection bin, the final cylinder share, the pixel share and the
  dressing attempt kept. A scene whose final cylinder bin differs from its selection bin is flagged, not dropped.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("occlusion", HERE.parent / "occlusion.py")
occ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(occ)
SPENT = range(0, 220)                              # 0-2 dev/held-out, 3-99 sealed, 100-219 fit/training/I09 test


def _selected(path: Path) -> list[dict]:
    rows = list(csv.DictReader(open(path)))
    bad = sorted({int(r["seed"]) for r in rows if int(r["seed"]) in SPENT})
    if bad:
        raise SystemExit(f"seeds {bad} are reserved or spent: refusing")
    return rows


def _others_blender() -> list[str]:
    out = subprocess.run(["pgrep", "-af", "blender"], capture_output=True, text=True).stdout.splitlines()
    return [line for line in out if "pgrep" not in line and "render_subset" not in line]


class _Recording:
    """The Blender renderer, unchanged, plus the observation's target-alone depth of each main call."""

    def __init__(self, inner):
        self.inner, self.name, self.version = inner, inner.name, inner.version
        self.current, self.pixel = None, {}

    def render(self, spec, frames):
        drawn = self.inner.render(spec, frames)
        alone = [i for i, f in enumerate(frames) if f.only]
        if alone and self.current is not None:     # the main call: [obs depth, views..., alone per view per target]
            import numpy as np
            i = alone[0]                           # observation view, first (only) target
            obs_cam, wide = frames[0].camera, frames[i].camera
            left, above = int(round(wide.cx - obs_cam.cx)), int(round(wide.cy - obs_cam.cy))
            depth, theirs_d = drawn[0][1], drawn[i][1]
            theirs = theirs_d > 0
            h, w = depth.shape
            inside = np.s_[above:above + h, left:left + w]
            in_image = theirs[inside]
            shown = in_image & (np.abs(depth - theirs_d[inside]) < 1e-3)
            self.pixel[self.current] = dict(
                target_px=int(theirs.sum()), in_image_px=int(in_image.sum()), shown_px=int(shown.sum()),
                hidden_of_in_image=None if not in_image.any() else round(1 - shown.sum() / in_image.sum(), 3))
        return drawn


def render(args) -> int:
    import sgi.synth.realistic.generate as G
    from sgi.config import load_config
    from sgi.synth.realistic.blender_renderer import BlenderRenderer
    assert Path(G.__file__).resolve().is_relative_to(Path.cwd().resolve()), "run from the scenes worktree"
    rows, cfg = _selected(args.selected), load_config()
    by_seed = defaultdict(list)
    for r in rows:
        by_seed[int(r["seed"])].append(r["twin_id"])
    renderer = _Recording(BlenderRenderer())       # 64 samples, OPTIX: as the v1 sets
    original = G._toured

    def toured(scene, dressed, renderer_, depth, th):
        renderer_.current = scene.scene_id
        return original(scene, dressed, renderer_, depth, th)
    G._toured = toured
    args.out.mkdir(parents=True, exist_ok=True)
    log = open(args.out / "render_log.jsonl", "a")
    for seed in sorted(by_seed):
        while _others_blender():
            print(f"seed {seed}: another Blender is running, waiting 30 s", flush=True)
            time.sleep(30)
        begun, remaining, failed = time.time(), list(by_seed[seed]), {}
        while remaining:
            try:                                   # scenes already rendered in this loop are skipped as unchanged
                counts = G.generate(args.out / f"seed{seed}", cfg, seed, "real", renderer, only=remaining)
                break
            except RuntimeError as error:          # "<real_id>: no dressing passed the scene check in 5 attempts"
                rid = str(error).split(":", 1)[0]
                twin = rid.removeprefix("real_")
                if twin not in remaining:
                    raise
                failed[twin] = str(error)
                remaining.remove(twin)
                print(f"seed {seed}: {rid} failed, going on without it: {error}", flush=True)
        took = time.time() - begun
        print(f"seed {seed}: {counts if remaining else 'none rendered'} in {took:.0f} s", flush=True)
        for twin in by_seed[seed]:
            rid = f"real_{twin}"
            log.write(json.dumps(dict(seed=seed, id=rid, failed=failed.get(twin), seed_s=round(took, 1),
                                      pixel=renderer.pixel.get(rid))) + "\n")
        log.flush()
    return 0


WANT = {"occluded 15-40": 6, "occluded 40-70": 6, "control_person": 6, "control_object": 6}


def assemble(args) -> int:
    """Measure every rendered pick from its final scene.json, keep per set the first WANT (selection order) whose
    folder exists and, for a person target, whose final cylinder bin is still the one it was picked for; copy only
    those into --scenes with scene_id s<seed>_real_<twin id>. assembled.csv lists every pick and whether it is used."""
    from sgi.scene import load_scene
    rows = _selected(args.selected)
    pixel = {}
    if (args.out / "render_log.jsonl").is_file():
        for line in open(args.out / "render_log.jsonl"):
            e = json.loads(line)
            if e.get("pixel"):
                pixel[(e["seed"], e["id"])] = e["pixel"]
    out, used = [], defaultdict(int)
    for r in rows:
        seed, twin = int(r["seed"]), r["twin_id"]
        rid, name = f"real_{twin}", f"s{seed}_real_{twin}"
        src = args.out / f"seed{seed}" / r["split"] / rid
        entry = dict(name=name, role_in_set=r["role_in_set"], situation=r["situation"], sel_bin=r["bin"],
                     sel_hidden=r["hidden_real"], final_hidden="", final_bin="", attempt="", pixel_hidden="",
                     flag="", use=False, requests="")
        if not (src / "scene.json").is_file():     # its dressing never passed the check: a spare takes its place
            out.append({**entry, "flag": "NOT RENDERED"})
            continue
        data = json.loads((src / "scene.json").read_text())
        t = data["ground_truth"]["targets"][0]
        final = occ.hidden_share(data, t["person_id"])[0] if t["kind"] == "person" else None
        fin_bin = ("" if final is None else "unhidden" if final < 0.05 else "15-40" if 0.15 <= final < 0.40
                   else "40-70" if 0.40 <= final <= 0.70 else "other")
        changed = t["kind"] == "person" and fin_bin != r["bin"]
        use = not changed and used[r["role_in_set"]] < WANT[r["role_in_set"]]
        used[r["role_in_set"]] += use
        out.append({**entry, "final_hidden": "" if final is None else round(final, 3), "final_bin": fin_bin,
                    "attempt": json.loads((src / "render_job.json").read_text())["attempt"],
                    "pixel_hidden": (pixel.get((seed, rid)) or {}).get("hidden_of_in_image"),
                    "flag": "BIN CHANGED" if changed else "", "use": use,
                    "requests": " | ".join(f"{e['episode_id']}: {e['request']}" + (f" [{e['context']}]"
                                           if e.get("context") else "") for e in data["episodes"])})
        if use:
            dst = args.scenes / name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            data["scene_id"] = name
            (dst / "scene.json").write_text(json.dumps(data, indent=1))
            load_scene(dst)                        # loads, with all its files
    with open(HERE / "assembled.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"used per set: {dict(used)} (wanted {WANT}); flagged: {sum(bool(o['flag']) for o in out)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("render", "assemble"))
    ap.add_argument("--selected", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scenes", type=Path, default=HERE / "run_scenes")
    args = ap.parse_args(argv)
    return render(args) if args.action == "render" else assemble(args)


if __name__ == "__main__":
    sys.exit(main())
