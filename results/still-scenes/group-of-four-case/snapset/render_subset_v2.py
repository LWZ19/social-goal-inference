"""SNAP_AMENDMENT_1's render driver. WRITTEN, NOT RUN. Uses the GPU: announce it first (models none, calls none).

Run from the scenes worktree, which has Blender (.tools/) and the prepared assets (assets/realistic/_cache):
    cd .-scenes
    G=results/2026-11-01-push/group4/snapset
    PYTHONPATH= .venv/bin/python $G/render_subset_v2.py render   --selected $G/selected_v2.csv --out $G/real_v2
    PYTHONPATH= .venv/bin/python $G/render_subset_v2.py assemble --selected $G/selected_v2.csv --out $G/real_v2 \
        --scenes $G/run_scenes_v2

render: one (seed, configuration) after another, never two Blenders from here, and waiting while any other Blender
  runs. For each, `sgi.synth.recipes._GROUP_BEARING_DEG` is set in this process to the row's (b2, b3, b4) and put back
  after; then sgi.synth.realistic.generate.generate(out/seed<N>_<config>, cfg, N, "real", only=<its twin ids>), which
  replays the whole seed under that patch on the CPU and renders only the named scenes. render_job.json does not hold
  the bearing, so manifest_v2.jsonl does: one line per scene with seed, configuration, bearings, ids, the dressing
  attempt kept, the cylinder shares on the rendered scene.json, and Blender's pixel shares:
    - the target's: target drawn alone vs the whole scene, observation view, read from frames generate() draws anyway;
    - set B only, one extra depth-only Blender call per scene: the hidden person drawn alone (their object group
      changed to a new one in the kept dressing's description; nothing else changes, and the whole-scene depth of
      that call is checked against the scene's depth.npy), giving their pixel share hidden by anything and by the
      target.
  A scene whose dressing never passes the check (5 attempts) is logged as failed and the rest go on.
assemble: SNAP_AMENDMENT_1's pixel rule, before any plan call. Per set and situation (and bin), in selected order, the
  first that meet it are used, primaries before spares: occluded 15-40 needs a pixel share in [0.10, 0.45], 40-55 in
  [0.35, 0.60], A and B a target under 0.10, B also its hidden person at 0.10 or more; the occluded set keeps at most
  2 per (situation, bearing). A primary that fails is replaced by the next spare of its cell; with none left it stays,
  labelled with both bins. Fallback B layouts (places, weak) are used as they are. The used scenes are copied to
  --scenes as s<seed><config>_real_<twin id>, with scene_id rewritten to that name; assembled_v2.csv lists every row.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
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
SAME_M = 1e-3                                      # as the generator's _SAME_DEPTH_M
WANT = {("occluded", "15-40"): 6, ("occluded", "40-55"): 6, ("A", ""): 6, ("B", ""): 6}
PIXEL_OK = {"15-40": (0.10, 0.45), "40-55": (0.35, 0.60)}


def _selected(path: Path) -> list[dict]:
    rows = list(csv.DictReader(open(path)))
    bad = sorted({int(r["seed"]) for r in rows if int(r["seed"]) in SPENT})
    if bad:
        raise SystemExit(f"seeds {bad} are reserved or spent: refusing")
    return rows


def _others_blender() -> list[str]:
    out = subprocess.run(["pgrep", "-af", "blender"], capture_output=True, text=True).stdout.splitlines()
    return [line for line in out if "pgrep" not in line and "render_subset" not in line]


def _crop_share(depth, alone, left, above):
    """In-image pixels of what `alone` draws, those of them the whole scene shows, and the crop in image coords."""
    import numpy as np
    h, w = depth.shape
    crop = alone[above:above + h, left:left + w]
    theirs = crop > 0
    shown = theirs & (np.abs(depth - crop) < SAME_M)
    return int(theirs.sum()), int(shown.sum()), crop


class _Recording:
    """The Blender renderer, unchanged; keeps, per scene of a main call, the observation's target-alone crop and the
    dressing that was kept (from the _toured hook)."""

    def __init__(self, inner):
        self.inner, self.name, self.version = inner, inner.name, inner.version
        self.current, self.target, self.dressed = None, {}, {}

    def render(self, spec, frames):
        drawn = self.inner.render(spec, frames)
        alone = [i for i, f in enumerate(frames) if f.only]
        if alone and self.current is not None:     # main call: [obs depth, views..., alone per view per target]
            i = alone[0]                           # the observation's view, its first (only) target
            obs, wide = frames[0].camera, frames[i].camera
            left, above = int(round(wide.cx - obs.cx)), int(round(wide.cy - obs.cy))
            in_image, shown, crop = _crop_share(drawn[0][1], drawn[i][1], left, above)
            self.target[self.current] = dict(in_image_px=in_image, shown_px=shown, crop=crop,
                                             hidden=None if not in_image else round(1 - shown / in_image, 3))
        return drawn


def _hidden_person_pixel(renderer: _Recording, rid: str, scene_dir: Path, person: str) -> dict:
    """One depth-only call: the whole scene, and `person` alone in a new object group, through the observation."""
    import numpy as np
    import sgi.synth.realistic.generate as G
    from sgi.geometry import Camera
    from sgi.synth.realistic.spec import Frame
    dressed = renderer.dressed.get(rid)
    if dressed is None:
        return dict(error="no kept dressing in this process (scene skipped as unchanged?)")
    spec = dressed.spec
    groups = [p.group for p in spec.people] + [getattr(x, "group", 0) for x in (*spec.parts, *spec.models)]
    new = 1 + max(groups)
    who = next(p for p in spec.people if p.person_id == person)
    spec2 = dataclasses.replace(spec, people=tuple(dataclasses.replace(p, group=new) if p.person_id == person else p
                                                   for p in spec.people))
    ob = json.loads((scene_dir / "scene.json").read_text())["observation"]
    k = ob["intrinsics"]
    cam = Camera(k["fx"], k["fy"], k["cx"], k["cy"], ob["width"], ob["height"],
                 np.array(ob["T_world_camera"], dtype=np.float64))
    wide, left, above = G._wide(cam, [(who.x, who.y, 0.0, 0.5, 0.5, who.height + 0.1)])
    drawn = renderer.inner.render(spec2, [Frame(cam, beauty=False), Frame(wide, beauty=False, only=(new,))])
    depth = np.load(scene_dir / "depth.npy")
    in_image, shown, crop = _crop_share(depth, drawn[1][1], left, above)
    target = renderer.target.get(rid, {}).get("crop")
    by_target = None
    if target is not None and in_image:
        mine = crop > 0
        by_target = round(float((mine & (target > 0) & (target < crop - SAME_M)).sum()) / in_image, 3)
    return dict(person=person, in_image_px=in_image, shown_px=shown,
                hidden=None if not in_image else round(1 - shown / in_image, 3), hidden_by_target=by_target,
                whole_depth_equal=bool(np.array_equal(drawn[0][1], depth)))


def render(args) -> int:
    import sgi.synth.realistic.generate as G
    import sgi.synth.recipes as R
    from sgi.config import load_config
    from sgi.synth.realistic.blender_renderer import BlenderRenderer
    assert Path(G.__file__).resolve().is_relative_to(Path.cwd().resolve()), "run from the scenes worktree"
    rows, cfg = _selected(args.selected), load_config()
    runs = defaultdict(list)
    for r in rows:
        runs[(int(r["seed"]), r["config"], float(r["b2"]), float(r["b3"]), float(r["b4"]))].append(r)
    renderer = _Recording(BlenderRenderer())       # 64 samples, OPTIX: as the v1 sets
    original, default = G._toured, dict(R._GROUP_BEARING_DEG)

    def toured(scene, dressed, renderer_, depth, th):
        renderer_.current = scene.scene_id
        renderer_.dressed[scene.scene_id] = dressed
        return original(scene, dressed, renderer_, depth, th)
    G._toured = toured
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = open(args.out / "manifest_v2.jsonl", "a")
    for (seed, config, b2, b3, b4), picks in sorted(runs.items()):
        while _others_blender():
            print(f"{seed} {config}: another Blender is running, waiting 30 s", flush=True)
            time.sleep(30)
        bearings = {2: b2, 3: b3, 4: b4}
        out = args.out / f"seed{seed}_{config}"
        begun, remaining, failed = time.time(), sorted({r["twin_id"] for r in picks}), {}
        R._GROUP_BEARING_DEG.clear()
        R._GROUP_BEARING_DEG.update(bearings)
        try:
            while remaining:
                try:                               # scenes rendered earlier in this loop are skipped as unchanged
                    G.generate(out, cfg, seed, "real", renderer, only=remaining)
                    break
                except RuntimeError as error:      # "<real_id>: no dressing passed the scene check in 5 attempts"
                    twin = str(error).split(":", 1)[0].removeprefix("real_")
                    if twin not in remaining:
                        raise
                    failed[twin] = str(error)
                    remaining.remove(twin)
                    print(f"{seed} {config}: real_{twin} failed, going on without it: {error}", flush=True)
            for r in picks:
                rid = f"real_{r['twin_id']}"
                scene_dir = out / r["split"] / rid
                entry = dict(scene_name=r["scene_name"], seed=seed, config=config, bearings=bearings,
                             twin_id=r["twin_id"], split=r["split"], set=r["set"], bin=r["bin"],
                             situation=r["situation"], spare=r["spare"] == "True", failed=failed.get(r["twin_id"]))
                if (scene_dir / "scene.json").is_file():
                    data = json.loads((scene_dir / "scene.json").read_text())
                    t = data["ground_truth"]["targets"][0]
                    entry["attempt"] = json.loads((scene_dir / "render_job.json").read_text())["attempt"]
                    if t["kind"] == "person":
                        entry["cyl_target_hidden"] = round(occ.hidden_share(data, t["person_id"])[0], 3)
                    target = renderer.target.get(rid)
                    entry["pixel_target_hidden"] = None if target is None else target["hidden"]
                    if r["set"] == "B" and r["b_person"]:
                        entry["b_person"] = r["b_person"]
                        entry["pixel_b_person"] = _hidden_person_pixel(renderer, rid, scene_dir, r["b_person"])
                manifest.write(json.dumps(entry) + "\n")
            manifest.flush()
        finally:
            R._GROUP_BEARING_DEG.clear()
            R._GROUP_BEARING_DEG.update(default)
        print(f"{seed} {config}: {len(picks)} picks, {len(failed)} failed, {time.time() - begun:.0f} s", flush=True)
    return 0


def _pixel_ok(r, m) -> bool:
    if r["note"].startswith("fallback"):           # places: used as they are, weak by construction
        return True
    px = m.get("pixel_target_hidden")
    if px is None:
        return False
    if r["set"] == "occluded":
        lo, hi = PIXEL_OK[r["bin"]]
        return lo <= px <= hi
    if px >= 0.10:
        return False
    if r["set"] == "B":
        return ((m.get("pixel_b_person") or {}).get("hidden") or 0.0) >= 0.10
    return True


def assemble(args) -> int:
    from sgi.scene import load_scene
    rows = _selected(args.selected)
    manifest = {}
    for line in open(args.out / "manifest_v2.jsonl"):
        e = json.loads(line)
        manifest[e["scene_name"]] = e               # the last line of a scene wins
    cells = defaultdict(list)                      # (set, bin, situation) -> rows, primaries then spares; set B
    for r in rows:                                 # is one pool: its spares need not share a situation
        cells[(r["set"], r["bin"], r["situation"] if r["set"] != "B" else "")].append(r)
    used, pair, report = defaultdict(int), defaultdict(int), []
    for cell, members in cells.items():
        primaries = [r for r in members if r["spare"] != "True"]
        spares = [r for r in members if r["spare"] == "True"]
        queue = list(spares)
        for r in primaries:
            chosen, why = r, ""
            m = manifest.get(r["scene_name"], {})
            if m.get("failed") or "attempt" not in m or not _pixel_ok(r, m):
                why = "failed" if m.get("failed") or "attempt" not in m else "pixel out of bin"
                chosen = None
                while queue:
                    s = queue.pop(0)
                    ms = manifest.get(s["scene_name"], {})
                    cap = s["set"] == "occluded" and pair[(s["situation"], s["bearing"])] >= 2
                    if "attempt" in ms and not ms.get("failed") and _pixel_ok(s, ms) and not cap:
                        chosen = s
                        break
                if chosen is None and "attempt" in m and not m.get("failed"):
                    chosen, why = r, why + "; no spare left, kept and labelled"
            if chosen is None:
                report.append(dict(scene_name=r["scene_name"], cell="/".join(cell), use=False, note=why))
                continue
            pair[(chosen["situation"], chosen["bearing"])] += chosen["set"] == "occluded"
            used[(chosen["set"], chosen["bin"])] += 1
            mc = manifest[chosen["scene_name"]]
            report.append(dict(scene_name=chosen["scene_name"], cell="/".join(cell), use=True,
                               note=(f"replaces {r['scene_name']} ({why})" if chosen is not r else why),
                               cyl=mc.get("cyl_target_hidden"), pixel=mc.get("pixel_target_hidden"),
                               b_pixel=(mc.get("pixel_b_person") or {}).get("hidden"),
                               bearings=json.dumps(mc["bearings"]), weak=r["note"].startswith("fallback")))
            src = args.out / f"seed{chosen['seed']}_{chosen['config']}" / chosen["split"] / f"real_{chosen['twin_id']}"
            dst = args.scenes / chosen["scene_name"]
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            data = json.loads((dst / "scene.json").read_text())
            data["scene_id"] = chosen["scene_name"]
            (dst / "scene.json").write_text(json.dumps(data, indent=1))
            load_scene(dst)                        # loads, with all its files
    with open(HERE / "assembled_v2.csv", "w", newline="") as f:
        fields = ["scene_name", "cell", "use", "note", "cyl", "pixel", "b_pixel", "bearings", "weak"]
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(report)
    print(f"used per set: {dict(used)} (wanted {WANT}); scenes in {args.scenes}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("render", "assemble"))
    ap.add_argument("--selected", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scenes", type=Path, default=HERE / "run_scenes_v2")
    args = ap.parse_args(argv)
    if args.action == "assemble":
        args.scenes.mkdir(parents=True, exist_ok=True)
    return render(args) if args.action == "render" else assemble(args)


if __name__ == "__main__":
    sys.exit(main())
