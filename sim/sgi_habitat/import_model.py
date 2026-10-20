"""Bring a downloaded 3D model into the simulator: one .glb under data/habitat/objects/, and the line to paste
into a scenario's `objects:` (docs/habitat.md).

    python -m sgi_habitat.import_model ~/Downloads/coffee_machine.zip [--name coffee_machine] [--z-up]

The input is a model file (.glb, .gltf, .obj, .stl, .ply, .off, .dae) or a .zip or folder holding one, as
Sketchfab and other sites give them; textures beside a .gltf or .obj come along. The model is written as one
.glb with its textures inside. Its size is printed, and when the numbers look like centimetres or millimetres
rather than metres, the printed line scales it with `height_m` to the height it would have in metres.
--z-up turns a model made z up (as many .obj and .stl files are) to y up, as glTF and the simulator have it.
.fbx and .blend cannot be read here: export them as glTF Binary (.glb) from Blender first.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "habitat" / "objects"
READABLE = (".glb", ".gltf", ".obj", ".stl", ".ply", ".off", ".dae")
UNREADABLE = (".fbx", ".blend", ".max", ".3ds", ".c4d", ".usd", ".usdz")


def find_model(folder: Path) -> Path:
    """The model file in an unpacked download: the readable file of the most preferred kind, then the largest."""
    found = [p for p in folder.rglob("*") if p.suffix.lower() in READABLE and "__MACOSX" not in p.parts]
    if not found:
        other = sorted({p.suffix.lower() for p in folder.rglob("*") if p.suffix.lower() in UNREADABLE})
        hint = f" (it holds {', '.join(other)}: export it as .glb from Blender)" if other else ""
        raise SystemExit(f"error: no model file ({', '.join(READABLE)}) in {folder}{hint}")
    return min(found, key=lambda p: (READABLE.index(p.suffix.lower()), -p.stat().st_size))


def units_guess(height: float) -> tuple[str, float]:
    """What the numbers of a model's height most likely are, and the factor to metres. Things people meet in a
    room stand between a few centimetres and a few metres."""
    if height > 300:
        return "millimetres", 0.001
    if height > 8:
        return "centimetres", 0.01
    return "metres", 1.0


def import_model(source: Path, name: str, z_up: bool, out: Path = OUT) -> tuple[Path, np.ndarray]:
    with tempfile.TemporaryDirectory() as tmp:
        if source.is_dir():
            model = find_model(source)
        elif source.suffix.lower() == ".zip":
            with zipfile.ZipFile(source) as archive:
                archive.extractall(tmp)
            model = find_model(Path(tmp))
        elif source.suffix.lower() in UNREADABLE:
            raise SystemExit(f"error: {source.suffix} cannot be read here: open it in Blender and use "
                             "File > Export > glTF 2.0, format glTF Binary (.glb)")
        else:
            model = source
        scene = trimesh.load(str(model), force="scene")
        if not scene.geometry:
            raise SystemExit(f"error: {model.name} holds no geometry")
        if z_up:
            scene.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{name}.glb"
        target.write_bytes(scene.export(file_type="glb"))
        for stale in folder.glob("*.object_config.json"):          # written again for the new file when loaded
            stale.unlink()
        lo, hi = scene.bounds
    return target, hi - lo


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sgi_habitat.import_model", description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path, help="a model file, or a .zip or folder holding one")
    parser.add_argument("--name", help="the folder and file name under data/habitat/objects/ (default: from the source)")
    parser.add_argument("--z-up", action="store_true", help="the model was made z up: turn it to y up")
    args = parser.parse_args(argv)
    source = args.source.expanduser()
    if not source.exists():
        print(f"error: {source} does not exist", file=sys.stderr)
        return 2
    name = args.name or "".join(c if c.isalnum() or c in "-_" else "_" for c in source.stem.lower())
    target, (width, height, depth) = import_model(source, name, args.z_up)
    units, factor = units_guess(height)
    print(f"wrote {target.relative_to(REPO) if target.is_relative_to(REPO) else target}")
    print(f"size as stored: {width:.3g} wide, {height:.3g} high, {depth:.3g} deep: most likely {units}")
    if max(width, depth) > 2 * height and min(width, depth) > 0.9 * height and not args.z_up:
        print("  its height is its smallest size: if it should stand up, import it again with --z-up")
    size = f"height_m: {height * factor:.2f}" if factor != 1.0 else "scale: 1.0"
    path = target.relative_to(REPO) if target.is_relative_to(REPO) else target
    print("paste under objects: in a scenario, with x and y where it should stand:")
    print(f"  - {{model: {path}, x: 5.0, y: -2.0, {size}, yaw_deg: 0}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
