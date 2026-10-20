"""Colour a scene that came without colours, by the height and the facing of each surface (docs/habitat.md).

    python -m sgi_habitat.paint_stage IN.gltf OUT.glb [--style clay|colours] [--floor-y 0.15]
                                       [--region X0 X1 Y0 Y1] [--counter X0 X1 Y0 Y1 ...] [--keep-ceiling]

--style clay (the default) gives the two tones of Sketchfab's clay view: the floor and every top a pale warm grey,
everything else one terracotta, with the shape shown by the lighting. --style colours guesses a colour for each
kind of surface, as below.

Some downloads (Sketchfab's conversions among them) hold geometry and normals but no material colours, and
render white. Each face is given the colour a person would guess for it: the floor wood inside --region (the
pipeline's x and y; the whole scene when it is left out) and paving outside it, table tops light wood, counter
tops pale stone, tall upright faces wall paint, low furniture dark wood, and the rest of the fittings dark metal.
Upright faces along the edge of --region, or outside it, are wall at any height. Inside each --counter box, the
upright faces below the counter top are the counter's wooden front, its tops stone, and what stands on it metal.
Faces looking down from above 2.4 m, the ceiling, are dropped unless --keep-ceiling: the camera that follows
the robot looks down from above. Heights are taken from --floor-y, the height of the floor in the file.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from trimesh.visual.material import PBRMaterial

CLAY = {"floor", "ground", "table_top", "counter_top"}  # the pale faces of the clay style; the rest are terracotta
CLAY_PALE, CLAY_BODY = ((214, 211, 204), 0.8), ((214, 160, 146), 0.7)    # lit, they come out as the clay view
PALETTE = {                                         # name: (RGB, roughness)
    "floor": ((150, 108, 72), 0.55), "ground": ((128, 126, 120), 0.9), "table_top": ((186, 146, 100), 0.5),
    "counter_top": ((226, 224, 218), 0.3), "wall": ((226, 216, 196), 0.85), "furniture": ((92, 62, 42), 0.6),
    "fitting": ((70, 72, 78), 0.35), "counter_front": ((128, 82, 48), 0.5),
}
EDGE_M = 0.35                                       # upright faces this near the edge of --region are wall


def _inside(x, y, box, margin: float = 0.0):
    x0, x1, y0, y1 = box
    return (x >= x0 + margin) & (x <= x1 - margin) & (y >= y0 + margin) & (y <= y1 - margin)


def classify(centres: np.ndarray, normals: np.ndarray, floor_y: float, region, counters=()) -> np.ndarray:
    """A PALETTE name, or "ceiling", for each face of a y-up mesh."""
    h = centres[:, 1] - floor_y
    up, down = normals[:, 1] > 0.8, normals[:, 1] < -0.8
    upright = np.abs(normals[:, 1]) < 0.3
    x, y = centres[:, 0], -centres[:, 2]                        # the pipeline's (x, y) is Habitat's (x, -z)
    inside = np.ones(len(h), bool) if region is None else _inside(x, y, region)
    edge = np.zeros(len(h), bool) if region is None else ~_inside(x, y, region, EDGE_M)
    kind = np.full(len(h), "fitting", dtype=object)
    kind[h < 0.6] = "furniture"
    kind[upright & (h > 1.6)] = "wall"
    kind[up & (h >= 0.6) & (h < 0.85)] = "table_top"
    kind[up & (h >= 0.85) & (h < 1.2)] = "counter_top"
    kind[upright & edge & (h > 0.08)] = "wall"
    for box in counters:
        on = _inside(x, y, box)
        kind[on & (h < 1.2)] = "counter_front"
        kind[on & up & (h >= 0.85) & (h < 1.2)] = "counter_top"
        kind[on & (h >= 1.2) & ~up] = "fitting"
    kind[up & (h < 0.08)] = np.where(inside, "floor", "ground")[up & (h < 0.08)]
    kind[down & (h > 2.4)] = "ceiling"
    return kind


def paint(source: Path, floor_y: float, region, keep_ceiling: bool, counters=(),
          style: str = "clay") -> tuple[trimesh.Scene, dict]:
    mesh = trimesh.load(str(source), force="mesh")              # every part in the file's own frame, as one mesh
    kinds = classify(mesh.triangles_center, mesh.face_normals, floor_y, region, counters)
    palette = PALETTE
    if style == "clay":
        palette = {"pale": CLAY_PALE, "body": CLAY_BODY}
        kinds = np.where(kinds == "ceiling", "ceiling", np.where(np.isin(kinds, list(CLAY)), "pale", "body"))
    scene, counts = trimesh.Scene(), {}
    for name, (rgb, roughness) in palette.items():
        faces = np.flatnonzero(kinds == name)
        counts[name] = len(faces)
        if not len(faces):
            continue
        part = mesh.submesh([faces], append=True)
        part.visual = trimesh.visual.TextureVisuals(material=PBRMaterial(
            name=name, baseColorFactor=[*(c / 255 for c in rgb), 1.0], metallicFactor=0.0, roughnessFactor=roughness,
            doubleSided=True))
        scene.add_geometry(part, node_name=name)
    counts["ceiling"] = int((kinds == "ceiling").sum())
    if keep_ceiling and counts["ceiling"]:
        part = mesh.submesh([np.flatnonzero(kinds == "ceiling")], append=True)
        part.visual = trimesh.visual.TextureVisuals(material=PBRMaterial(name="ceiling", baseColorFactor=[0.93, 0.93, 0.92, 1.0]))
        scene.add_geometry(part, node_name="ceiling")
    return scene, counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sgi_habitat.paint_stage", description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--style", choices=("clay", "colours"), default="clay")
    parser.add_argument("--floor-y", type=float, default=0.0, help="the height of the floor in the file")
    parser.add_argument("--region", type=float, nargs=4, metavar=("X0", "X1", "Y0", "Y1"),
                        help="the room, in the pipeline's x and y: floor inside it, paving outside")
    parser.add_argument("--counter", type=float, nargs=4, action="append", default=[], metavar=("X0", "X1", "Y0", "Y1"),
                        help="the footprint of a counter, in the pipeline's x and y; may be given several times")
    parser.add_argument("--keep-ceiling", action="store_true")
    args = parser.parse_args(argv)
    scene, counts = paint(args.source, args.floor_y, args.region, args.keep_ceiling, args.counter, args.style)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(scene.export(file_type="glb"))
    for stale in args.out.parent.glob(args.out.stem + ".stage_config.json"):    # written again when loaded
        stale.unlink()
    print(f"wrote {args.out}: " + ", ".join(f"{name} {n}" for name, n in counts.items())
          + ("" if args.keep_ceiling else " (ceiling dropped)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
