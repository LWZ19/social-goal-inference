"""Build a small café as a GLB stage for Habitat: the example of a scene made outside ReplicaCAD.

    python -m sgi_habitat.make_cafe [--out data/habitat/stages/cafe.glb]

A room of 10 x 8 m with walls but no ceiling, a coffee counter with a machine and a menu board along the north
wall, four tables with chairs, and plants. It is written as Habitat expects any stage, from Blender or
elsewhere: glTF binary, metres, y up, the floor's top at y = 0. In the pipeline's world (sim/scenarios/cafe.yaml)
Habitat's (x, z) is (x, -y), so the room spans x 0..10 and y -8..0, and the counter lies along y = -0.3..-1.0.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import trimesh
from trimesh.visual.material import PBRMaterial

W, D, WALL_H, WALL_T = 10.0, 8.0, 2.8, 0.15        # room width (x), depth (z), wall height and thickness


def _part(mesh: trimesh.Trimesh, rgb, roughness: float = 0.8) -> trimesh.Trimesh:
    material = PBRMaterial(baseColorFactor=[*(c / 255 for c in rgb), 1.0], metallicFactor=0.0, roughnessFactor=roughness)
    mesh.visual = trimesh.visual.TextureVisuals(material=material)
    return mesh


def box(x0, x1, y0, y1, z0, z1, rgb, roughness=0.8) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=[x1 - x0, y1 - y0, z1 - z0])
    mesh.apply_translation([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
    return _part(mesh, rgb, roughness)


def cylinder(x, z, radius, y0, y1, rgb, sections=24) -> trimesh.Trimesh:
    mesh = trimesh.creation.cylinder(radius=radius, height=y1 - y0, sections=sections)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0]))   # its axis was z: now y
    mesh.apply_translation([x, (y0 + y1) / 2, z])
    return _part(mesh, rgb)


def table_with_chairs(x, z):
    parts = [cylinder(x, z, 0.4, 0.72, 0.76, (120, 80, 50)), cylinder(x, z, 0.05, 0.0, 0.72, (40, 40, 40)),
             cylinder(x, z, 0.25, 0.0, 0.03, (40, 40, 40))]
    for angle in (0.0, math.pi):
        cx, cz = x + 0.65 * math.cos(angle), z + 0.65 * math.sin(angle)
        parts += [box(cx - 0.2, cx + 0.2, 0.42, 0.46, cz - 0.2, cz + 0.2, (60, 90, 120))]
        back = cx + 0.18 * math.cos(angle)
        parts += [box(back - 0.02, back + 0.02, 0.46, 0.9, cz - 0.2, cz + 0.2, (60, 90, 120))]
        parts += [box(cx - 0.18, cx - 0.15, 0.0, 0.42, cz - 0.18, cz - 0.15, (30, 30, 30)),
                  box(cx + 0.15, cx + 0.18, 0.0, 0.42, cz + 0.15, cz + 0.18, (30, 30, 30)),
                  box(cx - 0.18, cx - 0.15, 0.0, 0.42, cz + 0.15, cz + 0.18, (30, 30, 30)),
                  box(cx + 0.15, cx + 0.18, 0.0, 0.42, cz - 0.18, cz - 0.15, (30, 30, 30))]
    return parts


def cafe() -> trimesh.Scene:
    parts = [
        box(0, W, -0.1, 0.0, 0, D, (190, 160, 120)),                                   # wooden floor
        box(-WALL_T, 0, 0, WALL_H, 0, D, (235, 230, 220)),                               # west wall
        box(W, W + WALL_T, 0, WALL_H, 0, D, (235, 230, 220)),                            # east wall
        box(-WALL_T, W + WALL_T, 0, WALL_H, -WALL_T, 0, (225, 215, 200)),                # north wall
        box(-WALL_T, 3.5, 0, WALL_H, D, D + WALL_T, (235, 230, 220)),                    # south wall, with a door
        box(5.0, W + WALL_T, 0, WALL_H, D, D + WALL_T, (235, 230, 220)),
        # the coffee counter along the north wall, its front at z = 1.0
        box(3.5, 6.5, 0.0, 1.0, 0.3, 1.0, (70, 45, 30)),
        box(3.45, 6.55, 1.0, 1.05, 0.25, 1.05, (220, 220, 215), roughness=0.3),          # counter top
        box(4.6, 5.1, 1.05, 1.55, 0.4, 0.8, (30, 30, 30), roughness=0.3),                 # espresso machine
        box(4.65, 5.05, 1.25, 1.35, 0.8, 0.85, (180, 180, 180), roughness=0.2),
        cylinder(5.6, 0.6, 0.05, 1.05, 1.17, (250, 250, 250)),                            # cups
        cylinder(5.75, 0.6, 0.05, 1.05, 1.17, (250, 250, 250)),
        box(5.9, 6.3, 1.05, 1.3, 0.45, 0.75, (200, 60, 40)),                               # till
        box(3.8, 6.2, 1.7, 2.5, 0.0, 0.05, (35, 50, 40)),                                 # menu board
        box(3.5, 6.5, 0.0, 1.0, 0.0, 0.3, (50, 35, 25)),                                  # back counter
        # plants
        cylinder(0.5, 0.5, 0.25, 0.0, 0.45, (150, 80, 50)), cylinder(0.5, 0.5, 0.35, 0.45, 1.3, (50, 120, 50)),
        cylinder(9.5, 0.5, 0.25, 0.0, 0.45, (150, 80, 50)), cylinder(9.5, 0.5, 0.35, 0.45, 1.3, (50, 120, 50)),
    ]
    for x, z in ((1.8, 3.8), (1.8, 6.3), (8.2, 3.6), (8.2, 6.0)):
        parts += table_with_chairs(x, z)
    scene = trimesh.Scene()
    for i, mesh in enumerate(parts):
        scene.add_geometry(mesh, node_name=f"part_{i:02d}")
    return scene


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sgi_habitat.make_cafe", description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("data/habitat/stages/cafe.glb"))
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(cafe().export(file_type="glb"))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
