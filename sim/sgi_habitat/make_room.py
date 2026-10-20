"""Build an empty room as a GLB stage, to furnish with `objects:` in a scenario (docs/habitat.md).

    python -m sgi_habitat.make_room [--width 10] [--depth 8] [--door 2.0] [--no-walls] [--floor-rgb R G B]
                                    [--out data/habitat/stages/room.glb]

A floor of width (x) by depth metres, walls 2.8 m high with no ceiling, and a doorway in the south wall. In the
pipeline's world the room spans x 0..width and y -depth..0: the north wall is at y = 0, the doorway at
y = -depth. The floor's top is at height 0, as every stage needs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import trimesh

from sgi_habitat.make_cafe import WALL_H, WALL_T, box

FLOOR, WALL = (175, 160, 140), (235, 232, 225)


def room(width: float, depth: float, door: float, walls: bool = True, floor=FLOOR) -> trimesh.Scene:
    left = (width - door) / 2
    parts = [box(0, width, -0.1, 0.0, 0, depth, floor, roughness=0.6)]
    if walls:
        parts += [box(-WALL_T, 0, 0, WALL_H, 0, depth, WALL), box(width, width + WALL_T, 0, WALL_H, 0, depth, WALL),
                  box(-WALL_T, width + WALL_T, 0, WALL_H, -WALL_T, 0, WALL),
                  box(-WALL_T, left, 0, WALL_H, depth, depth + WALL_T, WALL),
                  box(left + door, width + WALL_T, 0, WALL_H, depth, depth + WALL_T, WALL)]
    scene = trimesh.Scene()
    for i, mesh in enumerate(parts):
        scene.add_geometry(mesh, node_name=f"part_{i:02d}")
    return scene


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sgi_habitat.make_room", description=__doc__.split("\n\n")[0])
    parser.add_argument("--width", type=float, default=10.0)
    parser.add_argument("--depth", type=float, default=8.0)
    parser.add_argument("--door", type=float, default=2.0, help="width of the doorway in the south wall")
    parser.add_argument("--no-walls", action="store_true", help="a bare floor: open ground, as for an outdoor concert")
    parser.add_argument("--floor-rgb", type=int, nargs=3, default=FLOOR, metavar=("R", "G", "B"))
    parser.add_argument("--out", type=Path, default=Path("data/habitat/stages/room.glb"))
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(room(args.width, args.depth, args.door, not args.no_walls, tuple(args.floor_rgb)).export(file_type="glb"))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
