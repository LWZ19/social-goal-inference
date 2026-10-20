"""Put PBR texture images on a model whose download keeps them as separate files (docs/habitat.md, "A concert").

    python -m sgi_habitat.texture_model IN.glb OUT.glb --base BASE.png [--roughness R.png] [--metallic M.png]
                                        [--normal N.png] [--metallic-factor 0.35] [--size 2048]

Every part of the model gets one glTF material from the images: base colour, metallic-roughness (packed from the
roughness and metallic maps, as glTF has it: green roughness, blue metallic) and normal, each scaled to --size.
The model has to carry texture coordinates. --metallic-factor scales the metallic map down: lit without
reflections, as the simulator lights it, a fully metal surface renders black.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial

Image.MAX_IMAGE_PIXELS = None                      # texture maps of 8k and more are common


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sgi_habitat.texture_model", description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--base", type=Path, required=True, help="the base colour image")
    parser.add_argument("--roughness", type=Path)
    parser.add_argument("--metallic", type=Path)
    parser.add_argument("--normal", type=Path)
    parser.add_argument("--metallic-factor", type=float, default=0.35)
    parser.add_argument("--size", type=int, default=2048, help="side of every texture, in pixels")
    args = parser.parse_args(argv)
    size = (args.size, args.size)

    def image(path, mode):
        return None if path is None else Image.open(path).convert(mode).resize(size, Image.LANCZOS)

    base, rough, metal, normal = (image(args.base, "RGB"), image(args.roughness, "L"), image(args.metallic, "L"),
                                  image(args.normal, "RGB"))
    packed = None
    if rough is not None or metal is not None:
        packed = Image.merge("RGB", (Image.new("L", size, 255), rough or Image.new("L", size, 255),
                                     metal or Image.new("L", size, 0)))
    scene = trimesh.load(str(args.source), force="scene")
    for mesh in scene.geometry.values():
        uv = getattr(mesh.visual, "uv", None)
        if uv is None:
            raise SystemExit(f"error: {args.source} has a part without texture coordinates; it cannot take textures")
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=PBRMaterial(
            baseColorTexture=base, metallicRoughnessTexture=packed, normalTexture=normal,
            metallicFactor=args.metallic_factor if packed is not None else 0.0, roughnessFactor=1.0, doubleSided=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(scene.export(file_type="glb"))
    for stale in args.out.parent.glob(args.out.stem + ".object_config.json"):     # written again when loaded
        stale.unlink()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
