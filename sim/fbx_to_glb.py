"""Convert an .fbx model to .glb with Assimp, in the sgi-tools environment (envs/sgi-tools.yml).

    conda run -n sgi-tools python sim/fbx_to_glb.py IN.fbx OUT.glb

Assimp is called through ctypes, so nothing but the library is needed. The .glb keeps the geometry, the UVs and
the materials the .fbx names; textures that come as separate image files are not linked by most .fbx downloads,
and are put on afterwards (docs/habitat.md, "A concert"). Bring the result in with
`python -m sgi_habitat.import_model` or name it under `objects:` directly.
"""
import ctypes
import os
import sys
from pathlib import Path

TRIANGULATE, GEN_NORMALS = 0x8, 0x20


def convert(source: Path, target: Path) -> None:
    lib = ctypes.CDLL(os.path.join(os.environ.get("CONDA_PREFIX", sys.prefix), "lib", "libassimp.so"))
    lib.aiImportFile.restype = ctypes.c_void_p
    lib.aiImportFile.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    lib.aiExportScene.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    lib.aiReleaseImport.argtypes = [ctypes.c_void_p]
    lib.aiGetErrorString.restype = ctypes.c_char_p
    scene = lib.aiImportFile(str(source).encode(), TRIANGULATE | GEN_NORMALS)
    if not scene:
        raise SystemExit(f"error: Assimp could not read {source}: {lib.aiGetErrorString().decode()}")
    try:
        if lib.aiExportScene(scene, b"glb2", str(target).encode(), 0) != 0:
            raise SystemExit(f"error: Assimp could not write {target}: {lib.aiGetErrorString().decode()}")
    finally:
        lib.aiReleaseImport(scene)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__.split("\n\n")[1])
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"wrote {sys.argv[2]}")
