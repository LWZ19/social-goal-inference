"""Download 3D models listed in the repository's asset manifest, for placing in a scene (docs/habitat.md).

    python -m sgi_habitat.fetch_models coffee_brewer plant_tall cooler_water
    python -m sgi_habitat.fetch_models --list

Each model goes to data/habitat/objects/<id>/, every file checked against the SHA-256 the manifest gives
(assets/realistic/manifest/models.json). The file to name in a scenario's `objects:` is printed. A file already
there with the right hash is not fetched again. Each model keeps its licence: the manifest names it, and
assets/realistic/ATTRIBUTION.md says what crediting it takes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "assets" / "realistic" / "manifest" / "models.json"
OUT = REPO / "data" / "habitat" / "objects"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(model: dict, out: Path) -> Path:
    """Download one model's files; the path of its main file (.gltf or .glb)."""
    folder = out / model["id"]
    main = None
    for source in model["sources"]:
        name = source.get("file") or source["url"].rsplit("/", 1)[-1]
        path = folder / name
        if not (path.is_file() and _sha256(path) == source["sha256"]):
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(source["url"], timeout=120) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != source["sha256"]:
                raise RuntimeError(f"{model['id']}: {source['url']} does not have the manifest's SHA-256")
            path.write_bytes(data)
        if main is None and path.suffix in (".gltf", ".glb"):
            main = path
    if main is None:
        raise RuntimeError(f"{model['id']}: no .gltf or .glb among its files")
    return main


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sgi_habitat.fetch_models", description=__doc__.split("\n\n")[0])
    parser.add_argument("ids", nargs="*", help="model ids of the manifest")
    parser.add_argument("--list", action="store_true", help="list the models of the manifest and fetch nothing")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    models = {m["id"]: m for m in json.loads(MANIFEST.read_text())}
    if args.list or not args.ids:
        for m in models.values():
            w, d, h = m["size"]
            print(f"{m['id']:18s} {m['kind']:10s} {w:.2f} x {d:.2f} x {h:.2f} m  {m['title']}  [{m['licence'].split(' (')[0]}]")
        return 0
    unknown = [i for i in args.ids if i not in models]
    if unknown:
        print(f"error: not in the manifest: {', '.join(unknown)} (see --list)", file=sys.stderr)
        return 2
    for model_id in args.ids:
        path = fetch(models[model_id], args.out)
        print(f"{model_id}: {path.relative_to(REPO) if path.is_relative_to(REPO) else path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
