"""Prepare the things of the realistic set: its models, materials, HDRIs and fonts. Runs inside Blender,
started by `python -m sgi.synth.realistic.assets prepare` through blender.run_script. After "--":

    --manifest DIR     the manifest folder: models.json, materials.json, hdris.json, fonts.json are read
    --downloads DIR    the downloads, one folder per manifest id: DIR/<id>/<file of each source>
    --prepared DIR     where the prepared files go: DIR/models/, DIR/materials/, DIR/hdris/, DIR/fonts/
    --stamp FILE       written last: {"models": [...], "materials": [...], "hdris": [...], "fonts": [...]}, every id
                       of these classes that is now prepared and intact, those of an earlier run among them
    --only ID ...      prepare these alone; an id of another class is passed over
    --force            prepare them again although they are intact

An asset is prepared when it has a receipt `<prepared>/<class>/<id>.json` that was made from the manifest
entry as it is now (its sources, its `prepare` parameters and what else is read of it) and whose `files` are
all there with the size and SHA-256 the receipt gives. Such an asset is left alone; any other is made again.
The receipt keeps those values of the entry themselves, under `made_from`, so that the `verify` action can
tell an asset whose entry was changed after it was made, and say what changed. A receipt also holds what
library.py needs to load the asset, so the prepared folder stands by itself.

What is made, by class, with the `prepare` parameters of an entry that say how:

    models     <id>.blend: one mesh object named <id>, metres at life size, the middle of its base at the origin,
               its front along +X, its textures packed. Parameters: `scale` (to life size) and `front` (the
               way it faces as imported: "+X", "-X", "+Y" or "-Y"); `file` (the source to import, when there
               are several a model could be); `keep` or `drop` (objects of the file, by name); `keep_box` (two
               corners, as imported: only the loose pieces whose middle lies between them are kept, to take
               one thing out of a mesh that holds several); `max_tris` (a denser mesh is thinned to this, 150 000 if
               not given) and
               `texture_px` (a larger texture is shrunk to this, 1024 if not given); `glass` (materials to
               swap for the library's glass, beside every one that transmits), `plain` ({material: colour},
               to paint over one that shows a name or is of the wrong colour) and `band` ({material: {colour,
               from, to}}, a second colour round one that `plain` paints, between two heights in metres over
               the model's base: a sleeve round a cup). The size measured has to be the entry's `size`.
    materials  for the kind "pbr", <id>/<role>.<ext> for each of `maps` ({role: file}, roles color, roughness,
               metallic, height), taken from the entry's folder or from the zip that `archive` names there.
               The kinds "plain", "glass" and "screen" are made in code and have a receipt alone.
    hdris      <id>.hdr or .exr, as fetched. Parameter: `strength`, what the strength it is shown at is
               multiplied by (1 if not given): one sky is several times as bright as another.
    fonts      <id>.ttf, or the entry's `file`, as fetched, with its licence text beside it.

A download is read only when it is the file the manifest names, by its SHA-256. The script ends with an
error, after writing the stamp of what did go well, when any asset could not be prepared.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import traceback
import urllib.parse
import zipfile
from pathlib import Path, PurePosixPath

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import library  # noqa: E402

CLASSES = ("models", "materials", "hdris", "fonts")
# What of a manifest entry its prepared form depends on. A receipt keeps these as they were when the asset was
# made, and assets.py, which cannot import this file, compares them with the manifest under the same names
# (PREPARED_FROM there).
READ_OF_AN_ENTRY = ("sources", "prepare", "kind", "tile_m", "size", "file")
# raised when the same entry would be prepared otherwise: all is remade
MADE_BY = 1
MODEL_SUFFIXES = (".gltf", ".glb", ".fbx", ".obj")
TURN_TO_X = {"+X": 0.0, "-Y": 90.0, "-X": 180.0, "+Y": -90.0}        # degrees about Z that bring that front onto +X
MAP_ROLES = ("color", "roughness", "metallic", "height")
SIZE_TOLERANCE_M = 0.002
MAX_MEMBER_BYTES = 256 << 20                       # of one file taken out of an archive


def arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="prepare_models.py")
    for name in ("--manifest", "--downloads", "--prepared", "--stamp"):
        parser.add_argument(name, type=Path, required=True)
    parser.add_argument("--only", nargs="*", default=None, metavar="ID")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv[argv.index("--") + 1:] if "--" in argv else [])


# ------------------------------------------------------------------------------------------------- files
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while block := file.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, data: bytes) -> None:
    """Whole or not at all, so a file that is there is a file that was finished."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_bytes(data)
    os.replace(part, path)


def entries(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []


def downloaded(downloads: Path, entry: dict) -> dict[str, Path]:
    """The sources of an entry by their path in its folder, each checked to be the file the manifest names."""
    files = {}
    for source in entry["sources"]:                # named as assets.py names it: `file`, else the address's last part
        name = source.get("file") or urllib.parse.unquote(PurePosixPath(urllib.parse.urlsplit(source["url"]).path).name)
        path = downloads / entry["id"] / name
        if not path.is_file() or sha256(path) != source["sha256"]:
            raise ValueError(f"{path} is not fetched, or is not the file the manifest names: run the fetch action")
        files[name] = path
    return files


def one_of(files: dict[str, Path], suffixes: tuple[str, ...], entry: dict) -> str:
    """The one source of an entry that ends in one of the suffixes."""
    found = [name for name in files if name.lower().endswith(suffixes)]
    if len(found) != 1:
        raise ValueError(f"{entry['id']} needs one source ending in {' or '.join(suffixes)}, and has {found or 'none'}")
    return found[0]


def made_from(entry: dict) -> dict:
    """Everything of an entry that its prepared form depends on, as its receipt keeps it: the entry's own
    values, null for what it has not, and the number of this way of preparing. JSON reads them back equal."""
    return {"by": MADE_BY, **{key: entry.get(key) for key in READ_OF_AN_ENTRY}}


def is_intact(prepared: Path, name: str, entry: dict) -> bool:
    try:
        receipt = json.loads((prepared / name / f"{entry['id']}.json").read_text(encoding="utf-8"))
        return receipt["made_from"] == made_from(entry) and all(
            (prepared / rel).stat().st_size == facts["bytes"] and sha256(prepared / rel) == facts["sha256"]
            for rel, facts in receipt["files"].items())
    except (OSError, ValueError, KeyError, TypeError):
        return False


# ------------------------------------------------------------------------------------------------ models
def bounds(mesh) -> tuple[np.ndarray, np.ndarray]:
    points = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", points)
    points = points.reshape(-1, 3)
    return points.min(axis=0), points.max(axis=0)


def triangles(mesh) -> int:
    mesh.calc_loop_triangles()
    return len(mesh.loop_triangles)


def import_model(path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix in (".gltf", ".glb"):
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    elif suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    else:
        raise ValueError(f"no importer for {path.name}: expected one of {', '.join(MODEL_SUFFIXES)}")


def one_mesh(name: str, given: dict) -> "bpy.types.Object":
    """Everything that was imported as one mesh object in world coordinates, at the origin and with no parent;
    what is not a mesh, and what `keep` and `drop` leave out, is gone."""
    bpy.context.view_layer.update()
    keep, drop = given.get("keep"), set(given.get("drop", ()))
    named = {ob.name for ob in bpy.data.objects}
    unknown = sorted((set(keep or ()) | drop) - named)
    if unknown:
        raise ValueError(f"prepare names {', '.join(unknown)}, and the file holds {', '.join(sorted(named))}")
    meshes = []
    for ob in list(bpy.data.objects):
        left_out = ob.name in drop or (keep is not None and ob.name not in keep)
        if ob.type != "MESH" or not len(ob.data.vertices) or left_out:
            continue
        placed = ob.matrix_world.copy()
        ob.data = ob.data.copy()                   # two objects may share one mesh: each gets its own place
        ob.data.transform(placed)
        if placed.determinant() < 0:
            ob.data.flip_normals()
        meshes.append(ob)
    if not meshes:
        raise ValueError("the file holds no mesh that is to be kept")
    for ob in list(bpy.data.objects):
        if ob not in meshes:
            bpy.data.objects.remove(ob)
    for ob in meshes:
        ob.parent = None
        ob.matrix_world = Matrix.Identity(4)
    if len(meshes) > 1:
        with bpy.context.temp_override(active_object=meshes[0], object=meshes[0], selected_objects=meshes,
                                       selected_editable_objects=meshes):
            bpy.ops.object.join()
    whole = meshes[0]
    whole.name = whole.data.name = name
    return whole


def keep_in_box(mesh, corners) -> None:
    """Take out of a mesh every loose piece whose middle does not lie between two corners."""
    low, high = (np.minimum(corners[0], corners[1]), np.maximum(corners[0], corners[1]))
    work = bmesh.new()
    work.from_mesh(mesh)
    work.verts.ensure_lookup_table()
    seen, gone = set(), []
    for start in work.verts:
        if start.index in seen:
            continue
        piece, edge = [], [start]
        seen.add(start.index)
        while edge:
            vert = edge.pop()
            piece.append(vert)
            for link in vert.link_edges:
                other = link.other_vert(vert)
                if other.index not in seen:
                    seen.add(other.index)
                    edge.append(other)
        points = np.array([vert.co[:] for vert in piece])
        middle = (points.min(axis=0) + points.max(axis=0)) / 2
        if not (np.all(middle >= low) and np.all(middle <= high)):
            gone += piece
    bmesh.ops.delete(work, geom=gone, context="VERTS")
    if not work.verts:
        work.free()
        raise ValueError("keep_box leaves nothing of the mesh")
    work.to_mesh(mesh)
    work.free()


def thin(ob, most: int) -> None:
    """Collapse the mesh to about `most` triangles."""
    have = triangles(ob.data)
    if have <= most:
        return
    modifier = ob.modifiers.new("thin", "DECIMATE")
    modifier.ratio = most / have
    bpy.context.view_layer.update()
    graph = bpy.context.evaluated_depsgraph_get()
    thinned = bpy.data.meshes.new_from_object(ob.evaluated_get(graph), preserve_all_data_layers=True, depsgraph=graph)
    ob.modifiers.remove(modifier)
    old, ob.data = ob.data, thinned
    name = old.name
    bpy.data.meshes.remove(old)                    # first, or its name is still taken
    thinned.name = name


def shader_of(material):
    if material is None or not material.use_nodes:
        return None
    return next((node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"), None)


def banded(coat, colour, band: dict) -> None:
    """Put a band of another colour round a plain coat, between two heights over the model's base. The heights
    are read from the mesh's own coordinates, which are metres at life size once the model is normalised and
    which a placed model's scale leaves as they are."""
    low, high = float(band["from"]), float(band["to"])
    if not 0.0 <= low < high:
        raise ValueError(f"prepare.band runs from {band['from']!r} to {band['to']!r}: expected two heights in metres "
                         "over the base, the first below the second")
    tree = coat.node_tree
    height = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(tree.nodes.new("ShaderNodeTexCoord").outputs["Object"], height.inputs[0])
    above, below, within = (tree.nodes.new("ShaderNodeMath") for _ in range(3))
    above.operation, below.operation, within.operation = "GREATER_THAN", "LESS_THAN", "MULTIPLY"
    above.inputs[1].default_value, below.inputs[1].default_value = low, high
    for edge in (above, below):
        tree.links.new(height.outputs["Z"], edge.inputs[0])
    tree.links.new(above.outputs[0], within.inputs[0])
    tree.links.new(below.outputs[0], within.inputs[1])
    mix = tree.nodes.new("ShaderNodeMix")              # its factor is socket 0; its colours are 6 and 7 in, and 2 out
    mix.data_type = "RGBA"
    mix.inputs[6].default_value = (*library.linear(colour), 1.0)
    mix.inputs[7].default_value = (*library.linear(band["colour"]), 1.0)
    tree.links.new(within.outputs[0], mix.inputs[0])
    tree.links.new(mix.outputs[2], tree.nodes["Principled BSDF"].inputs["Base Color"])


def restyle(ob, given: dict) -> dict:
    """Swap the materials that transmit, and those named as glass, for the library's glass, paint over the ones
    named as plain, and put a band round those of them that are to have one. What was swapped, for the receipt."""
    glass_names, plain = set(given.get("glass", ())), dict(given.get("plain", {}))
    bands = dict(given.get("band", {}))
    have = {slot.material.name for slot in ob.material_slots if slot.material is not None}
    unknown = sorted((glass_names | set(plain) | set(bands)) - have)
    if unknown:
        raise ValueError(f"prepare names the materials {', '.join(unknown)}, and the model has "
                         + ", ".join(sorted(have)))
    if set(bands) - set(plain):
        raise ValueError(f"prepare.band names {', '.join(sorted(set(bands) - set(plain)))}, which prepare.plain does "
                         "not paint: a band goes round a plain coat")
    glass, swapped = None, {"glass": [], "plain": [], "band": []}
    for slot in ob.material_slots:
        material, shader = slot.material, shader_of(slot.material)
        if material is None:
            continue
        if material.name in plain:
            swapped["plain"].append(material.name)
            coat = bpy.data.materials.new(material.name + "_plain")
            coat.use_nodes = True
            inputs = coat.node_tree.nodes["Principled BSDF"].inputs
            inputs["Base Color"].default_value = (*library.linear(plain[material.name]), 1.0)
            inputs["Roughness"].default_value = 0.5
            if material.name in bands:
                swapped["band"].append(material.name)
                banded(coat, plain[material.name], bands[material.name])
            slot.material = coat
        elif material.name in glass_names or (shader is not None and not shader.inputs["Transmission Weight"].is_linked
                                              and shader.inputs["Transmission Weight"].default_value > 0.5):
            swapped["glass"].append(material.name)
            glass = glass or library.window_glass("glass")
            slot.material = glass
    return swapped


def pack_textures(edge: int) -> list[str]:
    """Every image a material still uses, no larger than `edge` on a side and packed into the file; the names of
    those that were shrunk. Images that nothing uses any more are dropped."""
    bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    shrunk = []
    for image in list(bpy.data.images):
        if image.type != "IMAGE" or image.source not in ("FILE", "GENERATED"):
            continue
        width, height = image.size
        if not width or not height:
            raise ValueError(f"the texture {image.name} does not load")
        if max(width, height) > edge:
            factor = edge / max(width, height)
            image.scale(max(1, round(width * factor)), max(1, round(height * factor)))
            shrunk.append(f"{image.name} {width}x{height}")
            image.pack()                           # the pixels as they are now
        elif image.packed_file is None:
            image.pack()                           # the file as it was fetched; one that came packed stays as it is
        if image.packed_file is None:
            raise ValueError(f"the texture {image.name} could not be packed into the model's file")
    return shrunk


def build_model(entry: dict, files: dict[str, Path]) -> dict:
    """The model of an entry as the one object of an empty file, normalised; what was measured and done."""
    given = entry.get("prepare", {})
    candidates = [name for name in files if name.lower().endswith(MODEL_SUFFIXES)]
    name = given.get("file") or (candidates[0] if len(candidates) == 1 else None)
    if name not in files:
        raise ValueError(f"prepare.file is {given.get('file')!r}, and the sources a model could be are {candidates}")
    if given.get("front", "-Y") not in TURN_TO_X:
        raise ValueError(f"prepare.front is {given.get('front')!r}: expected one of {', '.join(TURN_TO_X)}")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import_model(files[name])
    ob = one_mesh(entry["id"], given)
    if "keep_box" in given:
        keep_in_box(ob.data, np.array(given["keep_box"], dtype=float))
    as_imported = triangles(ob.data)
    thin(ob, int(given.get("max_tris", 150_000)))
    swapped = restyle(ob, given)
    scale = float(given.get("scale", 1.0))
    ob.data.transform(Matrix.Rotation(math.radians(TURN_TO_X[given.get("front", "-Y")]), 4, "Z")
                      @ Matrix.Diagonal((scale, scale, scale, 1.0)))
    low, high = bounds(ob.data)
    ob.data.transform(Matrix.Translation(Vector((-(low[0] + high[0]) / 2, -(low[1] + high[1]) / 2, -low[2]))))
    ob.data.update()
    low, high = bounds(ob.data)
    shrunk = pack_textures(int(given.get("texture_px", 1024)))
    return {"size": [round(float(side), 4) for side in high - low], "triangles": triangles(ob.data),
            "triangles_as_imported": as_imported, "swapped": swapped, "shrunk": shrunk,
            "materials": sorted(slot.material.name for slot in ob.material_slots if slot.material is not None)}


def make_model(entry: dict, downloads: Path, prepared: Path) -> tuple[dict, list[str]]:
    facts = build_model(entry, downloaded(downloads, entry))
    if any(abs(measured - stated) > SIZE_TOLERANCE_M for measured, stated in zip(facts["size"], entry["size"])):
        raise ValueError(f"the prepared model measures {facts['size']} (along its front, across, up) and the manifest "
                         f"gives its size as {entry['size']}: the manifest has to give what is measured")
    rel = f"models/{entry['id']}.blend"
    (prepared / "models").mkdir(parents=True, exist_ok=True)
    part = prepared / "models" / f"{entry['id']}.part.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(part), compress=True, copy=True, relative_remap=False)
    os.replace(part, prepared / rel)
    return {**facts, "blend": rel}, [rel]


# --------------------------------------------------------------------------------------------- materials
def mean_linear(path: Path) -> list[float]:
    """The mean colour of a picture in scene-linear values: what a neutral material divides its colour map by."""
    image = bpy.data.images.load(str(path), check_existing=False)
    pixels = np.empty(image.size[0] * image.size[1] * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    bpy.data.images.remove(image)
    shown = pixels.reshape(-1, 4)[:, :3].astype(np.float64)
    lin = np.where(shown <= 0.04045, shown / 12.92, ((shown + 0.055) / 1.055) ** 2.4)
    return [round(float(value), 5) for value in lin.mean(axis=0)]


def make_material(entry: dict, downloads: Path, prepared: Path) -> tuple[dict, list[str]]:
    given, kind = entry.get("prepare", {}), entry["kind"]
    facts = {"kind": kind, "tile_m": float(entry["tile_m"]), "prepare": given}
    if kind in ("plain", "glass", "screen"):
        return facts, []
    if kind != "pbr":
        raise ValueError(f"kind is {kind!r}: expected pbr, plain, glass or screen")
    files, maps = downloaded(downloads, entry), given.get("maps", {})
    if "color" not in maps or set(maps) - set(MAP_ROLES):
        raise ValueError(f"prepare.maps is {maps!r}: expected a file for color, and for any of "
                         + ", ".join(MAP_ROLES[1:]))
    archive = zipfile.ZipFile(files[given["archive"]]) if "archive" in given else None
    written = {}
    for role, name in maps.items():
        if archive is not None:                    # read by name and written under ours: no path of the zip is followed
            if archive.getinfo(name).file_size > MAX_MEMBER_BYTES:
                raise ValueError(f"{name} in {given['archive']} is larger than a texture is")
            data = archive.read(name)
        else:
            data = files[name].read_bytes()
        rel = f"materials/{entry['id']}/{role}{PurePosixPath(name).suffix.lower()}"
        write(prepared / rel, data)
        written[role] = rel
    return {**facts, "maps": written, "mean_linear": mean_linear(prepared / written["color"])}, list(written.values())


# ------------------------------------------------------------------------------------- HDRIs and fonts
def make_hdri(entry: dict, downloads: Path, prepared: Path) -> tuple[dict, list[str]]:
    files = downloaded(downloads, entry)
    name = one_of(files, (".hdr", ".exr"), entry)
    rel = f"hdris/{entry['id']}{PurePosixPath(name).suffix.lower()}"
    write(prepared / rel, files[name].read_bytes())
    strength = float(entry.get("prepare", {}).get("strength", 1.0))
    if not (math.isfinite(strength) and strength > 0):
        raise ValueError(f"prepare.strength is {strength!r}: expected a number above 0")
    return {"file": rel, "strength": strength}, [rel]


def make_font(entry: dict, downloads: Path, prepared: Path) -> tuple[dict, list[str]]:
    files = downloaded(downloads, entry)
    name = one_of(files, (".ttf", ".otf"), entry)
    rel = entry.get("file", f"fonts/{entry['id']}.ttf")
    if os.path.isabs(rel) or os.path.normpath(rel).split(os.sep)[0] == os.pardir:
        raise ValueError(f"file is {rel!r}: expected a path under the prepared folder")
    write(prepared / rel, files[name].read_bytes())
    made = [rel]
    for other in files:                            # its licence, which goes wherever the font goes
        if other.lower().endswith(".txt"):
            made.append(f"fonts/{entry['id']}.{PurePosixPath(other).name}")
            write(prepared / made[-1], files[other].read_bytes())
    return {"file": rel}, made


MAKERS = {"models": make_model, "materials": make_material, "hdris": make_hdri, "fonts": make_font}


def main(argv: list[str]) -> None:
    given = arguments(argv)
    stamp, failed = {name: [] for name in CLASSES}, []
    for name in CLASSES:
        for entry in entries(given.manifest / f"{name}.json"):
            chosen = given.only is None or entry["id"] in given.only
            if not (chosen and given.force) and is_intact(given.prepared, name, entry):
                stamp[name].append(entry["id"])
                continue
            if not chosen:
                continue
            print(f"preparing {name} {entry['id']}", flush=True)
            try:
                facts, made = MAKERS[name](entry, given.downloads, given.prepared)
                files = {rel: {"sha256": sha256(given.prepared / rel), "bytes": (given.prepared / rel).stat().st_size}
                         for rel in made}
                receipt = {"id": entry["id"], "class": name, "made_from": made_from(entry), "files": files, **facts}
                write(given.prepared / name / f"{entry['id']}.json", (json.dumps(receipt, indent=1) + "\n").encode())
                stamp[name].append(entry["id"])
            except Exception as error:             # one asset that fails does not stop the others; all are named below
                traceback.print_exc()
                failed.append(f"{name} {entry['id']}: {type(error).__name__}: {error}")
    write(given.stamp, (json.dumps(stamp, indent=1) + "\n").encode())
    if failed:
        raise RuntimeError("could not prepare:\n  " + "\n  ".join(failed))


if __name__ == "__main__":
    main(sys.argv)
