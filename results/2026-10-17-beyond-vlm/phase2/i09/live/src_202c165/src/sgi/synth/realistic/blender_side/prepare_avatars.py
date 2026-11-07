"""Prepare the people of the realistic set: every avatar of the manifest baked in every pose into static meshes.

Run inside Blender by sgi.synth.realistic.blender.run_script, with these arguments after "--":

    --manifest DIR     the manifest folder; avatars.json and poses.json are read from it
    --downloads DIR    the downloads, one folder per manifest id: DIR/<id>/<file of each source>
    --prepared DIR     where the prepared files go (below)
    --stamp FILE       written last: {"avatars": [...], "poses": [...]}, every id that is now prepared and intact,
                       those of earlier runs included
    --only ID ...      prepare these alone; an id of another asset class is passed over
    --force            prepare them again although they are intact

A failure is an exception, which run_script turns into an error; no stamp is written then.

What is made, under the prepared folder:

    poses/<pose id>.json       the pose as read from its frame of its animation file, independent of any avatar's
                               rest pose: the pelvis rotation, and every other bone's rotation against its parent
    avatars/<avatar id>.blend  the avatar in every pose, one static mesh each, in the frame people_library states
                               (origin on the ground under the pelvis, chest along +X, metres), textures packed
    avatars/<avatar id>.json   what was measured on those meshes, and what the file was made from

The avatars are Microsoft Rocketbox characters. As imported, an avatar faces -Y with its feet on z = 0 and
its pelvis over the origin, the texture paths of the FBX lead nowhere, and hair and eyelash opacity is wired
wrongly; so the images are found by file name in the avatar's download folder, made no larger than
`texture_px` on a side, and every material is rebuilt. An avatar whose entry gives `prepare.recolour` has the
hue band of one garment in one colour texture turned to another hue first (recolour).

A pose is taken from another skeleton than the avatar's, so it is turned, per avatar, until the mean heading
of pelvis and chest is the facing direction. A pose made for a thing in the hand (`prepare.prop`) is baked
twice: with empty hands, and with the thing as part of the mesh.

What is measured on each baked mesh, at the avatar's natural size, and noted per pose (the larger of the mesh
with and without its thing): its height; how far it gets from its upright axis (`reach`), and of that what is
behind (`back`), beside (`side`) and ahead (`front`), the last also for the feet alone (`front_feet`, up to
FEET_TOP_M) and for the rest of the body (`front_body`); and of the scene check's nine samples on that axis
the fewest it shows from any side, at each of SIZES (`axis_seen_at`). From the last the note has, per avatar,
`max_width_scale`: how far the avatar may be widened and still show one sample more than the check needs
(max_width_scale). The manifest states these for whoever casts people: an avatar's entry its
`max_width_scale`, a pose's entry under `measured` the furthest any avatar reaches in it and the fewest samples
any avatar shows in it. A person placed `height` tall with a `build` is the mesh times height / natural height
and the build across and in depth; keeping that factor within the avatar's `max_width_scale` keeps the sample
to spare on the tallest figure, at which it is read. What the shortest figure shows at each build is in the
poses' `axis_seen_at`. Upwards the samples are counted on the mesh times height / natural height. build.py
stretches a person upwards until the top of the pose is at their height instead, which is the same in an
upright pose and up to 4 % more where the head is bowed: the head then stands a little higher on the axis
than it was counted at, and never lower.

Each run decides what to make by comparing what a prepared file says it was made from (the checksums of its
downloads, its `prepare` parameters, the poses, and MADE_BY) with what the manifest says now, and a prepared
avatar's file with the checksum noted beside it. The parts that need no Blender (arguments, to_make,
made_from, recolour, noted, max_width_scale) can be loaded and called outside it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # Blender does not put a script's folder on sys.path

import people_library  # noqa: E402

try:
    import bpy
    from mathutils import Matrix, Quaternion, Vector
    from mathutils.bvhtree import BVHTree
except ImportError:                                # outside Blender, where only the plain parts are called
    bpy = None

MADE_BY = 2                                        # raise it when a change here makes other files from the same input

PELVIS, CHEST, HEAD = "Bip01 Pelvis", "Bip01 Spine2", "Bip01 Head"
FETCH = "python -m sgi.synth.realistic.assets fetch"

# The scene check samples a person's upright axis at these heights, takes a sample as seen when the depth there
# is within the tolerance of the axis, and a person as visible with three seen samples (sgi.gt.derive,
# sgi.config.Thresholds). Stated again here, since nothing of sgi is imported.
AXIS_HEIGHTS_M = (0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7)
AXIS_TOLERANCE_M = 0.3
AXIS_NEEDED = 3
AXIS_KEPT = AXIS_NEEDED + 1                        # what an avatar is to show within its max_width_scale: one to spare
# The axis is looked at from this many sides, every degree. A hanging arm hides the trunk from a few degrees off
# the body's side and a hand from a narrower angle still, which a round of sixteen sides steps over; a round of
# 120 still counted more than this one in 130 of 2 700 cases (30 avatars, 30 poses, three sizes).
AXIS_DIRECTIONS = 360
CAMERA_HEIGHT_M, CAMERA_PITCH_RAD = 1.5, math.radians(10.0)     # the robot's camera, as the scene format mounts it
CAMERA_DISTANCES_M = (1.8, 2.5, 4.0, 7.0, 10.0)    # how far from the person it is looked from
# (height, build): what the scenes scale a person to. The shortest and the tallest figure, each at the narrowest,
# the plain and the widest build; and the tallest, whom a build widens the most, at the builds between as well:
# an avatar's max_width_scale is read off those.
SIZES = ((1.58, 0.92), (1.58, 1.0), (1.58, 1.15),
         (1.84, 0.92), (1.84, 1.0), (1.84, 1.03), (1.84, 1.06), (1.84, 1.09), (1.84, 1.12), (1.84, 1.15))
FEET_TOP_M = 0.30                                  # at an avatar's natural size: lower than this are feet and ankles

# Where a thing sits in a hand, found by eye on the feasibility spike's close-ups (metres, degrees).
_PHONE = {"along": 0.045, "lift": 0.010, "grip_along": 0.03, "grip_lift": 0.035, "face_head_deg": 60.0}
_PHONE_AT_EAR = {"along": -0.005, "lift": 0.012, "face_head_deg": 0.0, "lie": "fingers", "shift_across": 0.045}
_CUP = {"out": 0.004, "along": 0.015}


# ----------------------------------------------------------------------------------- the parts that need no Blender
def arguments(argv: list[str]) -> argparse.Namespace:
    """What was passed after "--" on Blender's command line."""
    parser = argparse.ArgumentParser(prog="prepare_avatars.py", description="Bake the avatars of the manifest in "
                                     "every pose of the manifest.")
    for name in ("--manifest", "--downloads", "--prepared", "--stamp"):
        parser.add_argument(name, type=Path, required=True)
    parser.add_argument("--only", nargs="*", default=None, metavar="ID")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv[argv.index("--") + 1:] if "--" in argv else [])


def to_make(avatars: list[str], poses: list[str], intact: set[str], only: list[str] | None,
            force: bool) -> tuple[list[str], list[str]]:
    """The avatars and the poses a run has to make, each in the order given.

    Without `only` every id is named. A named id is made when it is not intact, or when `force` is given.
    A pose is baked into every avatar's file, so naming a pose names every avatar; and an avatar that is made
    needs every pose, so the poses that are not intact are made with it, named or not."""
    named_poses = [pose for pose in poses if only is None or pose in only]
    all_avatars = only is None or bool(named_poses)
    make_avatars = [avatar for avatar in avatars
                    if (all_avatars or avatar in only) and (force or avatar not in intact)]
    make_poses = [pose for pose in poses
                  if (force and pose in named_poses)
                  or (pose not in intact and (pose in named_poses or bool(make_avatars)))]
    return make_avatars, make_poses


def _digest(data) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def pose_made_from(pose: dict) -> str:
    """What decides a prepared pose: its file, by checksum, and the frame and the thing its entry names."""
    return _digest({"by": MADE_BY, "files": [source["sha256"] for source in pose["sources"]],
                    "prepare": pose["prepare"]})


def made_from(avatar: dict, poses: dict[str, str]) -> str:
    """What decides a prepared avatar: its files, by checksum, its `prepare` parameters, and the poses it is baked
    in (`poses`: pose id to pose_made_from). What is measured or judged afterwards (height, tags) is not part."""
    return _digest({"by": MADE_BY, "files": [source["sha256"] for source in avatar["sources"]],
                    "prepare": avatar["prepare"], "poses": poses})


def size_name(height: float, build: float) -> str:
    """How a size is named wherever the axis samples are counted per size: "<height>x<build>"."""
    return f"{height:.2f}x{build:.2f}"


def noted(rows: dict[str, dict]) -> dict:
    """What is noted of a pose, from what was measured on its mesh with empty hands (`rows[""]`) and, where the
    pose is made for a thing, on its mesh with the thing (`rows[<thing>]`): the height with empty hands; of every
    way of reaching the larger of the two, since a scene has to leave room for either; of the axis samples the
    fewer."""
    seen_at = {size: min(row["axis_seen_at"][size] for row in rows.values()) for size in rows[""]["axis_seen_at"]}
    furthest = {key: max(row[key] for row in rows.values())
                for key in ("reach", "back", "front", "front_feet", "front_body", "side")}
    return {"height": rows[""]["height"], "reach": furthest.pop("reach"),
            "reach_with": {held: row["reach"] for held, row in rows.items()}, **furthest,
            "axis_seen": min(seen_at.values()), "axis_seen_at": seen_at}


def max_width_scale(rows: dict[str, dict], natural: float) -> float:
    """The most an avatar may be widened and still show AXIS_KEPT of the axis samples from every side in every
    pose: the largest height / natural height x build, among the measured sizes of the tallest figure, up to
    which it does so at every build. `rows` is what was measured per pose, each with its "axis_seen_at".

    It is read at the tallest figure, on whom every sample lies on the body and whom a build widens the most; a
    step of the measured builds is as fine as it gets, and it errs to the narrow side. 0.0 for an avatar that
    does not show so much even at the narrowest build."""
    tallest = max(height for height, _build in SIZES)
    widest = 0.0
    for build in sorted(build for height, build in SIZES if height == tallest):
        if min(row["axis_seen_at"][size_name(tallest, build)] for row in rows.values()) < AXIS_KEPT:
            break
        widest = tallest / natural * build
    return widest


def _smooth(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def recolour(pixels: np.ndarray, hue_from_deg, hue_to_deg: float, min_saturation: float = 0.2, min_value: float = 0.1,
             saturation_gain: float = 1.0, value_gain: float = 1.0, edge_deg: float = 12.0) -> np.ndarray:
    """The picture with one band of hues turned to another hue: a garment dyed, its folds and shading kept.

    `pixels` is (..., 4) RGBA between 0 and 1 as the image file encodes it. A pixel is changed as far as its
    hue lies inside `hue_from_deg` = (low, high), its saturation above `min_saturation` and its value above
    `min_value`; the change fades in over `edge_deg` degrees and a tenth of saturation and value, so a blend
    between two cloths gets no hard line. Its new hue is `hue_to_deg`, its saturation and value those it had
    times the gains. Skin (hues about 10 to 40 degrees), greys and near-black are outside any band worth
    using, and a pixel outside the band comes back bit for bit, its opacity always."""
    low, high = (float(v) for v in hue_from_deg)
    picture = np.asarray(pixels, dtype=np.float32)
    rgb = picture[..., :3].astype(np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    value = rgb.max(axis=-1)
    spread = value - rgb.min(axis=-1)
    safe = np.where(spread > 0, spread, 1.0)
    hue = np.where(value == r, ((g - b) / safe) % 6.0, np.where(value == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0))
    hue = np.where(spread > 0, hue * 60.0, 0.0)
    saturation = np.where(value > 0, spread / np.where(value > 0, value, 1.0), 0.0)
    weight = (_smooth((hue - low) / edge_deg) * _smooth((high - hue) / edge_deg)
              * _smooth((saturation - min_saturation) / 0.1) * _smooth((value - min_value) / 0.1))
    # the dyed colour: HSV back to RGB at the one new hue
    new_s, new_v = np.clip(saturation * saturation_gain, 0.0, 1.0), np.clip(value * value_gain, 0.0, 1.0)
    sector = (float(hue_to_deg) % 360.0) / 60.0
    step, part = int(sector) % 6, sector - int(sector)
    p, q, t = new_v * (1.0 - new_s), new_v * (1.0 - new_s * part), new_v * (1.0 - new_s * (1.0 - part))
    dyed = np.stack(((new_v, q, p, p, t, new_v)[step], (t, new_v, new_v, q, p, p)[step],
                     (p, p, t, new_v, new_v, q)[step]), axis=-1)
    mixed = rgb + weight[..., None] * (dyed - rgb)
    out = picture.copy()
    out[..., :3] = np.where(weight[..., None] > 0, mixed, rgb).astype(np.float32)
    return out


def _read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: Path, data) -> None:
    """Whole or not at all: a reader never finds half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(partial, path)


def _entries(path: Path) -> list[dict]:
    return _read(path) if path.is_file() else []


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def downloaded(downloads: Path, entry: dict) -> dict[str, Path]:
    """The downloaded files of a manifest entry by name, each present and with the bytes the manifest states."""
    found = {}
    for source in entry["sources"]:
        name = source.get("file") or urllib.parse.unquote(source["url"].rsplit("/", 1)[1])
        path = Path(downloads) / entry["id"] / name
        if not path.is_file():
            raise FileNotFoundError(f"{entry['id']}: its download {path} is missing; get it with `{FETCH}`")
        if _sha256(path) != source["sha256"]:
            raise ValueError(f"{entry['id']}: {path} has not the SHA-256 the manifest states ({source['sha256']}); "
                             f"remove it and get it again with `{FETCH}`")
        found[name] = path
    return found


def _pose_is_intact(folder: Path, pose: dict) -> bool:
    try:
        return _read(folder / f"{pose['id']}.json")["made_from"] == pose_made_from(pose)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _avatar_is_intact(folder: Path, avatar: dict, poses: dict[str, str]) -> bool:
    """Whether the avatar's prepared file is the one its note describes, made from what the manifest says now."""
    try:
        note, blend = _read(folder / f"{avatar['id']}.json"), folder / f"{avatar['id']}.blend"
        return (note["made_from"] == made_from(avatar, poses) and blend.stat().st_size == note["blend"]["bytes"]
                and _sha256(blend) == note["blend"]["sha256"])
    except (OSError, ValueError, KeyError, TypeError):
        return False


# ------------------------------------------------------------------------------------------------------ poses
def _reset() -> "bpy.types.Scene":
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.context.scene


def _import_fbx(path: Path) -> list:
    before = set(bpy.data.objects)
    bpy.ops.wm.fbx_import(filepath=str(path))
    return [ob for ob in bpy.data.objects if ob not in before]


def _turn(matrix: "Matrix") -> "Quaternion":
    return matrix.to_3x3().normalized().to_quaternion()


def read_pose(arm) -> dict:
    """The armature's pose now, independent of its rest pose: the pelvis rotation in the world, the pelvis height
    against its rest height, and every other bone's rotation against its parent's."""
    bpy.context.view_layer.update()
    to_world = arm.matrix_world
    world = {bone.name: to_world @ bone.matrix for bone in arm.pose.bones}
    local = {bone.name: list(_turn(world[bone.parent.name].inverted() @ world[bone.name]))
             for bone in arm.pose.bones if bone.parent is not None}
    return {"pelvis_rot": list(_turn(world[PELVIS])), "pelvis_z": world[PELVIS].translation.z,
            "pelvis_rest_z": (to_world @ arm.data.bones[PELVIS].matrix_local).translation.z, "local": local}


def extract_poses(poses: list[dict], downloads: Path, folder: Path) -> None:
    """Read each pose from its frame of its animation file and write it to `folder`; a file is opened once."""
    by_file: dict[Path, list[dict]] = {}
    for pose in poses:
        files = downloaded(downloads, pose)
        name = pose["prepare"]["animation"]
        if name not in files:
            raise ValueError(f"{pose['id']}: prepare.animation {name!r} is none of its sources")
        by_file.setdefault(files[name], []).append(pose)
    for path, group in by_file.items():
        scene = _reset()
        arm = next((ob for ob in _import_fbx(path) if ob.type == "ARMATURE"), None)
        if arm is None or arm.animation_data is None or arm.animation_data.action is None:
            raise ValueError(f"{path} holds no animated skeleton")
        if PELVIS not in arm.pose.bones:
            raise ValueError(f"{path}: its skeleton has no bone {PELVIS!r}")
        first, last = (int(round(frame)) for frame in arm.animation_data.action.frame_range)
        for pose in group:
            frame = pose["prepare"]["frame"]
            if not first <= frame <= last:
                raise ValueError(f"{pose['id']}: frame {frame} is outside {first} to {last} of {path.name}")
            scene.frame_set(frame)
            _write(folder / f"{pose['id']}.json", {"made_from": pose_made_from(pose), "animation": path.name,
                                                   "frame": frame, **read_pose(arm)})
            print(f"prepare_avatars: pose {pose['id']} from {path.name} frame {frame}", flush=True)


def _rest_local(bone) -> "Matrix":
    return bone.parent.matrix_local.inverted() @ bone.matrix_local if bone.parent else bone.matrix_local.copy()


def apply_pose(arm, pose: dict | None, turn_back: float = 0.0) -> None:
    """Put the avatar in the pose, turned back about the vertical by `turn_back` radians. None is the pose it
    arrived in."""
    for bone in arm.pose.bones:
        bone.rotation_mode = "QUATERNION"
        bone.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()
    if pose is None:
        return
    bones = arm.data.bones
    for bone in arm.pose.bones:
        rest, turned = bones[bone.name], pose["local"].get(bone.name)
        if rest.parent is not None and turned is not None:
            bone.rotation_quaternion = _rest_local(rest).to_quaternion().inverted() @ Quaternion(turned)
    root, rest = arm.pose.bones[PELVIS], bones[PELVIS]
    to_world = arm.matrix_world
    world_turn = _turn(to_world)
    wanted = Quaternion((0.0, 0.0, 1.0), -turn_back) @ Quaternion(pose["pelvis_rot"])
    rest_turn = rest.matrix_local.to_quaternion()
    root.rotation_quaternion = rest_turn.inverted() @ (world_turn.inverted() @ wanted)
    # the pelvis as high, against its rest height, as the pose has it: bent knees lower it
    rest_z = (to_world @ rest.matrix_local).translation.z
    rise = Vector((0.0, 0.0, (pose["pelvis_z"] / pose["pelvis_rest_z"] - 1.0) * rest_z))
    root.location = rest_turn.inverted() @ (world_turn.inverted() @ rise / to_world.to_scale().x)
    bpy.context.view_layer.update()


def headings(arm) -> dict:
    """How far pelvis, chest and head have turned about the vertical from the pose of arrival (radians), and how
    far the head looks down."""
    to_world, out = arm.matrix_world, {}
    for key, name in (("pelvis", PELVIS), ("chest", CHEST), ("head", HEAD)):
        rest = _turn(to_world @ arm.data.bones[name].matrix_local)
        ahead = _turn(to_world @ arm.pose.bones[name].matrix) @ (rest.inverted() @ Vector((0.0, -1.0, 0.0)))
        out[key] = math.atan2(ahead.x, -ahead.y)
        if key == "head":
            out["head_down"] = math.atan2(-ahead.z, math.hypot(ahead.x, ahead.y))
    return out


def _joint(arm, name: str) -> "Vector":
    return (arm.matrix_world @ arm.pose.bones[name].matrix).translation.copy()


# -------------------------------------------------------------------------------------------- textures, materials
def _stem(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()


def _roles(material) -> dict:
    """Which texture the importer wired to what: {"color" | "normal" | "specular" | "alpha": the file's stem}."""
    roles = {}
    if not material.use_nodes:
        return roles
    for link in material.node_tree.links:
        node = link.from_node
        if node.type == "TEX_IMAGE" and node.image:
            stem, to = _stem(node.image.filepath or node.image.name), link.to_socket.name
            if link.to_node.type == "NORMAL_MAP":
                roles["normal"] = stem
            elif to == "Base Color":
                roles["color"] = stem
            elif to in ("Specular Tint", "Specular IOR Level", "Roughness"):
                roles["specular"] = stem
            elif to == "Alpha":
                roles["alpha"] = stem
    return roles


def _shrunk(path: Path, edge: int, folder: Path, change: dict | None) -> "bpy.types.Image":
    """The texture as a PNG in `folder`, no larger than `edge` on a side, recoloured first when `change` says how."""
    image = bpy.data.images.load(str(path), check_existing=False)
    width, height = image.size
    if not width or not height:
        raise ValueError(f"{path} does not load as an image")
    if change is not None:
        pixels = np.empty(width * height * 4, dtype=np.float32)
        image.pixels.foreach_get(pixels)
        image.pixels.foreach_set(recolour(pixels.reshape(-1, 4), **change).ravel())
        image.update()
    shrink = min(1.0, edge / max(width, height))
    if shrink < 1.0:
        image.scale(max(1, round(width * shrink)), max(1, round(height * shrink)))
    made = folder / f"{path.stem}.png"
    image.filepath_raw, image.file_format = str(made), "PNG"
    image.save()
    bpy.data.images.remove(image)
    return bpy.data.images.load(str(made), check_existing=False)


def rebuild_materials(meshes: list, files: dict[str, Path], edge: int, folder: Path, change: dict | None) -> dict:
    """Every material of the avatar anew, as a Principled BSDF over the avatar's own texture files.

    colour to Base Color; the normal map to Normal; the specular map, which is dark, to Roughness (rough where
    it is black, down to 0.42 where it reaches 0.22); a texture the importer wired to Alpha is hair, lashes or
    glasses: its colour is the colour and its own alpha the opacity. Returns {material: roles}."""
    by_stem = {path.stem.lower(): path for path in files.values() if path.suffix.lower() in (".tga", ".png", ".jpg")}
    loaded: dict[str, object] = {}
    recoloured = False

    def texture(stem: str, space: str):
        nonlocal recoloured
        if stem not in loaded:
            dye = None
            if change is not None and by_stem[stem].name == change["texture"]:
                dye, recoloured = {key: value for key, value in change.items() if key != "texture"}, True
            loaded[stem] = _shrunk(by_stem[stem], edge, folder, dye)
        loaded[stem].colorspace_settings.name = space
        return loaded[stem]

    done, report = set(), {}
    for mesh in meshes:
        for material in mesh.data.materials:
            if material is None or material.name in done:
                continue
            done.add(material.name)
            roles, base = _roles(material), material.name.lower()
            for role in ("color", "normal", "specular"):
                if roles.get(role) not in by_stem:       # not wired, or to a name with no file: go by the material's
                    guess = [stem for stem in sorted(by_stem) if stem.startswith(f"{base}_{role}")]
                    if guess and not ("alpha" in roles and role != "color"):
                        roles[role] = guess[0]
                    else:
                        roles.pop(role, None)
            if roles.get("alpha") not in by_stem:
                roles.pop("alpha", None)
            if "alpha" not in roles and "color" not in roles:
                raise ValueError(f"no colour texture among {sorted(by_stem)} for the material {material.name!r}")
            report[material.name] = dict(roles)
            material.use_nodes = True
            tree = material.node_tree
            tree.nodes.clear()
            out, shader = tree.nodes.new("ShaderNodeOutputMaterial"), tree.nodes.new("ShaderNodeBsdfPrincipled")
            tree.links.new(shader.outputs["BSDF"], out.inputs["Surface"])
            shader.inputs["Metallic"].default_value = 0.0
            shader.inputs["Roughness"].default_value = 0.6
            shader.inputs["Specular IOR Level"].default_value = 0.5

            def image_node(stem: str, space: str):
                node = tree.nodes.new("ShaderNodeTexImage")
                node.image = texture(stem, space)
                return node

            if "alpha" in roles:
                node = image_node(roles["alpha"], "sRGB")
                node.image.alpha_mode = "STRAIGHT"
                tree.links.new(node.outputs["Color"], shader.inputs["Base Color"])
                tree.links.new(node.outputs["Alpha"], shader.inputs["Alpha"])
                shader.inputs["Roughness"].default_value = 0.5
                shader.inputs["Specular IOR Level"].default_value = 0.3
            else:
                tree.links.new(image_node(roles["color"], "sRGB").outputs["Color"], shader.inputs["Base Color"])
                if "normal" in roles:
                    bump = tree.nodes.new("ShaderNodeNormalMap")
                    tree.links.new(image_node(roles["normal"], "Non-Color").outputs["Color"], bump.inputs["Color"])
                    tree.links.new(bump.outputs["Normal"], shader.inputs["Normal"])
                if "specular" in roles:
                    spread = tree.nodes.new("ShaderNodeMapRange")
                    spread.clamp = True
                    for name, value in (("From Min", 0.0), ("From Max", 0.22), ("To Min", 0.92), ("To Max", 0.42)):
                        spread.inputs[name].default_value = value
                    tree.links.new(image_node(roles["specular"], "Non-Color").outputs["Color"], spread.inputs["Value"])
                    tree.links.new(spread.outputs["Result"], shader.inputs["Roughness"])
            material.blend_method = "HASHED" if "alpha" in roles else "OPAQUE"
            material.use_backface_culling = False
            material[people_library.MATERIAL_KEY] = material.name
    if change is not None and not recoloured:
        raise ValueError(f"prepare.recolour names the texture {change['texture']!r}, which no material uses")
    for image in list(bpy.data.images):
        if image.users == 0:                       # the importer's own, which point at files that are not there
            bpy.data.images.remove(image)
    for image in loaded.values():
        image.pack()                               # the folder they were written to does not outlive the avatar
    return report


# --------------------------------------------------------------------------------------------- things in a hand
def _plain(materials: dict, name: str, color, roughness: float, emission: float = 0.0):
    """The plain material of that name, made once per avatar's file and kept in `materials`."""
    if name in materials:
        return materials[name]
    materials[name] = material = bpy.data.materials.new(name)
    material.use_nodes = True
    shader = material.node_tree.nodes["Principled BSDF"]
    shader.inputs["Base Color"].default_value = (*color, 1.0)
    shader.inputs["Roughness"].default_value = roughness
    if emission:
        shader.inputs["Emission Color"].default_value = (*color, 1.0)
        shader.inputs["Emission Strength"].default_value = emission
    material[people_library.MATERIAL_KEY] = name
    return material


def _join(name: str, meshes: list) -> "bpy.types.Mesh":
    """One mesh of several, each face keeping its material; the first mesh's layers (its UV map) are kept. The
    meshes given are used up."""
    import bmesh
    joined, materials = bmesh.new(), []
    for mesh in meshes:
        slots = []
        for material in mesh.materials:
            if material not in materials:
                materials.append(material)
            slots.append(materials.index(material))
        start = len(joined.faces)
        joined.from_mesh(mesh)
        joined.faces.ensure_lookup_table()
        for face in joined.faces[start:]:
            face.material_index = slots[face.material_index] if slots else 0
    out = bpy.data.meshes.new(name)
    joined.to_mesh(out)
    joined.free()
    for material in materials:
        out.materials.append(material)
    for mesh in meshes:
        bpy.data.meshes.remove(mesh)
    return out


def _slab(name: str, size) -> "bpy.types.Mesh":
    x, y, z = (side / 2 for side in size)
    mesh = bpy.data.meshes.new(name)
    corners = [(-x, -y, -z), (x, -y, -z), (x, y, -z), (-x, y, -z), (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)]
    mesh.from_pydata(corners, [], [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
    return mesh


def make_phone(materials: dict) -> "bpy.types.Mesh":
    """A dark slab 72 x 150 x 8 mm, its long side along +Y, with a lit screen on its +Z face; origin at its centre."""
    size = (0.072, 0.150, 0.008)
    body = _plain(materials, "phone_body", (0.02, 0.02, 0.025), 0.35)
    glow = _plain(materials, "phone_screen", (0.75, 0.85, 1.0), 0.2, emission=2.5)
    slab = _slab("phone", size)
    slab.materials.append(body)
    x, y, z = size[0] / 2 - 0.004, size[1] / 2 - 0.008, size[2] / 2 + 0.0004
    screen = bpy.data.meshes.new("phone_screen")
    screen.from_pydata([(-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)], [], [(0, 1, 2, 3)])
    screen.materials.append(glow)
    return _join("phone", [slab, screen])


def make_cup(materials: dict, segments: int = 24) -> "bpy.types.Mesh":
    """A paper cup 115 mm tall with a dark lid and a brown sleeve; its axis +Z, origin at half its height."""
    top, bottom, height = 0.041, 0.030, 0.115
    # (z, its radius or None for the taper's, the material of the wall above it)
    rings = [(-height / 2, bottom, 0), (-height * 0.18, None, 1), (height * 0.22, None, 0),
             (height / 2 - 0.012, None, 2), (height / 2, top + 0.002, 2)]
    verts, faces, slots = [], [], []
    for z, radius, _slot in rings:
        radius = radius if radius is not None else bottom + (top - bottom) * (z + height / 2) / height
        verts += [(radius * math.cos(2 * math.pi * i / segments), radius * math.sin(2 * math.pi * i / segments), z)
                  for i in range(segments)]
    for ring in range(len(rings) - 1):
        for i in range(segments):
            j, low, high = (i + 1) % segments, ring * segments, (ring + 1) * segments
            faces.append((low + i, low + j, high + j, high + i))
            slots.append(rings[ring][2])
    faces.append(tuple(reversed(range(segments))))
    slots.append(0)
    last = (len(rings) - 1) * segments
    faces.append(tuple(range(last, last + segments)))
    slots.append(2)
    mesh = bpy.data.meshes.new("cup")
    mesh.from_pydata(verts, [], faces)
    for name, color, roughness in (("cup_paper", (0.9, 0.88, 0.84), 0.6), ("cup_sleeve", (0.42, 0.27, 0.15), 0.8),
                                   ("cup_lid", (0.05, 0.05, 0.05), 0.4)):
        mesh.materials.append(_plain(materials, name, color, roughness))
    for face, slot in zip(mesh.polygons, slots):
        face.material_index, face.use_smooth = slot, True
    return mesh


def hand_frame(arm, side: str) -> dict:
    """The palm of a posed hand in the world, from joint positions alone: `fingers` from the wrist to the middle
    knuckle, `across` from the little finger's knuckle to the forefinger's, `normal` out of the palm."""
    wrist = _joint(arm, f"Bip01 {side} Hand")
    index, middle, ring, little = (_joint(arm, f"Bip01 {side} Finger{i}") for i in (1, 2, 3, 4))
    tip = _joint(arm, f"Bip01 {side} Finger22")
    fingers = (middle - wrist).normalized()
    across = index - little
    across = (across - across.dot(fingers) * fingers).normalized()
    normal = (across.cross(fingers) if side == "R" else fingers.cross(across)).normalized()
    curl = tip - middle
    return {"fingers": fingers, "across": across, "normal": normal,
            "centre": (wrist + index + middle + ring + little) / 5,
            "gripping": curl.length > 1e-6 and curl.normalized().dot(normal) > 0.8}


def _frame(origin, x, y, z) -> "Matrix":
    matrix = Matrix.Identity(4)
    for column, axis in enumerate((x, y, z)):
        matrix[0][column], matrix[1][column], matrix[2][column] = axis.x, axis.y, axis.z
    matrix.translation = origin
    return matrix


def phone_in_hand(arm, side: str, along: float, lift: float, grip_along: float = 0.03, grip_lift: float = 0.035,
                  face_head_deg: float = 60.0, lie: str = "auto", shift_across: float = 0.0) -> "Matrix":
    """Where a phone sits in a posed hand, screen out of the palm. On an open hand it lies on the palm, its long
    side along the fingers; in a gripping hand, along the knuckles in front of the fingers. It is then turned
    about its long side, by `face_head_deg` at most, so that the screen looks at the head."""
    hand = hand_frame(arm, side)
    grip = hand["gripping"] if lie == "auto" else lie == "across"
    z = hand["normal"].copy()
    long_side = hand["across"] if grip else hand["fingers"]
    y = (long_side - long_side.dot(z) * z).normalized()
    origin = hand["centre"] + y * (grip_along if grip else along) + z * ((grip_lift if grip else lift) + 0.004)
    origin = origin + hand["across"] * shift_across
    if face_head_deg:
        to_eyes = _joint(arm, HEAD) + Vector((0.0, 0.0, 0.08)) - origin
        to_eyes = to_eyes - to_eyes.dot(y) * y
        if to_eyes.length > 1e-4:
            to_eyes.normalize()
            angle = min(z.angle(to_eyes), math.radians(face_head_deg))
            z = Matrix.Rotation(angle if y.dot(z.cross(to_eyes)) > 0 else -angle, 3, y) @ z
    return _frame(origin, y.cross(z).normalized(), y, z)


def cup_in_hand(arm, side: str, out: float, along: float, radius: float = 0.036) -> "Matrix":
    """Where a cup sits in a gripping hand: upright, its side against the palm."""
    hand, up = hand_frame(arm, side), Vector((0.0, 0.0, 1.0))
    outward = hand["normal"] - hand["normal"].dot(up) * up
    outward = outward.normalized() if outward.length > 1e-4 else hand["normal"]
    origin = hand["centre"] + outward * (radius + out) + hand["fingers"] * along
    return _frame(origin, outward, up.cross(outward).normalized(), up)


def thing_in_hand(arm, pose: dict, materials: dict) -> "bpy.types.Mesh | None":
    """The thing the pose's hand is posed for, as a mesh in the world where the hand is now; None for empty hands."""
    kind, side = pose.get("prop"), pose.get("hand") or "R"
    if kind is None:
        return None
    if kind == "phone":
        how = _PHONE_AT_EAR if pose.get("hold") == "ear" else _PHONE
        mesh, where = make_phone(materials), phone_in_hand(arm, side, **how)
    elif kind == "cup":
        mesh, where = make_cup(materials), cup_in_hand(arm, side, **_CUP)
    else:
        raise ValueError(f"prepare.prop is {kind!r}: expected one of {people_library.THINGS} or null")
    mesh.transform(where)
    return mesh


# ------------------------------------------------------------------------------------------ baking and measuring
def _posed(ob) -> "bpy.types.Mesh":
    """The object's mesh as its armature deforms it now, in world coordinates."""
    graph = bpy.context.evaluated_depsgraph_get()
    mesh = bpy.data.meshes.new_from_object(ob.evaluated_get(graph), preserve_all_data_layers=True, depsgraph=graph)
    mesh.transform(ob.matrix_world)
    return mesh


def _points(mesh) -> np.ndarray:
    points = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", points)
    return points.reshape(-1, 3)


def axis_views(natural: float) -> dict[str, list]:
    """The scene check's lines of sight to the axis of an avatar `natural` metres tall, at each of SIZES: per size
    one view for every one of AXIS_DIRECTIONS sides and every one of CAMERA_DISTANCES_M, for the robot's camera
    (CAMERA_HEIGHT_M up, pitched CAMERA_PITCH_RAD down) looking at the person.

    Everything is in the mesh's own measure, the mesh being the person at their natural size: placed, it is
    `tall` times as high and `wide` times as broad and deep. A view is the eye and, per sample height, the ray's
    direction, its length to the sample, and how much depth along the camera's axis in the world one metre of
    that ray is worth in the mesh's measure. They are made once per avatar: every pose is looked at alike."""
    views = {}
    for height, build in SIZES:
        tall, wide = height / natural, height / natural * build
        looks = views[size_name(height, build)] = []
        for step in range(AXIS_DIRECTIONS):
            angle = 2 * math.pi * step / AXIS_DIRECTIONS
            x, y = math.cos(angle), math.sin(angle)
            ahead = (-x * math.cos(CAMERA_PITCH_RAD), -y * math.cos(CAMERA_PITCH_RAD), -math.sin(CAMERA_PITCH_RAD))
            for distance in CAMERA_DISTANCES_M:
                eye = Vector((x * distance / wide, y * distance / wide, CAMERA_HEIGHT_M / tall))
                rays = []
                for z in AXIS_HEIGHTS_M:
                    ray = Vector((0.0, 0.0, z / tall)) - eye
                    length = ray.length
                    aim = ray / length
                    depth = aim.x * wide * ahead[0] + aim.y * wide * ahead[1] + aim.z * tall * ahead[2]
                    rays.append((aim, length, depth))
                looks.append((eye, rays))
    return views


def axis_seen(mesh, views: dict[str, list]) -> dict[str, int]:
    """How many of the scene check's nine axis samples this body shows from its worst side, at each size the
    scenes may give it: {"<height>x<build>": the fewest over the views of that size} (axis_views).

    As the scene check does it: a sample is shown when the ray from the camera to it first meets the body at a
    depth, along the camera's axis, within the tolerance of the sample's own. The check needs AXIS_NEEDED shown
    samples; a body that shows no more than that from some side is seen from there with nothing to spare."""
    tree = BVHTree.FromPolygons([tuple(vertex.co) for vertex in mesh.vertices],
                                [tuple(face.vertices) for face in mesh.polygons])
    found = {}
    for size, looks in views.items():
        fewest = len(AXIS_HEIGHTS_M)
        for eye, rays in looks:
            seen = 0
            for aim, length, depth in rays:
                met = tree.ray_cast(eye, aim, length + 2.0)[3]         # how far along the ray the body is met
                if met is not None and abs((met - length) * depth) <= AXIS_TOLERANCE_M:
                    seen += 1
            if seen < fewest:
                fewest = seen
        found[size] = fewest
    return found


def _measure(mesh, views: dict[str, list]) -> dict:
    """What the scenes need to know of a baked mesh. Ahead of the axis, what the feet reach and what the rest of
    the body reaches are told apart: knees, belly, hands and whatever they hold are what a counter before the
    person has to leave room for at their height, the toes only at the floor."""
    points = _points(mesh)
    feet = points[:, 2] <= FEET_TOP_M
    return {"height": float(points[:, 2].max()), "reach": float(np.hypot(points[:, 0], points[:, 1]).max()),
            "back": float(-points[:, 0].min()), "front": float(points[:, 0].max()),
            "front_feet": float(points[feet, 0].max()), "front_body": float(points[~feet, 0].max()),
            "side": float(np.abs(points[:, 1]).max()), "axis_seen_at": axis_seen(mesh, views)}


def bake(arm, meshes: list, pose_id: str, pose: dict, avatar: dict, natural: float, materials: dict,
         views: dict[str, list]) -> dict:
    """The avatar as it is posed now, as static mesh objects of the scene in the frame the scenes place people by;
    with the pose's thing a second object. Returns what was measured over both (`views`: axis_views)."""
    bpy.context.view_layer.update()
    body = [_posed(mesh) for mesh in meshes]
    body = body[0] if len(body) == 1 else _join(pose_id, body)
    thing = thing_in_hand(arm, pose, materials)
    pelvis = _joint(arm, PELVIS)
    low = float(_points(body)[:, 2].min())
    # the world as imported (facing -Y) to the frame of a placed person (facing +X, the ground point at the origin)
    to_frame = Matrix.Rotation(math.pi / 2, 4, "Z") @ Matrix.Translation((-pelvis.x, -pelvis.y, -low))
    body.transform(to_frame)
    made = {None: body}
    if thing is not None:
        thing.transform(to_frame)
        made[pose["prop"]] = _join(people_library.object_name(pose_id, pose["prop"]), [body.copy(), thing])
    rows = {}
    for prop, mesh in made.items():
        mesh.name = people_library.object_name(pose_id, prop)
        rows[prop or ""] = row = _measure(mesh, views)
        ob = bpy.data.objects.new(mesh.name, mesh)
        bpy.context.scene.collection.objects.link(ob)
        ob["avatar"], ob["pose"], ob["prop"] = avatar["id"], pose_id, prop or ""
        ob["natural_height"], ob["height"], ob["reach"] = natural, row["height"], row["reach"]
    turned, told = headings(arm), noted(rows)
    return {"height": told.pop("height"), "pelvis_height": float(pelvis.z - low), **told,
            "turn_deg": {key: round(math.degrees(turned[key]), 2) for key in ("pelvis", "chest", "head")},
            "head_down_deg": round(math.degrees(turned["head_down"]), 2)}


def make_avatar(avatar: dict, poses: list[tuple[dict, dict]], downloads: Path, folder: Path,
                marks: dict[str, str]) -> None:
    """Import the avatar, rebuild its materials, bake it in every pose and save its file and its note. `poses` is
    each pose's entry with the pose as read; `marks` is what each was made from, for the note."""
    started = time.time()
    files, prepare = downloaded(downloads, avatar), avatar["prepare"]
    if prepare["fbx"] not in files:
        raise ValueError(f"{avatar['id']}: prepare.fbx {prepare['fbx']!r} is none of its sources")
    _reset()
    new = _import_fbx(files[prepare["fbx"]])
    arm = next((ob for ob in new if ob.type == "ARMATURE"), None)
    meshes = [ob for ob in new if ob.type == "MESH"]
    if arm is None or not meshes or PELVIS not in arm.pose.bones:
        raise ValueError(f"{files[prepare['fbx']]} holds no skinned avatar with a bone {PELVIS!r}")
    for ob in new:
        if ob.type not in ("ARMATURE", "MESH"):
            bpy.data.objects.remove(ob)
    if arm.animation_data:
        arm.animation_data_clear()                 # the one-frame take the file comes with
    with tempfile.TemporaryDirectory(prefix="sgi_avatar_") as scratch:
        roles = rebuild_materials(meshes, files, int(prepare["texture_px"]), Path(scratch), prepare.get("recolour"))
    apply_pose(arm, None)
    standing = np.concatenate([_points(mesh) for mesh in (_posed(ob) for ob in meshes)])
    natural = float(standing[:, 2].max() - standing[:, 2].min())
    rows, things, views = {}, {}, axis_views(natural)
    for entry, pose in poses:
        apply_pose(arm, pose)
        raw = headings(arm)                        # the animation's figure faces where it likes: the mean of two angles
        apply_pose(arm, pose, turn_back=math.atan2(math.sin(raw["pelvis"]) + math.sin(raw["chest"]),
                                                   math.cos(raw["pelvis"]) + math.cos(raw["chest"])))
        rows[entry["id"]] = bake(arm, meshes, entry["id"], {**pose, **entry["prepare"]}, avatar, natural, things,
                                 views)
    widest = max_width_scale(rows, natural)
    for ob in [arm, *meshes]:
        bpy.data.objects.remove(ob)
    folder.mkdir(parents=True, exist_ok=True)
    blend, making = folder / f"{avatar['id']}.blend", folder / f"{avatar['id']}.making.blend"
    note = folder / f"{avatar['id']}.json"
    note.unlink(missing_ok=True)                   # until the new file is whole, there is no prepared avatar
    bpy.ops.wm.save_as_mainfile(filepath=str(making), compress=True, check_existing=False, copy=True)
    os.replace(making, blend)
    _write(note, {"avatar": avatar["id"], "made_from": made_from(avatar, marks),
                  "blend": {"bytes": blend.stat().st_size, "sha256": _sha256(blend)}, "natural_height": natural,
                  "max_width_scale": widest, "materials": roles, "texture_px": int(prepare["texture_px"]),
                  "poses": rows})
    print(f"prepare_avatars: avatar {avatar['id']}: {natural:.3f} m, {len(rows)} poses, widened {widest:.3f} at "
          f"the most, {blend.stat().st_size / 1e6:.1f} MB, {time.time() - started:.1f} s", flush=True)


def main(argv: list[str]) -> None:
    given = arguments(argv)
    avatars, poses = _entries(given.manifest / "avatars.json"), _entries(given.manifest / "poses.json")
    both = {entry["id"] for entry in avatars} & {entry["id"] for entry in poses}
    if both:
        raise ValueError(f"an avatar and a pose share the id {sorted(both)[0]!r}; --only could not tell them apart")
    pose_folder = given.prepared / people_library.POSES
    avatar_folder = given.prepared / people_library.AVATARS
    marks = {pose["id"]: pose_made_from(pose) for pose in poses}

    def intact() -> tuple[list[str], list[str]]:
        return ([avatar["id"] for avatar in avatars if _avatar_is_intact(avatar_folder, avatar, marks)],
                [pose["id"] for pose in poses if _pose_is_intact(pose_folder, pose)])

    whole_avatars, whole_poses = intact()
    make_avatars, make_poses = to_make([avatar["id"] for avatar in avatars], [pose["id"] for pose in poses],
                                       set(whole_avatars) | set(whole_poses), given.only, given.force)
    print(f"prepare_avatars: {len(make_poses)} of {len(poses)} poses and {len(make_avatars)} of {len(avatars)} "
          "avatars to make", flush=True)
    extract_poses([pose for pose in poses if pose["id"] in make_poses], given.downloads, pose_folder)
    if make_avatars:
        read = [(pose, _read(pose_folder / f"{pose['id']}.json")) for pose in poses]
        for avatar in avatars:
            if avatar["id"] in make_avatars:
                make_avatar(avatar, read, given.downloads, avatar_folder, marks)
    whole_avatars, whole_poses = intact()
    _write(given.stamp, {"avatars": whole_avatars, "poses": whole_poses})
    print(f"prepare_avatars: {len(whole_avatars)} of {len(avatars)} avatars and {len(whole_poses)} of {len(poses)} "
          "poses are prepared", flush=True)


if __name__ == "__main__":
    main(sys.argv)
