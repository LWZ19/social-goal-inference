"""The prepared people of the realistic set, inside Blender: a person of the library as one static mesh.

prepare_avatars.py bakes every avatar of the manifest in every pose and saves one file per avatar,
`<prepared>/avatars/<avatar id>.blend`, with its textures packed. Each object in such a file is one person in
one pose, named after the pose; a pose made for a thing in the hand (a phone, a cup) is there a second time
with the thing, named `<pose>+<thing>` (object_name). Every such mesh is in the frame the scenes place people by:

    metres, +Z up, the feet on z = 0
    the origin on the ground under the pelvis joint
    the chest facing +X

so a person is placed by a location and a rotation about Z, and sized by a scale: across and in depth by the
height that is wanted over the avatar's natural height, times the build, which is what every measurement of a
pose is stated for; and upward so that the top of the mesh, which a bowed head leaves some centimetres under
the natural height, comes to the height that is wanted (build.py's add_person does both). Each object says
what it is in custom properties:
"avatar", "pose", "prop" ("" for empty hands), "natural_height" (the avatar's own height, standing upright,
as the manifest states it), "height" (the top of this mesh, a little lower where the head is bowed) and
"reach" (the furthest this mesh gets from the upright axis through its origin).

add_person appends on demand. Every call gives a new object with a mesh of its own, so the caller may change
it; the materials and images are brought once per avatar, however often and in however many poses the avatar
is placed. Made for Blender; outside it (the tests load the preparing script to call its plain parts) the
module imports, and only the names can be used.
"""
from __future__ import annotations

from pathlib import Path

try:
    import bpy
except ImportError:                                # outside Blender: the names below are all there is to use
    bpy = None

AVATARS = "avatars"                                # under the prepared folder: <avatar id>.blend, <avatar id>.json
POSES = "poses"                                    # under the prepared folder: <pose id>.json, the pose as read
THINGS = ("phone", "cup")                          # what a hand can hold
MATERIAL_KEY = "prepared_name"                     # on each prepared material: its name, whatever Blender renames it to

_prepared: Path | None = None
_materials: dict[str, dict] = {}                   # avatar id -> the materials it has in this scene, by MATERIAL_KEY


def object_name(pose: str, prop: str | None = None) -> str:
    """The name, in an avatar's prepared file, of the avatar in `pose` holding `prop`."""
    return pose if prop is None else f"{pose}+{prop}"


def avatar_file(prepared: Path | str, avatar: str) -> Path:
    return Path(prepared) / AVATARS / f"{avatar}.blend"


def open_people(prepared: str) -> None:
    """Take the people from this prepared folder from now on. To be called once per scene, before add_person:
    it also forgets the materials of an earlier scene of the same process."""
    global _prepared
    _prepared = Path(prepared)
    _materials.clear()


def _alive(data) -> bool:
    """Whether a remembered datablock is still in the file; one of a scene since thrown away is not."""
    try:
        return bool(data.name)
    except ReferenceError:
        return False


def _share_materials(avatar: str, ob) -> None:
    """Give the object the materials this avatar already has in the scene, and drop the copies it brought.

    Appending an object brings its materials and their packed images again each time. A person has six or
    seven 1024 px images, so an avatar placed in two poses would hold them twice."""
    known = _materials.setdefault(avatar, {})
    for slot, material in enumerate(ob.data.materials):
        if material is None:
            continue
        name = material.get(MATERIAL_KEY, material.name)
        first = known.get(name)
        if first is None or not _alive(first):
            known[name] = material
        elif first != material:
            ob.data.materials[slot] = first
            if material.users == 0:
                images = [node.image for node in material.node_tree.nodes if node.type == "TEX_IMAGE" and node.image]
                bpy.data.materials.remove(material)
                for image in images:
                    if image.users == 0:
                        bpy.data.images.remove(image)


def add_person(avatar: str, pose: str, prop: str | None = None) -> "bpy.types.Object":
    """The avatar in the pose as a new object of the current scene: a static mesh at its natural height, the
    ground point under its pelvis at the origin, chest facing +X.

    `prop` is None, "phone" or "cup", and puts that thing in the hand of a pose that was made for it; such a
    pose may also be asked for with empty hands. ValueError for an avatar or a pose that is not prepared,
    for a thing that is none of these, and for a pose that does not hold the thing. RuntimeError before
    open_people."""
    if _prepared is None:
        raise RuntimeError("open_people(prepared) has to be called before add_person")
    if prop is not None and prop not in THINGS:
        raise ValueError(f"a person can hold {' or '.join(THINGS)}, not {prop!r}")
    name, blend = object_name(pose, prop), avatar_file(_prepared, avatar)
    if not blend.is_file():
        raise ValueError(f"no prepared avatar {avatar!r}: there is no {blend}")
    with bpy.data.libraries.load(str(blend), link=False) as (there, wanted):
        if name not in there.objects:
            if pose not in there.objects:
                raise ValueError(f"the avatar {avatar!r} is not prepared in a pose {pose!r}")
            held = [other.split("+", 1)[1] for other in there.objects if other.startswith(pose + "+")]
            raise ValueError(f"the pose {pose!r} is not one that holds a {prop}; it holds "
                             f"{' or '.join(held) if held else 'nothing'}")
        wanted.objects = [name]
    ob = wanted.objects[0]
    _share_materials(avatar, ob)
    ob.name = f"{avatar}:{name}"
    bpy.context.scene.collection.objects.link(ob)
    ob.location, ob.rotation_euler, ob.scale = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    bpy.context.view_layer.update()                # matrix_world of a freshly appended object is stale until this runs
    return ob
