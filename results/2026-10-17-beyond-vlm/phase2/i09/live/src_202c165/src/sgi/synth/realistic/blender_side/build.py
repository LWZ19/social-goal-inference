"""Building the scene of a job: the room, the parts, the models, the people and the light. Runs inside Blender.

What is built is the scene description as sgi.synth.realistic.spec writes it to JSON, in the world's own
coordinates: metres, Z up, the floor on z = 0. Nothing here imports sgi; the few tables that both sides need
are copied (FACES, the panel of a ceiling fitting, the colours of the plain materials).

The room. The floor is one face on z = 0 and the ceiling one flat mesh at the room's height. Each wall is one
flat mesh exactly on its edge of the floor's outline, painted in the wall's colour above the dado height and in
the dado's below it. Where two walls meet in a corner that points out of the room, and at the floor and the
ceiling, a wall runs on by a millimetre behind its neighbour, so that no ray slips between two meshes. What
stands off a wall is no deeper than 2 cm: a skirting board, and a rail where the two colours meet. Every wall,
the floor and the ceiling are single objects, because a box-projected texture jumps where two objects join.

Ceiling fittings stand where the placeholder's Room draws its light panels: one in the middle of every square
of `lights` metres, counted from the first corner along the first edge. A fitting is a rectangle of the
ceiling's own mesh that glows towards the camera only, with a trim round it, and an area light just under it
that the camera does not see. So the ceiling stays one plane in the depth, and no sample is spent on a glowing
box. A room without fittings is lit by the same lights, and shows none.

Openings. A window is a hole in its wall's mesh, lined by a reveal, with a frame, a mullion and a pane of
glass set back in it and the world seen through it; all of that lies behind the wall's face, and only a sill
stands 4 cm into the room. The daylight it sends in is an area light just inside the wall, which lights the room
and does not show: not to the camera, not in a mirror, not through glass. A door is a leaf in a frame on the
whole wall, with a handle, 4.5 cm deep in all.

Parts. A box, a cylinder or a sphere, its size baked into its mesh (a box-projected material needs an object
scale of 1), its edges rounded off by `bevel`. Its material is one of the asset library's by name, with the
part's `tint` where it has one (the colour that a material without one of its own comes out in), or "paint":
the solid's own colour, with its `top` colour on the upper face. A box's label, panel and band are not drawn:
lettering comes as signs, each an image on a rectangle 1 mm proud of one face of a box. A part with `glass`
set, or of the material "glass", is a pane: glass to the camera, clear to shadow rays.

The asset library (library.py beside this file, another task's) is asked for only when a job names its
prepared folder. Without one, a named material is a plain coat of a fitting colour, the world is a plain sky,
and a model or a person is an error.

Every object made for a part, a model or a person carries that item's object group in the custom property
GROUP, the objects under a model's parent too; what belongs to the room carries none.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Matrix

GROUP = "sgi_group"                                # the custom property that holds an object's group
PAINT, GLASS = "paint", "glass"
# A box's faces, as sgi.synth.render._FACES has them: the axis of the face's normal and its sign, the axis the
# lettering runs along and its sign, the axis that is up and its sign. A box's +x is its front, +y its left.
FACES = {"front": (0, 1.0, 1, 1.0, 2, 1.0), "back": (0, -1.0, 1, -1.0, 2, 1.0), "left": (1, 1.0, 0, -1.0, 2, 1.0),
         "right": (1, -1.0, 0, 1.0, 2, 1.0), "top": (2, 1.0, 1, 1.0, 0, -1.0)}
SIGN_PROUD_M = 0.001

PANEL_M = (1.2, 0.3)                               # a ceiling fitting, as sgi.synth.render._LIGHT_PANEL_M
PANEL_TRIM_M = 0.02                                # the frame round it, in the ceiling's own plane
LIGHT_DROP_M = 0.002                               # its light hangs this far under the ceiling; lights write no depth
UNLIT_SPACING_M = 2.4                              # how far apart the lights of a room without fittings are
SKIRTING_M = (0.12, 0.014)                         # height and depth of the skirting board
SKIRTING_COLOUR = (96, 86, 76)
RAIL_M = (0.05, 0.018)                             # and of the rail at the dado height
RAIL_DARKER = 0.8                                  # the rail is the dado's colour times this, as the placeholder's
SEAL_M = 0.001                                     # how far a wall runs on behind the floor, the ceiling and a corner
ON_A_WALL_M = 0.3                                  # an opening further than this from every wall is on none
WALL_THICK_M = 0.2                                 # as a window's reveal shows it
FRAME_COLOUR = (236, 236, 232)                     # window frames and door frames: white gloss
GLASS_SHEEN = 0.1                                  # how rough what a pane mirrors is; what is behind it stays sharp
CONTACT_GAIN = 3.0                                 # see contact_shaded: floor and walls alike get the full gain
EXPOSURE = -0.5                                    # stops, for every scene: fittings of a real wattage, one exposure

# The presets of a description's `lighting`.
#   fitting        the power of the ceiling fittings in watts per square metre of floor; the fittings of a room
#                  share what its floor asks for, so that rooms with fittings closer together or further apart
#                  are equally bright. 9 W/m2 is a 52 W panel every 2.4 m.
#   bounces        the diffuse bounces of light (Blender's factory default is 4; 3 changes the picture hardly at
#                  all). The shade round a person, a table, a wall's foot and a corner comes from `contact_shaded`,
#                  and the pools of light under the fittings from `spread` and `lamp_scale`, not from this.
#   spread         how widely a fitting shines, in degrees (180 is all round below the ceiling)
#   lamp_scale     the area light is this share of the glowing panel's size: a smaller source, a crisper shadow
#   floor_shade, wall_shade   contact_shaded: how dark the floor and the walls get next to what stands on or
#                  against them (a share), and within what distance in metres (`_m`)
#   glow           how bright a fitting itself looks
#   daylight       the power of the light a window sends in, in watts per square metre of window
#   sky            the strength of what a window shows while the job names no HDRI: a sky of the colours
#                  `horizon` and `zenith` over ground of the colour `ground`
#   hdri           the strength a named HDRI is shown at
# Neither the sky nor an HDRI lights the room.
# The generator draws with `ceiling` alone (dress.py's _LIGHTING). `daylight` and `evening` were only adjusted to
# the tests of this module, with no picture looked at.
_DAY_SKY = dict(horizon=(0.62, 0.76, 1.0), zenith=(0.2, 0.42, 0.95), ground=(0.2, 0.22, 0.19))
_DUSK_SKY = dict(horizon=(0.75, 0.45, 0.35), zenith=(0.08, 0.1, 0.3), ground=(0.05, 0.05, 0.06))
_FITTED = dict(bounces=3, spread=165.0, lamp_scale=0.5, floor_shade_m=0.5, floor_shade=0.6,
               wall_shade_m=0.4, wall_shade=0.12)
LIGHTING = {
    "ceiling": dict(fitting=9.0, fitting_tone=(1.0, 0.97, 0.92), glow=7.0, daylight=60.0,
                    daylight_tone=(0.86, 0.93, 1.0), sky=6.0, hdri=6.0, **_FITTED, **_DAY_SKY),
    "daylight": dict(fitting=6.0, fitting_tone=(1.0, 0.98, 0.95), glow=6.0, daylight=50.0,
                     daylight_tone=(0.9, 0.95, 1.0), sky=9.0, hdri=6.0, **_FITTED, **_DAY_SKY),
    "evening": dict(fitting=9.0, fitting_tone=(1.0, 0.88, 0.72), glow=7.0, daylight=0.0,
                    daylight_tone=(1.0, 1.0, 1.0), sky=1.0, hdri=0.6, **_FITTED, **_DUSK_SKY),
}

# What a named material is without the asset library: a plain coat. sRGB colour, roughness, metallic.
PLAIN = {
    "floor_tile": ((200, 192, 176), 0.17, 0.0), "floor_terrazzo": ((186, 182, 172), 0.17, 0.0),
    "floor_laminate": ((176, 140, 100), 0.26, 0.0), "ceiling_tile": ((243, 242, 236), 0.9, 0.0),
    "wall_paint": ((233, 227, 213), 0.8, 0.0), "wood": ((168, 124, 82), 0.5, 0.0),
    "wood_dark": ((96, 66, 44), 0.5, 0.0), "laminate_white": ((236, 236, 232), 0.35, 0.0),
    "steel": ((170, 174, 180), 0.35, 1.0), "painted_metal": ((92, 104, 120), 0.4, 0.0),
    "plastic_dark": ((48, 50, 54), 0.4, 0.0), "plastic_light": ((214, 214, 210), 0.4, 0.0),
    "fabric": ((88, 104, 132), 0.9, 0.0), "stone": ((150, 148, 142), 0.5, 0.0), "cork": ((186, 148, 104), 0.9, 0.0),
    "paper": ((244, 242, 232), 0.8, 0.0), "cardboard": ((178, 142, 98), 0.8, 0.0),
    "ceramic": ((240, 238, 232), 0.15, 0.0), "screen": ((28, 34, 44), 0.15, 0.0)}

_BOX_FACES = ((0, 3, 2, 1), (4, 5, 6, 7), (1, 2, 6, 5), (3, 0, 4, 7), (2, 3, 7, 6), (0, 1, 5, 4))   # bottom, top, sides
_TOP_FACE = 1

_materials: dict = {}                              # the materials of the scene being built, each made once
_library = None                                    # the asset library's module, once a job has opened it


# ------------------------------------------------------------------------------------------------ scene, materials

def new_scene():
    """An empty scene with factory settings, and nothing remembered of an earlier one."""
    global _library
    bpy.ops.wm.read_factory_settings(use_empty=True)
    _materials.clear()
    _library = None
    return bpy.context.scene


def open_assets(prepared: str | None) -> None:
    """Take models, people, materials and HDRIs from this prepared folder; with None, do without them."""
    global _library
    _library = None
    if prepared is not None:
        import library                             # beside this file; loaded only for a job that has prepared assets
        library.open_library(prepared)
        _library = library


def _assets(wanted: str):
    if _library is None:
        raise RuntimeError(f"the job places {wanted}, which comes from the prepared assets, and names no prepared "
                           "folder")
    return _library


def linear(rgb255) -> tuple[float, float, float]:
    """An sRGB 0..255 colour as scene-linear Rec.709."""
    out = []
    for value in rgb255:
        c = value / 255.0
        out.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return tuple(out)


def paint(rgb255, rough: float = 0.45, bump_m: float = 0.0003, metallic: float = 0.0):
    """A painted, laminated or moulded surface of one colour. Its roughness wanders a little and it has a fine
    relief, which keep a face from reading as a flat fill; both follow the object's own coordinates."""
    key = ("paint", tuple(int(v) for v in rgb255), round(rough, 3), round(bump_m, 5), round(metallic, 3))
    if key not in _materials:
        material = bpy.data.materials.new("paint_%02x%02x%02x" % key[1])
        material.use_nodes = True
        nodes, links = material.node_tree.nodes, material.node_tree.links
        shader = nodes["Principled BSDF"]
        shader.inputs["Base Color"].default_value = (*linear(key[1]), 1.0)
        shader.inputs["Metallic"].default_value = metallic
        coords = nodes.new("ShaderNodeTexCoord")

        def noise(scale: float, detail: float):
            node = nodes.new("ShaderNodeTexNoise")
            node.inputs["Scale"].default_value, node.inputs["Detail"].default_value = scale, detail
            links.new(coords.outputs["Object"], node.inputs["Vector"])
            return node.outputs["Fac"]

        wander = nodes.new("ShaderNodeMapRange")
        wander.inputs["From Min"].default_value, wander.inputs["From Max"].default_value = 0.3, 0.7
        wander.inputs["To Min"].default_value = max(rough - 0.07, 0.02)
        wander.inputs["To Max"].default_value = min(rough + 0.07, 1.0)
        links.new(noise(9.0, 4.0), wander.inputs["Value"])
        links.new(wander.outputs["Result"], shader.inputs["Roughness"])
        if bump_m > 0:
            relief = nodes.new("ShaderNodeBump")
            relief.inputs["Distance"].default_value = bump_m
            links.new(noise(54.0, 2.0), relief.inputs["Height"])
            links.new(relief.outputs["Normal"], shader.inputs["Normal"])
        _materials[key] = material
    return _materials[key]


def named(name: str, tint=None):
    """A material of the asset library by its name, its base colour multiplied by `tint` (sRGB 0..255) where one
    is given. Without the library: a plain coat of the material's usual colour, or of the tint."""
    if name == GLASS:
        return glass()
    tint = None if tint is None else tuple(int(v) for v in tint)
    if _library is not None:
        key = ("library", name, tint)
        if key not in _materials:
            _materials[key] = _library.material(name, tint)
        return _materials[key]
    if name not in PLAIN:
        raise ValueError(f"unknown material {name!r}: expected {PAINT!r}, {GLASS!r} or one of {', '.join(PLAIN)}")
    colour, rough, metallic = PLAIN[name]
    return paint(colour if tint is None else tint, rough, metallic=metallic)


def contact_shaded(material, distance: float, strength: float):
    """A copy of a surface whose base colour is darkened where something is near it: ambient occlusion within
    `distance` metres, at `strength` (0 none, 1 black in a crevice). It is what the soft light of a ceiling full
    of panels does not give: a person, a table leg or a wall meets the floor in a shade, and the foot of a wall,
    its top under the ceiling and a corner get the full strength too, not only things that stand on the floor.
    A surface that is not a Principled coat has no base colour to darken and is returned as it is. Only the
    colour changes; the surface stays where and what it was, and the depth is the same."""
    key = ("shaded", material.name, round(distance, 3), round(strength, 3))
    if key not in _materials:
        tree = material.node_tree
        shader = None if tree is None else next((node for node in tree.nodes if node.type == "BSDF_PRINCIPLED"), None)
        if shader is None:
            _materials[key] = material
            return material
        copy = material.copy()
        for each in list(copy.keys()):                    # the copy is its own material, not another bearer of the mark
            del copy[each]
        tree = copy.node_tree
        shader = next(node for node in tree.nodes if node.type == "BSDF_PRINCIPLED")
        base = shader.inputs["Base Color"]
        occlusion = tree.nodes.new("ShaderNodeAmbientOcclusion")
        occlusion.inputs["Distance"].default_value = float(distance)
        occlusion.samples = 8
        occlusion.inside = False
        occlusion.only_local = False
        # What a thin leg or a shoe shuts out of the sky above a point of floor is a fifth of it at most, so the
        # shut-out share (a wall's foot, top and corners too) is multiplied by CONTACT_GAIN before it is taken
        # off: 1 where open, 1 - strength shut in.
        shut = tree.nodes.new("ShaderNodeMath")
        shut.operation = "SUBTRACT"
        shut.inputs[0].default_value = 1.0
        tree.links.new(occlusion.outputs["AO"], shut.inputs[1])
        gained = tree.nodes.new("ShaderNodeMath")
        gained.operation, gained.use_clamp = "MULTIPLY", True
        gained.inputs[1].default_value = CONTACT_GAIN
        tree.links.new(shut.outputs[0], gained.inputs[0])
        eased = tree.nodes.new("ShaderNodeMath")
        eased.operation = "MULTIPLY_ADD"
        eased.inputs[1].default_value, eased.inputs[2].default_value = -float(strength), 1.0
        tree.links.new(gained.outputs[0], eased.inputs[0])
        multiply = tree.nodes.new("ShaderNodeMix")
        multiply.data_type, multiply.blend_type = "RGBA", "MULTIPLY"
        multiply.inputs[0].default_value = 1.0
        if base.links:
            tree.links.new(base.links[0].from_socket, multiply.inputs[6])
        else:
            multiply.inputs[6].default_value = base.default_value
        tree.links.new(eased.outputs[0], multiply.inputs[7])
        tree.links.new(multiply.outputs[2], base)
        _materials[key] = copy
    return _materials[key]


def glass(tint=(0.995, 1.0, 0.995), ior: float = 1.45, sheen: float = GLASS_SHEEN):
    """A pane: glass to the camera and to glossy rays, plainly clear to shadow and diffuse rays, so that light
    comes through it without caustic noise. A camera ray is refracted, not let through, so the pane is a
    surface in the depth pass.

    It is thin glass. What is behind it is seen sharp and straight through: a refraction of index 1, which
    bends nothing. What it mirrors comes off a coat of its own, as much of it as glass of the index `ior`
    mirrors at that angle of view, with the roughness `sheen`: a ceiling fitting is a soft light in the pane
    and not a second fitting."""
    key = ("glass", tint, ior, sheen)
    if key not in _materials:
        material = bpy.data.materials.new("glass")
        material.use_nodes = True
        nodes, links = material.node_tree.nodes, material.node_tree.links
        nodes.clear()
        through = nodes.new("ShaderNodeBsdfRefraction")
        through.inputs["Color"].default_value = (*tint, 1.0)
        through.inputs["Roughness"].default_value, through.inputs["IOR"].default_value = 0.0, 1.0
        mirror = nodes.new("ShaderNodeBsdfGlossy")
        mirror.inputs["Roughness"].default_value = sheen
        # How much is mirrored, by Schlick: `head_on` of the light where the pane is looked at squarely, all of
        # it at a glancing view. A pane's two faces are both looked at from their own side, the far one from
        # inside the glass, where a node that knows of indices would mirror everything.
        head_on = ((ior - 1.0) / (ior + 1.0)) ** 2
        glancing = nodes.new("ShaderNodeMath")     # (1 - |cos|) ** 5
        glancing.operation = "POWER"
        links.new(nodes.new("ShaderNodeLayerWeight").outputs["Facing"], glancing.inputs[0])
        glancing.inputs[1].default_value = 5.0
        share = nodes.new("ShaderNodeMath")
        share.operation = "MULTIPLY_ADD"
        links.new(glancing.outputs[0], share.inputs[0])
        share.inputs[1].default_value, share.inputs[2].default_value = 1.0 - head_on, head_on
        pane = nodes.new("ShaderNodeMixShader")
        links.new(share.outputs[0], pane.inputs["Fac"])
        links.new(through.outputs["BSDF"], pane.inputs[1])
        links.new(mirror.outputs["BSDF"], pane.inputs[2])
        clear = nodes.new("ShaderNodeBsdfTransparent")
        clear.inputs["Color"].default_value = (*tint, 1.0)
        path = nodes.new("ShaderNodeLightPath")
        either = nodes.new("ShaderNodeMath")
        either.operation = "MAXIMUM"
        links.new(path.outputs["Is Shadow Ray"], either.inputs[0])
        links.new(path.outputs["Is Diffuse Ray"], either.inputs[1])
        mix = nodes.new("ShaderNodeMixShader")
        links.new(either.outputs[0], mix.inputs["Fac"])
        links.new(pane.outputs["Shader"], mix.inputs[1])
        links.new(clear.outputs["BSDF"], mix.inputs[2])
        out = nodes.new("ShaderNodeOutputMaterial")
        links.new(mix.outputs["Shader"], out.inputs["Surface"])
        _materials[key] = material
    return _materials[key]


def sign_material(path: Path, emission: float):
    """The picture at `path` as a surface, mapped by the mesh's UVs; with `emission`, glowing with that strength
    as a lit screen does."""
    key = ("sign", str(path), round(emission, 4))
    if key not in _materials:
        material = bpy.data.materials.new(Path(path).stem)
        material.use_nodes = True
        nodes, links = material.node_tree.nodes, material.node_tree.links
        picture = nodes.new("ShaderNodeTexImage")
        picture.image = bpy.data.images.load(str(Path(path).resolve()), check_existing=True)
        picture.interpolation, picture.extension = "Cubic", "EXTEND"
        shader = nodes["Principled BSDF"]
        shader.inputs["Roughness"].default_value = 0.2 if emission > 0 else 0.5
        links.new(picture.outputs["Color"], shader.inputs["Base Color"])
        if emission > 0:
            links.new(picture.outputs["Color"], shader.inputs["Emission Color"])
            shader.inputs["Emission Strength"].default_value = emission
        _materials[key] = material
    return _materials[key]


def fitting_glow(strength: float, tone):
    """What a ceiling fitting shows: it glows towards the camera, on its front only, and lights nothing; the
    area light under it does the lighting."""
    key = ("glow", round(strength, 4), tuple(tone))
    if key not in _materials:
        material = bpy.data.materials.new("fitting_glow")
        material.use_nodes = True
        nodes, links = material.node_tree.nodes, material.node_tree.links
        nodes.clear()
        glow = nodes.new("ShaderNodeEmission")
        glow.inputs["Color"].default_value = (*tone, 1.0)
        facing = nodes.new("ShaderNodeMath")       # strength * (1 - backfacing)
        facing.operation = "MULTIPLY_ADD"
        links.new(nodes.new("ShaderNodeNewGeometry").outputs["Backfacing"], facing.inputs[0])
        facing.inputs[1].default_value, facing.inputs[2].default_value = -strength, strength
        seen = nodes.new("ShaderNodeMath")
        seen.operation = "MULTIPLY"
        links.new(facing.outputs[0], seen.inputs[0])
        links.new(nodes.new("ShaderNodeLightPath").outputs["Is Camera Ray"], seen.inputs[1])
        links.new(seen.outputs[0], glow.inputs["Strength"])
        out = nodes.new("ShaderNodeOutputMaterial")
        links.new(glow.outputs["Emission"], out.inputs["Surface"])
        material.cycles.emission_sampling = "NONE"     # no light sample is spent on it
        _materials[key] = material
    return _materials[key]


# ---------------------------------------------------------------------------------------------------- mesh objects

def frame(origin, x_axis, z: float = 0.0) -> Matrix:
    """A frame standing on the plane at height z: its origin at (x, y), +X along the level unit vector, +Z up."""
    c, s = float(x_axis[0]), float(x_axis[1])
    return Matrix(((c, -s, 0.0, float(origin[0])), (s, c, 0.0, float(origin[1])), (0.0, 0.0, 1.0, float(z)),
                   (0.0, 0.0, 0.0, 1.0)))


def _linked(scene, name: str, mesh, matrix, group):
    made = bpy.data.objects.new(name, mesh)
    if matrix is not None:
        made.matrix_world = matrix
    if group is not None:
        made[GROUP] = int(group)
    scene.collection.objects.link(made)
    return made


def mesh_object(scene, name: str, verts, faces, materials=(), slots=None, smooth=None, matrix=None, uvs=None,
                group=None):
    """A mesh object from vertices and faces. `slots` gives each face its material, `smooth` says per face whether
    it is shaded round, `uvs` holds for every face one (u, v) per corner in the face's own order."""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(map(float, v)) for v in verts], [], [tuple(face) for face in faces])
    if uvs is not None:
        layer = mesh.uv_layers.new(name="UVMap")
        layer.data.foreach_set("uv", [float(value) for face in uvs for corner in face for value in corner])
    for material in materials:
        mesh.materials.append(material)
    if slots is not None:
        mesh.polygons.foreach_set("material_index", [int(slot) for slot in slots])
    if smooth is not None:
        mesh.polygons.foreach_set("use_smooth", [bool(flag) for flag in smooth])
    mesh.update()
    return _linked(scene, name, mesh, matrix, group)


def blocks(scene, name: str, boxes, material, matrix, bevel_m: float = 0.0):
    """Several boxes, each (size, centre) in the frame `matrix`, as one mesh object: object coordinates run on
    across the pieces, so a box-projected texture does not jump from one to the next."""
    pieces = bmesh.new()
    for size, centre in boxes:
        bmesh.ops.create_cube(pieces, size=1.0, matrix=Matrix.Translation(centre) @ Matrix.Diagonal((*size, 1.0)))
    mesh = bpy.data.meshes.new(name)
    pieces.to_mesh(mesh)
    pieces.free()
    mesh.materials.append(material)
    made = _linked(scene, name, mesh, matrix, None)
    bevel(made, bevel_m, min(min(size) for size, _ in boxes))
    return made


def bevel(thing, width: float, smallest: float) -> None:
    """Round off the sharp edges of an object by `width`, and by no more than a quarter of its smallest side."""
    width = min(width, 0.25 * smallest)
    if width < 0.0004:
        return
    rounding = thing.modifiers.new("bevel", "BEVEL")
    rounding.width, rounding.segments = width, 2
    rounding.limit_method, rounding.angle_limit = "ANGLE", math.radians(40.0)


def _frustum(r0: float, r1: float, height: float, segments: int):
    """A cylinder that may taper: radius r0 at z = 0 and r1 at z = height, its vertices on the two circles.
    Returns vertices, faces and, per face, 0 for the side, 1 for the upper end, 2 for the lower."""
    angles = 2.0 * math.pi * np.arange(segments) / segments
    ring = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    verts = np.concatenate([np.column_stack([r0 * ring, np.zeros(segments)]),
                            np.column_stack([r1 * ring, np.full(segments, height)])])
    faces = [(k, (k + 1) % segments, segments + (k + 1) % segments, segments + k) for k in range(segments)]
    faces += [tuple(range(segments, 2 * segments)), tuple(reversed(range(segments)))]
    return verts, faces, [0] * segments + [1, 2]


def _ball(radius: float, segments: int):
    """A sphere about the origin, its vertices on it: segments // 2 bands from pole to pole, `segments` round."""
    rings = max(segments // 2, 4)
    verts = [(0.0, 0.0, 1.0)]
    for i in range(1, rings):
        theta = math.pi * i / rings
        z, r = math.cos(theta), math.sin(theta)
        verts += [(r * math.cos(2 * math.pi * j / segments), r * math.sin(2 * math.pi * j / segments), z)
                  for j in range(segments)]
    verts.append((0.0, 0.0, -1.0))
    faces = [(0, 1 + j, 1 + (j + 1) % segments) for j in range(segments)]
    for i in range(rings - 2):
        for j in range(segments):
            a, b = 1 + i * segments + j, 1 + i * segments + (j + 1) % segments
            faces.append((a, a + segments, b + segments, b))
    last, start = len(verts) - 1, 1 + (rings - 2) * segments
    faces += [(last, start + (j + 1) % segments, start + j) for j in range(segments)]
    return np.array(verts) * radius, faces


# ----------------------------------------------------------------------------------------------------------- parts

def _coats(part: dict) -> list:
    """The materials of a part's mesh: one, and for a painted solid with a colour of its own on top a second.
    A named material takes the part's tint, where it has one; a description written before parts had a tint
    holds none. Glass and paint take no tint."""
    solid = part["solid"]
    if part["glass"] or part["material"] == GLASS:
        return [glass()]
    if part["material"] != PAINT:
        return [named(part["material"], part.get("tint"))]
    coats = [paint(solid["color"])]
    if solid.get("top") is not None:
        coats.append(paint(solid["top"]))
    return coats


def add_part(scene, part: dict, name: str, signs: dict, job_dir: Path, segments: int) -> list:
    """A part's solid as a mesh object, with a further object for each of its signs. Returns the objects."""
    solid, group, coats = part["solid"], part["group"], _coats(part)
    shape = solid.get("shape")
    if part["signs"] and shape != "box":
        raise ValueError(f"{name} carries a sign and is a {shape}: a sign lies on a face of a box")
    if shape == "box":
        hx, hy, sz = solid["sx"] / 2, solid["sy"] / 2, solid["sz"]
        verts = [(-hx, -hy, 0), (hx, -hy, 0), (hx, hy, 0), (-hx, hy, 0),
                 (-hx, -hy, sz), (hx, -hy, sz), (hx, hy, sz), (-hx, hy, sz)]
        slots = [len(coats) - 1 if face == _TOP_FACE else 0 for face in range(len(_BOX_FACES))]
        # The object's origin is the middle of the underside and its +X the box's front: the mesh is the box in
        # its own frame, at its real size.
        placed = frame((solid["cx"], solid["cy"]), (math.cos(solid["yaw"]), math.sin(solid["yaw"])), solid["z0"])
        made = [mesh_object(scene, name, verts, _BOX_FACES, coats, slots, matrix=placed, group=group)]
        bevel(made[0], part["bevel"], min(solid["sx"], solid["sy"], sz))
        for number, sign in enumerate(part["signs"]):
            made.append(_add_sign(scene, f"{name}_sign_{number}", solid, sign, signs, job_dir, placed, group))
        return made
    if shape == "cylinder":
        top = solid["radius"] if solid.get("top_radius") is None else solid["top_radius"]
        low, high = max(solid["radius"], 1e-4), max(top, 1e-4)             # a point would leave faces without area
        verts, faces, kinds = _frustum(low, high, solid["height"], segments)
        slots = [len(coats) - 1 if kind == 1 else 0 for kind in kinds]
        made = mesh_object(scene, name, verts, faces, coats, slots, smooth=[kind == 0 for kind in kinds],
                           matrix=Matrix.Translation((solid["x"], solid["y"], solid["z0"])), group=group)
        bevel(made, part["bevel"], min(solid["height"], 2 * low, 2 * high))
        return [made]
    if shape == "sphere":
        verts, faces = _ball(solid["radius"], segments)
        return [mesh_object(scene, name, verts, faces, coats[:1], smooth=[True] * len(faces),
                            matrix=Matrix.Translation((solid["x"], solid["y"], solid["z"])), group=group)]
    raise ValueError(f"{name} has the unknown shape {shape!r}: expected a box, a cylinder or a sphere")


def _add_sign(scene, name: str, box: dict, sign: dict, signs: dict, job_dir: Path, placed: Matrix, group: int):
    """A sign's picture on its rectangle of a face of the box, 1 mm proud of it, the right way up to whoever
    reads the face from outside. `placed` is the box's own frame, its origin the middle of the underside."""
    if sign["face"] not in FACES:
        raise ValueError(f"{name} is on the {sign['face']!r} face: expected one of {', '.join(FACES)}")
    picture = signs.get(_art_key(sign["art"]))
    if picture is None:
        raise ValueError(f"the job holds no picture for the art of {name}: {sign['art']}")
    normal, outward, a_axis, a_sign, b_axis, b_sign = FACES[sign["face"]]
    size, middle = (box["sx"], box["sy"], box["sz"]), (0.0, 0.0, box["sz"] / 2)
    left, bottom, right, top = sign["rect"]
    corners = []
    for across, up in ((left, bottom), (right, bottom), (right, top), (left, top)):   # round it as its reader sees it
        point = [0.0, 0.0, 0.0]
        point[normal] = middle[normal] + outward * (size[normal] / 2 + SIGN_PROUD_M)
        point[a_axis] = middle[a_axis] + a_sign * (across - 0.5) * size[a_axis]
        point[b_axis] = middle[b_axis] + b_sign * (up - 0.5) * size[b_axis]
        corners.append(tuple(point))
    material = sign_material(job_dir / picture, float(sign["emission"]))
    return mesh_object(scene, name, corners, [(0, 1, 2, 3)], [material], uvs=[[(0, 0), (1, 0), (1, 1), (0, 1)]],
                       matrix=placed, group=group)


def _art_key(art: dict) -> str:
    """An art as a key: both the job's list of pictures and the signs of its parts hold the art as JSON."""
    return json.dumps(art, sort_keys=True)


def sign_pictures(listed: dict) -> dict:
    """The job's "signs", {picture file: art}, as {art key: picture file}."""
    return {_art_key(art): name for name, art in listed.items()}


# ----------------------------------------------------------------------------------------------- models and people

def _tag(scene, thing, group: int) -> None:
    """Give an object of the asset library, and every object under it, the group, and see that all of them are
    in the scene."""
    for each in (thing, *thing.children_recursive):
        each[GROUP] = int(group)
        if not each.users_collection:
            scene.collection.objects.link(each)


def _stood(x: float, y: float, z: float, yaw: float, scale) -> Matrix:
    """Where a thing of the asset library goes: scaled along its own axes, turned about Z, then moved."""
    return Matrix.Translation((x, y, z)) @ Matrix.Rotation(yaw, 4, "Z") @ Matrix.Diagonal((*scale, 1.0))


def add_model(scene, model: dict):
    """A model of the asset library, scaled along its own axes and stood with its bottom centre on (x, y, z),
    its front along yaw. The library gives it at life size, its bottom centre at the origin, its front along +X."""
    thing = _assets("a model").add_model(model["asset"])
    thing.matrix_world = _stood(model["x"], model["y"], model["z"], model["yaw"], model["scale"])
    _tag(scene, thing, model["group"])
    return thing


def _crown(thing) -> float | None:
    """How high above its origin the mesh of a library object reaches, as the library gives it; None for an
    object that is no mesh."""
    if thing.type != "MESH" or not thing.data.vertices:
        return None
    heights = np.empty(3 * len(thing.data.vertices), dtype=np.float32)
    thing.data.vertices.foreach_get("co", heights)
    return float(heights[2::3].max())


def add_person(scene, person: dict, heights: dict):
    """A person of the asset library, stood with the ground point under the pelvis on (x, y), chest along yaw.

    The library gives the avatar in its pose at the avatar's natural size, with that ground point at the origin
    and the chest to +X; the job states the natural height. Across and in depth the person is sized as their
    height is to the natural one, times their build: the library's measurements of a pose (its reach, how far
    it stands before and behind the axis) are stated for that. Upward they are sized so that the top of the
    mesh, as it is in this pose, comes to their height: a head bowed over a phone is some centimetres under
    the natural height, and the person's height is where the twin's figure ends, whatever the pose."""
    if person["avatar"] not in heights:
        raise ValueError(f"the job does not state the natural height of the avatar {person['avatar']!r}")
    thing = _assets("a person").add_person(person["avatar"], person["pose"], person["prop"])
    natural = heights[person["avatar"]]
    wide = person["height"] / natural * person["build"]
    tall = person["height"] / (_crown(thing) or natural)
    thing.matrix_world = _stood(person["x"], person["y"], 0.0, person["yaw"], (wide, wide, tall))
    _tag(scene, thing, person["group"])
    return thing


# ------------------------------------------------------------------------------------------------------------ room

def _inside(point, polygon) -> bool:
    x, y = point
    inside = False
    for (x0, y0), (x1, y1) in zip(polygon, [*polygon[1:], polygon[0]]):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def _walls(polygon) -> list[dict]:
    """Per edge of the outline: its ends a and b with the room on the left of a -> b, its length, its frame (+X
    along it, +Y into the room), and whether the corner at each end points out of the room."""
    def turns_left(p, q, r) -> bool:               # going round anticlockwise: the corner at q points out of the room
        return float((q - p)[0] * (r - q)[1] - (q - p)[1] * (r - q)[0]) > 0

    area = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(polygon, [*polygon[1:], polygon[0]]))
    corners = [np.array(corner) for corner in (polygon if area > 0 else polygon[::-1])]    # anticlockwise from here on
    count, walls = len(corners), []
    for k in range(count):
        before, a, b, after = corners[k - 1], corners[k], corners[(k + 1) % count], corners[(k + 2) % count]
        length = float(np.linalg.norm(b - a))
        along = (b - a) / length
        walls.append({"a": a, "b": b, "length": length, "along": along, "inward": np.array([-along[1], along[0]]),
                      "frame": frame(a, along), "out_a": turns_left(before, a, b), "out_b": turns_left(a, b, after)})
    return walls


def _on_wall(opening: dict, walls: list[dict]) -> tuple[int, float, float]:
    """The wall an opening is on, and where on it the opening begins and ends, along the wall's own frame."""
    place = np.array([opening["x"], opening["y"]])
    facing = np.array([math.cos(opening["yaw"]), math.sin(opening["yaw"])])        # an opening's +x is into the room
    best, nearest = None, ON_A_WALL_M
    for number, wall in enumerate(walls):
        along = float((place - wall["a"]) @ wall["along"])
        away = abs(float((place - wall["a"]) @ wall["inward"]))
        if 0.0 <= along <= wall["length"] and away < nearest and float(wall["inward"] @ facing) > 0.5:
            best, nearest = (number, along), away
    if best is None:
        raise ValueError(f"the {opening['kind']} at ({opening['x']:.2f}, {opening['y']:.2f}) is on no wall of the room "
                         "that faces as it does")
    number, along = best
    length = walls[number]["length"]
    start, end = max(along - opening["width"] / 2, 0.05), min(along + opening["width"] / 2, length - 0.05)
    if end - start < 0.2:
        raise ValueError(f"the {opening['kind']} at ({opening['x']:.2f}, {opening['y']:.2f}) does not fit on its wall")
    return number, start, end


def add_room(scene, room: dict, light: dict) -> dict:
    """The floor, the ceiling with its fittings and their lights, the walls with their trims, and the openings.
    Returns how many fittings and openings there are."""
    polygon = [(float(x), float(y)) for x, y in room["polygon"]]
    if len(polygon) < 3:
        raise ValueError(f"a room's polygon needs at least 3 corners, not {len(polygon)}")
    height = float(room["height"])
    first, second = np.array(polygon[0]), np.array(polygon[1])
    along = (second - first) / np.linalg.norm(second - first)
    across = np.array([-along[1], along[0]])
    # The floor's own frame is the room's, as the placeholder has it: its origin the first corner, its +X the
    # first edge. Fittings are laid out in it, and a box-projected floor follows the walls.
    local = [(float((np.array(corner) - first) @ along), float((np.array(corner) - first) @ across))
             for corner in polygon]
    area = sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(local, [*local[1:], local[0]])) / 2
    order = list(range(len(local))) if area > 0 else list(reversed(range(len(local))))
    mesh_object(scene, "room_floor", [(a, b, 0.0) for a, b in local], [order],
                [contact_shaded(named(room["floor"]), light["floor_shade_m"], light["floor_shade"])],
                matrix=frame(first, along))

    spacing = float(room["lights"])
    shown, spacing = spacing > 0, spacing if spacing > 0 else UNLIT_SPACING_M
    a_all, b_all = [a for a, _ in local], [b for _, b in local]
    fittings = [((i + 0.5) * spacing, (j + 0.5) * spacing)
                for i in range(math.floor(min(a_all) / spacing), math.ceil(max(a_all) / spacing))
                for j in range(math.floor(min(b_all) / spacing), math.ceil(max(b_all) / spacing))]
    fittings = [(a, b) for a, b in fittings if _inside(first + a * along + b * across, polygon)]
    if not fittings:                               # a room too narrow for the middle of any square: one light, unseen
        fittings, shown = [(sum(a_all) / len(a_all), sum(b_all) / len(b_all))], False
    glow = fitting_glow(light["glow"], light["fitting_tone"])
    _add_ceiling(scene, local, list(reversed(order)), frame(first, along, height), fittings if shown else [],
                 [named(room["ceiling"]), paint((225, 225, 222), rough=0.4), glow])
    # The fittings share the power that the floor's area asks for, so a room is as bright whether the squares
    # fit it evenly or leave a strip over.
    watts = light["fitting"] * abs(area) / len(fittings)
    for number, (a, b) in enumerate(fittings):
        lamp = bpy.data.lights.new(f"fitting_{number}", "AREA")
        lamp.shape = "RECTANGLE"
        lamp.size, lamp.size_y = (PANEL_M[0] * light["lamp_scale"], PANEL_M[1] * light["lamp_scale"])
        lamp.energy, lamp.color = watts, light["fitting_tone"]
        lamp.spread = math.radians(light["spread"])
        holder = bpy.data.objects.new(f"fitting_{number}", lamp)       # an area light shines along its own -Z: down
        holder.matrix_world = frame(first, along, height - LIGHT_DROP_M) @ Matrix.Translation((a, b, 0.0))
        holder.visible_camera = False
        scene.collection.objects.link(holder)

    walls = _walls(polygon)
    holes: dict[int, list] = {number: [] for number in range(len(walls))}
    for opening in room["openings"]:
        if opening["kind"] not in ("window", "door"):
            raise ValueError(f"unknown kind of opening {opening['kind']!r}: expected a window or a door")
        number, start, end = _on_wall(opening, walls)
        low = max(float(opening["z0"]), 0.0)
        high = min(low + float(opening["height"]), height - 0.05)
        holes[number].append({"kind": opening["kind"], "s0": start, "s1": end, "z0": low, "z1": high})
    for number, wall in enumerate(walls):
        _add_wall(scene, number, wall, holes[number], room, height, light)
    return {"fittings": len(fittings) if shown else 0, "openings": len(room["openings"])}


def _add_ceiling(scene, local, order, placed: Matrix, fittings, materials) -> None:
    """The ceiling as one flat mesh, facing down. It is cut along the edges of every fitting and of its trim, and
    the faces inside take the glow and the trim's paint: the fittings lie in the ceiling's own plane."""
    cut = bmesh.new()
    corners = [cut.verts.new((a, b, 0.0)) for a, b in local]
    cut.faces.new([corners[index] for index in order])
    lines = set()
    for a, b in fittings:
        for axis, centre, half in ((0, a, PANEL_M[0] / 2), (1, b, PANEL_M[1] / 2)):
            for reach in (half, half + PANEL_TRIM_M):
                lines |= {(axis, round(centre - reach, 6)), (axis, round(centre + reach, 6))}
    for axis, value in sorted(lines):
        point, normal = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        point[axis], normal[axis] = value, 1.0
        bmesh.ops.bisect_plane(cut, geom=cut.verts[:] + cut.edges[:] + cut.faces[:], dist=1e-6, plane_co=point,
                               plane_no=normal)
    for face in cut.faces:
        middle = face.calc_center_median()
        off = [(abs(middle.x - a) - PANEL_M[0] / 2, abs(middle.y - b) - PANEL_M[1] / 2) for a, b in fittings]
        if any(da < 0 and db < 0 for da, db in off):
            face.material_index = 2
        elif any(da < PANEL_TRIM_M and db < PANEL_TRIM_M for da, db in off):
            face.material_index = 1
    mesh = bpy.data.meshes.new("room_ceiling")
    cut.to_mesh(mesh)
    cut.free()
    for material in materials:
        mesh.materials.append(material)
    _linked(scene, "room_ceiling", mesh, placed, None)


def _add_wall(scene, number: int, wall: dict, holes: list, room: dict, height: float, light: dict) -> None:
    """One wall in its own frame (x along it, +y into the room): its face with the windows cut out, the skirting
    board and the dado rail broken where an opening is in their way, and what the openings are made of."""
    length, placed = wall["length"], wall["frame"]
    windows = [hole for hole in holes if hole["kind"] == "window"]
    doors = [hole for hole in holes if hole["kind"] == "door"]
    dado = room["dado"]
    dado_height = min(max(float(room["dado_height"]), 0.0), height)
    # The face runs on by a millimetre behind the floor, the ceiling and the wall of a corner that points out of
    # the room, where nothing of it shows; at a corner that points in, it ends on the corner.
    x_low, x_high = (-SEAL_M if wall["out_a"] else 0.0), (length + SEAL_M if wall["out_b"] else length)
    xs = sorted({x_low, x_high, *(edge for hole in windows for edge in (hole["s0"], hole["s1"]))})
    zs = sorted({-SEAL_M, height + SEAL_M, *([dado_height] if dado is not None else []),
                 *(edge for hole in windows for edge in (hole["z0"], hole["z1"]))})
    verts, faces, slots = [], [], []
    for x0, x1 in zip(xs[:-1], xs[1:]):
        for z0, z1 in zip(zs[:-1], zs[1:]):
            xm, zm = (x0 + x1) / 2, (z0 + z1) / 2
            if any(hole["s0"] < xm < hole["s1"] and hole["z0"] < zm < hole["z1"] for hole in windows):
                continue
            base = len(verts)
            verts += [(x0, 0.0, z0), (x0, 0.0, z1), (x1, 0.0, z1), (x1, 0.0, z0)]          # facing +y, into the room
            faces.append((base, base + 1, base + 2, base + 3))
            slots.append(1 if dado is not None and zm < dado_height else 0)
    wall_paint = named("wall_paint", room["wall"])
    coats = [contact_shaded(each, light["wall_shade_m"], light["wall_shade"])
             for each in [wall_paint] + ([named("wall_paint", dado)] if dado is not None else [])]
    mesh_object(scene, f"room_wall_{number}", verts, faces, coats, slots, matrix=placed)

    def clear_of(blocked, low: float, high: float) -> list[tuple[float, float]]:
        """The stretches of the wall, from `low` to `high` along it, that none of these openings takes."""
        free, at = [], low
        for hole in sorted(blocked, key=lambda hole: hole["s0"]):
            if hole["s0"] - at > 0.01:
                free.append((at, hole["s0"]))
            at = max(at, hole["s1"])
        return free + ([(at, high)] if high - at > 0.01 else [])

    # At a corner that points into the room a trim runs on by its own depth, to meet its neighbour's.
    def ends(depth: float) -> tuple[float, float]:
        return (0.0 if wall["out_a"] else -depth), (length if wall["out_b"] else length + depth)

    def strips(runs, depth: float, tall: float, level: float) -> list:
        """A trim of this depth and height along each stretch, its back on the wall's face, its middle at `level`."""
        return [((x1 - x0, depth, tall), ((x0 + x1) / 2, depth / 2, level)) for x0, x1 in runs]

    skirting_h, skirting_d = SKIRTING_M
    runs = clear_of([hole for hole in holes if hole["z0"] < skirting_h], *ends(skirting_d))
    if runs:
        blocks(scene, f"room_skirting_{number}", strips(runs, skirting_d, skirting_h, skirting_h / 2),
               paint(SKIRTING_COLOUR, rough=0.35, bump_m=0.0002), placed)
    rail_h, rail_d = RAIL_M
    if dado is not None and rail_h < dado_height < height - rail_h:
        low, high = dado_height - rail_h / 2, dado_height + rail_h / 2
        runs = clear_of([hole for hole in holes if hole["z0"] < high and hole["z1"] > low], *ends(rail_d))
        if runs:
            blocks(scene, f"room_rail_{number}", strips(runs, rail_d, rail_h, dado_height),
                   paint([int(RAIL_DARKER * value) for value in dado], rough=0.4), placed, bevel_m=0.004)
    for count, hole in enumerate(windows):
        _add_window(scene, f"room_window_{number}_{count}", hole, placed, wall_paint, light)
    for count, hole in enumerate(doors):
        _add_door(scene, f"room_door_{number}_{count}", hole, placed)


def _add_window(scene, name: str, hole: dict, placed: Matrix, wall_paint, light: dict) -> None:
    """What fills a window's hole, in the wall's frame: the reveal, the frame with its mullion, the pane, the
    sill, and the daylight it lets in. Only the sill stands before the wall's face (y > 0)."""
    s0, s1, z0, z1 = hole["s0"], hole["s1"], hole["z0"], hole["z1"]
    middle, level, width, tall = (s0 + s1) / 2, (z0 + z1) / 2, s1 - s0, z1 - z0
    deep = WALL_THICK_M
    # The reveal lines the hole through the wall's thickness: four faces that begin on the hole's own edges.
    verts = [(s0, 0.0, z0), (s1, 0.0, z0), (s1, 0.0, z1), (s0, 0.0, z1),
             (s0, -deep, z0), (s1, -deep, z0), (s1, -deep, z1), (s0, -deep, z1)]
    mesh_object(scene, f"{name}_reveal", verts, [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)], [wall_paint],
                matrix=placed)
    bar, back, front = 0.055, -0.10, -0.01         # the frame: bars this wide, from 10 cm to 1 cm behind the wall
    thick, at, between = front - back, (front + back) / 2, tall - 2 * bar
    # The upright bars stand between the level ones: two boxes of one mesh that ran through each other would
    # show their rounded edges inside one another.
    bars = [((width, thick, bar), (middle, at, z0 + bar / 2)), ((width, thick, bar), (middle, at, z1 - bar / 2)),
            ((bar, thick, between), (s0 + bar / 2, at, level)), ((bar, thick, between), (s1 - bar / 2, at, level))]
    if width > 1.3:
        bars.append(((0.045, thick - 0.02, between), (middle, at, level)))                  # a mullion
    white = paint(FRAME_COLOUR, rough=0.3, bump_m=0.0002)
    blocks(scene, f"{name}_frame", bars, white, placed, bevel_m=0.004)
    blocks(scene, f"{name}_pane", [((width - 0.02, 0.006, tall - 0.02), (middle, at, level))], glass(), placed)
    blocks(scene, f"{name}_sill", [((width + 0.1, 0.04, 0.035), (middle, 0.02, z0 - 0.0175))], white, placed,
           bevel_m=0.004)
    if light["daylight"] > 0 and width > 0.3 and tall > 0.3:
        lamp = bpy.data.lights.new(f"{name}_daylight", "AREA")
        lamp.shape, lamp.size, lamp.size_y = "RECTANGLE", width - 0.1, tall - 0.1
        lamp.energy, lamp.color = light["daylight"] * width * tall, light["daylight_tone"]
        lamp.spread = math.radians(150.0)
        holder = bpy.data.objects.new(f"{name}_daylight", lamp)
        # Its -Z, the way it shines, is the wall's +y; it stands just clear of the sill.
        holder.matrix_world = placed @ Matrix(((1, 0, 0, middle), (0, 0, -1, 0.045), (0, 1, 0, level), (0, 0, 0, 1)))
        # It lights the room and is not to be seen: looked at, mirrored, or through glass (a ray that has passed
        # a pane, or was mirrored off a pane's back, is a transmission ray) it would be a flat white board.
        holder.visible_camera = holder.visible_glossy = holder.visible_transmission = False
        scene.collection.objects.link(holder)


def _add_door(scene, name: str, hole: dict, placed: Matrix) -> None:
    """A door on its wall, in the wall's frame: a frame, a wooden leaf in it and a lever handle. The wall behind
    it is whole, and nothing of the door stands further than 4.5 cm before it."""
    s0, s1, z0, z1 = hole["s0"], hole["s1"], hole["z0"], hole["z1"]
    middle, width, tall = (s0 + s1) / 2, s1 - s0, z1 - z0
    jamb, deep = 0.06, 0.035                       # the frame: 6 cm wide, 3.5 cm proud of the wall
    blocks(scene, f"{name}_frame", [((jamb, deep, tall), (s0 + jamb / 2, deep / 2, z0 + tall / 2)),
                                    ((jamb, deep, tall), (s1 - jamb / 2, deep / 2, z0 + tall / 2)),
                                    ((width - 2 * jamb, deep, jamb), (middle, deep / 2, z1 - jamb / 2))],
           paint(FRAME_COLOUR, rough=0.3, bump_m=0.0002), placed, bevel_m=0.004)
    leaf_w, leaf_h = width - 2 * jamb, tall - jamb - 0.005
    blocks(scene, f"{name}_leaf", [((leaf_w, 0.025, leaf_h), (middle, 0.0175, z0 + 0.005 + leaf_h / 2))],
           named("wood"), placed, bevel_m=0.003)   # the leaf: from 0.5 to 3 cm before the wall
    grip, at = min(z0 + 1.02, z1 - 0.2), middle + leaf_w / 2 - 0.07
    plate, stem, lever = (0.045, 0.004, 0.16), (0.018, 0.006, 0.018), (0.12, 0.005, 0.018)
    blocks(scene, f"{name}_handle", [(plate, (at, 0.032, grip)), (stem, (at, 0.037, grip)),
                                     (lever, (at - 0.045, 0.0425, grip))], named("steel"), placed)


# ----------------------------------------------------------------------------------------------------------- world

def set_world(scene, hdri: str | None, light: dict) -> None:
    """What the windows show and glossy things reflect: the named HDRI of the asset library, or a plain sky.
    It is seen and lights nothing: the fittings and the daylight of the windows light the room."""
    if hdri is not None and _library is not None:
        _library.set_world(hdri, light["hdri"])
        world = scene.world
        if world is None:
            raise RuntimeError(f"the asset library set no world for the HDRI {hdri!r}")
    else:
        world = bpy.data.worlds.new("sky")
        world.use_nodes = True
        nodes, links = world.node_tree.nodes, world.node_tree.links
        up = nodes.new("ShaderNodeSeparateXYZ")    # of the direction looked in: -1 straight down, 1 straight up
        links.new(nodes.new("ShaderNodeTexCoord").outputs["Generated"], up.inputs[0])
        shade = nodes.new("ShaderNodeValToRGB")    # the ground below, the horizon's colour level, the zenith's overhead
        ground = (*light["ground"], 1.0)
        stops = shade.color_ramp.elements          # a new ramp has two stops, at 0 and at 1
        stops[0].position, stops[0].color = 0.0, ground
        stops[1].position, stops[1].color = 1.0, (*light["zenith"], 1.0)
        stops.new(0.49).color = ground
        stops.new(0.5).color = (*light["horizon"], 1.0)
        scale = nodes.new("ShaderNodeMath")
        scale.operation = "MULTIPLY_ADD"
        scale.inputs[1].default_value = scale.inputs[2].default_value = 0.5
        links.new(up.outputs["Z"], scale.inputs[0])
        links.new(scale.outputs[0], shade.inputs["Fac"])
        links.new(shade.outputs["Color"], nodes["Background"].inputs["Color"])
        nodes["Background"].inputs["Strength"].default_value = light["sky"]
        scene.world = world
    seen = world.cycles_visibility
    seen.camera = seen.transmission = seen.glossy = True
    seen.diffuse = seen.scatter = False
    world.cycles.sampling_method = "NONE"


# ----------------------------------------------------------------------------------------------------------- build

def build(scene, job: dict, job_dir: Path) -> dict:
    """Everything of the job's description as objects of the scene. Returns what was built, for facts.json, with
    the exposure the pictures are to be taken at."""
    spec = job["spec"]
    if spec["lighting"] not in LIGHTING:
        raise ValueError(f"unknown lighting preset {spec['lighting']!r}: expected one of {', '.join(LIGHTING)}")
    light = LIGHTING[spec["lighting"]]
    scene.cycles.diffuse_bounces = int(light["bounces"])
    open_assets(job.get("prepared"))
    set_world(scene, spec.get("hdri"), light)
    built = add_room(scene, spec["room"], light)
    pictures, segments = sign_pictures(job.get("signs", {})), int(job["settings"]["segments"])
    for number, part in enumerate(spec["parts"]):
        add_part(scene, part, f"part_{number:03d}", pictures, Path(job_dir), segments)
    for model in spec["models"]:
        add_model(scene, model)
    for person in spec["people"]:
        add_person(scene, person, job.get("heights", {}))
    bpy.context.view_layer.update()
    return {**built, "materials": "plain" if _library is None else "library", "objects": len(scene.objects),
            "parts": len(spec["parts"]), "models": len(spec["models"]), "people": len(spec["people"]),
            "lighting": spec["lighting"], "exposure": EXPOSURE}
