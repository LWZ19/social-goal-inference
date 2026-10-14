"""The prepared things of the realistic set, inside Blender: models, materials and the world behind the windows.

prepare_models.py leaves, for each model, material and HDRI of the manifest, a receipt
`<prepared>/<class>/<id>.json` beside what it made. Everything here is read from those receipts, so the
prepared folder is all this module needs: it never opens the manifest or a download.

    open_library(prepared)         take the assets from this folder from now on; once per scene, before the rest
    add_model(asset)               a model as one mesh object: life size in metres, the middle of its base at the
                                   origin, its front along +X, no parent and no children
    add_person(avatar, pose, prop) a person, from people_library.py beside this file when that is there
    material(name, tint)           a material of the library, for a mesh whose scale is 1
    set_world(hdri, strength)      what shows through an opening and in reflections

A material is laid on a surface by the object's own coordinates, in metres, so it needs no UV map and has the
same grain on every object; an object that is scaled would stretch it, so a part's size belongs in its mesh.
A texture set is box-projected at the real size of its tile. `tint` is a colour as a scene writes one, three
numbers from 0 to 255, and multiplies the base colour. A material that is made in code (painted metal,
plastic, laminate, paper, ceramic) or that the manifest calls neutral (wall paint, fabric, stone, the terrazzo
floor) is white before the tint, so the tint is the colour it comes out in, and without a tint it takes the
colour the manifest gives it: a neutral texture keeps its pattern, lighter or darker as the colour asks. Any
other material has a colour of its own, which a tint only shifts.

A model brought twice shares one mesh, and a material asked for twice is one material: both are found again
by name, so a scene thrown away between two calls costs nothing but the bringing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy

_HERE = str(Path(__file__).resolve().parent)       # Blender puts no script's folder on the path
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
try:
    import people_library as _people               # the people's half, written beside this one
    _no_people = None
except Exception as error:                         # not written yet, or not loading: the things are still served
    _people, _no_people = None, error

MARK = "sgi_library"                               # on each datablock made here: what it is, to find it again
_SKY = (0.42, 0.55, 0.78)                          # the world without an HDRI: a plain clear sky, scene-linear
_prepared: Path | None = None


def open_library(prepared: str) -> None:
    """Take models, materials, HDRIs and people from this prepared folder from now on."""
    global _prepared
    _prepared = Path(prepared)
    if _people is not None:
        _people.open_people(str(prepared))


def add_person(avatar: str, pose: str, prop: str | None = None) -> "bpy.types.Object":
    """A person of the library as a static mesh at its natural height, the ground point under its pelvis at
    the origin, chest facing +X; `prop` is None, "phone" or "cup". people_library.py has the rest."""
    if _people is None:
        raise RuntimeError(f"people_library.py in {_HERE} could not be loaded ({_no_people!r}): this library has no "
                           "people")
    return _people.add_person(avatar, pose, prop)


def _receipt(name: str, asset: str) -> dict:
    """What the preparation recorded of one asset."""
    if _prepared is None:
        raise RuntimeError("open_library(prepared) has to be called first")
    path = _prepared / name / f"{asset}.json"
    if not path.is_file():
        known = sorted(found.stem for found in (_prepared / name).glob("*.json"))
        raise ValueError(f"nothing is prepared as {asset!r} among the {name} of {_prepared}: there are "
                         f"{', '.join(known) or 'none'}")
    return json.loads(path.read_text(encoding="utf-8"))


def _found(blocks, mark: str):
    """The datablock made here under a mark, if it is still in the file."""
    return next((block for block in blocks if block.get(MARK) == mark), None)


# ------------------------------------------------------------------------------------------------ models
def add_model(asset: str) -> "bpy.types.Object":
    """A model of the library as a new object of the current scene, at the origin: one mesh, life size in
    metres, the middle of its base at the origin, its front along +X. It has no parent and no children, so
    it is placed, scaled and hidden as one thing. ValueError for a model that is not prepared."""
    facts = _receipt("models", asset)
    mark = f"model:{asset}"
    mesh = _found(bpy.data.meshes, mark)
    if mesh is None:
        with bpy.data.libraries.load(str(_prepared / facts["blend"]), link=False) as (there, wanted):
            if asset not in there.meshes:
                raise ValueError(f"{facts['blend']} holds no mesh {asset!r}")
            wanted.meshes = [asset]                # its materials and their packed images come with it
        mesh = wanted.meshes[0]
        mesh[MARK] = mark
    made = bpy.data.objects.new(asset, mesh)
    made[MARK] = mark
    bpy.context.scene.collection.objects.link(made)
    bpy.context.view_layer.update()                # matrix_world of a new object is stale until this runs
    return made


# --------------------------------------------------------------------------------------------- materials
def linear(colour) -> tuple[float, float, float]:
    """A colour of 0 to 255 a channel, as a picture file holds it, in the scene-linear values a shader takes."""
    def one(value: float) -> float:
        value = min(max(float(value) / 255.0, 0.0), 1.0)
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    return one(colour[0]), one(colour[1]), one(colour[2])


def _noise(tree, scale: float, detail: float):
    coordinates, noise = tree.nodes.new("ShaderNodeTexCoord"), tree.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value, noise.inputs["Detail"].default_value = scale, detail
    tree.links.new(coordinates.outputs["Object"], noise.inputs["Vector"])
    return noise.outputs["Fac"]


def _finish(tree, shader, roughness: float, bump_m: float, grain: float) -> None:
    """Roughness that wanders a little and a fine bump: what keeps a face of one colour from reading as a flat
    fill. `grain` is how many times the wandering repeats in a metre."""
    spread = tree.nodes.new("ShaderNodeMapRange")
    for name, value in (("From Min", 0.3), ("From Max", 0.7), ("To Min", max(roughness - 0.07, 0.02)),
                        ("To Max", min(roughness + 0.07, 1.0))):
        spread.inputs[name].default_value = value
    tree.links.new(_noise(tree, grain, 4.0), spread.inputs["Value"])
    tree.links.new(spread.outputs["Result"], shader.inputs["Roughness"])
    if bump_m > 0:
        bump = tree.nodes.new("ShaderNodeBump")
        bump.inputs["Distance"].default_value = bump_m
        tree.links.new(_noise(tree, grain * 6.0, 2.0), bump.inputs["Height"])
        tree.links.new(bump.outputs["Normal"], shader.inputs["Normal"])


def window_glass(name: str, colour=(0.991, 1.0, 0.991), ior: float = 1.45) -> "bpy.types.Material":
    """Glass as a building or a display case has it: glass to the camera and to glossy rays, plainly clear to
    shadow and diffuse rays, so light comes through a pane without caustic noise and what stands behind it is
    lit. Cycles still writes it into the depth pass as a surface. `colour` is scene-linear."""
    made = bpy.data.materials.new(name)
    made.use_nodes = True
    tree = made.node_tree
    tree.nodes.clear()
    glass, clear = tree.nodes.new("ShaderNodeBsdfGlass"), tree.nodes.new("ShaderNodeBsdfTransparent")
    glass.inputs["Color"].default_value = clear.inputs["Color"].default_value = (*colour, 1.0)
    glass.inputs["Roughness"].default_value, glass.inputs["IOR"].default_value = 0.0, ior
    path, either = tree.nodes.new("ShaderNodeLightPath"), tree.nodes.new("ShaderNodeMath")
    either.operation = "MAXIMUM"
    tree.links.new(path.outputs["Is Shadow Ray"], either.inputs[0])
    tree.links.new(path.outputs["Is Diffuse Ray"], either.inputs[1])
    mix, out = tree.nodes.new("ShaderNodeMixShader"), tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(either.outputs[0], mix.inputs["Fac"])
    tree.links.new(glass.outputs["BSDF"], mix.inputs[1])
    tree.links.new(clear.outputs["BSDF"], mix.inputs[2])
    tree.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return made


def _texture_set(name: str, facts: dict, tint) -> "bpy.types.Material":
    """A texture set on any mesh: the object's coordinates, divided by the real size of one tile, into
    box-projected images. Relief comes from the height map through a Bump node, which needs no tangents; a
    tangent-space normal map would need a UV map."""
    given, maps = facts.get("prepare", {}), facts["maps"]
    made = bpy.data.materials.new(name)
    made.use_nodes = True
    tree = made.node_tree
    shader = tree.nodes["Principled BSDF"]
    coordinates, mapping = tree.nodes.new("ShaderNodeTexCoord"), tree.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (1.0 / facts["tile_m"],) * 3
    tree.links.new(coordinates.outputs["Object"], mapping.inputs["Vector"])

    def image(role: str, colour: bool):
        node = tree.nodes.new("ShaderNodeTexImage")
        node.image = bpy.data.images.load(str(_prepared / maps[role]), check_existing=True)
        node.image.colorspace_settings.name = "sRGB" if colour else "Non-Color"
        node.projection, node.projection_blend = "BOX", float(given.get("blend", 0.0))
        tree.links.new(mapping.outputs["Vector"], node.inputs["Vector"])
        return node.outputs["Color"]

    factor = None                                  # what the colour map is multiplied by, channel by channel
    if given.get("neutral"):                       # white on the whole, then the colour asked for or its own
        wanted = linear(tint if tint is not None else given.get("colour", (255, 255, 255)))
        factor = tuple(want / max(mean, 1e-4) for want, mean in zip(wanted, facts["mean_linear"]))
    elif tint is not None:
        factor = linear(tint)
    shade = image("color", True)
    if factor is not None:
        multiply = tree.nodes.new("ShaderNodeMix")
        multiply.data_type, multiply.blend_type = "RGBA", "MULTIPLY"
        # A pale colour asked of a dark texture is a factor far above 1, and the texture's lightest grains would
        # come out lighter than white, which no surface is.
        multiply.clamp_result = True
        multiply.inputs[0].default_value = 1.0     # its factor; the colour sockets of a Mix node are 6 and 7, out 2
        tree.links.new(shade, multiply.inputs[6])
        multiply.inputs[7].default_value = (*factor, 1.0)
        shade = multiply.outputs[2]
    tree.links.new(shade, shader.inputs["Base Color"])
    if "roughness" in maps:
        rough = image("roughness", False)
        if given.get("roughness_scale", 1.0) != 1.0:
            scaled = tree.nodes.new("ShaderNodeMath")
            scaled.operation, scaled.use_clamp = "MULTIPLY", True
            scaled.inputs[1].default_value = float(given["roughness_scale"])
            tree.links.new(rough, scaled.inputs[0])
            rough = scaled.outputs[0]
        tree.links.new(rough, shader.inputs["Roughness"])
    if "metallic" in maps:
        tree.links.new(image("metallic", False), shader.inputs["Metallic"])
    if "height" in maps and given.get("bump_m", 0.001) > 0:
        bump = tree.nodes.new("ShaderNodeBump")
        bump.inputs["Distance"].default_value = float(given.get("bump_m", 0.001))
        tree.links.new(image("height", False), bump.inputs["Height"])
        tree.links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    return made


def _plain(name: str, facts: dict, tint) -> "bpy.types.Material":
    """A painted, laminated, moulded or glazed surface of one colour, made in code: there is no texture to
    repeat, only roughness that wanders and a fine bump."""
    given = facts.get("prepare", {})
    made = bpy.data.materials.new(name)
    made.use_nodes = True
    shader = made.node_tree.nodes["Principled BSDF"]
    colour = tint if tint is not None else given.get("colour", (200, 200, 200))
    shader.inputs["Base Color"].default_value = (*linear(colour), 1.0)
    shader.inputs["Metallic"].default_value = float(given.get("metallic", 0.0))
    shader.inputs["Coat Weight"].default_value = float(given.get("coat", 0.0))
    _finish(made.node_tree, shader, float(given.get("roughness", 0.45)), float(given.get("bump_m", 0.0003)),
            float(given.get("grain", 9.0)))
    return made


def _glass(name: str, facts: dict, tint) -> "bpy.types.Material":
    colour = linear(facts.get("prepare", {}).get("colour", (245, 251, 249)))
    if tint is not None:
        colour = tuple(own * by for own, by in zip(colour, linear(tint)))
    return window_glass(name, colour, float(facts.get("prepare", {}).get("ior", 1.45)))


def _screen(name: str, facts: dict, tint) -> "bpy.types.Material":
    """A display that is on and shows nothing: dark glossy glass that glows a dim grey. What a screen shows is
    a sign on its face, drawn over this by whoever builds the part."""
    given = facts.get("prepare", {})
    made = bpy.data.materials.new(name)
    made.use_nodes = True
    shader = made.node_tree.nodes["Principled BSDF"]
    glow = linear(given.get("glow", (96, 100, 106)))
    if tint is not None:
        glow = tuple(own * by for own, by in zip(glow, linear(tint)))
    shader.inputs["Base Color"].default_value = (*linear(given.get("colour", (28, 34, 44))), 1.0)
    shader.inputs["Roughness"].default_value = float(given.get("roughness", 0.15))
    shader.inputs["Emission Color"].default_value = (*glow, 1.0)
    shader.inputs["Emission Strength"].default_value = float(given.get("emission", 1.0))
    return made


_KINDS = {"pbr": _texture_set, "plain": _plain, "glass": _glass, "screen": _screen}


def material(name: str, tint: tuple[int, int, int] | None = None) -> "bpy.types.Material":
    """A material of the library by its name, for a mesh whose object has scale 1; `tint`, three numbers from
    0 to 255, multiplies its base colour (see the top of this file for which materials it colours outright).
    The same name and tint give the same material again. ValueError for a name the library has not."""
    mark = "material:" + name + ("" if tint is None else "#%02x%02x%02x" % tuple(int(value) for value in tint[:3]))
    made = _found(bpy.data.materials, mark)
    if made is None:
        facts = _receipt("materials", name)
        if facts.get("kind") not in _KINDS:
            raise ValueError(f"the material {name!r} is of the kind {facts.get('kind')!r}: expected one of "
                             + ", ".join(_KINDS))
        made = _KINDS[facts["kind"]](mark[len("material:"):], facts, tint)
        made[MARK] = mark
    return made


# ------------------------------------------------------------------------------------------------- world
def set_world(hdri: str | None, strength: float) -> None:
    """Give the current scene its world: an HDRI of the library, or with None a plain clear sky, at `strength`.

    An HDRI is shown at `strength` times the factor its manifest entry gives it (prepare.strength, 1 where it
    gives none): the skies were photographed at exposures of their own, and at one strength for all of them
    one is a street on a bright day and the next a white board.

    The world is what an opening shows and what glossy surfaces reflect. It lights nothing: it is hidden
    from diffuse rays, because the rooms are lit by their fittings, and daylight through a window would be
    the noisiest light in the picture. Whoever wants it to light sets scene.world.cycles_visibility.diffuse."""
    scene = bpy.context.scene
    facts = None if hdri is None else _receipt("hdris", hdri)                         # before anything is made
    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    tree = world.node_tree
    background = tree.nodes["Background"]
    if facts is None:
        background.inputs["Strength"].default_value = float(strength)
        background.inputs["Color"].default_value = (*_SKY, 1.0)
    else:
        background.inputs["Strength"].default_value = float(strength) * float(facts.get("strength", 1.0))
        picture = tree.nodes.new("ShaderNodeTexEnvironment")
        picture.image = bpy.data.images.load(str(_prepared / facts["file"]), check_existing=True)
        tree.links.new(picture.outputs["Color"], background.inputs["Color"])
    world[MARK] = f"world:{hdri}"
    old, scene.world = scene.world, world          # the scene's world first: Cycles tags it at each setting below
    seen = world.cycles_visibility
    seen.camera = seen.glossy = seen.transmission = True
    seen.diffuse = seen.scatter = False
    world.cycles.sampling_method = "NONE"
    if old is not None and old.get(MARK) and old.users == 0:
        bpy.data.worlds.remove(old)
