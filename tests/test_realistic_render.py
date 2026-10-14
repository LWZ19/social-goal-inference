"""The Blender renderer: a scene description built and drawn by Blender, measured against the numpy ray caster.

The tests marked needs_blender start the checkout's Blender and are skipped where there is none. They place
no model and no person of the asset library, and without prepared assets the Blender side paints a named
material plainly, so they run on a machine that has fetched Blender alone; with prepared assets the same
tests draw the library's materials. One of them places a model and a person through a stand-in for the
library, written here. Those marked needs_assets draw what the asset library itself holds and are skipped
until the assets are prepared: every model in the box its size names, every pose at the height asked for
with what it holds, people over their ground points and facing their yaw as the scene check sees them, every
material and every HDRI, and signs lettered by the sign module in the library's font. The other tests put a
function of their own in the place of the run in Blender.
"""
import ast
import dataclasses
import json
import math
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sgi.config import load_config
from sgi.geometry import Camera, RobotPose, camera_on_robot, ground_hits, in_image, project
from sgi.gt.derive import is_visible
from sgi.inputs import ObservationData
from sgi.scene import Person
from sgi.synth import realistic
from sgi.synth.realistic import blender, blender_renderer, signs
from sgi.synth.realistic.blender import BLENDER_SIDE, BlenderMissing
from sgi.synth.realistic.blender_renderer import LIGHTING, SIGN_FONT, BlenderRenderer
from sgi.synth.realistic.library import AvatarAsset, Library, load_library
from sgi.synth.realistic.spec import (PAINT, Frame, ModelRef, Opening, Part, PersonRef, RoomSpec, SceneSpec, Sign,
                                      SignArt, to_json)
from sgi.synth.realistic.stub import StubRenderer
from sgi.synth.render import Box, Cylinder, Sphere, render

needs_blender = pytest.mark.skipif(not blender.binary_available(), reason="no Blender of the 4.5 series here")

LENS = dict(fx=554.3, fy=554.3, cx=320.0, cy=240.0, width=640, height=480)
MOUNT = dict(forward_m=0.2, height_m=1.5, pitch_down_rad=math.radians(10.0))


def camera_at(x: float = 0.0, y: float = 0.0, yaw: float = 0.0, **lens) -> Camera:
    return Camera(**{**LENS, **lens}, T_world_camera=camera_on_robot(RobotPose(x, y, yaw), **MOUNT))


CAMERA = camera_at()
SMALL = dict(fx=277.15, fy=277.15, cx=160.0, cy=120.0, width=320, height=240)      # the same view at half the size
# A lens with pixels that are not square and a centre off the middle, from another place: what a lens shift, a
# pixel aspect or a pose got wrong would show here and not through CAMERA.
ASKEW = camera_at(1.0, -1.5, 0.6, fx=600.0, fy=520.0, cx=250.3, cy=301.7)
# CAMERA with its image grown by whole pixels, as the generator grows one to hold all of a target.
WIDENED = camera_at(cx=520.0, cy=340.0, width=990, height=630)

ONE_MM = 1e-3
SKIRTING_TOP_M, RAIL_HALF_M, TRIM_DEPTH_M = 0.15, 0.05, 0.02      # what the room may add to a wall, and where


def turned(cx: float, cy: float, yaw: float, sx: float, sy: float) -> tuple[tuple[float, float], ...]:
    c, s = math.cos(yaw), math.sin(yaw)
    return tuple((cx + c * x - s * y, cy + s * x + c * y)
                 for x, y in ((-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2)))


# A room that no axis of the world runs along, and one with a corner that points into it, given clockwise.
TURNED_ROOM = RoomSpec(polygon=turned(3.5, 0.0, 0.3, 11.0, 9.0), height=2.8, floor="floor_tile",
                       ceiling="ceiling_tile", wall=(214, 222, 228), dado=(150, 170, 180), dado_height=1.2, lights=2.0)
L_ROOM = RoomSpec(polygon=((-2.0, 4.0), (7.4, 4.0), (7.4, 2.2), (9.0, 2.2), (9.0, -4.0), (-2.0, -4.0)), height=3.0,
                  floor="floor_terrazzo", ceiling="ceiling_tile", wall=(233, 227, 213), dado=None, dado_height=0.95,
                  lights=2.4)
PLAIN_ROOM = RoomSpec(polygon=((-2.0, -4.0), (8.0, -4.0), (8.0, 4.0), (-2.0, 4.0)), height=2.8, floor="floor_tile",
                      ceiling="ceiling_tile", wall=(214, 222, 228), dado=(150, 170, 180), dado_height=1.2, lights=2.0)


def a_dozen_parts() -> tuple[Part, ...]:
    """Boxes, cylinders and spheres with sharp edges: turned, raised, tapered, of glass, and in four groups."""
    sharp = dict(bevel=0.0)
    return (
        Part(Box(4.0, -1.6, 0.3, 0.6, 1.2, 1.1, color=(150, 110, 70)), "wood", group=1, **sharp),
        Part(Box(4.0, -1.6, 0.3, 0.66, 1.26, 0.04, z0=1.1, color=(60, 60, 64)), "stone", group=1, **sharp),
        Part(Box(5.5, 1.2, -0.7, 0.5, 0.5, 2.0, color=(200, 60, 50), top=(240, 240, 240)), PAINT, group=2, **sharp),
        Part(Box(6.5, -0.4, 1.3, 1.6, 0.05, 0.9, z0=1.0), "glass", glass=True, **sharp),
        Part(Box(3.2, 1.6, 0.0, 0.4, 0.4, 0.45, color=(40, 90, 160)), PAINT, **sharp),
        Part(Box(6.8, 2.6, 0.5, 0.3, 1.0, 0.6, z0=0.9, color=(90, 90, 90)), "painted_metal", **sharp),
        Part(Cylinder(4.6, 0.2, 0.25, 1.2, color=(170, 174, 180)), "steel", group=3, **sharp),
        Part(Cylinder(5.6, -2.4, 0.3, 2.6, color=(230, 228, 220)), PAINT, **sharp),
        Part(Cylinder(3.4, -0.5, 0.2, 0.5, color=(120, 90, 60), top=(60, 120, 60), top_radius=0.28), PAINT, group=4,
             **sharp),
        Part(Cylinder(6.2, 1.9, 0.18, 0.7, z0=0.3, top_radius=0.08), "plastic_dark", **sharp),
        Part(Sphere(3.4, -0.5, 0.78, 0.26, color=(40, 140, 60)), PAINT, group=4, **sharp),
        Part(Sphere(5.0, 0.9, 0.3, 0.3, color=(220, 200, 40)), PAINT, **sharp),
        Part(Sphere(6.0, -1.2, 1.9, 0.22, color=(250, 250, 250)), "plastic_light", **sharp))


def blender_draws(spec: SceneSpec, frames, **options) -> list:
    return BlenderRenderer(**options).render(spec, frames)


def depth_only(*cameras: Camera, only=None) -> list[Frame]:
    return [Frame(camera, beauty=False, only=only) for camera in cameras]


def grown(mask: np.ndarray, reach: int = 1) -> np.ndarray:
    """The mask, and every pixel within `reach` pixels of it, level, upright or on the slant."""
    rows, cols = mask.shape
    for _ in range(reach):
        padded = np.pad(mask, 1)
        mask = np.logical_or.reduce([padded[i:i + rows, j:j + cols] for i in range(3) for j in range(3)])
    return mask


def near_an_edge(depth: np.ndarray, reach: int = 2, bend: float = 1e-4) -> np.ndarray:
    """The pixels within `reach` pixels of a step or a crease of the depth image.

    Across a flat surface 1 / depth runs straight from pixel to pixel, whatever the surface's slant, so where it
    bends by more than `bend` per metre between a pixel and its two neighbours one surface ends: at a silhouette,
    at a corner, or where a round thing turns away from the eye. A pixel that nothing is seen in counts as
    endlessly far."""
    rows, cols = depth.shape
    inverse = np.divide(1.0, depth, out=np.zeros(depth.shape), where=depth > 0)
    edge = np.zeros(depth.shape, dtype=bool)
    down = np.abs(inverse[:-2] - 2 * inverse[1:-1] + inverse[2:]) > bend
    across = np.abs(inverse[:, :-2] - 2 * inverse[:, 1:-1] + inverse[:, 2:]) > bend
    for shift in range(3):                         # the pixel in the middle, and the one on each side of it
        edge[shift:rows - 2 + shift] |= down
        edge[:, shift:cols - 2 + shift] |= across
    return grown(edge, reach)


def world_points(depth: np.ndarray, camera: Camera) -> np.ndarray:
    """The world point seen in every pixel, (H, W, 3); the camera's own place where nothing is seen."""
    v, u = np.meshgrid(np.arange(camera.height) + 0.5, np.arange(camera.width) + 0.5, indexing="ij")
    z = depth.astype(np.float64)
    right, down = (u - camera.cx) * z / camera.fx, (v - camera.cy) * z / camera.fy
    return (np.stack([right, down, z, np.ones_like(z)], axis=-1) @ camera.T_world_camera.T)[..., :3]


def from_a_wall(points: np.ndarray, room: RoomSpec) -> np.ndarray:
    """How far each point is, level, from the nearest wall of the room."""
    corners = np.asarray(room.polygon, dtype=np.float64)
    nearest = np.full(points.shape[:-1], np.inf)
    for a, b in zip(corners, np.roll(corners, -1, axis=0)):
        along = np.clip(((points[..., :2] - a) @ (b - a)) / ((b - a) @ (b - a)), 0.0, 1.0)
        nearest = np.minimum(nearest, np.linalg.norm(points[..., :2] - a - along[..., None] * (b - a), axis=-1))
    return nearest


def on_a_trim(depth: np.ndarray, camera: Camera, room: RoomSpec) -> np.ndarray:
    """The pixels that show a skirting board or a dado rail: no deeper than 2 cm, at the foot of a wall and at the
    height where its two colours meet. The ray caster's wall is bare there."""
    points = world_points(depth, camera)
    height = points[..., 2]
    banded = height < SKIRTING_TOP_M
    if room.dado is not None:
        banded |= np.abs(height - room.dado_height) < RAIL_HALF_M
    return (depth > 0) & banded & (from_a_wall(points, room) < TRIM_DEPTH_M + ONE_MM)


def depth_at(depth: np.ndarray, camera: Camera, point) -> float:
    u, v, _ = project(camera, point)
    assert in_image(camera, u, v)
    return float(depth[math.floor(v), math.floor(u)])


def colour_at(picture: np.ndarray, camera: Camera, point, reach: int = 1) -> np.ndarray:
    """The mean colour of the few pixels round the one a world point projects into."""
    u, v, _ = project(camera, point)
    row, col = math.floor(v), math.floor(u)
    return picture[row - reach:row + reach + 1, col - reach:col + reach + 1].reshape(-1, 3).mean(axis=0)


def camera_z(camera: Camera, point) -> float:
    return project(camera, point)[2]


# ---------------------------------------------------------------------------------------------- depth, in Blender

@needs_blender
@pytest.mark.parametrize("room", [TURNED_ROOM, L_ROOM], ids=["a turned room", "an L-shaped room"])
def test_depth_matches_the_numpy_ray_caster_away_from_silhouettes(room):
    spec = SceneSpec(room, parts=a_dozen_parts(), seed=3)
    cameras = (CAMERA, ASKEW, WIDENED)
    drawn = blender_draws(spec, depth_only(*cameras))
    cast = StubRenderer().render(spec, depth_only(*cameras))
    for camera, (picture, ours), (_, theirs) in zip(cameras, drawn, cast, strict=True):
        assert picture is None and ours.dtype == np.float32 and ours.shape == (camera.height, camera.width)
        assert np.array_equal(ours > 0, theirs > 0) and (ours > 0).all()     # no hit by one that the other misses
        trim = on_a_trim(ours, camera, room)
        compared = ~near_an_edge(theirs) & ~trim
        assert compared.mean() > 0.75 and trim.mean() < 0.05                 # most of the picture is compared
        apart = np.abs(ours.astype(np.float64) - theirs)
        assert apart[compared].max() < ONE_MM
        assert np.median(apart[compared]) < 2e-5
        # A trim stands before its wall and never behind it, and the walls themselves are on the room's outline:
        # a wall 5 cm inside it would show in every pixel of it above.
        assert (theirs - ours)[trim & ~near_an_edge(theirs)].min() > -ONE_MM


@needs_blender
def test_depth_is_plane_depth_not_ray_length():
    room = RoomSpec(polygon=((-3.0, 6.0), (15.0, 6.0), (15.0, -6.0), (-3.0, -6.0)), height=3.2, floor="floor_laminate",
                    ceiling="ceiling_tile", wall=(233, 227, 213), dado=None, dado_height=0.95, lights=2.4)
    cameras = (CAMERA, ASKEW)
    drawn = blender_draws(SceneSpec(room), depth_only(*cameras))
    for camera, (_, depth) in zip(cameras, drawn, strict=True):
        xy, plane_z = ground_hits(camera)                                   # where each pixel's ray meets z = 0
        points = np.dstack([np.nan_to_num(xy, nan=1e6), np.zeros(depth.shape)])
        inside = (xy[..., 0] > -3.0) & (xy[..., 0] < 15.0) & (np.abs(xy[..., 1]) < 6.0)
        floor = np.isfinite(plane_z) & inside & (from_a_wall(points, room) > 0.05)   # clear of the skirting boards
        assert floor.mean() > 0.3
        assert np.abs(depth[floor] - plane_z[floor]).max() < 0.2e-3
        # The length of the ray is longer than the depth by a part that grows away from the middle of the picture.
        v, u = np.nonzero(floor)
        ray = plane_z[floor] * np.sqrt(1.0 + ((u + 0.5 - camera.cx) / camera.fx) ** 2
                                       + ((v + 0.5 - camera.cy) / camera.fy) ** 2)
        assert np.abs(depth[floor] - ray).max() > 0.5


@needs_blender
def test_only_draws_the_named_groups_without_the_room():
    spec = SceneSpec(TURNED_ROOM, parts=a_dozen_parts())
    frames = [Frame(CAMERA, beauty=False), Frame(CAMERA, beauty=False, only=(1,)),
              Frame(CAMERA, beauty=False, only=(4, 3)), Frame(WIDENED, beauty=False, only=(1, 2)),
              Frame(CAMERA, beauty=False, only=()), Frame(CAMERA, beauty=False), Frame(CAMERA, only=(1,))]
    ((_, whole), (_, boxes), (_, round_things), (_, wide), (_, nothing), (_, whole_again),
     (cut_out, pictured)) = blender_draws(spec, frames, samples=8)
    cast = [depth for _, depth in StubRenderer().render(spec, frames)]
    assert not nothing.any()
    assert np.array_equal(whole_again, whole)                               # showing all again gives the first image
    assert cut_out.shape == (480, 640, 3) and np.array_equal(pictured, boxes)   # a picture of some groups alone, too
    assert (boxes > 0).sum() > 5000 and (round_things > 0).sum() > 3000 and (wide > 0).sum() > 10000
    # Flat things have the ray caster's outline to the pixel. A round one is a prism of 128 sides inside the round
    # shape, which passes it by less than a hundredth of a pixel: here and there along the outline a pixel centre
    # falls between the two, and nowhere else do they differ.
    assert np.array_equal(boxes > 0, cast[1] > 0) and np.array_equal(wide > 0, cast[3] > 0)
    theirs = cast[2] > 0
    outline = theirs & grown(~theirs)                                       # their pixels beside one that is not theirs
    missed = theirs & (round_things == 0)
    assert not ((round_things > 0) & ~theirs).any() and not (missed & ~outline).any()
    assert outline.sum() > 500 and missed.sum() < 0.02 * outline.sum()
    for alone, theirs in ((boxes, cast[1]), (round_things, cast[2]), (wide, cast[3])):
        compared = (alone > 0) & (theirs > 0) & ~near_an_edge(theirs)
        assert compared.sum() > 0.6 * (theirs > 0).sum()
        assert np.abs(alone - theirs)[compared].max() < ONE_MM
    in_view = np.s_[100:580, 200:840]                                       # CAMERA's own pixels in the grown image
    for alone in (boxes, round_things, wide[in_view]):                      # what stands free shows as in the whole
        free = (alone > 0) & (np.abs(cast[0] - np.where(alone > 0, alone, -1.0)) < ONE_MM)
        assert free.sum() > 3000 and np.abs(alone - whole)[free].max() < 1e-5


@needs_blender
def test_two_runs_give_the_same_depth():
    spec = SceneSpec(TURNED_ROOM, parts=a_dozen_parts(), seed=5)
    frames = depth_only(CAMERA, ASKEW)
    first, second = blender_draws(spec, frames), blender_draws(spec, frames)
    for (_, a), (_, b) in zip(first, second, strict=True):
        assert np.array_equal(a, b)


@needs_blender
def test_one_process_renders_all_frames(monkeypatch):
    runs = []

    def counted(script, args, **options):
        runs.append(Path(script))
        return blender.run_script(script, args, **options)

    monkeypatch.setattr(blender_renderer, "run_script", counted)
    spec = SceneSpec(PLAIN_ROOM, parts=a_dozen_parts())
    cameras = [camera_at(1.0, 0.0, turn * math.tau / 9, **SMALL) for turn in range(8)] + [CAMERA]
    frames = [Frame(camera, beauty=number % 3 == 0) for number, camera in enumerate(cameras)]
    renderer = BlenderRenderer(samples=8)
    drawn = renderer.render(spec, frames)
    assert runs == [BLENDER_SIDE / "job.py"] and len(drawn) == 9            # nine frames, one start of Blender
    cast = StubRenderer().render(spec, depth_only(*cameras))
    for number, (camera, (picture, depth), (_, theirs)) in enumerate(zip(cameras, drawn, cast, strict=True)):
        assert depth.shape == (camera.height, camera.width) and depth.dtype == np.float32
        if number % 3 == 0:                                                 # a picture where one was asked for
            assert picture.shape == (camera.height, camera.width, 3) and picture.dtype == np.uint8
            assert 40 < picture.mean() < 230                                # of a lit room: not black, not burnt out
        else:
            assert picture is None
        compared = ~near_an_edge(theirs) & ~on_a_trim(depth, camera, PLAIN_ROOM)
        assert np.abs(depth - theirs)[compared].max() < ONE_MM              # each through its own camera, in order
    facts = renderer.last_facts
    assert facts["blender"] == renderer.version and facts["device"] in ("OPTIX", "CUDA", "CPU")
    assert len(facts["seconds"]["frames"]) == 9 and all(seconds > 0 for seconds in facts["seconds"]["frames"])


@needs_blender
def test_without_a_device_of_the_kind_asked_for_the_cpu_draws_the_same_depth(monkeypatch):
    # "NONE" is a kind of device that Cycles knows and that no machine has one of: what a machine without a
    # graphics card answers to OPTIX.
    monkeypatch.setattr(blender_renderer, "DEVICES", (*blender_renderer.DEVICES, "NONE"))
    spec = SceneSpec(PLAIN_ROOM, parts=a_dozen_parts())
    by_default, without, on_the_cpu = BlenderRenderer(), BlenderRenderer(device="NONE"), BlenderRenderer(device="CPU")
    ((_, usual),) = by_default.render(spec, depth_only(CAMERA))
    ((_, fallen_back),) = without.render(spec, depth_only(CAMERA))
    ((_, asked_for),) = on_the_cpu.render(spec, depth_only(CAMERA))
    assert (without.last_facts["device"], on_the_cpu.last_facts["device"]) == ("CPU", "CPU")
    assert np.array_equal(fallen_back, asked_for) and np.array_equal(usual > 0, fallen_back > 0)
    assert np.abs(usual - fallen_back)[~near_an_edge(usual)].max() < 1e-4


@needs_blender
def test_a_bevel_rounds_the_edges_and_leaves_the_faces_where_they_are():
    crate = Box(4.0, 0.0, 0.5, 0.8, 0.8, 0.8, color=(150, 110, 70))
    drum = Cylinder(4.0, 1.4, 0.4, 0.8, color=(90, 104, 120))
    parts = (Part(crate, PAINT, group=1, bevel=0.03), Part(drum, PAINT, group=2, bevel=0.03))
    spec = SceneSpec(PLAIN_ROOM, parts=parts)
    frames = depth_only(CAMERA, only=(1,)) + depth_only(CAMERA, only=(2,))
    (_, box_alone), (_, drum_alone) = blender_draws(spec, frames)
    (_, cast_box), (_, cast_drum) = StubRenderer().render(spec, frames)       # which draws both with sharp edges
    for alone, theirs in ((box_alone, cast_box), (drum_alone, cast_drum)):
        both = (alone > 0) & (theirs > 0)
        assert (alone > 0).sum() > 5000 and not ((alone > 0) & (theirs == 0)).any()   # it stays inside its solid
        assert 0 < ((theirs > 0) & (alone == 0)).sum() < 0.08 * (theirs > 0).sum()    # and its corners are cut back
        behind = (alone - theirs)[both]
        assert behind.min() > -ONE_MM and behind.max() > 0.005              # nowhere in front; rounded off somewhere
        assert (np.abs(behind) < ONE_MM).mean() > 0.8                       # and the faces are where they were:
        assert np.quantile(behind, 0.99) < 0.03                             # all but an edge's rim, by under the bevel


# ------------------------------------------------------------------------------------------- pictures, in Blender

QUARTERS = {"top left": (220, 30, 30), "top right": (30, 200, 30), "bottom left": (30, 30, 220),
            "bottom right": (240, 240, 240)}


def four_quarters(art: SignArt) -> np.ndarray:
    """A test picture that says which way round it is: red and green above, blue and white below."""
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:32, :32], image[:32, 32:] = QUARTERS["top left"], QUARTERS["top right"]
    image[32:, :32], image[32:, 32:] = QUARTERS["bottom left"], QUARTERS["bottom right"]
    return image


def reads_as(colour: np.ndarray) -> str:
    """Which quarter of the test picture a colour of a photograph shows: the white one where the colour is all
    but grey, else the one whose channel leads. A bright light pales a colour and does not change which leads."""
    seen = np.asarray(colour, dtype=np.float64)
    if seen.max() - seen.min() < 0.2 * seen.max():
        return "bottom right"
    return ("top left", "top right", "bottom left")[int(np.argmax(seen))]


@needs_blender
def test_a_sign_is_drawn_on_its_face_the_right_way_round():
    art, drawn = SignArt("label", text="WAY ROUND", width_m=0.8, height_m=0.8), []

    def draw(asked: SignArt) -> np.ndarray:
        drawn.append(asked)
        return four_quarters(asked)

    def cube(y: float, yaw: float, face: str, **fields) -> Part:
        return Part(Box(6.0, y, yaw, 0.8, 0.8, 0.8, z0=0.6, color=(235, 235, 235)), PAINT, bevel=0.0,
                    signs=(Sign(face, art, **fields),))

    # Five cubes in a row, each with another face turned to the camera, which stands at x = 0 and looks along +x:
    # to whoever stands there, left is +y. Two low ones nearer by carry the sign on top, to be read from their front.
    sides = {"front": (-2.2, math.pi), "left": (-1.1, math.pi / 2), "right": (0.0, -math.pi / 2), "back": (1.1, 0.0)}
    half = cube(2.2, math.pi, "front", rect=(0.5, 0.0, 1.0, 1.0))           # the picture on the right half alone
    low = Part(Box(3.0, 1.0, math.pi, 0.8, 0.8, 0.5, color=(235, 235, 235)), PAINT, bevel=0.0,
               signs=(Sign("top", art),))
    lit = dataclasses.replace(low, solid=dataclasses.replace(low.solid, cy=-1.0),
                              signs=(Sign("top", art, emission=4.0),))
    # emission 4.0: the fittings light a cube's top more strongly than they did, so a glow of 2.0 no longer
    # clears the margin below
    spec = SceneSpec(PLAIN_ROOM, parts=(*(cube(y, yaw, face) for face, (y, yaw) in sides.items()), half, low, lit))
    ((picture, _),) = BlenderRenderer(samples=16, draw=draw).render(spec, [Frame(CAMERA)])
    assert drawn == [art]                                                   # one drawing for the seven signs of it
    quarter = {"top left": (0.2, 0.2), "top right": (-0.2, 0.2), "bottom left": (0.2, -0.2),
               "bottom right": (-0.2, -0.2)}                                # to the left (+y) and up, from the middle
    for face, (y, _) in sides.items():
        for name, (left, up) in quarter.items():
            assert reads_as(colour_at(picture, CAMERA, (5.6, y + left, 1.0 + up))) == name, (face, name)
    for name, (left, up) in quarter.items():                                # on a top, up is away from the reader
        assert reads_as(colour_at(picture, CAMERA, (3.0 + up, 1.0 + left, 0.5))) == name, ("top", name)
    for name, (left, up) in quarter.items():                                # squeezed into the right half of the face
        assert reads_as(colour_at(picture, CAMERA, (5.6, 2.2 - 0.2 + left / 2, 1.0 + up))) == name, ("half", name)
    bare = colour_at(picture, CAMERA, (5.6, 2.2 + 0.2, 1.0))                # the left half keeps the cube's own paint
    assert bare.max() - bare.min() < 25 and bare.mean() > 90
    printed = colour_at(picture, CAMERA, (3.0 - 0.2, 1.0 + 0.2, 0.5))       # the same blue, printed and glowing
    glowing = colour_at(picture, CAMERA, (3.0 - 0.2, -1.0 + 0.2, 0.5))
    assert reads_as(printed) == reads_as(glowing) == "bottom left" and glowing.sum() > printed.sum() + 30


@needs_blender
def test_glass_is_seen_through_in_the_picture_and_is_a_surface_in_the_depth():
    red = Part(Box(5.5, 0.0, 0.0, 0.4, 2.0, 1.8, color=(215, 30, 30)), PAINT, bevel=0.0)
    pane = Part(Box(4.0, 0.0, 0.0, 0.01, 0.9, 0.7, z0=0.7), "glass", glass=True, bevel=0.0)
    spec = SceneSpec(PLAIN_ROOM, parts=(red, pane))
    ((picture, depth),) = blender_draws(spec, [Frame(CAMERA)], samples=32)
    ((_, cast),) = StubRenderer().render(spec, depth_only(CAMERA))           # which draws the pane solid
    behind_the_pane, beside_it = (5.3, 0.0, 1.05), (5.3, 0.8, 1.05)          # both on the red thing's near face
    on_the_pane = CAMERA.T_world_camera[:3, 3] + (np.array(behind_the_pane) - CAMERA.T_world_camera[:3, 3]) * (
        (3.995 - CAMERA.T_world_camera[0, 3]) / (5.3 - CAMERA.T_world_camera[0, 3]))
    assert depth_at(depth, CAMERA, behind_the_pane) == pytest.approx(camera_z(CAMERA, on_the_pane), abs=ONE_MM)
    assert depth_at(depth, CAMERA, beside_it) == pytest.approx(camera_z(CAMERA, beside_it), abs=ONE_MM)
    assert camera_z(CAMERA, behind_the_pane) - camera_z(CAMERA, on_the_pane) > 1.0
    compared = ~near_an_edge(cast) & ~on_a_trim(depth, CAMERA, PLAIN_ROOM)
    assert np.abs(depth - cast)[compared].max() < ONE_MM                    # the depth is the ray caster's all over
    through, bare = colour_at(picture, CAMERA, behind_the_pane), colour_at(picture, CAMERA, beside_it)
    for seen in (through, bare):                                            # red, through the glass as beside it
        assert seen[0] > 90 and seen[0] > 2.5 * seen[1] and seen[0] > 2.5 * seen[2]
    # A clear pane: the old tint of (0.92, 0.97, 0.95) let through 0.92 of the red and came out near 0.90.
    assert through[0] > 0.95 * bare[0]


@needs_blender
def test_the_floor_is_darker_next_to_a_tall_box_than_away_from_it_and_the_depth_is_the_same():
    box = Part(Box(4.0, 0.0, 0.0, 0.6, 0.6, 1.8, color=(200, 200, 200)), PAINT, bevel=0.0)
    spec = SceneSpec(PLAIN_ROOM, parts=(box,))
    ((picture, depth),) = blender_draws(spec, [Frame(CAMERA)], samples=64)
    ((_, cast),) = StubRenderer().render(spec, depth_only(CAMERA))
    foot, further, open_floor = (3.66, 0.0, 0.0), (3.3, 0.0, 0.0), (3.3, 1.6, 0.0)     # the face is at x = 3.7
    near, mid, free = (colour_at(picture, CAMERA, point, reach=1).mean() for point in (foot, further, open_floor))
    assert near < 0.8 * mid and near < 0.8 * free, (near, mid, free)       # the contact shade at the box's foot
    compared = ~near_an_edge(cast) & ~on_a_trim(depth, CAMERA, PLAIN_ROOM)
    assert np.abs(depth - cast)[compared].max() < ONE_MM                    # a shade moves no surface


CONTACT_SCRIPT = '''"""Run in Blender: contact_shaded on a coat with a Principled node and on one with none."""
import sys

sys.path.insert(0, {side!r})
import bpy

import build

bpy.ops.wm.read_factory_settings(use_empty=True)
build._materials.clear()
coat = bpy.data.materials.new("coat")
coat.use_nodes = True
coat["sgi_library"] = "material:coat"
bare = bpy.data.materials.new("bare")                         # no node tree: nothing to darken
assert bare.node_tree is None
before = len(bpy.data.materials)
assert build.contact_shaded(bare, 0.5, 0.6) is bare
assert len(bpy.data.materials) == before, "a material with no Principled node left a copy behind"
shaded = build.contact_shaded(coat, 0.5, 0.6)
assert shaded is not coat and shaded.name != coat.name
assert "sgi_library" not in shaded.keys() and "sgi_library" in coat.keys()
assert any(node.type == "AMBIENT_OCCLUSION" for node in shaded.node_tree.nodes)
assert not any(node.type == "AMBIENT_OCCLUSION" for node in coat.node_tree.nodes)
assert build.contact_shaded(coat, 0.5, 0.6) is shaded         # the same coat twice is one material
'''


@needs_blender
def test_contact_shading_leaves_a_coat_without_a_principled_node_as_it_is_and_copies_one_with(tmp_path):
    script = tmp_path / "contact_script.py"
    script.write_text(CONTACT_SCRIPT.format(side=str(BLENDER_SIDE)), encoding="utf-8")
    blender.run_script(script, [], timeout_s=300)


@needs_blender
def test_a_tint_is_the_colour_of_a_material_that_has_none_of_its_own_and_paint_ignores_it():
    red, pale, blue = (200, 40, 40), (236, 232, 222), (40, 60, 210)

    def block(y: float, material: str, tint=None, colour=(128, 128, 128)) -> Part:
        return Part(Box(4.0, y, math.pi, 0.34, 0.34, 0.5, color=colour), material, bevel=0.004, tint=tint)

    spots = {"metal": -1.0, "red metal": -0.5, "stone": 0.0, "pale stone": 0.5, "paint": 1.0}
    parts = (block(spots["metal"], "painted_metal"), block(spots["red metal"], "painted_metal", red),
             block(spots["stone"], "stone"), block(spots["pale stone"], "stone", pale),
             block(spots["paint"], PAINT, red, colour=blue))
    spec = SceneSpec(PLAIN_ROOM, parts=parts)
    ((picture, depth),) = blender_draws(spec, [Frame(CAMERA)], samples=32)
    ((_, cast),) = StubRenderer().render(spec, depth_only(CAMERA))
    seen = {}
    for name, y in spots.items():                                           # the middle of the face turned to CAMERA
        spot = (4.0 - 0.17, y, 0.25)
        assert depth_at(depth, CAMERA, spot) == pytest.approx(camera_z(CAMERA, spot), abs=0.01), name
        seen[name] = colour_at(picture, CAMERA, spot, reach=4)
    assert seen["red metal"][0] > 1.8 * seen["red metal"][1] and seen["red metal"][0] > 1.8 * seen["red metal"][2]
    assert seen["metal"][0] < 1.2 * seen["metal"][2]                        # as the library has it: no red in it
    assert seen["pale stone"].mean() > seen["stone"].mean() + 40            # a pale pillar and a dark counter top
    assert seen["pale stone"].max() - seen["pale stone"].min() < 40         # pale, and of no hue
    assert seen["paint"][2] > 1.8 * seen["paint"][0]                        # the solid's own blue, whatever the tint
    compared = ~near_an_edge(cast) & ~on_a_trim(depth, CAMERA, PLAIN_ROOM)
    assert np.abs(depth - cast)[compared].max() < ONE_MM                    # a tint moves no surface


@needs_blender
def test_a_window_is_a_hole_closed_by_glass_and_a_door_stays_within_5_cm_of_its_wall():
    window = Opening("window", x=8.0, y=1.6, yaw=math.pi, width=1.8, z0=0.9, height=1.3)
    door = Opening("door", x=8.0, y=-1.8, yaw=math.pi, width=0.9, z0=0.0, height=2.1)
    room = dataclasses.replace(PLAIN_ROOM, dado=None, openings=(window, door))
    bare = dataclasses.replace(room, openings=())
    camera = camera_at(2.0, 0.0, 0.0)                                       # 5.8 m from the wall with the two of them
    ((picture, depth),) = blender_draws(SceneSpec(room), [Frame(camera)], samples=32)
    ((_, wall),) = StubRenderer().render(SceneSpec(bare), depth_only(camera))    # the wall whole, as the stub has it
    assert (depth > 0).all()                                                # the glass closes the hole
    points = world_points(wall, camera)
    on_the_wall = np.abs(points[..., 0] - 8.0) < ONE_MM
    y, z = points[..., 1], points[..., 2]
    hole = on_the_wall & (np.abs(y - 1.6) < 0.9 - 0.01) & (z > 0.9 + 0.01) & (z < 2.2 - 0.01)
    round_it = on_the_wall & (np.abs(y - 1.6) < 1.05) & (z > 0.8) & (z < 2.3)
    leaf = on_the_wall & (np.abs(y + 1.8) < 0.45 - 0.08) & (z > 0.05) & (z < 2.1 - 0.08) & (np.abs(z - 1.02) > 0.15)
    doorway = on_the_wall & (np.abs(y + 1.8) < 0.5) & (z < 2.15)
    elsewhere = on_the_wall & ~round_it & ~doorway & (z > SKIRTING_TOP_M)
    assert hole.sum() > 10000 and leaf.sum() > 5000 and elsewhere.sum() > 20000
    proud = wall - depth                                                    # how far something stands before the wall
    assert np.abs(proud[elsewhere]).max() < ONE_MM
    assert proud[hole].max() < ONE_MM and proud[hole].min() > -0.25         # in the hole all is behind the wall's face
    assert np.median(proud[hole]) < -0.03                                   # and most of it is the pane, set back
    assert proud[round_it].max() < 0.055                                    # a sill at the most, 5 cm proud
    assert proud[doorway].max() < 0.055 and proud[doorway].min() > -ONE_MM  # the door: on the wall, 5 cm at the most
    assert proud[leaf].min() > 0.01                                         # and a leaf that stands off it
    outside = colour_at(picture, camera, (8.0, 1.6 - 0.45, 1.9), reach=3)   # through the upper part of one light
    plaster = colour_at(picture, camera, (8.0, 0.0, 1.9), reach=3)
    wood = colour_at(picture, camera, (8.0, -1.8, 1.5), reach=3)
    assert outside.mean() > 120 and outside[2] > outside[0] + 10            # the sky, as long as no HDRI is named
    assert np.abs(outside - plaster).max() > 25 and np.abs(wood - plaster).max() > 25
    assert wood[0] > wood[2] + 15                                           # a wooden leaf, not a painted wall


@needs_blender
@pytest.mark.parametrize("lighting", ["ceiling", "daylight"])               # the presets with daylight from a window
def test_a_window_seen_through_glass_shows_the_outside_and_not_the_light_it_sends_in(lighting):
    # The daylight of a window is an area light just inside the wall, as large as the window. The camera does not
    # see it; a ray that has gone through a pane in the room must not see it either, or the pane shows the window
    # as a flat white board. The pane stands before the window's left half; the camera is level with z = 1.5 m, so
    # the plain sky is above that on the far wall and its ground below.
    window = Opening("window", x=8.0, y=0.0, yaw=math.pi, width=3.2, z0=0.9, height=1.4)
    room = dataclasses.replace(PLAIN_ROOM, dado=None, openings=(window,))
    pane = Part(Box(5.5, 0.9, 0.0, 0.01, 1.8, 1.9, z0=0.3), "glass", glass=True, bevel=0.0)
    camera = camera_at(2.0, 0.0, 0.0)
    ((picture, depth),) = blender_draws(SceneSpec(room, parts=(pane,), lighting=lighting), [Frame(camera)], samples=32)
    views = {}
    for name, y, behind_the_pane in (("through the pane", 0.75, True), ("beside it", -0.75, False)):
        sky, ground = (8.0, y, 2.0), (8.0, y, 1.15)                         # two points of the window's hole
        for point in (sky, ground):                                         # the pane is in the way of one view only
            assert (depth_at(depth, camera, point) < camera_z(camera, point) - 1.0) == behind_the_pane, name
        views[name] = colour_at(picture, camera, sky, reach=6), colour_at(picture, camera, ground, reach=6)
    for name, (sky, ground) in views.items():
        assert sky[2] > sky[0] + 8, name                                    # a blue sky,
        assert sky.mean() > ground.mean() + 25, name                        # over darker ground: no flat board
    for through, beside in zip(views["through the pane"], views["beside it"], strict=True):
        assert np.abs(through - beside).max() < 30                          # and the same view as beside the pane


@needs_blender
@pytest.mark.parametrize("lighting", ["ceiling", "daylight"])               # as above
def test_a_pane_mirrors_the_outside_of_a_window_and_not_the_light_it_sends_in(lighting):
    # A pane mirrors twice: off its front, and off its back by a ray that has gone into the glass and comes out of
    # it again. Neither may show the window's light: that lay as a white board over whoever stood behind the glass
    # of a counter. The window is in the wall behind the camera; the pane stands before something dark, so that
    # what is seen in it is what it mirrors.
    window = Opening("window", x=-2.0, y=0.0, yaw=0.0, width=1.8, z0=0.9, height=1.3)
    room = dataclasses.replace(PLAIN_ROOM, dado=None, openings=(window,))
    pane = Part(Box(4.0, 0.0, 0.0, 0.01, 1.0, 1.0, z0=0.9), "glass", glass=True, bevel=0.0)
    dark = Part(Box(4.5, 0.0, 0.0, 0.4, 2.0, 2.1, color=(12, 12, 12)), PAINT, bevel=0.0)
    camera = camera_at(2.0, 0.0, 0.0)
    eye, near_face = camera.T_world_camera[:3, 3], 3.995

    def mirrored(point) -> np.ndarray:
        """Where on the pane's near face the camera sees a point of the room behind it."""
        image = np.array([2 * near_face - point[0], point[1], point[2]])
        return eye + (image - eye) * (near_face - eye[0]) / (image[0] - eye[0])

    ((picture, depth),) = blender_draws(SceneSpec(room, parts=(pane, dark), lighting=lighting), [Frame(camera)],
                                        samples=32)
    in_the_sky, on_the_ground = mirrored((-2.0, 0.45, 1.9)), mirrored((-2.0, 0.45, 1.2))   # in one light of the window
    beside_the_pane = (4.3, 0.8, 1.5)                                       # on the dark thing's near face
    for spot in (in_the_sky, on_the_ground):
        assert depth_at(depth, camera, spot) == pytest.approx(camera_z(camera, spot), abs=ONE_MM)
    assert depth_at(depth, camera, beside_the_pane) == pytest.approx(camera_z(camera, beside_the_pane), abs=ONE_MM)
    sky, ground, bare = (colour_at(picture, camera, spot, reach=6).mean()
                         for spot in (in_the_sky, on_the_ground, beside_the_pane))
    assert sky > 1.4 * ground                                               # the sky over darker ground: no white board
    assert bare < ground < bare + 55                                        # and a faint picture over the dark thing


@needs_blender
def test_a_small_and_a_large_room_are_exposed_alike():
    small = RoomSpec(polygon=((-1.0, -2.0), (4.5, -2.0), (4.5, 2.0), (-1.0, 2.0)), height=2.7, floor="floor_laminate",
                     ceiling="ceiling_tile", wall=(233, 227, 213), dado=None, dado_height=0.95, lights=2.4)
    large = dataclasses.replace(small, polygon=((-4.0, -7.0), (14.0, -7.0), (14.0, 7.0), (-4.0, 7.0)), height=3.2)
    sparse = dataclasses.replace(large, lights=3.0)                         # fewer fittings, each of more watts
    unlit = dataclasses.replace(small, lights=0.0)                          # no fitting to be seen, and lit as well
    camera = camera_at(**SMALL)
    brightness = []
    for room, far_wall in ((small, 4.5), (large, 14.0), (sparse, 14.0), (unlit, 4.5)):
        ((picture, _),) = blender_draws(SceneSpec(room), [Frame(camera)], samples=32)
        wall = colour_at(picture, camera, (far_wall, 0.3, 1.4), reach=4)
        floor = colour_at(picture, camera, (2.5, 0.0, 0.0), reach=4)
        assert 130 < wall.mean() < 225 and 80 < floor.mean() < 225          # lit, and neither dim nor burnt out
        brightness.append(wall.mean())
    assert max(brightness) - min(brightness) < 30                           # one exposure serves them all


@needs_blender
@pytest.mark.parametrize("lighting", LIGHTING)
def test_every_lighting_preset_lights_the_room(lighting):
    window = Opening("window", x=8.0, y=1.0, yaw=math.pi, width=1.8, z0=0.9, height=1.3)
    room = dataclasses.replace(PLAIN_ROOM, openings=(window,))
    camera = camera_at(**SMALL)
    ((picture, depth),) = blender_draws(SceneSpec(room, parts=a_dozen_parts(), lighting=lighting), [Frame(camera)],
                                        samples=16)
    assert 50 < picture.mean() < 225 and (depth > 0).all()
    assert (picture.max(axis=2) < 12).mean() < 0.02                         # nothing much is black


STAND_IN = '''"""A stand-in for the asset library, put where the job's scripts look for it: boxes of known sizes."""
import json
import sys
import types

import bpy

sys.path.insert(0, {side!r})
library = types.ModuleType("library")
library.asked = []
LIFE = {{"crate_a": (0.5, 0.8, 1.0)}}                # along its front, across, up
BODY = (0.3, 0.5, 1.8)                             # in depth, across, and its natural height


def block(name, size, parent=None):
    half_x, half_y, top = size[0] / 2, size[1] / 2, size[2]
    corners = [(x, y, z) for z in (0.0, top) for x, y in ((-half_x, -half_y), (half_x, -half_y), (half_x, half_y),
                                                           (-half_x, half_y))]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(corners, [], faces)
    made = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(made)
    made.parent = parent
    return made


def open_library(prepared):
    library.asked.append(["open_library", prepared])


def add_model(asset):
    library.asked.append(["add_model", asset])
    parent = bpy.data.objects.new(asset, None)     # one parent object, the mesh under it
    bpy.context.scene.collection.objects.link(parent)
    block(asset + "_mesh", LIFE[asset], parent)
    return parent


def add_person(avatar, pose, prop=None):
    library.asked.append(["add_person", avatar, pose, prop])
    return block(avatar, BODY)


def material(name, tint=None):
    library.asked.append(["material", name, None if tint is None else list(tint)])
    return bpy.data.materials.new(name)


def set_world(hdri, strength):
    library.asked.append(["set_world", hdri])
    bpy.context.scene.world = bpy.data.worlds.new(hdri)


for name in ("open_library", "add_model", "add_person", "material", "set_world"):
    setattr(library, name, globals()[name])
sys.modules["library"] = library
import job  # noqa: E402

arguments = sys.argv[sys.argv.index("--") + 1:]
job.main(arguments)
with open(arguments[arguments.index("--job") + 1] + "/asked.json", "w", encoding="utf-8") as out:
    json.dump(library.asked, out)
'''


@needs_blender
def test_a_model_stands_by_its_bottom_centre_and_a_person_by_the_ground_point_under_the_pelvis(monkeypatch, tmp_path):
    stand_in = tmp_path / "scripts" / "with_a_stand_in_library.py"
    stand_in.parent.mkdir()
    stand_in.write_text(STAND_IN.format(side=str(BLENDER_SIDE)), encoding="utf-8")
    prepared = tmp_path / "assets" / "_cache" / "prepared"
    people = {"avatar_a": AvatarAsset("avatar_a", "female", 1.8, frozenset())}
    monkeypatch.setattr(blender_renderer, "run_script",
                        lambda script, args, **options: blender.run_script(stand_in, args, **options))
    library = Library({}, people, {}, {}, ("street_a",), {}, prepared)
    monkeypatch.setattr(blender_renderer, "load_library", lambda: library)
    monkeypatch.setattr(blender_renderer, "is_prepared", lambda library: True)
    # A crate of 0.5 x 0.8 x 1.0 m, stretched to another size along each of its own axes, turned, and stood on a
    # step; a body of 0.3 x 0.5 x 1.8 m as tall as 1.62 m, and a quarter wider than that makes it.
    crate = ModelRef("crate_a", x=4.2, y=-1.0, z=0.3, yaw=0.7, scale=(1.2, 0.9, 1.1), size=(0.6, 0.72, 1.1), group=5)
    person = PersonRef("p0", "avatar_a", "waiting_a", x=4.5, y=1.2, yaw=2.5, height=1.62, build=1.25, prop="cup",
                       group=6)
    step = Part(Box(4.2, -1.0, 0.7, 0.9, 0.9, 0.3, color=(90, 90, 90)), "stone", bevel=0.0)
    spec = SceneSpec(PLAIN_ROOM, parts=(step,), models=(crate,), people=(person,), hdri="street_a")
    frames = depth_only(CAMERA) + depth_only(CAMERA, only=(5,)) + depth_only(CAMERA, only=(6,))
    renderer = BlenderRenderer(work_dir=tmp_path / "jobs", keep=True)
    (_, whole), (_, crate_alone), (_, person_alone) = renderer.render(spec, frames)
    shrunk = 1.62 / 1.8
    body = Box(4.5, 1.2, 2.5, 0.3 * shrunk * 1.25, 0.5 * shrunk * 1.25, 1.62)
    filled = Box(4.2, -1.0, 0.7, 0.6, 0.72, 1.1, z0=0.3)                    # the box the model's `size` names
    sky = render(CAMERA, antialias=False)[1]
    for alone, solid in ((crate_alone, filled), (person_alone, body)):
        theirs = render(CAMERA, [solid], antialias=False)[1]
        theirs = np.where(theirs != sky, theirs, 0.0)
        assert (theirs > 0).sum() > 4000 and np.array_equal(alone > 0, theirs > 0)
        compared = (theirs > 0) & ~near_an_edge(theirs)
        assert np.abs(alone - theirs)[compared].max() < ONE_MM
    free = (crate_alone > 0) | (person_alone > 0)                           # and both are in the whole picture
    assert np.abs(np.where(crate_alone > 0, crate_alone, person_alone) - whole)[free].max() < 1e-5
    asked = json.loads((renderer.last_job / "asked.json").read_text(encoding="utf-8"))
    assert asked[0] == ["open_library", str(prepared)]                      # before anything is asked of it
    assert ["add_model", "crate_a"] in asked and ["add_person", "avatar_a", "waiting_a", "cup"] in asked
    assert ["set_world", "street_a"] in asked and ["material", "stone", None] in asked
    assert ["material", "wall_paint", [214, 222, 228]] in asked and ["material", "floor_tile", None] in asked


# ----------------------------------------------------------------------------- in Blender, with the prepared assets

needs_assets = pytest.mark.skipif(not blender.available(), reason="no Blender of the 4.5 series with prepared assets")


def reach_of(mask: np.ndarray) -> tuple[int, int, int, int]:
    """The first and the last row, and the first and the last column, that a mask fills."""
    rows, cols = np.nonzero(mask.any(axis=1))[0], np.nonzero(mask.any(axis=0))[0]
    return int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])


@needs_assets
def test_a_model_of_the_library_fills_the_box_its_size_names():
    library = load_library()
    kinds = [kind for kind in ("bench", "lockers", "cooler", "plant", "office_chair") if library.models_of(kind)][:4]
    assert kinds, "the library has no model that stands on the floor"
    stretches = ((1.0, 1.0, 1.0), (1.1, 0.9, 1.05), (0.9, 1.1, 0.95))        # along its front, across, up
    models = []
    for number, kind in enumerate(kinds):
        asset, scale = library.models_of(kind)[0], stretches[number % len(stretches)]
        models.append(ModelRef(asset.id, x=5.0, y=-2.1 + 1.4 * number, z=0.0, yaw=math.pi + 0.4 * number, scale=scale,
                               size=tuple(side * by for side, by in zip(asset.size, scale)), group=number + 1))
    spec = SceneSpec(PLAIN_ROOM, models=tuple(models))
    frames = [Frame(CAMERA, beauty=False, only=(model.group,)) for model in models]
    drawn, cast = blender_draws(spec, frames), StubRenderer().render(spec, frames)    # the stub draws each as its box
    for model, (_, ours), (_, theirs) in zip(models, drawn, cast, strict=True):
        shown, box = ours > 0, theirs > 0
        assert box.sum() > 1500 and shown.sum() > 0.15 * box.sum(), model.asset      # there, and no speck in its box
        assert (shown & ~grown(box, 2)).sum() <= 0.01 * box.sum(), model.asset       # and it stays in it
        top, bottom, left, right = reach_of(shown)
        box_top, box_bottom, box_left, box_right = reach_of(box)
        tall, wide = box_bottom - box_top, box_right - box_left
        assert top - box_top < 0.3 * tall and box_bottom - bottom < 0.3 * tall, model.asset   # from its foot to its top
        assert left - box_left < 0.3 * wide and box_right - right < 0.3 * wide, model.asset   # and from side to side


@needs_assets
def test_a_person_of_the_library_stands_on_their_ground_point_at_their_height_and_build():
    library = load_library()
    first, second = list(library.avatars.values())[:2]

    def waiting(avatar) -> str:
        return next(pose.id for pose in library.poses.values()
                    if "waiting" in pose.tags and pose.sex in (None, avatar.sex) and pose.reach <= 0.45)

    def person(group: int, avatar, **fields) -> PersonRef:                   # each where the others are, facing CAMERA
        return PersonRef(f"p{group}", avatar.id, waiting(avatar), x=4.5, y=0.0, yaw=math.pi, group=group, **fields)

    people = (person(1, first, height=1.62, build=0.92), person(2, first, height=1.62, build=1.15),
              person(3, second, height=1.84))
    spec = SceneSpec(PLAIN_ROOM, people=people)
    drawn = blender_draws(spec, [Frame(CAMERA, beauty=False, only=(each.group,)) for each in people])
    reaches = []
    for each, (_, alone) in zip(people, drawn, strict=True):
        for up in (0.9, 1.0, 1.2):                                          # the body stands round the upright axis
            on_the_axis = camera_z(CAMERA, (4.5, 0.0, up))
            assert on_the_axis - 0.45 < depth_at(alone, CAMERA, (4.5, 0.0, up)) < on_the_axis, (each.avatar, up)
        assert depth_at(alone, CAMERA, (4.5, 0.0, each.height + 0.1)) == 0.0, each.avatar    # and ends at its height
        top, bottom, left, right = reach_of(alone > 0)
        crown, a_bowed_head, feet = (project(CAMERA, (4.5, 0.0, up))[1] for up in (each.height + 0.03,
                                                                                    each.height - 0.15, 0.0))
        assert crown - 1 <= top <= a_bowed_head, each.avatar
        assert abs(bottom - feet) < 0.08 * (feet - crown), each.avatar      # the feet on the floor, round the point
        reaches.append((bottom - top, right - left))
    (slim_tall, slim_wide), (broad_tall, broad_wide), _ = reaches
    assert abs(slim_tall - broad_tall) <= 2                                 # a build widens and does not heighten
    assert broad_wide / slim_wide == pytest.approx(1.15 / 0.92, abs=0.07)


@needs_assets
def test_with_prepared_assets_the_library_gives_the_materials_and_the_view_from_the_window():
    library = load_library()
    window = Opening("window", x=8.0, y=1.6, yaw=math.pi, width=1.8, z0=0.9, height=1.3)
    room = dataclasses.replace(PLAIN_ROOM, openings=(window,))
    camera = camera_at(2.0, 0.0, 0.0)
    ((_, cast),) = StubRenderer().render(SceneSpec(PLAIN_ROOM, parts=a_dozen_parts()), depth_only(camera))
    points = world_points(cast, camera)
    on_the_wall = np.abs(points[..., 0] - 8.0) < ONE_MM
    hole = on_the_wall & (np.abs(points[..., 1] - 1.6) < 0.8) & (points[..., 2] > 1.0) & (points[..., 2] < 2.1)
    round_it = on_the_wall & (np.abs(points[..., 1] - 1.6) < 1.05) & (points[..., 2] > 0.8) & (points[..., 2] < 2.3)
    assert hole.sum() > 4000                                                # the window, where no part stands before it
    renderer, views = BlenderRenderer(samples=32), []
    for hdri in (None, library.hdris[0]):
        ((picture, depth),) = renderer.render(SceneSpec(room, parts=a_dozen_parts(), hdri=hdri), [Frame(camera)])
        assert renderer.last_facts["materials"] == "library"
        compared = ~near_an_edge(cast) & ~on_a_trim(depth, camera, room) & ~round_it
        assert compared.mean() > 0.6 and np.abs(depth - cast)[compared].max() < ONE_MM   # a texture moves no surface
        assert 50 < picture.mean() < 225
        views.append(picture[hole].astype(np.float64))
    assert views[1].mean() > 50 and np.abs(views[1] - views[0]).mean() > 8  # the HDRI, not the plain sky, outside


# ------------------------------------------------- in Blender, the whole library: what the parts give one another

THRESHOLDS = load_config().thresholds               # the scene check's own: what it calls a person it can see
SIDE_ON = camera_at(4.5, -3.8, math.pi / 2)         # CAMERA looks along +x; this one looks along +y, from the side


def seen_from(depth: np.ndarray, camera: Camera, x: float, y: float, yaw: float) -> np.ndarray:
    """The points a depth image shows, (N, 3), in the frame of something that stands on (x, y) and faces yaw:
    how far ahead of it, how far to its left, how far above the floor."""
    points = world_points(depth, camera)[depth > 0]
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = points[:, 0] - x, points[:, 1] - y
    return np.stack([c * dx + s * dy, c * dy - s * dx, points[:, 2]], axis=1)


def whole_in_the_picture(mask: np.ndarray) -> bool:
    return bool(mask.any() and not (mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()))


def held_in(pose) -> str | None:
    """What a person is given to hold in a pose: the thing the pose was made for, and nothing in any other."""
    return "phone" if "phone" in pose.tags else "cup" if "cup" in pose.tags else None


@needs_assets
def test_every_model_of_the_library_lies_in_the_box_its_size_names_at_its_pose():
    library = load_library()
    stretches = ((1.0, 1.0, 1.0), (1.15, 0.85, 1.1), (0.85, 1.15, 0.9), (1.1, 1.05, 0.85))    # to its front, across, up
    models = []
    for number, asset in enumerate(library.models.values()):
        scale = stretches[number % len(stretches)]
        size = tuple(side * by for side, by in zip(asset.size, scale))
        # Each stands where the others do, turned another way with its front towards the camera, and all but the
        # tall ones at some height, as a thing on a counter is.
        models.append(ModelRef(asset.id, x=3.6, y=0.3, z=0.0 if size[2] > 1.2 else 0.25 * (number % 3),
                               yaw=math.pi + 0.45 * (number % 5 - 2), scale=scale, size=size, group=number + 1))
    assert len(models) >= 14                                                # the library has a model of every kind
    spec = SceneSpec(PLAIN_ROOM, models=tuple(models))
    drawn = blender_draws(spec, [Frame(CAMERA, beauty=False, only=(model.group,)) for model in models])
    for model, (_, alone) in zip(models, drawn, strict=True):
        assert (alone > 0).sum() > 120 and whole_in_the_picture(alone > 0), model.asset
        own = seen_from(alone, CAMERA, model.x, model.y, model.yaw) - (0.0, 0.0, model.z)
        low = np.array([-model.size[0] / 2, -model.size[1] / 2, 0.0])
        high = np.array([model.size[0] / 2, model.size[1] / 2, model.size[2]])
        assert (own.min(axis=0) > low - 3 * ONE_MM).all() and (own.max(axis=0) < high + 3 * ONE_MM).all(), model.asset
        # And it fills that box: its front, its two sides, its foot and its top are each within a twentieth of the
        # box's side of the box's face. Its back is turned away.
        short = np.concatenate([(own.min(axis=0) - low)[1:], high - own.max(axis=0)]) / np.tile(high - low, 2)[1:]
        assert short.max() < 0.05, (model.asset, short)


@needs_assets
def test_a_person_is_as_tall_as_asked_in_every_pose_and_holds_what_the_pose_was_made_for():
    library = load_library()
    woman = next(avatar for avatar in library.avatars.values() if avatar.sex == "female")
    man = next(avatar for avatar in library.avatars.values() if avatar.sex == "male")
    people = []
    for avatar, height, build in ((woman, 1.58, 1.0), (man, 1.84, 0.95), (woman, 1.84, 1.1), (man, 1.62, 1.05)):
        for pose in library.poses.values():        # each where the others are, their right side half to the camera
            for prop in dict.fromkeys((held_in(pose), None)):
                people.append(PersonRef(f"p{len(people)}", avatar.id, pose.id, x=4.5, y=0.0, yaw=math.pi + 0.9,
                                        height=height, build=build, prop=prop, group=len(people) + 1))
    assert {held_in(pose) for pose in library.poses.values()} == {None, "phone", "cup"}
    drawn = blender_draws(SceneSpec(PLAIN_ROOM, people=tuple(people)),
                          [Frame(CAMERA, beauty=False, only=(each.group,)) for each in people])
    alone_of = {(each.avatar, each.pose, each.height, each.prop): alone for each, (_, alone) in zip(people, drawn)}
    for each, (_, alone) in zip(people, drawn, strict=True):
        assert whole_in_the_picture(alone > 0), (each.avatar, each.pose)
        up = seen_from(alone, CAMERA, each.x, each.y, each.yaw)[:, 2]
        # The top of the head is at the height asked for, whether the pose bows it or not: a pixel at 4.3 m is
        # 8 mm high, so the topmost pixel centre is that much under the crown at the most.
        assert up.max() == pytest.approx(each.height, abs=0.03), (each.avatar, each.pose, each.height)
        assert -0.005 < up.min() < 0.02, (each.avatar, each.pose)           # and the soles are on the floor
        if each.prop is not None:                   # the thing is there, before the same person with empty hands
            bare = alone_of[(each.avatar, each.pose, each.height, None)]
            nearer = (bare - alone > 0.005) | ((alone > 0) & (bare == 0))
            assert nearer.sum() >= 15, (each.avatar, each.pose, each.prop)
            assert ((alone - bare > 0.005) & (bare > 0)).sum() <= 2, (each.avatar, each.pose, each.prop)


@needs_assets
def test_a_person_stands_over_their_ground_point_faces_their_yaw_and_is_seen_by_the_scene_check():
    library = load_library()
    woman = next(avatar for avatar in library.avatars.values() if avatar.sex == "female")
    man = next(avatar for avatar in library.avatars.values() if avatar.sex == "male")
    staff = next(avatar for avatar in library.avatars.values() if any(tag.startswith("staff_") for tag in avatar.tags))
    empty = next(pose for pose in library.poses.values() if "waiting" in pose.tags and held_in(pose) is None)
    phone = next(pose for pose in library.poses.values() if held_in(pose) == "phone")
    cup = next(pose for pose in library.poses.values() if held_in(pose) == "cup")
    # Three people in a row across the room, each turned a little differently towards CAMERA; SIDE_ON sees the
    # row from its end, so each of them in profile.
    people = (PersonRef("p0", woman.id, empty.id, x=4.5, y=-1.0, yaw=math.pi + 0.3, height=1.62, build=0.95, group=1),
              PersonRef("p1", man.id, phone.id, x=4.5, y=0.3, yaw=math.pi - 0.25, height=1.84, build=1.0, prop="phone",
                        group=2),
              PersonRef("p2", staff.id, cup.id, x=4.5, y=1.6, yaw=math.pi, height=1.72, build=1.1, prop="cup", group=3))
    alone = [Frame(camera, beauty=False, only=(each.group,)) for each in people for camera in (CAMERA, SIDE_ON)]
    renderer = BlenderRenderer()
    (_, whole), *drawn = renderer.render(SceneSpec(PLAIN_ROOM, people=people), [Frame(CAMERA, beauty=False), *alone])
    assert renderer.last_facts["people"] == 3 and renderer.last_facts["materials"] == "library"
    observation = ObservationData(np.zeros((480, 640, 3), dtype=np.uint8), whole, CAMERA, RobotPose(0.0, 0.0, 0.0))
    for number, each in enumerate(people):
        (_, front), (_, side) = drawn[2 * number], drawn[2 * number + 1]
        assert whole_in_the_picture(front > 0) and whole_in_the_picture(side > 0), each.person_id
        # The scene check samples the upright axis through the ground point and calls them visible.
        assert is_visible(Person(id=each.person_id, x=each.x, y=each.y, yaw=each.yaw), observation, THRESHOLDS)
        for up in (0.9, 1.0, 1.2):                  # that axis is inside the body, from the front and from the side
            for camera, seen in ((CAMERA, front), (SIDE_ON, side)):
                on_the_axis = camera_z(camera, (each.x, each.y, up))
                assert on_the_axis - 0.35 < depth_at(seen, camera, (each.x, each.y, up)) < on_the_axis - 0.02, (
                    each.person_id, up)
        facing, profile = (seen_from(seen, camera, each.x, each.y, each.yaw)
                           for seen, camera in ((front, CAMERA), (side, SIDE_ON)))
        # From the front the outline is as wide to their left of the ground point as to their right.
        assert abs(facing[:, 1].max() + facing[:, 1].min()) / 2 < 0.05, each.person_id
        # In profile the feet say where they stand and which way they face: the heels a little behind the point
        # under the pelvis, the toes well ahead of it.
        feet = profile[profile[:, 2] < 0.04 * each.height, 0]
        assert 0.14 < feet.max() < 0.33 and 0.02 < -feet.min() < 0.14, (each.person_id, feet.min(), feet.max())
        # And what a hand holds up is before the chest, about 0.15 m ahead of it; empty hands hang by the body.
        hands = profile[(profile[:, 2] > 0.5 * each.height) & (profile[:, 2] < 0.8 * each.height), 0]
        if each.prop is None:
            assert hands.max() < 0.21, (each.person_id, hands.max())
        else:
            assert 0.25 < hands.max() < 0.42 and -hands.min() < 0.27, (each.person_id, hands.min(), hands.max())


# What each material is to look like in a picture, loosely: (the least and the most of its brightness, 0 to 255,
# and whether red leads blue, blue leads red, or neither does by much).
MATERIAL_LOOKS = {
    "floor_tile": (70, 235, None), "floor_terrazzo": (70, 235, None), "floor_laminate": (60, 225, "warm"),
    "ceiling_tile": (110, 250, None), "wall_paint": (110, 250, None), "wood": (50, 215, "warm"),
    "wood_dark": (15, 150, "warm"), "laminate_white": (130, 252, None), "steel": (40, 240, None),
    "painted_metal": (30, 200, None), "plastic_dark": (8, 110, None), "plastic_light": (110, 250, None),
    "fabric": (25, 200, None), "stone": (40, 220, None), "cork": (60, 225, "warm"), "paper": (130, 252, None),
    "cardboard": (60, 225, "warm"), "ceramic": (130, 252, None), "glass": (20, 250, None), "screen": (5, 130, None)}


def missing_texture(colours: np.ndarray, pale: bool = True) -> np.ndarray:
    """Which colours are the magenta that Cycles paints where an image of a material could not be read: red and
    blue alike, and green well under both. In bright light the view transform makes a lilac of it, about (208,
    172, 240), which `pale` counts too; no material of the library is of that hue, but a shirt may be."""
    colours = np.asarray(colours, dtype=np.float64)
    red, green, blue = colours[..., 0], colours[..., 1], colours[..., 2]
    least = np.minimum(red, blue)
    lead = least - green
    return (least > 60) & (np.abs(red - blue) < 80) & (lead > (25 if pale else 0.55 * least))


@needs_assets
def test_every_material_of_the_library_is_drawn_as_itself_and_none_as_a_missing_texture():
    library = load_library()
    assert set(library.materials) == set(MATERIAL_LOOKS)                    # the twenty names the parts share
    assert missing_texture(np.array([[255, 0, 255], [228, 40, 214], [208, 172, 240]])).all()
    assert not missing_texture(np.array([[200, 60, 50], [40, 40, 40], [150, 170, 180], [233, 227, 213], [168, 124, 82],
                                         [88, 104, 132], [139, 111, 90]])).any()
    # A box of each, in two rows before the camera, the far row on a shelf.
    parts, spots = [], {}
    for number, name in enumerate(library.materials):
        row, column = divmod(number, 10)
        x, y, z0 = 4.0 + 1.2 * row, 0.42 * (column - 4.5), 0.9 * row
        parts.append(Part(Box(x, y, math.pi, 0.34, 0.34, 0.5, z0=z0, color=(128, 128, 128)), name, bevel=0.004))
        spots[name] = (x - 0.17, y, z0 + 0.25)                              # the middle of the face it turns to CAMERA
    shelf = Part(Box(5.2, 0.0, math.pi, 0.4, 4.3, 0.9, color=(150, 150, 150)), PAINT, bevel=0.0)
    renderer = BlenderRenderer(samples=32)
    ((picture, depth),) = renderer.render(SceneSpec(PLAIN_ROOM, parts=(*parts, shelf)), [Frame(CAMERA)])
    assert renderer.last_facts["materials"] == "library"
    assert missing_texture(picture).mean() < 2e-4 and (picture.max(axis=2) < 12).mean() < 0.02
    looks = {}
    for name, spot in spots.items():
        assert depth_at(depth, CAMERA, spot) == pytest.approx(camera_z(CAMERA, spot), abs=0.01), name   # in plain view
        looks[name] = colour_at(picture, CAMERA, spot, reach=4)
        dimmest, brightest, tone = MATERIAL_LOOKS[name]
        assert dimmest < looks[name].mean() < brightest, (name, looks[name])
        if tone == "warm":
            assert looks[name][0] > looks[name][2] + 12, (name, looks[name])
    assert looks["wood"].mean() > looks["wood_dark"].mean() + 15            # the two woods, the two plastics
    assert looks["plastic_light"].mean() > looks["plastic_dark"].mean() + 60
    assert looks["laminate_white"].mean() > looks["painted_metal"].mean() + 30


@needs_assets
def test_no_model_and_no_avatar_of_the_library_shows_a_missing_texture_and_every_avatar_is_as_tall_as_asked():
    # A model's and an avatar's textures are packed into its own prepared file. Each is drawn alone, before the
    # plain sky, so that what the picture shows where the depth has something is that thing and nothing else.
    library = load_library()
    pose = next(pose for pose in library.poses.values() if "waiting" in pose.tags and held_in(pose) is None)
    models = tuple(ModelRef(asset.id, x=3.2, y=0.0, z=0.0 if asset.size[2] > 1.2 else 0.6, yaw=math.pi - 0.5,
                            scale=(1.0, 1.0, 1.0), size=asset.size, group=number + 1)
                   for number, asset in enumerate(library.models.values()))
    people = tuple(PersonRef(f"p{number}", avatar.id, pose.id, x=4.5, y=0.0, yaw=math.pi - 0.4,
                             height=(1.58, 1.7, 1.84)[number % 3], group=len(models) + number + 1)
                   for number, avatar in enumerate(library.avatars.values()))
    spec = SceneSpec(PLAIN_ROOM, models=models, people=people)
    frames = ([Frame(camera_at(**SMALL), only=(each.group,)) for each in models]
              + [Frame(CAMERA, only=(each.group,)) for each in people])
    drawn = blender_draws(spec, frames, samples=8)
    for each, (picture, depth) in zip((*models, *people), drawn, strict=True):
        name = getattr(each, "asset", None) or each.avatar
        thing = picture[depth > 0].astype(np.float64)
        assert len(thing) > 60, name
        assert missing_texture(thing, pale=False).mean() < 0.01, name      # of the full magenta; a top may be lilac
        assert thing.mean() > 14 and (thing.max(axis=1) < 6).mean() < 0.3, name       # lit, and not a black shape
    # Hair is cards of a picture that is clear in places, and the depth pass draws it where it is more than half
    # there: the crown of every avatar is still where the height asked for puts it.
    for each, (_, depth) in zip(people, drawn[len(models):], strict=True):
        up = seen_from(depth, CAMERA, each.x, each.y, each.yaw)[:, 2]
        assert whole_in_the_picture(depth > 0) and up.max() == pytest.approx(each.height, abs=0.03), each.avatar
        assert -0.005 < up.min() < 0.02, each.avatar


@needs_assets
def test_every_hdri_of_the_library_shows_a_view_through_a_window_that_is_neither_burnt_out_nor_dark():
    library = load_library()
    window = Opening("window", x=8.0, y=0.0, yaw=math.pi, width=2.4, z0=0.9, height=1.4)
    room = dataclasses.replace(PLAIN_ROOM, dado=None, openings=(window,))
    camera = camera_at(3.5, 0.0, 0.0)                                       # 4.3 m from the window, at z = 1.5 m
    v, u = np.meshgrid(np.arange(480) + 0.5, np.arange(640) + 0.5, indexing="ij")
    corners = [project(camera, (8.0, y, z)) for y in (1.0, -1.0) for z in (2.15, 1.05)]
    hole = ((u > corners[0][0]) & (u < corners[2][0]) & (v > corners[0][1]) & (v < corners[1][1])
            & (np.abs(u - 320.0) > 12))                                     # the two lights, without the mullion
    assert hole.sum() > 20000
    renderer = BlenderRenderer(samples=32)
    for lighting in ("ceiling", "daylight"):
        for hdri in library.hdris:
            ((picture, _),) = renderer.render(SceneSpec(room, lighting=lighting, hdri=hdri), [Frame(camera)])
            outside = picture[hole].astype(np.float64)
            plaster = colour_at(picture, camera, (8.0, 2.4, 1.6), reach=4)  # the wall beside the window
            # A street or a sky by day: neither dark nor the white board that a sky far too bright comes out as,
            # and of another colour than the pale wall round it.
            assert 70 < outside.mean() < 190, (lighting, hdri, outside.mean())
            assert (outside.mean(axis=1) > 235).mean() < 0.1, (lighting, hdri)
            assert np.abs(plaster - outside.mean(axis=0)).max() > 12, (lighting, hdri, plaster, outside.mean(axis=0))


@needs_assets
def test_a_sign_is_drawn_by_the_sign_module_in_the_librarys_font_the_right_way_round():
    library = load_library()
    # What the constructions put up in either signage: a made-up logo, and the placeholder's own words.
    arts = (SignArt("logo", text="coffee", ink=(240, 234, 220), ground=(60, 40, 30), width_m=1.6, height_m=0.5, seed=3),
            SignArt("label", text="GATE 7", ink=(245, 245, 245), ground=(30, 60, 110), width_m=1.6, height_m=0.5))
    boards = [Box(4.0, y, math.pi, 0.06, 1.6, 0.5, z0=1.3, color=art.ground) for art, y in zip(arts, (0.9, -0.9))]
    spec = SceneSpec(PLAIN_ROOM, parts=tuple(Part(board, PAINT, bevel=0.0, signs=(Sign("front", art),))
                                             for art, board in zip(arts, boards)))
    ((picture, _),) = BlenderRenderer(samples=32).render(spec, [Frame(CAMERA)])     # with the drawing it has by default
    rows, columns = 20, 64
    for art, board in zip(arts, boards):
        in_the_font = signs.draw(art, font=library.fonts[SIGN_FONT])
        assert not np.array_equal(in_the_font, signs.draw(art)), art.style  # the library's font is not Pillow's own
        # The picture, read off the board's face at the middle of each cell of a grid, against the drawing shrunk
        # to that grid. Whoever reads the board from CAMERA has +y on their left.
        seen = np.array([[picture[math.floor(v), math.floor(u)].mean()
                          for u, v, _ in (project(CAMERA, (3.969, board.cy + 0.8 - 1.6 * (column + 0.5) / columns,
                                                           1.8 - 0.5 * (row + 0.5) / rows))
                                          for column in range(columns))] for row in range(rows)])
        wanted = np.asarray(Image.fromarray(in_the_font).convert("L").resize((columns, rows), Image.BOX),
                            dtype=np.float64)

        def agreement(drawing: np.ndarray) -> float:
            return float(np.corrcoef(seen.ravel(), drawing.ravel())[0, 1])

        turned_round = max(agreement(wanted[:, ::-1]), agreement(wanted[::-1]), agreement(wanted[::-1, ::-1]))
        assert agreement(wanted) > 0.75 and agreement(wanted) > turned_round + 0.2, (art.style, turned_round)


def test_signs_are_lettered_in_the_librarys_font_once_the_assets_are_prepared(pretended, monkeypatch, tmp_path):
    asked = []

    def draw(art, font=None, px_per_m=512):
        asked.append((art, font))
        return np.zeros((8, 8, 3), dtype=np.uint8)

    stand_in = types.ModuleType("sgi.synth.realistic.signs")
    stand_in.draw = draw
    monkeypatch.setitem(sys.modules, "sgi.synth.realistic.signs", stand_in)
    monkeypatch.setattr(realistic, "signs", stand_in, raising=False)
    prepared = tmp_path / "assets" / "_cache" / "prepared"
    (prepared / "fonts").mkdir(parents=True)
    font = prepared / "fonts" / "sans.ttf"
    fonts = {SIGN_FONT: font, "sans_bold": prepared / "fonts" / "sans_bold.ttf"}
    library = Library({}, {}, {}, {}, (), fonts, prepared)
    monkeypatch.setattr(blender_renderer, "load_library", lambda: library)
    monkeypatch.setattr(blender_renderer, "is_prepared", lambda library: True)
    spec = lettered_scene()
    logo, menu = spec.parts[0].signs[0].art, spec.parts[0].signs[1].art
    BlenderRenderer().render(spec, depth_only(CAMERA))
    assert asked == [(logo, None), (menu, None)]                            # a library whose font file is not there
    font.write_bytes(b"a font")
    BlenderRenderer().render(spec, depth_only(CAMERA))
    assert asked[2:] == [(logo, font), (menu, font)]
    own = []                                                                # a drawing of the caller's is left alone
    BlenderRenderer(draw=lambda art: own.append(art) or np.zeros((8, 8, 3), dtype=np.uint8)).render(
        spec, depth_only(CAMERA))
    assert own == [logo, menu] and len(asked) == 4


def test_the_library_serves_the_people_through_the_one_module_the_scene_builder_asks():
    # build.py asks library.py alone for models, people, materials and the world; library.py hands the people on
    # to people_library.py beside it, which opens with it.
    assert "people_library" in imported(BLENDER_SIDE / "library.py")
    assert "people_library" not in imported(BLENDER_SIDE / "build.py")
    source = (BLENDER_SIDE / "library.py").read_text(encoding="utf-8")
    assert "_people.open_people(" in source and "_people.add_person(avatar, pose, prop)" in source


# ------------------------------------------------------------------------------------- the renderer, without Blender

class Pretended:
    """What stands in for the run in Blender: it writes what the job's script would, and keeps what it was asked."""

    def __init__(self, write: bool = True):
        self.calls, self.jobs, self.write = [], [], write

    def __call__(self, script, args, **options):
        folder = Path(args[list(args).index("--job") + 1])
        job = json.loads((folder / "job.json").read_text(encoding="utf-8"))
        self.calls.append((Path(script), [str(arg) for arg in args], options))
        self.jobs.append((job, {path.name: path.read_bytes() for path in sorted(folder.iterdir())}))
        for number, frame in enumerate(job["frames"] if self.write else ()):
            shape = (frame["height"], frame["width"])
            np.save(folder / f"{frame['name']}.depth.npy", np.full(shape, 2.0 + number, dtype=np.float32))
            if frame["beauty"]:
                Image.fromarray(np.full((*shape, 3), 40 + number, dtype=np.uint8)).save(folder / f"{frame['name']}.png")
        (folder / "facts.json").write_text(json.dumps({"blender": "4.5.14", "device": "CPU",
                                                       "seconds": {"frames": [0.5] * len(job["frames"])}}))
        return subprocess.CompletedProcess(["blender"], 0, stdout="done\n")


@pytest.fixture
def pretended(monkeypatch, tmp_path):
    """A renderer's surroundings without Blender or assets: a binary that answers 4.5.14, nothing prepared, and
    the run in Blender replaced."""
    run = Pretended()
    monkeypatch.setattr(blender_renderer, "find_blender", lambda: tmp_path / "blender")
    monkeypatch.setattr(blender_renderer, "blender_version", lambda binary: "4.5.14")
    monkeypatch.setattr(blender_renderer, "is_prepared", lambda library: False)
    monkeypatch.setattr(blender_renderer, "run_script", run)
    return run


def lettered_scene() -> SceneSpec:
    menu = SignArt("menu", text="coffee", width_m=0.6, height_m=0.9, seed=4)
    logo = SignArt("logo", text="coffee", width_m=0.5, height_m=0.3)
    board = Box(4.0, 0.0, math.pi, 0.06, 1.6, 0.35, z0=1.9, color=(30, 60, 90))
    parts = (Part(board, "painted_metal", group=2, signs=(Sign("front", logo), Sign("back", menu, emission=1.5)),
                  tint=(52, 56, 62)),
             Part(dataclasses.replace(board, cy=2.0), PAINT, signs=(Sign("front", logo, rect=(0.1, 0.1, 0.5, 0.9)),)),
             *a_dozen_parts())
    return SceneSpec(PLAIN_ROOM, parts=parts, lighting="evening", seed=11)


def test_the_renderer_is_named_blender_and_carries_its_version(pretended):
    renderer = BlenderRenderer()
    assert (BlenderRenderer.name, renderer.name, renderer.version) == ("blender", "blender", "4.5.14")
    assert (renderer.samples, renderer.device, renderer.keep) == (64, "OPTIX", False)
    assert BlenderRenderer(32, "CPU", None, True).samples == 32             # the order the plan gives them in
    assert LIGHTING[0] == "ceiling" and SceneSpec(PLAIN_ROOM).lighting in LIGHTING


def test_a_renderer_is_not_made_without_blender(monkeypatch):
    def missing():
        raise BlenderMissing("no Blender found. Fetch Blender 4.5 into the checkout with: ...")

    monkeypatch.setattr(blender_renderer, "find_blender", missing)
    with pytest.raises(BlenderMissing, match="Fetch Blender"):
        BlenderRenderer()


def test_the_job_holds_the_description_the_frames_the_settings_and_the_signs(pretended, tmp_path):
    spec, drawn = lettered_scene(), []

    def draw(art: SignArt) -> np.ndarray:
        drawn.append(art)
        return np.full((6, 9, 3), 10 * len(drawn), dtype=np.uint8)

    frames = [Frame(CAMERA), Frame(ASKEW, beauty=False), Frame(WIDENED, beauty=False, only=(2, 1))]
    renderer = BlenderRenderer(samples=48, device="CUDA", work_dir=tmp_path / "jobs", keep=True, draw=draw)
    results = renderer.render(spec, frames)
    ((script, args, options),), ((job, files),) = pretended.calls, pretended.jobs
    folder = renderer.last_job
    assert script == BLENDER_SIDE / "job.py" and args == ["--job", str(folder)] and folder.parent == tmp_path / "jobs"
    assert options["timeout_s"] >= 1800.0
    assert set(job) == {"spec", "frames", "settings", "prepared", "heights", "signs"}
    assert job["spec"] == to_json(spec) == json.loads(json.dumps(to_json(spec)))       # field for field
    assert [part["tint"] for part in job["spec"]["parts"][:3]] == [[52, 56, 62], None, None]   # a tint among them
    assert [frame["name"] for frame in job["frames"]] == ["000", "001", "002"]
    assert job["frames"][1] == {"name": "001", "fx": 600.0, "fy": 520.0, "cx": 250.3, "cy": 301.7, "width": 640,
                                "height": 480, "T_world_camera": ASKEW.T_world_camera.tolist(), "beauty": False,
                                "only": None}
    assert (job["frames"][0]["beauty"], job["frames"][2]["only"], job["frames"][2]["width"]) == (True, [2, 1], 990)
    assert job["settings"] == {"samples": 48, "device": "CUDA", "denoiser": "OPENIMAGEDENOISE", "view_transform": "AgX",
                               "look": "AgX - Punchy", "segments": 128}
    assert job["prepared"] is None and job["heights"] == {}                # nothing is prepared here
    # Each art once, though the logo is on two boards; the picture is in the folder before Blender starts.
    assert drawn == [spec.parts[0].signs[0].art, spec.parts[0].signs[1].art]
    assert job["signs"] == {"sign_000.png": to_json(spec)["parts"][0]["signs"][0]["art"],
                            "sign_001.png": to_json(spec)["parts"][0]["signs"][1]["art"]}
    assert set(files) == {"job.json", "sign_000.png", "sign_001.png"}
    for number, name in enumerate(("sign_000.png", "sign_001.png"), start=1):
        assert np.array_equal(np.asarray(Image.open(folder / name)), np.full((6, 9, 3), 10 * number, dtype=np.uint8))
    assert [(None if rgb is None else rgb.shape, depth.shape) for rgb, depth in results] == [
        ((480, 640, 3), (480, 640)), (None, (480, 640)), (None, (630, 990))]
    assert results[0][0].dtype == np.uint8 and int(results[0][0][0, 0, 0]) == 40
    assert [float(depth[0, 0]) for _, depth in results] == [2.0, 3.0, 4.0] and results[2][1].dtype == np.float32
    assert renderer.last_facts["device"] == "CPU"                           # what Blender said it used


def test_the_job_folder_is_removed_unless_it_is_to_be_kept(pretended, tmp_path):
    spec, frames = SceneSpec(PLAIN_ROOM), depth_only(CAMERA)
    passing = BlenderRenderer(work_dir=tmp_path / "jobs")
    passing.render(spec, frames)
    passing.render(spec, frames)
    assert len(pretended.jobs) == 2 and list((tmp_path / "jobs").iterdir()) == [] and passing.last_job is None
    keeping = BlenderRenderer(work_dir=tmp_path / "kept", keep=True)
    keeping.render(spec, frames)
    first = keeping.last_job
    keeping.render(spec, frames)
    assert first != keeping.last_job and sorted((tmp_path / "kept").iterdir()) == sorted([first, keeping.last_job])
    assert {path.name for path in first.iterdir()} == {"job.json", "000.depth.npy", "facts.json"}
    elsewhere = BlenderRenderer()                                           # no folder of its own: a temporary one
    elsewhere.render(spec, frames)
    assert not Path(pretended.calls[-1][1][1]).exists()


def test_no_frames_start_nothing(pretended):
    assert BlenderRenderer().render(lettered_scene(), []) == [] and pretended.calls == []


def test_a_frame_that_blender_did_not_write_is_an_error(pretended, tmp_path):
    pretended.write = False
    renderer = BlenderRenderer(work_dir=tmp_path / "jobs")
    with pytest.raises(RuntimeError, match="000.depth.npy"):
        renderer.render(SceneSpec(PLAIN_ROOM), depth_only(CAMERA))
    assert list((tmp_path / "jobs").iterdir()) == []                        # and nothing is left behind


def test_a_run_that_fails_leaves_no_folder_behind(pretended, monkeypatch, tmp_path):
    def failing(script, args, **options):
        raise RuntimeError("job.py failed in Blender with exit status 1; its log ends:\nValueError: no such thing")

    monkeypatch.setattr(blender_renderer, "run_script", failing)
    with pytest.raises(RuntimeError, match="no such thing"):
        BlenderRenderer(work_dir=tmp_path / "jobs").render(SceneSpec(PLAIN_ROOM), depth_only(CAMERA))
    assert list((tmp_path / "jobs").iterdir()) == []
    kept = BlenderRenderer(work_dir=tmp_path / "kept", keep=True)
    with pytest.raises(RuntimeError, match="no such thing"):
        kept.render(SceneSpec(PLAIN_ROOM), depth_only(CAMERA))
    assert (kept.last_job / "job.json").is_file()                           # kept, the job can be looked at


def test_what_blender_could_not_draw_is_refused_before_it_starts(pretended):
    with pytest.raises(ValueError, match="candlelight"):
        BlenderRenderer().render(SceneSpec(PLAIN_ROOM, lighting="candlelight"), depth_only(CAMERA))
    with pytest.raises(ValueError, match="METAL"):
        BlenderRenderer(device="METAL")
    with pytest.raises(ValueError, match="samples"):
        BlenderRenderer(samples=0)
    post = Part(Cylinder(4.0, 0.0, 0.2, 1.0), "steel", signs=(Sign("front", SignArt("plate")),))
    with pytest.raises(ValueError, match="box"):                            # a sign lies on a face of a box
        BlenderRenderer(draw=four_quarters).render(SceneSpec(PLAIN_ROOM, parts=(post,)), depth_only(CAMERA))
    crate = Part(Box(4.0, 0.0, 0.0, 1.0, 1.0, 1.0), "wood", signs=(Sign("bottom", SignArt("plate")),))
    with pytest.raises(ValueError, match="bottom"):
        BlenderRenderer(draw=four_quarters).render(SceneSpec(PLAIN_ROOM, parts=(crate,)), depth_only(CAMERA))
    assert pretended.calls == []


def test_models_and_people_need_the_prepared_assets(pretended, monkeypatch, tmp_path):
    plant = ModelRef("plant_a", x=5.0, y=1.0, z=0.0, yaw=0.0, scale=(1.0, 1.0, 1.0), size=(0.4, 0.8, 1.3))
    person = PersonRef("p0", "avatar_a", "waiting_a", x=5.0, y=2.0, yaw=math.pi, height=1.84)
    for spec in (SceneSpec(PLAIN_ROOM, models=(plant,)), SceneSpec(PLAIN_ROOM, people=(person,))):
        with pytest.raises(RuntimeError, match="sgi.synth.realistic.assets prepare"):
            BlenderRenderer().render(spec, depth_only(CAMERA))
    assert pretended.calls == []
    # Prepared, the job names the folder and the height each avatar has as it was made, to scale it from.
    prepared = tmp_path / "assets" / "_cache" / "prepared"
    people = {"avatar_a": AvatarAsset("avatar_a", "female", 1.73, frozenset()),
              "avatar_b": AvatarAsset("avatar_b", "male", 1.84, frozenset())}
    monkeypatch.setattr(blender_renderer, "load_library", lambda: Library({}, people, {}, {}, (), {}, prepared))
    monkeypatch.setattr(blender_renderer, "is_prepared", lambda library: True)
    BlenderRenderer().render(SceneSpec(PLAIN_ROOM, models=(plant,), people=(person,)), depth_only(CAMERA))
    job = pretended.jobs[0][0]
    assert job["prepared"] == str(prepared) and job["heights"] == {"avatar_a": 1.73}
    stranger = dataclasses.replace(person, avatar="avatar_z")
    with pytest.raises(ValueError, match="avatar_z"):
        BlenderRenderer().render(SceneSpec(PLAIN_ROOM, people=(stranger,)), depth_only(CAMERA))
    assert len(pretended.calls) == 1


def test_a_manifest_that_does_not_read_counts_as_nothing_prepared(pretended, monkeypatch):
    def broken():
        raise ValueError("assets/realistic/manifest/models.json: not JSON")

    monkeypatch.setattr(blender_renderer, "load_library", broken)
    BlenderRenderer().render(SceneSpec(PLAIN_ROOM), depth_only(CAMERA))
    assert pretended.jobs[0][0]["prepared"] is None


def test_the_default_drawing_is_the_sign_modules_own(pretended, monkeypatch):
    asked = []

    def draw(art, font=None, px_per_m=512):
        asked.append((art, font, px_per_m))
        return np.zeros((8, 8, 3), dtype=np.uint8)

    # The sign module is another task's and is asked for only when a sign is to be drawn: one is put in its place
    # here, whether or not it is there yet.
    stand_in = types.ModuleType("sgi.synth.realistic.signs")
    stand_in.draw = draw
    monkeypatch.setitem(sys.modules, "sgi.synth.realistic.signs", stand_in)
    monkeypatch.setattr(realistic, "signs", stand_in, raising=False)
    spec = lettered_scene()
    renderer = BlenderRenderer()
    renderer.render(SceneSpec(PLAIN_ROOM, parts=a_dozen_parts()), depth_only(CAMERA))
    assert asked == []                                                      # nothing lettered, nothing asked of it
    renderer.render(spec, depth_only(CAMERA))
    assert asked == [(spec.parts[0].signs[0].art, None, 512), (spec.parts[0].signs[1].art, None, 512)]


def test_a_drawing_that_is_no_picture_is_refused(pretended):
    for wrong in (np.zeros((8, 8), dtype=np.uint8), np.zeros((8, 8, 3), dtype=np.float64),
                  np.zeros((8, 8, 4), dtype=np.uint8)):
        with pytest.raises(ValueError, match="sign"):
            BlenderRenderer(draw=lambda art, wrong=wrong: wrong).render(lettered_scene(), depth_only(CAMERA))
    assert pretended.calls == []


# ------------------------------------------------------------------------------------------------ the two sides

def imported(path: Path, everywhere: bool = True) -> set[str]:
    """The first names of what a file imports: anywhere in it, or at its top alone, as it is loaded."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree) if everywhere else tree.body:
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module.split(".")[0])
    return names


def test_the_two_sides_share_files_only():
    mine = [BLENDER_SIDE / name for name in ("job.py", "build.py", "render.py")]
    for path in mine:                                                       # Blender's Python has none of these
        assert imported(path) & {"sgi", "pydantic", "PIL"} == set(), path.name
    assert "bpy" in imported(BLENDER_SIDE / "build.py") and "bpy" in imported(BLENDER_SIDE / "render.py")
    assert imported(Path(blender_renderer.__file__)) & {"bpy", "mathutils", "bmesh"} == set()   # and sgi has no bpy
    # The asset library's loaders are asked for when a job needs them, not when the scripts are loaded: a job
    # without a model, a person or a prepared material runs where there is no library at all.
    assert "library" in imported(BLENDER_SIDE / "build.py")
    for path in mine:
        assert "library" not in imported(path, everywhere=False), path.name
    assert "signs" not in {alias.name for node in ast.parse(Path(blender_renderer.__file__).read_text()).body
                           if isinstance(node, ast.ImportFrom) for alias in node.names}
