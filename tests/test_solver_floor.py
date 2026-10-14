import math
from dataclasses import replace

import numpy as np
import pytest

from sgi import solver
from sgi.config import Thresholds, load_config
from sgi.freespace import connected, robot_grid, static_grid
from sgi.geometry import Camera, RobotPose, camera_on_robot, floor_in_view, ground_hits, project
from sgi.gt.derive import is_visible
from sgi.inputs import ObservationData
from sgi.scene import load_scene
from sgi.synth.render import Box, Figure, render
from sgi.types import DetectedPerson
from tests.solver_layouts import (EDGE_AHEAD_M, EDGE_HALF_WIDTH_M, HERE, LENS, TH, cells, moved, moved_xy, mounted,
                                  mounted_camera, person)
from tests.test_run import placeholder, scene_dirs  # noqa: F401

MASKS = ("seen", "hidden", "strip", "blocked", "own", "passable", "standable", "selectable")
PLATE = Box(cx=4.25, cy=0.0, yaw=0.0, sx=0.5, sy=8.0, sz=0.10)       # 0.10 m high, from 4.0 to 4.5 m ahead, across the picture
PILLAR = Box(cx=5.1, cy=0.0, yaw=0.0, sx=0.2, sy=0.2, sz=1.0)        # its face 5.0 m ahead, 0.2 m wide
WALL = Box(cx=6.15, cy=0.0, yaw=0.0, sx=0.3, sy=16.0, sz=2.5)        # its face 6.0 m ahead, across the picture
BEFORE_WALL = [(4.0, -0.6), (4.5, 0.5), (3.4, 1.2), (5.0, -1.6)]     # people who stand between the camera and that wall


def at(floor, mask, x, y):
    """Whether the cell that holds a position of the frame of `HERE` is in a mask."""
    return bool(getattr(floor, mask)[floor.index(x, y)])


def where(floor, mask, spots):
    return [at(floor, mask, x, y) for x, y in spots]


def with_depth(observation, depth):
    return replace(observation, depth=depth.astype(np.float32))


def test_the_grid_lies_in_the_robots_frame_out_to_the_floor_range():
    robot = RobotPose(3.0, -1.0, math.radians(30.0))
    floor = solver.floor(mounted(robot), [], TH)
    assert floor.robot == robot and floor.offsets.shape == (481,)        # 12 m either way in cells of 0.05 m
    assert floor.offsets[[0, 240, 241, 480]] == pytest.approx([-12.0, 0.0, 0.05, 12.0], abs=1e-12)
    assert all(getattr(floor, mask).shape == (481, 481) and getattr(floor, mask).dtype == bool for mask in MASKS)
    centres = floor.centres()                                            # rows run to the robot's left, columns ahead of it
    assert centres.shape == (481, 481, 2) and centres[240, 240] == pytest.approx((3.0, -1.0), abs=1e-12)
    assert centres[240, 260] == pytest.approx((3.0 + math.cos(robot.yaw), -1.0 + math.sin(robot.yaw)), abs=1e-12)
    assert centres[260, 240] == pytest.approx((3.0 - math.sin(robot.yaw), -1.0 + math.cos(robot.yaw)), abs=1e-12)
    assert floor.index(3.0, -1.0) == (240, 240) and floor.index(*centres[17, 400]) == (17, 400)
    assert floor.index(3.0 + 0.024, -1.0) == (240, 240)                  # a cell holds what is nearer its centre than half a cell
    assert floor.index(3.0, 11.0) == (448, 360) and floor.index(3.0, 14.0) is None      # 12 and 15 m away, half left
    small = solver.floor(mounted(robot), [], Thresholds(floor_range_m=3.0, grid_resolution_m=0.1))
    assert small.offsets.shape == (61,) and small.offsets[[0, 30, 60]] == pytest.approx([-3.0, 0.0, 3.0], abs=1e-12)


def test_a_bare_floor_is_seen_where_the_picture_shows_it_and_the_strip_joins_it_to_the_robot():
    floor = solver.floor(mounted(), [], TH)
    assert (EDGE_AHEAD_M, EDGE_HALF_WIDTH_M) == pytest.approx((2.474, 1.443), abs=1e-3)    # where the picture ends
    # Seen: in the picture, which shows the floor from 2.474 m ahead and 0.569 m to a side per metre farther.
    assert where(floor, "seen", [(2.5, 0.0), (5.0, 0.0), (5.0, 2.8), (5.0, -2.8), (12.0, 0.0)]) == [True] * 5
    assert where(floor, "seen", [(2.45, 0.0), (5.0, 3.0), (-1.0, 0.0), (0.0, 0.0)]) == [False] * 4
    assert not at(floor, "seen", 11.95, 1.5) and at(floor, "seen", 11.9, 1.5)       # 12.04 and 11.99 m from the robot
    # The strip: under the picture, in the triangle from the robot to the picture's two bottom corners.
    assert where(floor, "strip", [(2.45, 0.0), (1.0, 0.0), (1.0, 0.5), (1.0, -0.5), (0.35, 0.0), (2.45, 1.4)]) == [True] * 6
    assert not at(floor, "strip", 1.0, 0.6)          # under the picture and between its side edges, but beside the triangle
    assert where(floor, "strip", [(2.5, 0.0), (-0.5, 0.0), (5.0, 0.0)]) == [False] * 3
    triangle_m2 = EDGE_AHEAD_M * EDGE_HALF_WIDTH_M
    assert floor.strip.sum() * TH.grid_resolution_m ** 2 == pytest.approx(triangle_m2, rel=0.02)
    assert not floor.hidden.any() and not floor.blocked.any() and not (floor.seen & floor.strip).any()
    # Own: within the robot's radius of where it stands.
    assert where(floor, "own", [(0.0, 0.0), (0.3, 0.0), (0.0, -0.3), (0.2, 0.2), (-0.25, 0.15)]) == [True] * 5
    assert where(floor, "own", [(0.3, 0.05), (0.25, 0.2), (0.35, 0.0)]) == [False] * 3 and floor.own.sum() == 113
    floor_cells = floor.seen | floor.strip | floor.own
    assert np.array_equal(floor.passable, floor_cells) and np.array_equal(floor.standable, floor_cells)
    assert np.array_equal(floor.selectable, floor_cells)                 # every one of them is joined to the robot


def under_the_picture(observation):
    """The centres of the grid cells of `HERE` in the triangle from the robot to the picture's bottom corners that
    the camera has under its lower edge, each with what the standard goal's rule says of it."""
    camera, steps, out = observation.camera, np.round(np.arange(-40, 60) * 0.05, 2), []
    for x in steps[(steps >= 0) & (steps < EDGE_AHEAD_M)]:
        for y in steps[np.abs(steps) <= x * EDGE_HALF_WIDTH_M / EDGE_AHEAD_M]:
            if project(camera, (x, y, 0.0))[1] >= camera.height:
                shown = floor_in_view(observation.depth, camera, (x, y), TH.floor_tolerance_m,
                                      TH.visibility_depth_tolerance_m, TH.lift_window)
                out.append(((float(x), float(y)), shown))
    return out


def test_the_strip_is_the_floor_the_standard_goal_counts_under_the_picture():
    bare = mounted()
    low = Box(cx=2.2, cy=0.0, yaw=0.0, sx=0.4, sy=0.6, sz=0.5)           # 2.0 m ahead: it stands at the foot of the columns
    stood = mounted(boxes=[low])                                         # 239 to 401, 0.31 m over the floor in the lowest row
    for observation, all_shown in ((bare, True), (stood, False)):
        floor, spots = solver.floor(observation, [], TH), under_the_picture(observation)
        shown = [xy for xy, is_shown in spots if is_shown]
        assert len(spots) > 1400 and (len(shown) == len(spots) if all_shown else 900 < len(shown) < 1200)
        assert all(at(floor, "strip", *xy) == is_shown for xy, is_shown in spots)
        assert floor.strip.sum() == len(shown)                           # and no cell beside those
    floor = solver.floor(stood, [], TH)
    assert where(floor, "strip", [(1.0, 0.0), (1.0, 0.1), (1.0, -0.1), (0.5, 0.0)]) == [False] * 4       # behind that thing
    assert where(floor, "strip", [(1.0, 0.3), (1.0, -0.3), (2.0, 1.0)]) == [True] * 3                    # beside it
    assert not at(floor, "passable", 1.0, 0.0) and at(floor, "selectable", 1.0, 0.3) and at(floor, "selectable", 0.0, 0.0)
    # A camera that looks up has no floor under its picture: the rays of its bottom corners miss the ground.
    up = Camera(**LENS, T_world_camera=camera_on_robot(HERE, 0.2, 1.5, math.radians(-30.0)))
    sky = solver.floor(ObservationData(*render(up), up, HERE), [], TH)
    assert not sky.strip.any() and not sky.seen.any() and np.array_equal(sky.selectable, sky.own)


def test_floor_hidden_from_the_camera_is_crossed_and_not_stood_on():
    # Over the plate and behind it, as far as the rays over its far edge land (4.81 m), the picture shows the plate:
    # 0.10 m up, which is 6.7 % nearer than the floor on that ray. Up to 4.10 m the ray meets its near face lower than
    # that, inside what counts as the floor's own depth (3 % and 0.02 m).
    floor = solver.floor(mounted(boxes=[PLATE]), [], TH)
    band = [(x, 0.0) for x in (4.2, 4.3, 4.5, 4.7, 4.8)] + [(4.5, 2.0), (4.5, -2.0)]
    assert where(floor, "hidden", band) == [True] * 7 and where(floor, "seen", band) == [False] * 7
    assert where(floor, "seen", [(3.95, 0.0), (4.05, 0.0), (4.9, 0.0), (5.5, 0.0)]) == [True] * 4
    assert not floor.blocked.any()                                       # 0.10 m is inside the floor tolerance: nothing stands
    assert where(floor, "passable", band) == [True] * 7 and where(floor, "standable", band) == [False] * 7
    assert where(floor, "selectable", band) == [False] * 7
    across = cells(floor.seen | floor.strip, floor, lambda x, y: (x > 4.19) & (x < 4.76))
    assert not across.any()                                              # no floor to stand on leads past the plate
    assert at(floor, "selectable", 3.9, 0.0) and at(floor, "selectable", 5.5, 0.0) and at(floor, "selectable", 9.0, 2.0)

    bare = mounted()
    ahead = ground_hits(bare.camera)[0][..., 0]                          # how far ahead each pixel shows the floor
    rows = (ahead > 4.0) & (ahead < 4.8)
    unknown = solver.floor(with_depth(bare, np.where(rows, 0.0, bare.depth)), [], TH)     # no depth there
    assert where(unknown, "hidden", [(4.2, 0.0), (4.6, 0.0)]) == [True] * 2 and not at(unknown, "standable", 4.4, 0.0)
    assert at(unknown, "selectable", 5.5, 0.0)
    nearer = solver.floor(with_depth(bare, np.where(rows, 0.93 * bare.depth, bare.depth)), [], TH)
    assert where(nearer, "hidden", [(4.2, 0.0), (4.6, 0.0)]) == [True] * 2 and at(nearer, "selectable", 5.5, 0.0)
    within = solver.floor(with_depth(bare, np.where(rows, 0.98 * bare.depth, bare.depth)), [], TH)
    assert where(within, "seen", [(4.2, 0.0), (4.6, 0.0)]) == [True] * 2                  # 2 % nearer: the floor's own depth
    # A depth beyond the floor's: there is no ground at the cell. It is neither seen nor hidden, and nothing crosses it.
    drop = solver.floor(with_depth(bare, np.where(rows, 1.3 * bare.depth, bare.depth)), [], TH)
    assert where(drop, "seen", [(4.2, 0.0), (4.6, 0.0)]) == [False] * 2 == where(drop, "hidden", [(4.2, 0.0), (4.6, 0.0)])
    assert at(drop, "seen", 5.5, 0.0) and not at(drop, "selectable", 5.5, 0.0) and at(drop, "selectable", 3.9, 0.0)


def test_what_stands_over_the_floor_and_every_person_block_the_cells_round_them():
    floor = solver.floor(mounted(boxes=[PILLAR]), [], TH)                # the picture shows its face, in the cells 5.0 m ahead
    assert where(floor, "blocked", [(4.6, 0.0), (5.0, 0.0), (5.0, 0.5), (4.7, 0.35), (4.65, -0.2)]) == [True] * 5
    assert where(floor, "blocked", [(4.55, 0.0), (5.0, 0.55), (4.7, 0.4), (4.0, 0.0)]) == [False] * 4
    assert at(floor, "seen", 4.6, 0.0) and not at(floor, "passable", 4.6, 0.0) and at(floor, "selectable", 4.55, 0.0)
    assert at(floor, "hidden", 6.5, 0.0) and at(floor, "passable", 6.5, 0.0) and not at(floor, "selectable", 6.5, 0.0)
    tight = solver.floor(mounted(boxes=[PILLAR]), [], Thresholds(standard_goal_clearance_m=0.2))
    assert at(tight, "blocked", 4.8, 0.0) and not at(tight, "blocked", 4.75, 0.0)
    # What is higher than the camera is in nobody's way, and hides no floor.
    overhead = Box(cx=4.5, cy=0.0, yaw=0.0, sx=1.0, sy=2.0, sz=0.2, z0=1.8)
    under = solver.floor(mounted(boxes=[overhead]), [], TH)
    assert not under.blocked.any() and not under.hidden.any() and at(under, "selectable", 4.5, 0.0)
    # A detected person blocks the cells within their radius and the robot's of their centre, body in the picture or not.
    stood = solver.floor(mounted(), [person(1, 5.0, 0.0)], TH)
    assert where(stood, "blocked", [(5.55, 0.0), (4.45, 0.0), (5.4, 0.35), (5.0, -0.55), (5.0, 0.0)]) == [True] * 5
    assert where(stood, "blocked", [(5.6, 0.0), (5.4, 0.4), (4.4, 0.0)]) == [False] * 3
    assert stood.blocked.sum() == 377                                    # the cells within 11 cells' width of a centre
    wide = solver.floor(mounted(), [person(1, 5.0, 0.0)], Thresholds(person_radius_m=0.35, robot_radius_m=0.25))
    assert at(wide, "blocked", 5.6, 0.0) and not at(wide, "blocked", 5.65, 0.0)


def test_the_robots_own_disc_is_floor_whatever_the_picture_shows():
    close = solver.floor(mounted(), [person(1, 0.4, 0.0)], TH)          # someone 0.4 m ahead of the robot
    assert at(close, "blocked", 0.2, 0.0) and at(close, "selectable", 0.2, 0.0) and at(close, "selectable", 0.0, 0.0)
    assert at(close, "strip", 0.35, 0.0) and not at(close, "passable", 0.35, 0.0)
    bare = mounted()
    dark = solver.floor(with_depth(bare, np.zeros_like(bare.depth)), [], TH)       # a picture with no depth at all
    assert not dark.seen.any() and not dark.strip.any() and dark.hidden.any()
    assert np.array_equal(dark.selectable, dark.own) and dark.selectable.sum() == 113 and dark.selectable[240, 240]


def wall_with_people(robot=HERE, turn=0.0, shift=(0.0, 0.0)):
    figures = [Figure(x, y, math.pi) for x, y in BEFORE_WALL]
    observation = mounted(boxes=[WALL], figures=figures)
    if robot is not HERE:                                                # the same picture, taken somewhere else
        observation = replace(observation, camera=mounted_camera(robot), robot=robot)
    people = [person(n, *moved_xy(xy, turn, shift), math.pi + turn) for n, xy in enumerate(BEFORE_WALL, 1)]
    return observation, people


def test_no_cell_behind_a_wall_is_a_place_though_people_stand_before_it():
    observation, people = wall_with_people()
    floor = solver.floor(observation, people, TH)
    behind = lambda x, y: x > 6.0                                        # past the wall's face
    assert not cells(floor.selectable, floor, behind).any() and not cells(floor.standable, floor, behind).any()
    assert cells(floor.hidden, floor, behind).sum() > 10000              # the wall and the bodies hide what is there
    # The wall's own depth is that of the floor up to 0.2 m behind its face (3 % of 6 m and 0.02 m): seen, and blocked.
    assert cells(floor.seen, floor, behind).any() and not cells(floor.seen, floor, lambda x, y: x > 6.21).any()
    assert cells(floor.blocked, floor, lambda x, y: floor.seen & behind(x, y)).all()
    assert not cells(floor.selectable, floor, lambda x, y: x > 5.6).any()            # nor within the clearance of its face
    assert where(floor, "selectable", [(5.55, 0.0), (5.55, 2.0), (3.0, 0.0), (4.5, -0.3)]) == [True] * 4
    shadows = cells(floor.hidden & ~floor.blocked, floor, lambda x, y: x < 5.5)      # the bodies hide floor before the wall
    assert shadows.sum() > 200 and all(at(floor, "blocked", x, y) for x, y in BEFORE_WALL)


def test_the_same_picture_gives_the_same_floor_wherever_the_world_frame_lies():
    here = solver.floor(*wall_with_people(), TH)
    turn, shift = math.radians(40.0), (3.1, -2.3)
    there = solver.floor(*wall_with_people(moved(HERE, turn, shift), turn, shift), TH)
    assert there.robot == moved(HERE, turn, shift) and np.array_equal(there.offsets, here.offsets)
    for mask in MASKS:                                                   # a cell on the edge of a pixel may fall either way
        assert (getattr(here, mask) != getattr(there, mask)).mean() < 1e-4, mask
    assert there.centres()[100, 300] == pytest.approx(moved_xy(here.centres()[100, 300], turn, shift), abs=1e-9)


def oracle_people(truth, observation, th):
    seen = [p for p in truth.people if is_visible(p, observation, th)]
    return [DetectedPerson(number=n, xy=(p.x, p.y), yaw=p.yaw, pixel=(0.0, 0.0)) for n, p in enumerate(seen, 1)]


def shows_floor(observation, x, y, th):
    """Whether the picture shows the floor at each ground position: it projects into the picture, and the depth
    at that pixel is the position's own to the tolerance of the scene check."""
    camera = observation.camera
    to_camera = np.linalg.inv(camera.T_world_camera)
    xc, yc, zc = (to_camera[k, 0] * x + to_camera[k, 1] * y + to_camera[k, 3] for k in range(3))
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = camera.fx * xc / zc + camera.cx, camera.fy * yc / zc + camera.cy
    inside = (zc > 0) & (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
    depth = observation.depth[np.where(inside, v, 0).astype(int), np.where(inside, u, 0).astype(int)]
    return inside & np.isfinite(depth) & (np.abs(depth - zc) <= th.floor_depth_max_rel_error * np.abs(zc) + 0.02)


def test_on_every_placeholder_scene_no_place_lies_outside_the_free_space(placeholder):
    th = load_config().thresholds
    folders = scene_dirs(placeholder / "eval") + scene_dirs(placeholder / "examples")
    assert len(folders) >= 14
    places = found = stood = true_floor = 0
    for folder in folders:
        inputs, truth = load_scene(folder)
        observation, robot = inputs.observation, inputs.observation.robot
        floor = solver.floor(observation, oracle_people(truth, observation, th), th)
        assert floor.selectable[240, 240]                                # the nearest place is where the robot stands
        static = static_grid(truth, th.grid_resolution_m)
        outside = [xy for xy in floor.centres()[floor.selectable] if not static.is_free(*xy)]
        assert outside == [], inputs.scene_id
        places += int(floor.selectable.sum())
        # Of the floor the robot's centre can truly reach and the picture truly shows, within range: how much of it
        # the rule calls floor, and how much a place. The truth keeps 0.30 m from what stands and the rule 0.40 m.
        grid = robot_grid(truth, robot, th)
        ny, nx = grid.free.shape
        x, y = np.meshgrid(grid.x0 + (np.arange(nx) + 0.5) * grid.resolution, grid.y0 + (np.arange(ny) + 0.5) * grid.resolution)
        true = connected(grid, (robot.x, robot.y)) & shows_floor(observation, x, y, th)
        true &= np.hypot(x - robot.x, y - robot.y) < th.floor_range_m
        held = [floor.index(*xy) for xy in zip(x[true], y[true])]
        rows, cols = np.array(held).T
        true_floor += len(held)
        found += int((floor.seen | floor.strip)[rows, cols].sum())
        stood += int(floor.selectable[rows, cols].sum())
    # Measured on the set as it is generated now: 240,032 places, 0.992 of that floor called floor and 0.946 a place.
    assert places > 150000 and found / true_floor > 0.98 and stood / true_floor > 0.92
