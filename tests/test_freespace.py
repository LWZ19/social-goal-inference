from dataclasses import replace

import numpy as np
import pytest

from tests.helpers import ROBOT, floor_observation, scene_dict, write_scene
import sgi.freespace as fs
from sgi.config import Thresholds
from sgi.scene import Occupancy, Person, load_scene

TH = Thresholds()                    # robot_radius_m 0.30, person_radius_m 0.25, grid_resolution_m 0.05
START = (ROBOT.x, ROBOT.y)           # (1.0, 2.0)

@pytest.fixture(scope="module")
def base(tmp_path_factory):          # bounds x -9..11, y -8..12; one obstacle x 0.5..1.5, y 6..7; no people
    folder = write_scene(tmp_path_factory.mktemp("scene"), scene_dict(), floor_observation())
    return load_scene(folder)[1]     # a frozen GroundTruth; tests derive variants with replace()

def test_static_grid_from_polygons(base):
    g = fs.static_grid(base, 0.05)
    assert (g.x0, g.y0, g.resolution) == (-9.0, -8.0, 0.05)
    assert g.free.shape == (400, 400) and g.free.dtype == bool and int(g.free.sum()) == 159600   # 20 x 20 cells blocked
    assert g.index(1.025, 6.525) == (290, 200) and g.center(290, 200) == pytest.approx((1.025, 6.525))   # (iy, ix)
    assert g.is_free(1.025, 6.525) is False and g.is_free(1.025, 5.975) is True   # inside the obstacle, just below it
    assert g.index(11.01, 2.0) is None and g.index(1.0, -8.01) is None
    assert g.is_free(-9.6, 2.025) is False                   # outside: index -12 must not wrap to the far side
    tri = fs.static_grid(replace(base, bounds=[(0.0, 0.0), (1.02, 0.0), (0.0, 0.73)], obstacles=()), 0.05)
    assert (tri.x0, tri.y0, tri.free.shape) == (0.0, 0.0, (15, 21))               # 14.6 by 20.4 cells, rounded up
    assert int(tri.free.sum()) == 150 and tri.is_free(0.475, 0.375) and not tri.is_free(0.525, 0.375)

def occ(img, origin, resolution=0.05):
    return dict(occupancy=Occupancy(image="occupancy.png", resolution=resolution, origin=origin), occupancy_image=img)

def test_occupancy_image_gives_the_polygon_grid_and_a_flipped_one_does_not(base):
    img = np.zeros((430, 440), np.uint8)     # x -10..12, y -8.5..13 at 0.05 m; row 0 is the top, y 12.95..13
    img[20:420, 20:420] = 255                # the bounds: x -9..11, y -8..12
    img[120:140, 210:230] = 0                # the obstacle: x 0.5..1.5, y 6..7
    want = fs.static_grid(base, 0.05).free
    got = fs.static_grid(replace(base, obstacles=(), **occ(img, [-10.0, -8.5])), 0.05)
    assert np.array_equal(got.free, want)                    # the obstacle now comes from the image alone
    flipped = fs.static_grid(replace(base, obstacles=(), **occ(img[::-1], [-10.0, -8.5])), 0.05)
    assert int((flipped.free != want).sum()) == 4800         # strip y -8..-7.5 (4000), obstacle old and new (800)
    assert flipped.is_free(1.025, 6.525) and not flipped.is_free(1.025, -1.975)   # obstacle now at y -2.5..-1.5

def test_occupancy_threshold_both_forms_and_coarser_pixels(base):
    img = np.full((40, 60), 255, np.uint8)   # x 0..3, y 5..7: white over the obstacle polygon as well
    img[0, 0], img[0, 1] = 249, 250          # top row, y 6.95..7: x 0..0.05 and x 0.05..0.10
    g = fs.static_grid(replace(base, **occ(img, [0.0, 5.0])), 0.05)
    assert g.free.shape == (400, 400) and int(g.free.sum()) == 1999      # 2400 pixels: 1 dark, 400 under the polygon
    assert not g.is_free(0.025, 6.975) and g.is_free(0.075, 6.975) and g.is_free(0.025, 5.025)
    assert not g.is_free(1.025, 6.525)                                   # a white pixel inside the obstacle polygon
    assert not g.is_free(3.025, 6.025) and not g.is_free(1.025, 4.975)   # inside the bounds, outside the image
    coarse = np.full((30, 50), 255, np.uint8)                # 0.1 m pixels: x 2..7, y -1..2
    coarse[0, 49] = 0                                        # top-right pixel: x 6.9..7, y 1.9..2
    g = fs.static_grid(replace(base, bounds=None, obstacles=(), **occ(coarse, [2.0, -1.0, 0.0], 0.1)), 0.05)
    assert (g.x0, g.y0, g.free.shape) == (2.0, -1.0, (60, 100)) and int(g.free.sum()) == 5996
    assert not g.is_free(6.975, 1.975) and not g.is_free(6.925, 1.925) and g.is_free(6.875, 1.975)
    with pytest.raises(ValueError):                          # neither bounds nor occupancy
        fs.static_grid(replace(base, bounds=None), 0.05)

def test_robot_grid_inflates_obstacles_and_the_grid_edge(base):
    g = fs.robot_grid(base, ROBOT, TH)
    assert not g.is_free(1.025, 5.75) and g.is_free(1.025, 5.60)      # 0.25 m and 0.40 m below the obstacle's edge
    assert not g.is_free(1.025, 5.725) and g.is_free(1.025, 5.675)    # centres 6 and 7 cells from an obstacle cell
    assert not g.is_free(-8.75, 2.025) and g.is_free(-8.60, 2.025)    # 0.25 m and 0.40 m from the grid edge
    assert int(g.free.sum()) == 149576                                # 388 x 388 minus the inflated obstacle's 968
    small = fs.robot_grid(base, ROBOT, Thresholds(robot_radius_m=0.1, grid_resolution_m=0.1))
    assert small.free.shape == (200, 200) and int(small.free.sum()) == 39064      # 198 x 198 minus 140

def test_people_block_and_the_robot_position_stays_free(base):
    g = fs.robot_grid(replace(base, people=(Person(id="p0", x=3.0, y=2.0, yaw=0.0),)), ROBOT, TH)
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        assert not g.is_free(3.0 + 0.5 * dx, 2.0 + 0.5 * dy) and g.is_free(3.0 + 0.6 * dx, 2.0 + 0.6 * dy)
    assert int(g.free.sum()) == 149576 - 384                 # a 0.55 m disc about (3.0, 2.0) itself, a cell corner
    near = fs.robot_grid(replace(base, people=(Person(id="p0", x=1.3, y=2.0, yaw=0.0),)), ROBOT, TH)
    assert near.is_free(1.0, 2.0) and near.is_free(1.2, 2.0) and near.is_free(0.8, 2.0)   # within 0.30 m of the robot
    assert not near.is_free(1.35, 2.0) and fs.reachable(near, START, (-2.025, 2.025)) is True
    wall = fs.robot_grid(base, replace(ROBOT, y=5.8), TH)    # the robot 0.2 m below the obstacle's edge
    assert wall.is_free(1.025, 5.825) and wall.is_free(1.025, 6.025) and fs.reachable(wall, (1.0, 5.8), START) is True

def test_connected_is_four_connected():
    g = fs.Grid(x0=0.0, y0=0.0, resolution=1.0, free=np.array([[True, False], [False, True]]))
    assert fs.connected(g, (0.5, 0.5)).tolist() == [[True, False], [False, False]]
    assert not fs.connected(g, (1.5, 0.5)).any() and not fs.connected(g, (-0.5, 0.5)).any()   # start blocked or outside

def test_reachable(base):
    def boxed(door):                 # walls 0.2 m thick around x 4..6, y 4..6; a doorway that wide in the south wall
        spans = [(3.8, 3.8, 5.0 - door / 2, 4.0), (5.0 + door / 2, 3.8, 6.2, 4.0), (3.8, 6.0, 6.2, 6.2),
                 (3.8, 4.0, 4.0, 6.0), (6.0, 4.0, 6.2, 6.0)]
        walls = tuple([(xa, ya), (xb, ya), (xb, yb), (xa, yb)] for xa, ya, xb, yb in spans)
        return fs.robot_grid(replace(base, obstacles=base.obstacles + walls), ROBOT, TH)
    goal, closed = (5.025, 5.025), boxed(0.0)
    assert closed.is_free(*goal) and fs.reachable(closed, START, goal) is False       # enclosed by obstacles
    assert int(closed.free.sum()) - int(fs.connected(closed, START).sum()) == 784     # the 1.4 m square left inside
    assert fs.reachable(boxed(0.5), START, goal) is False             # a doorway narrower than the robot
    assert fs.reachable(boxed(0.8), START, goal) is True
    assert fs.reachable(fs.robot_grid(base, ROBOT, TH), START, goal) is True
    assert fs.reachable(closed, START, (1.025, 6.525)) is False       # goal inside an obstacle
    assert fs.reachable(closed, START, (-9.6, 2.025)) is False and fs.reachable(closed, (30.0, 2.0), goal) is False
