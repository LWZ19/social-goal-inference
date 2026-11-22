import math
import numpy as np
import pytest
import sgi.geometry as g
from tests.helpers import INTRINSICS, MOUNT, ROBOT, camera_for, floor_observation

# Fixed case: robot at (1, 2) facing world +Y; camera 0.2 m ahead, 1.2 m up, pitched 20 degrees down.
# Columns: camera +X (right), +Y (down), +Z (optical axis), origin. 0.9396926... = cos 20, 0.3420201... = sin 20.
T_FIXED = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, -0.3420201433256687, 0.9396926207859084, 2.2],
                    [0.0, -0.9396926207859084, -0.3420201433256687, 1.2], [0.0, 0.0, 0.0, 1.0]])
OTHER = g.RobotPose(-2.0, 0.5, 3.0)

def test_camera_on_robot_columns():
    T = g.camera_on_robot(ROBOT, **MOUNT)
    assert T.shape == (4, 4) and T.dtype == np.float64 and T == pytest.approx(T_FIXED, abs=1e-12)

def test_wrap_angle_is_half_open():
    assert g.wrap_angle(-math.pi) == math.pi and g.wrap_angle(math.pi) == math.pi and g.wrap_angle(0.0) == 0.0
    assert g.wrap_angle(7.0) == pytest.approx(0.7168146928204138)         # 7 - 2 pi

@pytest.mark.parametrize("x, y, u, v", [                                  # four corners, the centre, the fixed case
    (0, 0, 0.0, 0.0), (1000, 0, 639.999, 0.0), (0, 1000, 0.0, 479.999), (1000, 1000, 639.999, 479.999),
    (500, 500, 320.0, 240.0), (626, 572, 400.64, 274.56)])
def test_point_pixel_corners_and_centre(x, y, u, v):
    assert g.point_to_pixel(x, y, 640, 480) == pytest.approx((u, v), abs=1e-9)
    point = g.pixel_to_point(np.float64(u), np.float64(v), 640, 480)
    assert point == (x, y) and all(type(c) is int for c in point)
    assert g.pixel_to_point(-5.0, 700.0, 640, 480) == (0, 1000)           # clipped to 0..1000

def test_make_poses_with_robot_yaw():
    pose_world, pose_robot = g.make_poses(ROBOT, np.array([1.5, 5.0]), 3 * math.pi)     # 3 pi is wrapped to pi
    assert pose_world == pytest.approx((1.5, 5.0, math.pi)) and pose_robot == pytest.approx((3.0, -0.5, math.pi / 2))
    assert all(type(c) is float for c in pose_world + pose_robot)
    assert g.to_world_xy(ROBOT, 3.0, -0.5) == pytest.approx((1.5, 5.0))   # back from the robot frame
    _, pose_robot = g.make_poses(OTHER, (0.0, 0.0), -3.0)                 # theta_robot = -6 + 2 pi
    assert pose_robot == pytest.approx((-2.050544997231, 0.212756232180, 0.283185307180), abs=1e-9)
    assert g.make_poses(OTHER, (0.3, 0.4), OTHER.yaw)[1][2] == 0.0        # theta_world = yaw keeps the heading
    assert g.bearing((0.0, 0.0), (-1.0, 0.0)) == math.pi and g.bearing((1.0, 1.0), (1.0, 1.0)) == 0.0

def test_fixed_case_from_pixel_and_depth_to_pose_robot():
    cam, u, v = g.Camera(**INTRINSICS, T_world_camera=T_FIXED), 400.64, 274.56          # the pixel of Point(626, 572)
    p = g.lift(np.full((480, 640), 3.0, np.float32), cam, u, v)
    assert p.shape == (3,) and p.dtype == np.float64                      # z stays as lifted: 1.8 mm under the floor
    assert p == pytest.approx([1.436442359733, 4.955104114459, -0.001826857765], abs=1e-9)
    pose_world, pose_robot = g.make_poses(ROBOT, p[:2], g.bearing(p[:2], (1.0, 6.0)))    # face a target at (1, 6)
    assert pose_world == pytest.approx((1.436442359733, 4.955104114459, 1.966458917087), abs=1e-9)
    assert pose_robot == pytest.approx((2.955104114459, -0.436442359733, 0.395662590292), abs=1e-9)

def test_lift_window_ignores_invalid_depth():
    cam, u, v = g.Camera(**INTRINSICS, T_world_camera=np.eye(4)), 386.516, 84.796       # world = camera frame
    depth = np.full((480, 640), 9.0, np.float32)
    depth[82:87, 384:389] = [0.0, np.nan, np.inf, -1.0, -np.inf]          # the window: rows 82..86, columns 384..388
    assert g.lift(depth, cam, u, v) is None                               # the 9.0 outside the window is not read
    depth[82, 384], depth[83, 385], depth[84, 386] = 2.0, 2.5, 4.0
    assert g.lift(depth, cam, u, v) == pytest.approx([0.3, -0.7, 2.5])    # median of 2.0, 2.5, 4.0
    assert g.lift(depth, cam, u, v, window=3)[2] == pytest.approx(3.25)   # 2.5 and 4.0: mean of the two

@pytest.mark.parametrize("u, v", [(0.0, 0.0), (639.999, 479.999), (320.0, 240.0), (123.4, 456.7)])
def test_lift_then_project_round_trip(u, v):
    cam, depth = camera_for(), np.full((480, 640), 2.5, np.float32)
    assert g.project(cam, g.lift(depth, cam, u, v)) == pytest.approx((u, v, 2.5), abs=1e-6)

def test_image_bounds_and_points_behind_the_camera():
    cam, depth = camera_for(), np.full((480, 640), 2.5, np.float32)       # camera at (1, 2.2, 1.2) looking along +Y
    assert g.in_image(cam, 0.0, 0.0) and g.in_image(cam, 639.999, 479.999)
    assert not g.in_image(cam, 640.0, 10.0) and not g.in_image(cam, 10.0, -0.001)
    assert g.lift(depth, cam, 640.5, 10.0) is None and g.lift(depth, cam, 10.0, -0.5) is None
    assert g.project(cam, (1.0, 0.0, 0.0)) is None                        # behind the camera
    assert g.project(g.Camera(**INTRINSICS, T_world_camera=np.eye(4)), (0.3, 0.1, 0.0)) is None    # Z = 0
    p = g.project(cam, np.array([10.0, 5.0, 0.0]))                        # in front but outside the image: still returned
    assert p == pytest.approx((1960.176173630568, 270.976502735876, 3.041563510191), abs=1e-9) and type(p[2]) is float

def test_ground_hits_agree_with_lift_on_floor_depth():
    obs = floor_observation()
    cam, depth = obs.camera, obs.depth
    xy, z_cam = g.ground_hits(cam)
    assert xy.shape == (480, 640, 2) and z_cam.shape == (480, 640)
    assert np.isnan(xy[:38]).all() and np.isnan(z_cam[:38]).all() and (depth[:38] == 0.0).all()    # horizon: v = 38.25
    assert np.isfinite(xy[38:]).all() and np.allclose(z_cam[38:], depth[38:], rtol=1e-6, atol=0.0)
    assert (*xy[240, 320], z_cam[240, 320]) == pytest.approx((1.003157037157, 5.487742344215, 3.499891391895), abs=1e-9)
    for row, col in [(40, 3), (240, 320), (470, 630)]:
        p = g.lift(depth, cam, col + 0.5, row + 0.5)
        assert p[:2] == pytest.approx(xy[row, col], rel=1e-6) and abs(p[2]) < 1e-6
    points = g.lift_all(depth, cam)                                       # one point per valid pixel, row-major
    assert points.shape == (282880, 3) and np.abs(points[:, 2]).max() < 1e-6
    assert np.allclose(points[:, :2], xy[38:].reshape(-1, 2), rtol=1e-6, atol=1e-6)
    assert g.lift_all(depth, cam, stride=4).shape == (17600, 3)

FIT = dict(radius_m=0.30, min_points=40, max_tilt=0.5, max_rms_m=0.06, min_spread_m=0.05, max_rms_ratio=0.5)
#     the defaults of Thresholds
P = (1.0, 6.0, 0.8)                                                       # 3.8 m ahead of the camera of camera_for()
BEHIND = g.RobotPose(1.0, 10.0, -math.pi / 2)                             # a robot that looks at P from the other side
everywhere = lambda hit: np.ones(hit.shape[:-1], bool)
FLOOR = ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), everywhere)

def seen(cam, *pieces):                                   # the depth image of flat pieces: (a point of it, its normal, the hits it keeps)
    R, t = cam.T_world_camera[:3, :3], cam.T_world_camera[:3, 3]
    v, u = np.mgrid[0:cam.height, 0:cam.width] + 0.5
    rays = np.stack([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, np.ones_like(u)], axis=-1) @ R.T   # world, per metre of camera Z
    depth = np.full(u.shape, np.inf)
    for point, normal, keep in pieces:
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (np.asarray(point) - t) @ np.asarray(normal) / (rays @ np.asarray(normal))
            depth = np.minimum(depth, np.where((z > 0) & keep(t + z[..., None] * rays), z, np.inf))
    return np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)

def wall(turn_deg, lean_deg=0.0):                         # a plane through P: upright and square to +Y, then turned about the vertical
    a, b = math.radians(turn_deg), math.radians(lean_deg)  # and leant back, its top away from the side its normal is on
    return P, (math.sin(a) * math.cos(b), -math.cos(a) * math.cos(b), math.sin(b)), everywhere

def apart_deg(a, b):                                      # between two ground directions
    return math.degrees(math.acos(np.clip(np.dot(a, b) / (np.hypot(*a) * np.hypot(*b)), -1.0, 1.0)))

@pytest.mark.parametrize("robot, turn_deg, normal", [
    (ROBOT, 0.0, (0.0, -1.0)),                            # a wall facing the camera
    (ROBOT, 50.0, (0.766044, -0.642788)), (ROBOT, -50.0, (-0.766044, -0.642788)), (ROBOT, 75.0, (0.965926, -0.258819)),
    (BEHIND, 0.0, (0.0, 1.0)), (BEHIND, 50.0, (-0.766044, 0.642788))])     # the same walls from behind: their other side
def test_surface_normal_is_the_walls_own_normal_on_the_cameras_side(robot, turn_deg, normal):
    cam = camera_for(robot)
    n = g.surface_normal_xy(seen(cam, wall(turn_deg), FLOOR), cam, P, **FIT)
    assert n.shape == (2,) and n.dtype == np.float64 and np.hypot(*n) == pytest.approx(1.0, abs=1e-12)     # no vertical part
    to_camera = cam.T_world_camera[:2, 3] - np.array(P[:2])
    assert apart_deg(n, normal) < 1.0 and np.dot(n, to_camera) > 0
    assert apart_deg(n, to_camera) == pytest.approx(abs(turn_deg), abs=1.0)        # the wall's normal, not the line of sight

@pytest.mark.parametrize("lean_deg, max_tilt, faced", [
    (20.0, 0.5, True), (-20.0, 0.5, True),                # a face leant back or forward: its normal rises or falls by sin 20 = 0.34
    (40.0, 0.5, False), (40.0, 0.7, True)])               # sin 40 = 0.64: more table top than face, unless the limit says otherwise
def test_a_leaning_face_gives_its_ground_direction_up_to_the_tilt_limit(lean_deg, max_tilt, faced):
    cam = camera_for()
    n = g.surface_normal_xy(seen(cam, wall(0.0, lean_deg)), cam, P, **{**FIT, "max_tilt": max_tilt})
    assert (n is not None) == faced
    if faced:
        assert n == pytest.approx((0.0, -1.0), abs=1e-3) and np.hypot(*n) == pytest.approx(1.0, abs=1e-12)

def test_a_floor_patch_and_a_table_top_are_no_face():
    obs = floor_observation()
    assert g.surface_normal_xy(obs.depth, obs.camera, (1.0, 5.0, 0.0), **FIT) is None
    top = ((1.0, 5.5, 0.75), (0.0, 0.0, 1.0), lambda hit: (np.abs(hit[..., 0] - 1.0) <= 0.7) & (np.abs(hit[..., 1] - 5.5) <= 0.5))
    depth = seen(obs.camera, FLOOR, top)                  # a top 1.4 m by 1.0 m, 0.75 m up, with nothing drawn under it
    on_top = g.lift_all(depth, obs.camera)
    assert (np.abs(on_top[:, 2] - 0.75) < 1e-4).sum() > 5000             # the camera, 1.2 m up, looks down on it
    assert g.surface_normal_xy(depth, obs.camera, (1.0, 5.5, 0.75), **FIT) is None
    for lifted in ({"max_tilt": 1.0}, {"max_tilt": 1.0, "max_rms_m": 1.0}):       # with no limit on the tilt: nothing to drop it to
        assert g.surface_normal_xy(depth, obs.camera, (1.0, 5.5, 0.75), **{**FIT, **lifted}) is None

@pytest.mark.parametrize("between_deg, limits, refused", [
    (60.0, {}, True),                                     # a sharp corner: its points are 0.068 m RMS from the best plane,
    (60.0, {"max_rms_ratio": 1.0}, True),                 # which is 0.93 of their spread across it: each limit refuses it
    (60.0, {"max_rms_m": 1.0}, True),                     # by itself
    (90.0, {}, True),                                     # a right angle: 0.056 m, inside the first limit, but 0.54 of the spread across
    (90.0, {"max_rms_ratio": 0.6}, False),                # with the second limit raised the corner's bisector comes back,
    (90.0, {"max_rms_ratio": 0.6, "max_rms_m": 0.05}, True),      # and the first limit, lowered, refuses that
    (120.0, {}, False)])                                  # an open corner: 0.039 m, 0.31 of the spread across: one bent face
def test_where_two_walls_meet_the_fit_is_refused_by_how_thick_its_points_lie(between_deg, limits, refused):
    cam, half = camera_for(), between_deg / 2             # the corner is the upright line through P, its edge towards the camera
    right, left = wall(90.0 - half), wall(half - 90.0)    # each wall runs back from the edge, on its own side of it
    depth = seen(cam, (*right[:2], lambda hit: hit[..., 0] >= P[0]), (*left[:2], lambda hit: hit[..., 0] <= P[0]))
    n = g.surface_normal_xy(depth, cam, P, **{**FIT, **limits})
    assert (n is None) == refused
    if not refused:
        assert n == pytest.approx((0.0, -1.0), abs=1e-3)
    for side in (right, left):                            # each wall alone is a face
        assert g.surface_normal_xy(seen(cam, side), cam, P, **{**FIT, **limits}) == pytest.approx(side[1][:2], abs=1e-3)

def strip(width):                                         # the wall through P, square to +Y, cut to an upright strip this wide
    return P, (0.0, -1.0, 0.0), lambda hit: np.abs(hit[..., 0] - P[0]) <= width / 2

def test_points_that_lie_along_a_line_span_no_surface():
    cam = camera_for()
    full = seen(cam, wall(0.0))
    col = math.floor(g.project(cam, P)[0])
    column, noisy = np.zeros_like(full), np.zeros_like(full)
    column[:, col] = full[:, col]                         # one column of pixels of the wall: an upright line through P
    near = g.lift_all(column, cam)
    assert (np.linalg.norm(near - P, axis=1) < FIT["radius_m"]).sum() >= 2 * FIT["min_points"]    # points enough: 92
    assert g.surface_normal_xy(column, cam, P, **FIT) is None
    noise = np.random.default_rng(0).normal(0.0, 0.01, (cam.height, 3)).astype(np.float32)
    noisy[:, col - 1:col + 2] = full[:, col - 1:col + 2] + noise                      # three columns, their depth 1 cm out
    assert g.surface_normal_xy(noisy, cam, P, **FIT) is None
    loose = g.surface_normal_xy(noisy, cam, P, **{**FIT, "min_spread_m": 0.0, "max_rms_ratio": 1.0})
    assert apart_deg(loose, (0.0, -1.0)) > 80.0           # what the least spread of such points is: a direction along the wall
    # Across a strip of a face the points spread by its width / sqrt(12): 0.029 m for 0.10 m, 0.087 m for 0.30 m. A whole
    # patch gives half the radius, 0.15 m; with P on the edge of the face 0.079 m, and on its corner 0.066 m.
    assert g.surface_normal_xy(seen(cam, strip(0.10)), cam, P, **FIT) is None
    assert g.surface_normal_xy(seen(cam, strip(0.10)), cam, P, **{**FIT, "min_spread_m": 0.02}) == pytest.approx((0.0, -1.0), abs=1e-3)
    assert g.surface_normal_xy(seen(cam, strip(0.30)), cam, P, **FIT) == pytest.approx((0.0, -1.0), abs=1e-3)
    edge = lambda hit: hit[..., 0] >= P[0]
    for keep in (edge, lambda hit: edge(hit) & (hit[..., 2] <= P[2])):                 # P on the face's edge, and on its corner
        assert g.surface_normal_xy(seen(cam, (P, (0.0, -1.0, 0.0), keep)), cam, P, **FIT) == pytest.approx((0.0, -1.0), abs=1e-3)

def test_too_few_valid_depth_pixels_near_the_point_give_no_normal():
    cam = camera_for()
    full = seen(cam, wall(0.0))
    u, v, _ = g.project(cam, P)
    row, col = math.floor(v), math.floor(u)
    rows, cols = np.ix_(row + 8 * np.arange(-2, 3), col + 8 * np.arange(-4, 4))
    sparse = np.zeros_like(full)
    sparse[rows, cols] = full[rows, cols]                 # 40 valid pixels, 8 apart each way: spread over the patch, within 0.24 m of P
    assert g.surface_normal_xy(sparse, cam, P, **FIT) == pytest.approx((0.0, -1.0), abs=1e-3)
    sparse[row, col] = np.nan                             # 39: one short of min_points
    assert g.surface_normal_xy(sparse, cam, P, **FIT) is None
    assert g.surface_normal_xy(sparse, cam, P, **{**FIT, "min_points": 39}) == pytest.approx((0.0, -1.0), abs=1e-3)
    assert g.surface_normal_xy(np.zeros_like(full), cam, P, **FIT) is None             # no depth at all
    assert g.surface_normal_xy(full, cam, (1.0, 4.0, 0.8), **FIT) is None              # a point in mid air, 2 m before the wall
    assert g.surface_normal_xy(full, cam, P, **{**FIT, "radius_m": 0.02}) is None      # a radius that holds 32 pixels

def test_floor_in_view_needs_the_ground_point_in_the_picture_and_floor_at_its_pixel():
    obs = floor_observation()                             # bare floor; the camera is at (1, 2.2), 1.2 m up, looking along +Y
    cam = obs.camera
    def in_view(xy, depth=obs.depth, floor_tolerance_m=0.15, depth_tolerance_m=0.3, **window):
        return g.floor_in_view(depth, cam, xy, floor_tolerance_m, depth_tolerance_m, **window)
    assert in_view((1.0, 5.0)) is True and in_view((2.5, 6.0)) and in_view(np.array([1.0, 3.6])) and in_view((1.0, 12.0))
    assert in_view((1.0, 0.0)) is False                   # behind the camera
    assert not in_view((10.0, 5.0))                       # ahead, but beside the picture
    def nearer(xy, share, half=3):                        # the depth round the pixel of that ground point, at this share of the floor's
        u, v, z = g.project(cam, (*xy, 0.0))
        row, col = math.floor(v), math.floor(u)
        depth = obs.depth.copy()
        depth[row - half:row + half + 1, col - half:col + half + 1] *= share
        return depth, z
    # What a ray shows at the share f of the floor's depth is 1.2 * (1 - f) m above the ground and z * (1 - f) nearer.
    depth, z = nearer((1.0, 5.0), 0.95)
    assert z == pytest.approx(3.0416, abs=1e-4)           # 0.06 m up and 0.15 m nearer: floor at that place, by both tolerances
    assert in_view((1.0, 5.0), depth) and not in_view((1.0, 5.0), depth, floor_tolerance_m=0.05)
    assert not in_view((1.0, 5.0), depth, depth_tolerance_m=0.1)
    assert not in_view((1.0, 5.0), nearer((1.0, 5.0), 0.85)[0])           # 0.18 m up: something stands there, or before it
    assert not in_view((1.0, 5.0), nearer((1.0, 5.0), 1.15)[0])           # 0.18 m under the ground plane: no floor there
    depth, z = nearer((1.0, 3.6), 0.88)                   # close by the ray is steep: 0.21 m nearer is 0.144 m up, under the
    assert z * 0.12 == pytest.approx(0.207, abs=1e-3)     # tolerance by the height the ray has there (1.2 * 0.12), not by
    assert in_view((1.0, 3.6), depth) and not in_view((1.0, 3.6), depth, floor_tolerance_m=0.14)     # 1.2 * (1 / 0.88 - 1) = 0.164
    beyond, z = nearer((1.0, 3.6), 1.15)                  # a depth beyond the floor's is no floor either. Close by, 0.26 m farther
    assert z * 0.15 == pytest.approx(0.259, abs=1e-3)     # is 0.18 m under the ground: the height alone says so there
    assert not in_view((1.0, 3.6), beyond) and in_view((1.0, 3.6), beyond, floor_tolerance_m=0.2)
    depth, z = nearer((1.0, 12.0), 0.95)                  # far off the ray grazes the floor: 0.06 m up is 0.48 m nearer, the foot
    assert z * 0.05 == pytest.approx(0.48, abs=0.01)      # of something that stands before the place and hides it
    assert not in_view((1.0, 12.0), depth) and in_view((1.0, 12.0), depth, depth_tolerance_m=0.5)
    beyond, _ = nearer((1.0, 12.0), 1.05)                 # and 0.53 m farther is 0.07 m under the ground: the depth alone says so
    assert not in_view((1.0, 12.0), beyond) and in_view((1.0, 12.0), beyond, depth_tolerance_m=0.6)
    stray = nearer((1.0, 5.0), 0.5, half=0)[0]            # one stray pixel: the depth is the median of the window, as for lift
    assert in_view((1.0, 5.0), stray) and not in_view((1.0, 5.0), stray, window=1)
    assert not in_view((1.0, 5.0), nearer((1.0, 5.0), 0.0)[0]) and not in_view((1.0, 5.0), nearer((1.0, 5.0), np.nan)[0])   # no valid depth

def test_floor_in_view_takes_a_place_under_the_pictures_lower_edge_from_the_foot_of_its_column():
    obs = floor_observation()                             # the lowest row shows the floor 1.27 m ahead of the camera, at y = 3.47
    cam, at = obs.camera, (1.3, 3.0)                      # 0.8 m ahead of the camera and 0.3 m to its right
    def in_view(xy, depth=obs.depth, floor_tolerance_m=0.15, **window):
        return g.floor_in_view(depth, cam, xy, floor_tolerance_m, 0.3, **window)
    u, v, _ = g.project(cam, (*at, 0.0))
    col = math.floor(u)
    assert v >= cam.height and (col, round(v)) == (463, 647)             # under the lower edge, between the side edges
    assert in_view(at) is True and in_view((1.0, 3.4)) and in_view(np.array([0.8, 2.6]))
    assert in_view((1.0, 2.2)) and in_view((1.0, 2.0))    # the strip counts as floor back to the camera's feet, and the robot's
    def foot(share, rows=slice(-5, None), column=col):    # the depth with those rows of a column at this share of the floor's
        depth = obs.depth.copy()
        depth[rows, column] *= share
        return depth
    assert not in_view(at, foot(0.8))                     # 0.24 m up: something stands at the foot of the column
    assert in_view(at, foot(0.8), floor_tolerance_m=0.25) and not in_view(at, foot(1.2))      # or 0.24 m under the ground: a drop
    assert in_view(at, foot(0.8, column=col - 1)) and in_view(at, foot(0.8, column=col + 1))  # the place's own column, no other
    assert not in_view(at, foot(0.8, rows=-5)) and not in_view(at, foot(0.8, rows=-1))        # every one of the lowest rows,
    assert in_view(at, foot(0.8, rows=-5), window=4) and in_view(at, foot(0.8, rows=-6))      # as many as the window, none above
    assert not in_view(at, foot(0.0, rows=-2)) and not in_view(at, foot(np.nan, rows=-4))     # and each with valid depth
    for x in (-0.5, 2.5):                                 # as near, but left and right of the picture: the foot of no column
        u, v, _ = g.project(cam, (x, 3.0, 0.0))
        assert v >= cam.height and not 0 <= u < cam.width and not in_view((x, 3.0))
    assert not in_view((1.0, 1.7))                        # behind the camera, 0.3 m behind the robot
    steep = g.Camera(**INTRINSICS, T_world_camera=g.camera_on_robot(ROBOT, 0.2, 1.2, math.radians(30.0)))
    bare = seen(steep, FLOOR)                             # pitched 30 degrees down, a camera has the horizon above its picture
    u, v, _ = g.project(steep, (1.0, 30.0, 0.0))
    assert v < 0 and 0 <= u < steep.width and not g.floor_in_view(bare, steep, (1.0, 30.0), 0.15, 0.3)    # above the upper edge
    assert g.floor_in_view(bare, steep, (1.0, 8.0), 0.15, 0.3) and g.floor_in_view(bare, steep, (1.0, 2.5), 0.15, 0.3)   # in it, under it
