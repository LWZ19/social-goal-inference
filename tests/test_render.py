import math
import time
from dataclasses import replace

import numpy as np
import pytest
from sgi.geometry import Camera, camera_on_robot, ground_hits, project
from sgi.inputs import ObservationData
from sgi.synth.render import Box, Cylinder, Figure, Room, Sphere, appearance, render
from tests.helpers import INTRINSICS, MOUNT, ROBOT, rendered_observation

# The front is turned 0.6 rad from facing the camera. Its centre projects to (421.51, 122.50), close to a pixel centre.
DESK = Box(cx=1.6, cy=6.7, yaw=-math.pi / 2 + 0.6, sx=0.6, sy=1.6, sz=1.2, color=(180, 130, 70))
FRONT_CENTRE = (DESK.cx + 0.3 * math.cos(DESK.yaw), DESK.cy + 0.3 * math.sin(DESK.yaw), 0.6)
LOW = replace(DESK, sz=0.8)                                 # low enough for the camera of the helpers to see its top
HALL = Room.rectangle(1.0, 6.0, 0.0, 10.0, 12.0)            # x from -4 to 6 and y from 0 to 12, 3 m high, round ROBOT
AHEAD = (1.0, 6.2)                                          # 4 m in front of the eye-level camera, on its optical axis
PERSON = Figure(*AHEAD, -math.pi / 2, color=(200, 40, 40))  # facing that camera
EYE_LEVEL = dict(forward_m=0.2, height_m=1.5, pitch_down_rad=math.radians(10.0))

def _view(boxes=(), figures=(), mount=MOUNT, **options):  # rendered_observation with render's options, from any mount
    cam = Camera(**INTRINSICS, T_world_camera=camera_on_robot(ROBOT, **mount))
    rgb, depth = render(cam, boxes, figures, **options)
    return ObservationData(rgb=rgb, depth=depth, camera=cam, robot=ROBOT)

def _shot(boxes=(), figures=(), **options):  # the view from eye level
    return _view(boxes, figures, EYE_LEVEL, **options)

def _world(obs, mask):  # world points of the masked pixels, lifted at pixel centres
    cam, (rows, cols) = obs.camera, np.nonzero(mask)
    z = obs.depth[rows, cols].astype(float)
    pc = np.stack([(cols + 0.5 - cam.cx) * z / cam.fx, (rows + 0.5 - cam.cy) * z / cam.fy, z, np.ones_like(z)])
    return (cam.T_world_camera @ pc)[:3].T

def _colours(obs, mask):  # the distinct colours of the masked pixels, sorted
    return np.unique(obs.rgb[mask], axis=0).tolist()

def _at(obs, p_world):  # the colour of the pixel a world point projects to
    u, v, _ = project(obs.camera, p_world)
    return obs.rgb[int(v), int(u)].astype(float)

def _is(pixel, base):  # whether a pixel shows the base colour under some lighting: the same colour, brighter or darker
    return bool(np.abs(pixel / pixel.max() - np.array(base) / max(base)).max() < 0.06)

def _runs(flags):  # the number of unbroken runs of true values
    return int(np.count_nonzero(np.diff(np.concatenate([[False], flags]).astype(int)) == 1))

def _grown(mask):  # the mask with the pixels next to it
    out = mask.copy()
    out[1:] |= mask[:-1]; out[:-1] |= mask[1:]; out[:, 1:] |= mask[:, :-1]; out[:, :-1] |= mask[:, 1:]
    return out

def test_empty_scene_is_the_ground_plane():
    obs = rendered_observation()
    assert (obs.rgb.shape, obs.rgb.dtype, obs.depth.shape, obs.depth.dtype) == ((480, 640, 3), np.uint8, (480, 640), np.float32)
    assert not obs.depth[:38].any() and (obs.depth[38:] > 0).all()  # horizon at v = 38.25; camera Z is 2846 m on row 38
    np.testing.assert_allclose(obs.depth[38:], ground_hits(obs.camera)[1][38:], rtol=1e-6, atol=1e-4)  # rtol: float32 spacing

def test_depth_is_camera_z_not_ray_length():
    obs = rendered_observation()
    hit = np.array([*ground_hits(obs.camera)[0][400, 80], 0.0])
    ray_length = np.linalg.norm(hit - obs.camera.T_world_camera[:3, 3])
    assert ray_length == pytest.approx(2.2026, abs=1e-3) and obs.depth[400, 80] == pytest.approx(1.9540, abs=1e-3)

def test_box_depth_at_the_centre_of_its_near_face():
    obs = rendered_observation(boxes=[DESK])
    u, v, z = project(obs.camera, FRONT_CENTRE)
    assert (int(u), int(v)) == (421, 122) and z == pytest.approx(4.2012, abs=1e-3)
    assert obs.depth[122, 421] == pytest.approx(z, abs=1e-3)  # the neighbouring pixels are 0.003 to 0.006 m off

def test_depth_of_round_shapes_is_camera_z():
    post, ball = Cylinder(1.0, 6.0, radius=0.3, height=1.0), Sphere(2.5, 6.0, 1.3, radius=0.4)
    obs, empty = rendered_observation(boxes=[post, ball]), rendered_observation()
    eye, centre = obs.camera.T_world_camera[:3, 3], np.array([ball.x, ball.y, ball.z])
    u, v, z = project(obs.camera, centre + ball.radius * (eye - centre) / np.linalg.norm(eye - centre))  # its nearest point
    assert np.linalg.norm(centre - eye) - ball.radius == pytest.approx(3.6866, abs=1e-3) and z == pytest.approx(3.1905, abs=1e-3)
    assert obs.depth[int(v), int(u)] == pytest.approx(z, abs=1e-3)
    p = _world(obs, obs.depth != empty.depth)
    on_ball, r = p[:, 0] > 1.8, np.hypot(p[:, 0] - post.x, p[:, 1] - post.y)
    assert on_ball.sum() > 9000 and np.abs(np.linalg.norm(p[on_ball] - centre, axis=1) - ball.radius).max() < 2e-3
    side = ~on_ball & (p[:, 2] < post.height - 2e-3)        # the rest of the post is its top, seen from above
    assert side.sum() > 9000 and np.abs(r[side] - post.radius).max() < 2e-3 and p[:, 2].min() > -2e-3
    assert (~on_ball & ~side).sum() > 300 and r[~on_ball].max() < post.radius + 2e-3 and p[~on_ball, 2].max() < post.height + 2e-3
    cone = rendered_observation(boxes=[replace(post, top_radius=0.1)])
    q = _world(cone, cone.depth != empty.depth)             # it tapers evenly from 0.3 m at the floor to 0.1 m at its top
    slope = q[:, 2] < post.height - 2e-3
    assert np.abs(np.hypot(q[slope, 0] - post.x, q[slope, 1] - post.y) - (0.3 - 0.2 * q[slope, 2])).max() < 2e-3

def test_flat_shading_gives_one_colour_per_face():
    # One ray a pixel: with more, the pixels along an edge mix the colours either side of it. The colours are those
    # of light from above and from the viewer's left; it used to come from the camera, which made the front the brighter.
    obs, empty = _view(boxes=[DESK], antialias=False), _view(antialias=False)
    assert _colours(empty, empty.depth == 0) == [[225, 235, 245]] and _colours(empty, empty.depth > 0) == [[184, 179, 166]]
    assert _colours(obs, obs.depth != empty.depth) == [[123, 89, 48], [137, 99, 53]]  # the desk's front, then its side face
    assert obs.rgb[122, 421].tolist() == [123, 89, 48]  # the centre of the front face, which is turned from the light
    low = _view(boxes=[LOW], antialias=False)
    assert _colours(low, low.depth != empty.depth) == [[123, 89, 48], [137, 99, 53], [161, 117, 63]]  # and its top, the brightest

def test_antialiasing_blends_outlines_and_leaves_depth_alone():
    scene = dict(boxes=[replace(DESK, label="DESK")], figures=[PERSON])
    crisp, smooth = _shot(antialias=False, **scene), _shot(**scene)
    assert np.array_equal(crisp.depth, smooth.depth)
    changed = np.abs(crisp.rgb.astype(int) - smooth.rgb).max(axis=2) > 4       # by more than a rounding of the shading
    assert 0.005 < changed.mean() < 0.06                    # outlines only
    edge = np.zeros_like(changed)                           # where the crisp picture steps from one colour to another
    for step in (np.s_[1:], np.s_[:, 1:]):
        back = np.s_[:-1] if step == np.s_[1:] else np.s_[:, :-1]
        jump = np.abs(crisp.rgb[step].astype(int) - crisp.rgb[back]).max(axis=2) > 4
        edge[step] |= jump; edge[back] |= jump
    assert not (changed & ~_grown(edge)).any()

@pytest.mark.parametrize("yaw", [0.0, 0.3, 0.6, -0.4])
def test_of_two_faces_in_one_plane_the_first_drawn_shows_all_over(yaw):
    # A counter and a side wall of a booth, in the booth's own frame, which stands at AHEAD turned by `turn`. The
    # wall's outer face lies in the plane of the counter's end, y = 0.8, and covers it from x = -0.6 to x = -0.32.
    turn = -math.pi + yaw                                   # at 0 that plane squarely faces the camera
    c, s = math.cos(turn), math.sin(turn)
    at = lambda x, y: (AHEAD[0] + c * x - s * y, AHEAD[1] + s * x + c * y)
    counter = Box(*at(-0.3, 0.0), turn, sx=0.6, sy=1.6, sz=1.0, color=(180, 130, 70))
    wall = Box(*at(-0.56, 0.77), turn, sx=0.48, sy=0.06, sz=1.0, color=(60, 90, 200))
    shot = lambda *boxes: _shot(list(boxes), antialias=False)
    counter_alone, wall_alone = shot(counter), shot(wall)
    seen = counter_alone.depth > 0
    p = _world(counter_alone, seen)
    x = c * (p[:, 0] - AHEAD[0]) + s * (p[:, 1] - AHEAD[1])                     # in the booth's frame
    y = -s * (p[:, 0] - AHEAD[0]) + c * (p[:, 1] - AHEAD[1])
    shared = np.zeros_like(seen)                            # the pixels well inside the part of the plane both boxes have
    shared[seen] = (np.abs(y - 0.8) < 1e-3) & (x > -0.59) & (x < -0.33) & (p[:, 2] > 0.02) & (p[:, 2] < 0.98)
    assert shared.sum() > 3000 and (counter_alone.rgb[shared] != wall_alone.rgb[shared]).any(axis=1).all()
    # The two ray parameters differ there by rounding alone, which used to decide between the boxes pixel by pixel.
    assert np.array_equal(shot(counter, wall).rgb[shared], counter_alone.rgb[shared])
    assert np.array_equal(shot(wall, counter).rgb[shared], wall_alone.rgb[shared])
    assert np.array_equal(shot(counter, wall).depth, shot(wall, counter).depth)   # depth is the same either way round

def test_raised_box_leaves_the_ground_visible_beneath_it():
    slab = Box(cx=1.3, cy=6.0, yaw=-math.pi / 2, sx=0.6, sy=1.2, sz=0.1, z0=0.8)
    obs, empty = rendered_observation(boxes=[slab]), rendered_observation()
    assert obs.depth[216, 361] == empty.depth[216, 361]  # the ground point (1.3, 6.0, 0.0), under the slab
    z = _world(obs, obs.depth != empty.depth)[:, 2]
    assert len(z) > 4000 and z.min() > 0.8 - 1e-3 and z.max() < 0.9 + 1e-3
    solid = rendered_observation(boxes=[replace(slab, z0=0.0, sz=0.9)])
    assert solid.depth[216, 361] < empty.depth[216, 361] - 0.2

def test_figure_occludes_the_ground_behind_it():
    fig = Figure(2.0, 12.0, -math.pi / 2)  # far enough for the head to be in view
    obs, empty = rendered_observation(figures=[fig]), rendered_observation()
    # Points on the body's axis, in the hips and at the waist: the surface is just in front, the ground far behind.
    # The figure was a cylinder and the lower height was 0.5 m; at that height its axis now runs between two legs.
    for z in (0.9, 1.1):
        u, v, zc = project(obs.camera, (fig.x, fig.y, z))
        assert 0.0 < zc - obs.depth[int(v), int(u)] <= 0.3 and empty.depth[int(v), int(u)] - zc > 1.0
    pts = _world(obs, obs.depth != empty.depth)
    r = np.hypot(pts[:, 0] - fig.x, pts[:, 1] - fig.y)
    # The cylinder filled its footprint and its sphere head was 0.12 m in radius; the body stays inside the footprint,
    # reaching nearly to its edge at the elbows, and the head reaches 0.132 m from the axis at the tip of its nose.
    assert 0.19 < r.max() <= fig.radius and 0.11 < r[pts[:, 2] > 1.5].max() < 0.135
    assert pts[:, 2].min() > -1e-3 and pts[:, 2].max() == pytest.approx(fig.height, abs=0.03)

@pytest.mark.parametrize("yaw", [-math.pi / 2, 0.0, 2.2, math.pi / 2])  # facing the camera, side on, obliquely away, away
@pytest.mark.parametrize("height", [1.55, 1.7, 1.9])
def test_a_figure_is_seen_along_its_axis_from_hips_to_head(yaw, height):
    obs = _shot(figures=[Figure(*AHEAD, yaw, height=height)])
    for z in (0.9, 1.2, 1.6 if height >= 1.7 else 1.5):     # the visibility rule samples the axis from 0.9 m up
        u, v, zc = project(obs.camera, (*AHEAD, z))
        assert 0.0 < zc - obs.depth[int(v), int(u)] <= 0.3, z
    u, v, zc = project(obs.camera, (*AHEAD, height + 0.05))
    assert obs.depth[int(v), int(u)] == 0.0                 # nothing above the top of the head

def test_figure_is_a_body_with_two_legs_two_arms_and_a_head():
    obs, empty = _shot(figures=[PERSON]), _shot()
    body = obs.depth != empty.depth
    row = lambda z: body[int(project(obs.camera, (*AHEAD, z))[1])]
    assert [_runs(row(z)) for z in (0.3, 0.6, 1.2, 1.48, 1.6)] == [2, 2, 1, 1, 1]  # shins and thighs apart; chest, neck, head
    width = lambda z: row(z).sum() * 4.0 / INTRINSICS["fx"]                         # about, in metres, at 4 m
    assert 0.38 < width(1.2) <= 0.45 and 0.07 < width(1.48) < 0.12 and 0.15 < width(1.6) < 0.22
    assert _is(_at(obs, (*AHEAD, 1.2)), PERSON.color)                               # the shirt
    for side in (-0.09, 0.09):
        assert _is(_at(obs, (AHEAD[0] + side, AHEAD[1], 0.5)), PERSON.trousers)     # a leg, each side of the axis
    for side in (-0.165, 0.165):                                                    # a hand hangs a little in front of each hip
        assert _is(_at(obs, (AHEAD[0] + side, AHEAD[1] - 0.075, 0.8)), PERSON.skin)

def test_figure_shows_which_way_it_faces():
    # The camera of the helpers cuts a figure this near off at the chest, where a light panel used to mark its front.
    # The front is now marked by the face, so the figure is seen from eye level, and no longer has the same outline
    # (hence the same depth) from the front as from the back: its nose, forearms and shoes point forward.
    toward, away, empty = _shot(figures=[PERSON]), _shot(figures=[replace(PERSON, yaw=math.pi / 2)]), _shot()
    diff = np.abs(toward.rgb.astype(int) - away.rgb.astype(int)).max(axis=2)
    assert (diff >= 60).sum() > 500
    assert not diff[~_grown((toward.depth != empty.depth) | (away.depth != empty.depth))].any()
    cheek, chest = (AHEAD[0] + 0.05, AHEAD[1], 1.56), (*AHEAD, 1.2)  # a pixel on the head: the face, then the hair
    assert _is(_at(toward, cheek), PERSON.skin) and _is(_at(away, cheek), PERSON.hair)
    assert _is(_at(toward, chest), PERSON.color) and _is(_at(away, chest), PERSON.color)
    side = _shot(figures=[replace(PERSON, yaw=0.0)])        # facing +x, the right of the image: face on the right, hair behind
    assert _is(_at(side, (AHEAD[0] + 0.06, AHEAD[1], 1.56)), PERSON.skin) and _is(_at(side, (AHEAD[0] - 0.06, AHEAD[1], 1.6)), PERSON.hair)

def test_face_has_two_eyes_and_a_mouth_under_hair():
    for distance in (4.0, 6.0):
        x, y = AHEAD[0], AHEAD[1] - 4.0 + distance
        obs = _shot(figures=[replace(PERSON, y=y)])
        level = lambda z: obs.rgb[int(project(obs.camera, (x, y, z))[1])].astype(float)   # the image row at a height
        u = int(project(obs.camera, (x, y, 1.6))[0])
        span = slice(u - int(0.07 * INTRINSICS["fx"] / distance), u + int(0.07 * INTRINSICS["fx"] / distance) + 1)
        skin = _at(obs, (x + 0.05, y, 1.555)).sum()         # a cheek
        dark = level(1.59)[span].sum(axis=1) < 0.6 * skin
        assert _runs(dark) == 2 and not dark[len(dark) // 2], distance              # an eye each side of the nose
        lips = level(1.52)[span].sum(axis=1) < 0.8 * skin
        assert _runs(lips) == 1 and abs(np.nonzero(lips)[0].mean() - len(lips) // 2) <= 1.5, distance   # one mouth, below it
        assert _is(_at(obs, (x, y, 1.685)), PERSON.hair) and _is(_at(obs, (x, y, 1.555)), PERSON.skin)

def test_appearance_tells_the_people_of_a_scene_apart():
    looks = [appearance(i) for i in range(14)]
    assert looks[3] == appearance(3) and all(sorted(look) == ["hair", "height", "skin", "trousers"] for look in looks)
    for one, next_one in zip(looks, looks[1:]):             # neighbours differ in all four
        assert all(one[field] != next_one[field] for field in one)
    assert all(1.55 <= look["height"] <= 1.9 for look in looks)   # the heights at which the axis is known to be seen
    obs, empty = _shot(figures=[Figure(*AHEAD, -math.pi / 2, color=(200, 40, 40), **looks[4])]), _shot()
    assert looks[4]["height"] == 1.84 and _world(obs, obs.depth != empty.depth)[:, 2].max() == pytest.approx(1.84, abs=0.03)

def test_figure_colours_its_shirt_trousers_skin_and_hair():
    base = _shot(figures=[PERSON], antialias=False)

    def heights(**fields):  # the world heights of the pixels the fields recolour; they move nothing
        other = _shot(figures=[replace(PERSON, **fields)], antialias=False)
        assert np.array_equal(other.depth, base.depth)
        return _world(base, (other.rgb != base.rgb).any(axis=2))[:, 2]

    shirt, trousers = heights(color=(250, 250, 0)), heights(trousers=(250, 250, 0))
    hair, skin = heights(hair=(250, 250, 0)), heights(skin=(250, 250, 0))
    assert len(shirt) > 3000 and 0.88 < shirt.min() < 0.95 and 1.4 < shirt.max() < 1.46      # hem to shoulders
    assert len(trousers) > 4000 and 0.04 < trousers.min() < 0.12 and 0.85 < trousers.max() < 0.95  # ankles to the hem
    assert len(hair) > 100 and hair.min() > 1.55 and hair.max() == pytest.approx(1.7, abs=0.01)    # beside the face, brows, fringe
    assert len(skin) > 1200 and skin.min() < 0.78 and skin.max() > 1.62                      # hands to forehead
    assert not ((skin > 1.1) & (skin < 1.3)).any()          # nothing between the elbows and the neckline of the shirt
    wide = _shot(figures=[replace(PERSON, radius=0.3)])
    p = _world(wide, wide.depth != _shot().depth)
    assert 0.26 < np.hypot(p[:, 0] - AHEAD[0], p[:, 1] - AHEAD[1]).max() <= 0.3

def test_label_changes_pixels_only_on_the_front_face():
    plain, labelled = rendered_observation(boxes=[DESK]), rendered_observation(boxes=[replace(DESK, label="L")])
    changed = (plain.rgb != labelled.rgb).any(axis=2)
    assert np.array_equal(plain.depth, labelled.depth) and changed.sum() > 300
    p = _world(plain, changed)
    c, s, dx, dy = math.cos(DESK.yaw), math.sin(DESK.yaw), p[:, 0] - DESK.cx, p[:, 1] - DESK.cy
    out, across = c * dx + s * dy, -s * dx + c * dy  # along the box's +x and +y
    assert np.abs(out - DESK.sx / 2).max() < 1e-3 and np.abs(across).max() < DESK.sy / 2
    assert p[:, 2].min() == pytest.approx(0.3, abs=0.02) and p[:, 2].max() == pytest.approx(0.9, abs=0.02)  # half of sz, centred
    assert across.mean() < (across.min() + across.max()) / 2  # the stem of the L is on the viewer's left
    assert p[:, 2].mean() < (p[:, 2].min() + p[:, 2].max()) / 2  # and its foot is at the bottom

@pytest.mark.parametrize("face, turn", [("front", 0.0), ("left", -math.pi / 2), ("back", math.pi), ("right", math.pi / 2)])
def test_label_and_panel_sit_on_the_named_face(face, turn):
    # A box 1.6 m square, turned so that the named face is the one DESK shows the camera.
    block = Box(cx=1.9, cy=7.2, yaw=DESK.yaw + turn, sx=1.6, sy=1.6, sz=1.2, color=(180, 130, 70))
    c, s = math.cos(DESK.yaw), math.sin(DESK.yaw)

    def changed_by(**fields):  # where the fields change the picture: out of the shown face, across it, and up
        plain, other = rendered_observation(boxes=[block]), rendered_observation(boxes=[replace(block, **fields)])
        changed = (plain.rgb != other.rgb).any(axis=2)
        assert np.array_equal(plain.depth, other.depth)
        p = _world(plain, changed)
        return c * (p[:, 0] - block.cx) + s * (p[:, 1] - block.cy), -s * (p[:, 0] - block.cx) + c * (p[:, 1] - block.cy), p[:, 2]

    out, across, up = changed_by(label="L", label_face=face)
    assert len(out) > 300 and np.abs(out - 0.8).max() < 1e-3
    assert across.mean() < (across.min() + across.max()) / 2 and up.mean() < (up.min() + up.max()) / 2   # an L the right way round
    assert up.min() == pytest.approx(0.3, abs=0.02) and up.max() == pytest.approx(0.9, abs=0.02)
    out, across, up = changed_by(label="L", label_face=face, label_height=0.8)
    assert up.min() == pytest.approx(0.12, abs=0.02) and up.max() == pytest.approx(1.08, abs=0.02)     # larger lettering
    out, across, up = changed_by(panel=(20, 30, 40), panel_rect=(0.5, 0.25, 0.9, 0.75), label_face=face)
    assert len(out) > 3000 and np.abs(out - 0.8).max() < 1e-3                                            # the viewer's right half
    assert across.min() == pytest.approx(0.0, abs=0.02) and across.max() == pytest.approx(0.64, abs=0.02)
    assert up.min() == pytest.approx(0.3, abs=0.02) and up.max() == pytest.approx(0.9, abs=0.02)
    out, across, up = changed_by(panel=(20, 30, 40), panel_rect=(0.5, 0.25, 0.9, 0.75), label="L", label_face=face)
    assert across.min() == pytest.approx(0.0, abs=0.02) and up.max() == pytest.approx(0.9, abs=0.02)     # lettering stays on the panel

def test_label_on_the_top_reads_from_the_front():
    mat = Box(cx=1.0, cy=5.0, yaw=-math.pi / 2, sx=1.2, sy=1.2, sz=0.02)   # on the floor, its front towards the camera
    plain, other = rendered_observation(boxes=[mat]), rendered_observation(boxes=[replace(mat, label="L", label_face="top")])
    p = _world(plain, (plain.rgb != other.rgb).any(axis=2))
    assert len(p) > 300 and np.abs(p[:, 2] - 0.02).max() < 1e-3
    assert p[:, 0].mean() < (p[:, 0].min() + p[:, 0].max()) / 2 and p[:, 1].mean() < (p[:, 1].min() + p[:, 1].max()) / 2

def test_band_and_top_colour_their_parts_of_a_box():
    plain = _view(boxes=[LOW], antialias=False)
    c, s = math.cos(LOW.yaw), math.sin(LOW.yaw)

    def changed_by(**fields):
        other = _view(boxes=[replace(LOW, **fields)], antialias=False)
        assert np.array_equal(plain.depth, other.depth)
        p = _world(plain, (plain.rgb != other.rgb).any(axis=2))
        dx, dy = p[:, 0] - LOW.cx, p[:, 1] - LOW.cy
        return np.abs(c * dx + s * dy), np.abs(-s * dx + c * dy), p[:, 2]

    out, across, up = changed_by(band=(30, 30, 30), band_range=(0.75, 1.0))
    assert len(up) > 1500 and up.min() == pytest.approx(0.6, abs=0.01) and up.max() < 0.8 + 1e-3
    assert (np.minimum(LOW.sx / 2 - out, LOW.sy / 2 - across) < 1e-3).all()     # on the upright faces, none on the top
    out, across, up = changed_by(top=(250, 250, 250))
    assert len(up) > 1000 and np.abs(up - 0.8).max() < 1e-3 and out.max() < LOW.sx / 2 + 1e-3 and across.max() < LOW.sy / 2 + 1e-3

def test_a_room_leaves_no_ray_without_a_hit():
    obs = _shot(room=HALL)
    assert (obs.depth > 0).all()
    p = _world(obs, obs.depth > 0)
    gaps = np.abs(np.stack([p[:, 0] + 4, p[:, 0] - 6, p[:, 1], p[:, 1] - 12, p[:, 2], p[:, 2] - 3]))
    assert gaps.min(axis=0).max() < 2e-3                    # every pixel on a wall, the floor or the ceiling
    assert sorted(set(gaps.argmin(axis=0).tolist())) == [0, 1, 3, 4, 5]         # all of them but the wall behind the camera
    assert p.min(axis=0) == pytest.approx((-4, 4.4776, 0), abs=2e-3) and p.max(axis=0) == pytest.approx((6, 12, 3), abs=2e-3)   # the nearest floor in view is 2.28 m ahead
    assert np.array_equal(_shot(room=HALL, antialias=False).depth, obs.depth)
    # Walls on the edges of any polygon: here the left of the far wall comes forward to y = 8.
    ell = _shot(room=replace(HALL, polygon=((-4, 0), (6, 0), (6, 12), (2, 12), (2, 8), (-4, 8))))
    assert (ell.depth > 0).all()
    wall = np.zeros((480, 640), dtype=bool)
    wall[200, [100, 600]] = True                            # a pixel left of the step in the wall, and one right of it
    assert _world(ell, wall)[:, 1] == pytest.approx([8.0, 12.0], abs=2e-3)

def test_ceiling_light_panels_sit_in_the_middle_of_their_squares():
    obs = _shot(room=HALL)                                  # squares of 2.4 m from the first corner, (-4, 0)
    lit = (obs.rgb == (255, 254, 246)).all(axis=2)
    p = _world(obs, lit)
    assert lit.sum() > 1000 and np.abs(p[:, 2] - 3.0).max() < 2e-3
    along, across = (p[:, 0] + 4.0) % 2.4, p[:, 1] % 2.4    # a panel is 1.2 m by 0.3 m: 0.6 to 1.8 m along, 1.05 to 1.35 m across
    assert along.min() == pytest.approx(0.6, abs=0.03) and along.max() == pytest.approx(1.8, abs=0.03)
    assert 1.05 - 2e-3 < across.min() < 1.15 and 1.25 < across.max() < 1.35 + 2e-3      # a pixel there spans 0.1 m across
    assert _at(obs, (2.0, 10.8, 3.0)).tolist() == [255, 254, 246] and _at(obs, (0.8, 9.6, 3.0)).tolist() == [226, 225, 219]
    dark = _shot(room=replace(HALL, lights=0.0))
    assert not (dark.rgb == (255, 254, 246)).all(axis=2).any() and np.array_equal(dark.depth, obs.depth)

def test_floor_tiles_change_colours_but_not_depth():
    tiled, plain = _shot(room=HALL), _shot(room=replace(HALL, tile=0.0))
    changed = (tiled.rgb != plain.rgb).any(axis=2)
    assert np.array_equal(tiled.depth, plain.depth) and changed.sum() > 50000
    assert np.abs(_world(tiled, changed)[:, 2]).max() < 0.02                    # floor pixels, and the row that blends floor and wall
    joint, middle = (0.8, 6.0, 0.0), (1.1, 6.3, 0.0)        # tiles of 0.6 m from the first corner, (-4, 0): a crossing, a centre
    assert (_at(tiled, joint) < _at(tiled, middle) - 20).all() and np.abs(_at(plain, joint) - _at(plain, middle)).max() <= 2
    patch = np.s_[300:470, 200:440]                         # floor more than 2 m from every wall, where none shades it
    assert len(_colours(plain, patch)) == 1 and len(_colours(tiled, patch)) > 20

def test_contact_shadows_darken_the_floor_beside_what_stands_on_it():
    room = replace(HALL, tile=0.0)
    empty = _shot(room=room)
    for scene, beside in ((dict(figures=[PERSON]), (AHEAD[0] + 0.3, AHEAD[1] - 0.1, 0.0)),
                          (dict(boxes=[Box(1.0, 6.2, 0.0, 0.6, 0.6, 0.8)]), (1.42, 6.0, 0.0)),
                          (dict(props=[Cylinder(1.0, 6.2, radius=0.3, height=0.8)]), (1.42, 6.1, 0.0))):
        obs = _shot(room=room, **scene)
        u, v, z = project(obs.camera, beside)
        assert obs.depth[int(v), int(u)] == empty.depth[int(v), int(u)] == pytest.approx(z, abs=0.02)   # still the floor
        assert (_at(obs, beside) < _at(empty, beside) - 12).all()
        assert np.array_equal(_at(obs, (3.0, 6.2, 0.0)), _at(empty, (3.0, 6.2, 0.0)))                   # 2 m away: untouched
    raised = _shot(room=room, boxes=[Box(1.0, 6.2, 0.0, 0.6, 0.6, 0.3, z0=2.2)])                        # a sign overhead
    assert np.array_equal(_at(raised, (1.42, 6.0, 0.0)), _at(empty, (1.42, 6.0, 0.0)))

def test_shadows_that_fall_together_leave_the_floor_some_of_its_light():
    room = replace(HALL, tile=0.0)
    under = (1.0, 6.0, 0.0)                                 # the floor below the front of the shelves, seen under the lowest
    shelf = lambda z0: Box(1.0, 6.2, 0.0, 0.6, 0.6, 0.05, z0=z0)
    floor, one = _at(_shot(room=room), under), _at(_shot(room=room, boxes=[shelf(0.3)]), under)
    four = _at(_shot(room=room, boxes=[shelf(z0) for z0 in (0.3, 0.4, 0.5, 0.6)]), under)
    assert one == pytest.approx(floor * 0.6625, abs=1.5)    # a shelf 0.3 m up takes 0.45 * (1 - 0.3 / 1.2) of the light below it
    assert four == pytest.approx(floor * 0.38, abs=1.5)     # the four together would leave 0.265 of it; no less than 0.38 is left

def test_unknown_shapes_and_faces_are_refused():
    with pytest.raises(TypeError, match="Figure"):
        rendered_observation(boxes=[PERSON])
    with pytest.raises(ValueError, match="label_face"):
        rendered_observation(boxes=[replace(DESK, label="L", label_face="bottom")])
    with pytest.raises(ValueError, match="polygon"):
        _shot(room=replace(HALL, polygon=((0, 0), (1, 1))))

def test_twenty_objects_render_in_under_three_seconds():
    boxes = [Box(cx=-3.5 + k, cy=9.0, yaw=0.3 * k, sx=0.6, sy=0.6, sz=1.0 + 0.1 * k, label="BOX") for k in range(10)]
    figures = [Figure(-1.7 + 0.6 * k, 5.0 + 0.2 * k, 0.6 * k) for k in range(10)]
    start = time.perf_counter()
    obs = rendered_observation(boxes=boxes, figures=figures)
    # The ten cylinders hid 100000 pixels with the boxes; ten bodies with arms, legs and a neck hide fewer.
    assert time.perf_counter() - start < 3.0 and (obs.depth != rendered_observation().depth).sum() > 60000

def test_a_furnished_room_with_twelve_people_renders_in_under_five_seconds():
    figures = [Figure(-2.0 + 0.55 * k, 5.0 + 0.5 * (k % 4), 0.5 * k) for k in range(12)]
    boxes = [Box(cx=-3.0 + 0.9 * k, cy=10.5, yaw=-math.pi / 2, sx=0.5, sy=0.8, sz=1.0 + 0.1 * k, label="STALL",
                 band=(40, 40, 40), panel=(250, 250, 240)) for k in range(10)]
    posts = [Cylinder(-3.5 + 0.9 * k, 8.5, radius=0.1, height=1.0) for k in range(10)]
    balls = [Sphere(-3.5 + 0.9 * k, 8.5, 1.2, radius=0.25) for k in range(10)]
    start = time.perf_counter()
    obs = _shot(boxes, figures, room=HALL, props=[*posts, *balls])
    assert time.perf_counter() - start < 5.0 and (obs.depth > 0).all()
    assert (obs.depth != _shot(room=HALL).depth).sum() > 60000
