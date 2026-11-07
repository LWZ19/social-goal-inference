"""Unit tests of tracker.py with a synthetic camera and depth (no GPU, no service)."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracker as T  # noqa: E402
from sgi.geometry import Camera  # noqa: E402


def camera():
    # camera at the origin, 1.2 m up, looking along +x (world z up); camera axes: x right, y down, z forward
    R = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)     # columns: camera x, y, z in the world
    Tm = np.eye(4)
    Tm[:3, :3] = R
    Tm[:3, 3] = [0, 0, 1.2]
    return Camera(fx=320.0, fy=320.0, cx=320.0, cy=240.0, width=640, height=480, T_world_camera=Tm)


def depth_far():
    return np.full((480, 640), 50.0, dtype=np.float32)


def test_new_tracks_and_stable_ids():
    tr, cam, dep = T.Tracker(), camera(), depth_far()
    tr.update([dict(xy=(4.0, 0.0), yaw=0.0), dict(xy=(5.0, 1.0), yaw=None)], cam, dep, 0.0)
    ids = sorted(t.id for t in tr.tracks)
    tr.update([dict(xy=(4.2, 0.1), yaw=0.1), dict(xy=(5.1, 1.0), yaw=None)], cam, dep, 0.5)
    assert sorted(t.id for t in tr.tracks) == ids
    a = next(t for t in tr.tracks if abs(t.xy[0] - 4.2) < 1e-9)
    assert a.yaw == 0.1 and a.misses == 0


def test_gate_starts_new_track():
    tr, cam, dep = T.Tracker(), camera(), depth_far()
    tr.update([dict(xy=(4.0, 0.0), yaw=None)], cam, dep, 0.0)
    tr.update([dict(xy=(4.0, 0.9), yaw=None)], cam, dep, 0.5)        # 0.9 m > gate 0.75: a new person
    assert len(tr.tracks) == 2


def test_in_view_misses_drop_after_three():
    tr, cam, dep = T.Tracker(), camera(), depth_far()
    tr.update([dict(xy=(4.0, 0.0), yaw=None)], cam, dep, 0.0)
    for k in range(1, 3):
        tr.update([], cam, dep, 0.5 * k)
        assert len(tr.tracks) == 1 and tr.tracks[0].misses == k
    tr.update([], cam, dep, 1.5)
    assert tr.tracks == [] and tr.dropped and tr.dropped[0][0] == 1


def test_out_of_view_is_stale_not_dropped():
    tr, cam, dep = T.Tracker(), camera(), depth_far()
    tr.update([dict(xy=(-3.0, 0.0), yaw=None)], cam, dep, 0.0)       # behind the camera: never in view
    for k in range(1, 6):
        tr.update([], cam, dep, 0.5 * k)
    assert len(tr.tracks) == 1 and tr.tracks[0].stale and tr.tracks[0].misses == 0


def test_occluded_is_stale():
    tr, cam = T.Tracker(), camera()
    dep = np.full((480, 640), 2.0, dtype=np.float32)                   # a wall 2 m ahead hides a person at 4 m
    tr.update([dict(xy=(4.0, 0.0), yaw=None)], cam, depth_far(), 0.0)
    for k in range(1, 5):
        tr.update([], cam, dep, 0.5 * k)
    assert len(tr.tracks) == 1 and tr.tracks[0].stale


def test_one_to_one_nearest_first():
    tr, cam, dep = T.Tracker(), camera(), depth_far()
    tr.update([dict(xy=(4.0, 0.0), yaw=None), dict(xy=(4.0, 0.6), yaw=None)], cam, dep, 0.0)
    ids = {round(t.xy[1], 1): t.id for t in tr.tracks}
    tr.update([dict(xy=(4.0, 0.55), yaw=None), dict(xy=(4.0, 0.05), yaw=None)], cam, dep, 0.5)
    got = {round(t.xy[1], 2): t.id for t in tr.tracks}
    assert got[0.05] == ids[0.0] and got[0.55] == ids[0.6]


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
