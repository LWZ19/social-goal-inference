"""E1d tracker (E1D_FROZEN.md): people tracks from the robot's own camera, in place of the sim's true positions.

Per re-solve frame (2 Hz): the detector service's raw answer -> A1's deployed people (pipeline.run_image, A1's frozen
setting, F5 facing; world x, y and yaw from the frame's depth and camera) -> association with the existing tracks.
- Association: nearest pairs first, one to one, within GATE_M; a detection left over starts a new track.
- A track not matched while it should be seen (its point 1.1 m up projects into the image within RANGE_M and the depth
  there is not nearer than its distance minus OCCLUDE_M) counts a miss; MISSES_DROP misses in a row drop it.
- A track not matched while out of view keeps its last position and yaw ("stale"): what a robot can know.
- Matched tracks take the new position and yaw (no smoothing).
Track ids are stable integers; the resolver matches them to the VLM's numbers at the first re-solve as it does with
oracle tracks. Uses only the frame (rgb via the detector, depth, camera). Pure Python + numpy, so it runs in the sgi env.
"""
from __future__ import annotations

import base64
import importlib.util
import json
import math
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

GATE_M = 0.75
MISSES_DROP = 3
RANGE_M = 12.0
OCCLUDE_M = 0.5
MARK_Z = 1.1
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402

A1 = paths.STILL / 'people-finder/early-detector/phase1c/a1'                  # A1's frozen people pipeline
_P = None
_S = None


def _a1():
    global _P, _S
    if _P is None:
        spec = importlib.util.spec_from_file_location('a1pipeline', A1 / 'pipeline.py')
        _P = importlib.util.module_from_spec(spec)
        sys.modules['a1pipeline'] = _P
        spec.loader.exec_module(_P)
        doc = json.loads((A1 / 'chosen.json').read_text())
        _S = doc.get('setting', doc)
    return _P, _S


def detect(rgb_png: bytes, url: str = 'http://127.0.0.1:8899/detect', timeout: float = 10.0) -> dict:
    """The detector service's raw answer for one frame."""
    req = urllib.request.Request(url, data=json.dumps({'rgb_png': base64.b64encode(rgb_png).decode()}).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def people_from_raw(raw: dict, depth: np.ndarray, cam) -> list[dict]:
    """A1's deployed people for one frame: [{'xy': (x, y), 'yaw': float | None, 'score': float}]."""
    P, S = _a1()
    return [dict(xy=tuple(p['xy']), yaw=p.get('yaw'), score=p.get('score')) for p in P.run_image(raw, depth, cam, S, 'F5')]


def in_view(cam, depth: np.ndarray, xy) -> bool:
    from sgi.geometry import project
    pr = project(cam, (xy[0], xy[1], MARK_Z))
    if pr is None:
        return False
    u, v, z = pr
    if not (0 <= u < cam.width and 0 <= v < cam.height):
        return False
    cx, cy = cam.T_world_camera[0, 3], cam.T_world_camera[1, 3]
    if math.hypot(xy[0] - cx, xy[1] - cy) > RANGE_M:
        return False
    dz = float(depth[int(v), int(u)])
    return not (np.isfinite(dz) and dz > 0 and dz < z - OCCLUDE_M)


@dataclass
class Track:
    id: int
    xy: tuple
    yaw: float | None
    last_seen_t: float
    misses: int = 0
    stale: bool = False


@dataclass
class Tracker:
    tracks: list = field(default_factory=list)
    next_id: int = 1
    dropped: list = field(default_factory=list)

    def update(self, people: list[dict], cam, depth: np.ndarray, t: float) -> list[Track]:
        pairs = sorted(((math.hypot(p['xy'][0] - tr.xy[0], p['xy'][1] - tr.xy[1]), i, j)
                        for i, p in enumerate(people) for j, tr in enumerate(self.tracks)), key=lambda x: (x[0], x[1], x[2]))
        used_p, used_t = set(), set()
        for d, i, j in pairs:
            if d <= GATE_M and i not in used_p and j not in used_t:
                used_p.add(i); used_t.add(j)
                tr = self.tracks[j]
                tr.xy, tr.yaw, tr.last_seen_t, tr.misses, tr.stale = people[i]['xy'], people[i]['yaw'], t, 0, False
        keep = []
        for j, tr in enumerate(self.tracks):
            if j not in used_t:
                if in_view(cam, depth, tr.xy):
                    tr.misses += 1
                    tr.stale = False
                    if tr.misses >= MISSES_DROP:
                        self.dropped.append((tr.id, t))
                        continue
                else:
                    tr.stale = True
            keep.append(tr)
        for i, p in enumerate(people):
            if i not in used_p:
                keep.append(Track(self.next_id, p['xy'], p['yaw'], t))
                self.next_id += 1
        self.tracks = keep
        return list(self.tracks)

    def as_people(self) -> list[dict]:
        """The tracks as the resolver's tracked people: [{'id', 'x', 'y', 'yaw', 'stale'}]."""
        return [dict(id=tr.id, x=float(tr.xy[0]), y=float(tr.xy[1]), yaw=tr.yaw, stale=tr.stale) for tr in self.tracks]
