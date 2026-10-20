"""Between Habitat's frames and the pipeline's (src/sgi/geometry.py).

Habitat's world is y-up, and its cameras look along -z with y up (OpenGL). The pipeline's world is z-up with
the floor at z = 0, and its cameras are optical frames: x right, y down, z forward. The two worlds share x;
the pipeline's y is Habitat's -z, and its z is Habitat's y above the floor. A heading `yaw` in the pipeline's
world, from +x towards +y, is the same angle about Habitat's +y, so Spot's `base_rot` and a humanoid's are the
pipeline's yaw as it stands.
"""
from __future__ import annotations

import math

import numpy as np

# Habitat camera axes (x right, y up, looking along -z) to optical axes (x right, y down, z forward).
_GL_TO_OPTICAL = np.diag([1.0, -1.0, -1.0, 1.0])


def hab_to_world_matrix(floor_y: float) -> np.ndarray:
    """The 4 x 4 transform from Habitat world coordinates to the pipeline's world, with the floor at z = 0."""
    return np.array([[1.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, -1.0, 0.0],
                     [0.0, 1.0, 0.0, -floor_y],
                     [0.0, 0.0, 0.0, 1.0]])


def to_world(p_hab, floor_y: float) -> tuple[float, float, float]:
    """A Habitat world point in the pipeline's world."""
    x, y, z = (float(v) for v in p_hab)
    return x, -z, y - floor_y


def to_hab(x: float, y: float, floor_y: float, z: float = 0.0) -> np.ndarray:
    """A point of the pipeline's world in Habitat's world; on the floor unless `z` is given."""
    return np.array([x, z + floor_y, -y], dtype=np.float32)


def yaw_of(direction_hab) -> float:
    """The pipeline yaw of a horizontal Habitat direction (dx, _, dz)."""
    return math.atan2(-float(direction_hab[2]), float(direction_hab[0]))


def heading_hab(yaw: float) -> np.ndarray:
    """The horizontal unit direction in Habitat's world that a pipeline yaw points along."""
    return np.array([math.cos(yaw), 0.0, -math.sin(yaw)], dtype=np.float32)


def camera_to_world(T_hab_sensor: np.ndarray, floor_y: float) -> np.ndarray:
    """T_world_camera of the pipeline (optical camera to its world) for a Habitat sensor's absolute transform."""
    return hab_to_world_matrix(floor_y) @ np.asarray(T_hab_sensor, dtype=np.float64) @ _GL_TO_OPTICAL


def intrinsics(width: int, height: int, hfov_deg: float) -> tuple[float, float, float, float]:
    """(fx, fy, cx, cy) of a Habitat pinhole camera: square pixels, the principal point at the image centre."""
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return fx, fx, width / 2.0, height / 2.0


def wrap(a: float) -> float:
    """An angle in (-pi, pi]."""
    return math.pi - ((math.pi - a) % (2 * math.pi))
