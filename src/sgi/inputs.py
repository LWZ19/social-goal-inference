"""What the pipeline is given for a scene. Holds no ground truth."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from sgi.geometry import Camera, RobotPose


@dataclass(frozen=True)
class ObservationData:
    rgb: np.ndarray                                # (H, W, 3) uint8
    depth: np.ndarray                              # (H, W) float32, metres, camera Z
    camera: Camera
    robot: RobotPose

    def sha256(self) -> str:
        """Hex SHA-256 of the RGB array's shape and bytes."""
        digest = hashlib.sha256(repr(self.rgb.shape).encode())
        digest.update(self.rgb.tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class TourFrameData:
    index: int
    rgb: np.ndarray
    T_world_camera: np.ndarray


@dataclass(frozen=True)
class EpisodeInputs:
    episode_id: str
    request: str
    context: str | None


@dataclass(frozen=True)
class SceneInputs:
    scene_id: str
    observation: ObservationData
    tour: tuple[TourFrameData, ...]
    episodes: tuple[EpisodeInputs, ...]
