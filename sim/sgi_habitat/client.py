"""The robot's view as the pipeline is given it, and the call to `python -m sgi.serve` (src/sgi/serve.py)."""
from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from sgi_habitat.frames import camera_to_world, intrinsics


@dataclass
class Observation:
    rgb: np.ndarray                # (H, W, 3) uint8
    depth: np.ndarray              # (H, W) float32, camera Z in metres; 0 where nothing was hit
    camera: dict                   # fx, fy, cx, cy, width, height, T_world_camera (4 x 4 list)
    robot: dict                    # x, y, yaw in the pipeline's world


def make_observation(rgba: np.ndarray, depth: np.ndarray, T_hab_sensor: np.ndarray, hfov_deg: float,
                     floor_y: float, robot_pose: tuple) -> Observation:
    height, width = depth.shape
    fx, fy, cx, cy = intrinsics(width, height, hfov_deg)
    camera = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": width, "height": height,
              "T_world_camera": camera_to_world(T_hab_sensor, floor_y).tolist()}
    x, y, yaw = robot_pose
    return Observation(rgb=np.ascontiguousarray(rgba[..., :3]), depth=depth.astype(np.float32), camera=camera,
                       robot={"x": x, "y": y, "yaw": yaw})


def body(observation: Observation, request: str, context: Optional[str] = None) -> dict:
    png, npy = io.BytesIO(), io.BytesIO()
    Image.fromarray(observation.rgb).save(png, format="PNG")
    np.save(npy, observation.depth, allow_pickle=False)
    return {"request": request, "context": context,
            "rgb_png": base64.b64encode(png.getvalue()).decode(),
            "depth_npy": base64.b64encode(npy.getvalue()).decode(),
            "camera": observation.camera, "robot": observation.robot}


class SgiError(RuntimeError):
    pass


class SgiClient:
    def __init__(self, url: str = "http://127.0.0.1:8765", timeout_s: float = 600.0):
        self.url, self.timeout_s = url.rstrip("/"), timeout_s

    def _call(self, path: str, payload: Optional[dict] = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(self.url + path, data=data,
                                         headers={"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            raise SgiError(f"{path}: {error.code} {error.read().decode(errors='replace')}") from error
        except (urllib.error.URLError, OSError) as error:
            raise SgiError(f"cannot reach the pipeline at {self.url}: {error} "
                           f"(start it with: python -m sgi.serve)") from error

    def health(self) -> dict:
        return self._call("/health")

    def goal(self, observation: Observation, request: str, context: Optional[str] = None) -> dict:
        return self._call("/goal", body(observation, request, context))
