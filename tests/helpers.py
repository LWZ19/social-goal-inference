import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
from sgi.geometry import Camera, RobotPose, camera_on_robot
from sgi.inputs import ObservationData

INTRINSICS = dict(fx=554.3, fy=554.3, cx=320.0, cy=240.0, width=640, height=480)
MOUNT = dict(forward_m=0.2, height_m=1.2, pitch_down_rad=math.radians(20.0))
ROBOT = RobotPose(1.0, 2.0, math.pi / 2)

def camera_for(robot: RobotPose = ROBOT) -> Camera:
    return Camera(**INTRINSICS, T_world_camera=camera_on_robot(robot, **MOUNT))

def floor_observation(robot: RobotPose = ROBOT) -> ObservationData:
    cam, pitch = camera_for(robot), MOUNT["pitch_down_rad"]
    # Closed form, not sgi.geometry: with no roll, a ray of row r drops k[r] metres per metre of camera Z.
    k = math.sin(pitch) + (np.arange(cam.height) + 0.5 - cam.cy) / cam.fy * math.cos(pitch)
    row_depth = np.where(k > 0, MOUNT["height_m"] / np.maximum(k, 1e-9), 0.0)
    depth = np.repeat(row_depth[:, None], cam.width, axis=1).astype(np.float32)
    return ObservationData(np.full((cam.height, cam.width, 3), 128, np.uint8), depth, cam, robot)

def scene_dict(**overrides) -> dict:
    """A valid minimal scene.json dict, built afresh on every call; overrides replace top-level keys."""
    footprint = lambda: [[0.5, 6.0], [1.5, 6.0], [1.5, 7.0], [0.5, 7.0]]
    scene = {
        "format_version": "0.1",
        "scene_id": "s0",
        "observation": {
            "rgb": "rgb.png",
            "depth": "depth.npy",
            "width": INTRINSICS["width"],
            "height": INTRINSICS["height"],
            "intrinsics": {k: INTRINSICS[k] for k in ("fx", "fy", "cx", "cy")},
            "T_world_camera": camera_for(ROBOT).T_world_camera.tolist(),
            "robot_pose_world": {"x": ROBOT.x, "y": ROBOT.y, "yaw": ROBOT.yaw},
        },
        "episodes": [{
            "episode_id": "e0",
            "request": "take me to the info desk",
            "ground_truth": {"request_type": "grounded", "target_ids": ["t0"], "situation": "no_one"},
        }],
        "ground_truth": {
            "people": [],
            "targets": [{"id": "t0", "kind": "place", "label": "info desk", "anchor": [1.0, 6.0, 1.0],
                         "footprint": footprint(), "front_yaw": -math.pi / 2}],
            "bounds": {"polygon": [[-9.0, -8.0], [11.0, -8.0], [11.0, 12.0], [-9.0, 12.0]]},
            "obstacles": [{"polygon": footprint()}],
        },
    }
    scene.update(overrides)
    return scene

def write_scene(tmp_path: Path, scene: dict, obs: ObservationData) -> Path:
    """Write the dict as it is (unvalidated) with obs as rgb.png and depth.npy; returns the scene folder."""
    folder = tmp_path / scene["scene_id"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "scene.json").write_text(json.dumps(scene, indent=2), encoding="utf-8")
    Image.fromarray(obs.rgb).save(folder / "rgb.png")
    np.save(folder / "depth.npy", obs.depth)
    return folder

def rendered_observation(boxes=(), figures=(), robot: RobotPose = ROBOT) -> ObservationData:
    from sgi.synth.render import render
    cam = camera_for(robot)
    rgb, depth = render(cam, boxes, figures)
    return ObservationData(rgb=rgb, depth=depth, camera=cam, robot=robot)
