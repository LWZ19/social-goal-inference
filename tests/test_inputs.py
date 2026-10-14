import dataclasses
import numpy as np
from sgi.geometry import Camera, RobotPose
from sgi.inputs import EpisodeInputs, ObservationData, SceneInputs, TourFrameData
from tests.helpers import floor_observation

def test_sha256_follows_the_rgb_array_only():
    obs = floor_observation()
    rgb, flat = obs.rgb.copy(), np.zeros((2, 4, 3), np.uint8)
    rgb[10, 20, 1] += 1                                                   # one channel of one pixel
    assert len(obs.sha256()) == 64 and obs.sha256() == floor_observation().sha256()
    assert dataclasses.replace(obs, rgb=rgb).sha256() != obs.sha256()
    assert dataclasses.replace(obs, depth=obs.depth + 1).sha256() == obs.sha256()
    assert dataclasses.replace(obs, rgb=flat).sha256() != dataclasses.replace(obs, rgb=flat.reshape(4, 2, 3)).sha256()

def test_dataclasses_are_frozen_with_the_shared_fields():
    classes = (Camera, RobotPose, ObservationData, TourFrameData, EpisodeInputs, SceneInputs)
    assert [[f.name for f in dataclasses.fields(c)] for c in classes] == [
        ["fx", "fy", "cx", "cy", "width", "height", "T_world_camera"], ["x", "y", "yaw"],
        ["rgb", "depth", "camera", "robot"], ["index", "rgb", "T_world_camera"],
        ["episode_id", "request", "context"], ["scene_id", "observation", "tour", "episodes"]]
    assert all(c.__dataclass_params__.frozen for c in classes)
