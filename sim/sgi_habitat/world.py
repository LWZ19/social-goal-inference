"""The simulator: one scene, the robot's RGB-D camera (agent 0) and a camera that watches from behind (agent 1)."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import habitat_sim
import magnum as mn
import numpy as np
from habitat_sim.utils.common import quat_from_magnum

ROBOT_RGB, ROBOT_DEPTH, VIEW_RGB = "sgi_rgb", "sgi_depth", "view_rgb"    # no prefix of an agent's camera params


@dataclass
class CameraSpec:
    width: int = 640
    height: int = 480
    hfov_deg: float = 90.0
    height_m: float = 1.2              # above the floor
    forward_m: float = 0.35            # ahead of the robot's base
    pitch_down_deg: float = 10.0


@dataclass
class WorldSpec:
    dataset: Optional[Path]            # a scene dataset file; None when `scene` is a stage file (.glb) of its own
    scene: str                         # a scene of the dataset, or the path of a .glb
    camera: CameraSpec
    view_size: tuple = (640, 480)      # (width, height) of the watching camera
    nav_radius_m: float = 0.3          # the navmesh is cut for a body this wide
    nav_height_m: float = 1.5


def _sensor(uuid: str, kind, width: int, height: int, hfov: float, position, pitch_rad: float):
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = uuid
    spec.sensor_type = kind
    spec.resolution = [height, width]
    spec.hfov = mn.Deg(hfov)
    spec.position = mn.Vector3(*position)
    spec.orientation = mn.Vector3(-pitch_rad, 0.0, 0.0)     # a negative rotation about x looks down
    if kind == habitat_sim.SensorType.COLOR:
        spec.clear_color = mn.Color4(0.90, 0.90, 0.89, 1.0)    # where nothing is hit (no ceiling): light, not black
    return spec


def stage_config_for(glb: Path) -> Path:
    """The stage config beside a .glb, written if there is none: y up, as glTF and Blender's glTF export have
    it. Without one, Habitat takes a bare .glb to be z up and lays it on its side. An existing file is kept,
    so it can be edited (for instance to name a simpler `collision_asset`)."""
    config = glb.with_name(glb.stem + ".stage_config.json")
    if not config.exists():
        config.write_text(json.dumps({"render_asset": glb.name, "collision_asset": glb.name,
                                      "up": [0, 1, 0], "front": [0, 0, -1], "requires_lighting": True}, indent=2))
    return config


@dataclass
class PlacedObject:
    """A model put into the scene (a scenario's `objects:`): its bounding box centred on (x, y) of the
    pipeline's world, standing on the floor there (or `z` above it), turned `yaw_deg` about the vertical, and
    scaled by `scale`, or to `height_m` tall when that is given."""
    model: Path                        # .glb, .gltf, .obj, .fbx, .dae: y up
    x: float
    y: float
    z: float = 0.0
    yaw_deg: float = 0.0
    scale: float = 1.0
    height_m: Optional[float] = None


def object_config_for(model: Path) -> Path:
    """The object config beside a model, written if there is none: y up, as glTF has it."""
    config = model.with_name(model.stem + ".object_config.json")
    if not config.exists():
        config.write_text(json.dumps({"render_asset": model.name, "collision_asset": model.name,
                                      "up": [0, 1, 0], "front": [0, 0, -1], "requires_lighting": True}, indent=2))
    return config


def _stage_floor(sim: habitat_sim.Simulator, x: float, z: float) -> float:
    """The height of the stage's own surface under a Habitat (x, z): objects already placed are looked through."""
    ray = habitat_sim.geo.Ray(mn.Vector3(x, 20.0, z), mn.Vector3(0, -1, 0))
    for hit in sim.cast_ray(ray, 100.0).hits:
        if hit.object_id == habitat_sim.stage_id:
            return float(hit.point[1])
    raise ValueError(f"no floor under ({x:.2f}, {-z:.2f}): the object is outside the scene")


def _template(templates, model: Path, scale: float) -> int:
    """The id of a template for `model` at `scale`, which keeps the model's own origin: by default Habitat moves
    it to the centre of mass, which would throw out a placement made from the model's bounding box."""
    template = templates.get_template_by_id(templates.load_configs(str(object_config_for(model)))[0])
    template.scale = mn.Vector3(scale, scale, scale)
    template.compute_COM_from_shape = False
    template.com = mn.Vector3(0, 0, 0)
    return templates.register_template(template, f"{model}@{scale:.6f}")


def place_objects(sim: habitat_sim.Simulator, objects) -> None:
    """Each model is measured as Habitat loads it (any format its importers read: .glb, .gltf, .obj, .fbx, .dae),
    then scaled, turned, and set down with its bounding box centred on (x, y) and its bottom on the floor."""
    templates = sim.get_object_template_manager()
    rigid = sim.get_rigid_object_manager()
    for o in objects:
        probe = rigid.add_object_by_template_id(_template(templates, o.model, 1.0))
        if probe is None:
            raise ValueError(f"{o.model}: Habitat could not load it")
        bb = probe.root_scene_node.cumulative_bb
        lo, hi = np.array(bb.min, dtype=float), np.array(bb.max, dtype=float)
        rigid.remove_object_by_id(probe.object_id)
        scale = o.height_m / (hi[1] - lo[1]) if o.height_m else o.scale
        obj = rigid.add_object_by_template_id(_template(templates, o.model, scale))
        turn = mn.Quaternion.rotation(mn.Deg(o.yaw_deg), mn.Vector3(0, 1, 0))
        obj.rotation = turn
        centre = turn.transform_vector(mn.Vector3(*(scale * (lo + hi) / 2)))
        x, z = o.x, -o.y                                     # the pipeline's (x, y) in Habitat's (x, z)
        floor = _stage_floor(sim, x, z)
        obj.translation = mn.Vector3(x - centre[0], floor + o.z - scale * lo[1], z - centre[2])


def studio_lights():
    """A key light from above and four softer ones from the sides, all directional (w = 0): each vector is the
    way its light travels, so a negative y shines down."""
    Light, Global = habitat_sim.gfx.LightInfo, habitat_sim.gfx.LightPositionModel.Global
    lights = [Light(vector=mn.Vector4(0.3, -1.0, 0.2, 0.0), color=mn.Color3(1.0, 1.0, 1.0), model=Global)]
    for x, z in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        lights.append(Light(vector=mn.Vector4(x, -0.4, z, 0.0), color=mn.Color3(0.55, 0.55, 0.55), model=Global))
    return lights


def make_sim(spec: WorldSpec, objects=()) -> habitat_sim.Simulator:
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_dataset_config_file = "default" if spec.dataset is None else str(spec.dataset)
    scene = Path(spec.scene)
    backend.scene_id = (str(stage_config_for(scene)) if spec.dataset is None and scene.suffix in (".glb", ".gltf")
                        else spec.scene)
    backend.enable_physics = True
    backend.gpu_device_id = 0
    if spec.dataset is None:                       # a stage file of its own: lit, or Habitat draws it unshaded
        backend.override_scene_light_defaults = True
        backend.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY
    cam = spec.camera
    pitch = math.radians(cam.pitch_down_deg)
    robot = habitat_sim.agent.AgentConfiguration(sensor_specifications=[
        _sensor(ROBOT_RGB, habitat_sim.SensorType.COLOR, cam.width, cam.height, cam.hfov_deg, [0, cam.height_m, 0], pitch),
        _sensor(ROBOT_DEPTH, habitat_sim.SensorType.DEPTH, cam.width, cam.height, cam.hfov_deg, [0, cam.height_m, 0], pitch),
    ])
    view_w, view_h = spec.view_size
    view = habitat_sim.agent.AgentConfiguration(sensor_specifications=[
        _sensor(VIEW_RGB, habitat_sim.SensorType.COLOR, view_w, view_h, 70.0, [0, 0, 0], 0.0),     # tilted by its agent
    ])
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [robot, view]))
    if spec.dataset is None:
        sim.set_light_setup(studio_lights())       # every side lit, as a clay view is; the default leaves faces dark
    place_objects(sim, objects)
    # Nothing is pushed about here, and only static furniture is cut out of the navmesh.
    for manager in (sim.get_rigid_object_manager(), sim.get_articulated_object_manager()):
        for obj in manager.get_objects_by_handle_substring("").values():
            obj.motion_type = habitat_sim.physics.MotionType.STATIC
    settings = habitat_sim.NavMeshSettings()
    settings.set_defaults()
    settings.agent_radius = spec.nav_radius_m
    settings.agent_height = spec.nav_height_m
    settings.include_static_objects = True
    if not sim.recompute_navmesh(sim.pathfinder, settings):
        raise RuntimeError(f"no navmesh could be computed for scene {spec.scene}")
    return sim


def robot_pathfinder(sim: habitat_sim.Simulator, clearance_m: float, height_m: float = 1.0):
    """A navmesh of its own for the robot, cut `clearance_m` back from every wall and piece of furniture: its
    centre keeps that far from them, which leaves room for a body longer than it is wide as it turns."""
    pathfinder = habitat_sim.nav.PathFinder()
    settings = habitat_sim.NavMeshSettings()
    settings.set_defaults()
    settings.agent_radius = clearance_m
    settings.agent_height = height_m
    settings.include_static_objects = True
    if not sim.recompute_navmesh(pathfinder, settings):
        raise RuntimeError(f"no navmesh could be computed for a robot {clearance_m} m in radius")
    return pathfinder


def floor_height(sim: habitat_sim.Simulator) -> float:
    """The height of the floor the navmesh mostly covers: the median, over many navigable points, of where a ray
    cast straight down from half a metre above the point meets the scene. The navmesh itself lies a little above
    the floor, by up to a cell's height."""
    sim.pathfinder.seed(0)
    heights = []
    for _ in range(200):
        p = sim.pathfinder.get_random_navigable_point()
        hits = sim.cast_ray(habitat_sim.geo.Ray(mn.Vector3(p[0], p[1] + 0.5, p[2]), mn.Vector3(0, -1, 0))).hits
        if hits:
            heights.append(hits[0].point[1])
    return float(np.median(heights))


def place_agent(sim: habitat_sim.Simulator, agent_id: int, position, yaw_about_y: float, pitch_up: float = 0.0) -> None:
    """Put an agent's body at a Habitat position, turned `yaw_about_y` about +y from looking along -z and then
    tilted `pitch_up` about its own x (negative: down)."""
    state = habitat_sim.AgentState()
    state.position = np.asarray(position, dtype=np.float32)
    turn = mn.Quaternion.rotation(mn.Rad(yaw_about_y), mn.Vector3(0, 1, 0))
    state.rotation = quat_from_magnum(turn * mn.Quaternion.rotation(mn.Rad(pitch_up), mn.Vector3(1, 0, 0)))
    sim.get_agent(agent_id).set_state(state, reset_sensors=False, infer_sensor_states=True)


def sensor_transform(sim: habitat_sim.Simulator, uuid: str) -> np.ndarray:
    """The absolute 4 x 4 transform of a sensor in Habitat's world."""
    return np.array(sim._sensors[uuid]._sensor_object.node.absolute_transformation())
