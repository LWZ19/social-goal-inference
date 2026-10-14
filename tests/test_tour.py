import dataclasses
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import sgi.synth.generate as gen
from sgi.config import ConfigError, Thresholds, load_config
from sgi.geometry import Camera, RobotPose, camera_on_robot
from sgi.gt.derive import prepare
from sgi.inputs import TourFrameData
from sgi.pipeline import Pipeline
from sgi.polygons import oriented_rectangle
from sgi.run import main, run, run_episode, summarise
from sgi.scene import SceneError, SceneFile, load_scene, save_scene
from sgi.synth.generate import _build_tour
from sgi.synth.recipes import Layout
from sgi.synth.render import Box, Figure, render
from sgi.types import GoalResult, Usage
from sgi.vlm.base import BackendResult, TransientError, VLMClient
from sgi.vlm.mock import MockBackend
from tests.helpers import INTRINSICS, ROBOT, camera_for, rendered_observation, scene_dict, write_scene

DESK = Box(cx=1.0, cy=6.5, yaw=-math.pi / 2, sx=1.0, sy=1.0, sz=1.2)      # fills scene_dict()'s footprint; front face at y = 6
POSES = [(-3.0, 6.0, 0), (-2.4641, 4.0, 30), (-1.0, 2.5359, 60), (1.0, 2.0, 90),     # ROBOT swung about the anchor (1, 6)
         (3.0, 2.5359, 120), (4.4641, 4.0, 150), (5.0, 6.0, 180), (4.4641, 8.0, 210)]

def desk_tour(anchor=(1.0, 6.0, 1.0), second=None, edit=None, boxes=(DESK,), figures=(), mount=None, **thresholds):
    doc = scene_dict()
    doc["ground_truth"]["targets"][0]["anchor"] = list(anchor)
    if second:
        doc["ground_truth"]["targets"].append({"id": "t1", "kind": "object", "label": "sign", "anchor": list(second)})
    if edit:
        edit(doc["ground_truth"])
    if mount:                                       # the observation's camera on another mount than the helpers'
        doc["observation"]["T_world_camera"] = camera_on_robot(ROBOT, *mount).tolist()
    scene = SceneFile.model_validate(doc)
    layout = Layout("desk", tuple(boxes), tuple(figures), ROBOT, scene.ground_truth, tuple(scene.episodes))
    toured, frames = _build_tour(layout, scene, Thresholds(**thresholds))
    return scene, toured, frames, [t.tour_frames for t in toured.ground_truth.targets]

def at(entry, x, y, deg):
    return np.array(entry.T_world_camera) == pytest.approx(camera_for(RobotPose(x, y, math.radians(deg))).T_world_camera, abs=1e-3)

def test_build_tour_swings_the_observation_camera_about_the_targets():
    scene, toured, frames, seen = desk_tour()
    assert scene.tour == [] and scene.ground_truth.targets[0].tour_frames is None      # the scene passed in is left alone
    assert [f.rgb for f in toured.tour] == [f"tour/{k:03d}.png" for k in range(8)]
    assert all(at(entry, *pose) for entry, pose in zip(toured.tour, POSES, strict=True))
    assert toured.tour[3].T_world_camera == scene.observation.T_world_camera           # frame 3 is the observation itself
    assert [f.shape for f in frames] == [(480, 640, 3)] * 8 and np.array_equal(frames[3], rendered_observation(boxes=[DESK]).rgb)
    assert seen == [ALL]                                            # the desk is whole in every frame, whichever side it is seen from
    _, two, _, seen = desk_tour(second=(1.0, 4.0, 1.0))             # C is the mean anchor (1, 5); nothing stands at the second
    assert at(two.tour[0], -2.0, 5.0, 0) and seen == [[0, 1, 2, 3, 4, 5, 6], []]     # frame 7: the desk's end, most of it off the image

ALL = list(range(8))
RED = (210, 40, 40)
HIGH = (0.2, 1.5, math.radians(10.0))               # the generator's mount, which sees a sign above people's heads

def mark(gt):                                       # the target as a mark on the desk's front: nothing stands wholly on its 0.2 m,
    gt["targets"][0].update(footprint=None, match_radius=0.2)      # so only its anchor can show it

def test_a_frame_lists_a_target_for_its_anchor_or_for_half_of_it():
    assert desk_tour(edit=mark)[3] == [[1, 2, 3, 4, 5]]             # the front face is edge-on from the sides and hidden from behind
    assert desk_tour(anchor=(1.0, 6.2, 1.0), edit=mark)[3] == [[2, 3, 4]]    # 0.2 m inside the desk: more than 0.3 m deep at 60 degrees
    assert desk_tour(anchor=(1.0, 6.2, 1.0), edit=mark, anchor_depth_tolerance_m=0.1)[3] == [[]]
    def screened(width):                            # a screen beside the desk: frame 0 looks along +x at the desk's end, 1 m wide,
        screen = Box(cx=0.35, cy=6.0 + width / 2, yaw=0.0, sx=0.1, sy=width, sz=2.0)      # from 3.3 m; the screen is 0.2 m nearer
        return desk_tour(boxes=(DESK, screen))[3]
    assert screened(0.3) == [ALL]                                   # 0.32 m of that end is hidden, and the anchor is edge-on
    assert screened(0.6) == [[1, 2, 3, 4, 5, 6, 7]]                 # 0.64 m of it: less than half the desk is left
    def someone(gt):                                # the target is a person in red where the desk's front was
        gt.update(people=[{"id": "p0", "x": 1.0, "y": 6.0, "yaw": -math.pi / 2}])
        gt["targets"][0].update(kind="person", person_id="p0", anchor=[1.0, 6.0, 1.1], footprint=None, front_yaw=None)
    def before(width):                              # the observation looks at them past something 1 m nearer, on its line of sight
        post = Box(cx=1.0, cy=5.0, yaw=0.0, sx=width, sy=0.08, sz=2.0)
        return desk_tour(anchor=(1.0, 6.0, 1.1), edit=someone, boxes=(post,), figures=(Figure(1.0, 6.0, -math.pi / 2, color=RED),))[3]
    assert before(0.08) == [ALL]                                    # a post hides their chest and little else
    assert before(0.5) == [[0, 1, 2, 4, 5, 6, 7]]                   # a screen hides all of them from frame 3

def test_the_share_of_a_thing_that_a_frame_shows_counts_what_the_image_cuts_off_and_what_stands_before_it():
    board = Box(cx=1.0, cy=6.0, yaw=-math.pi / 2, sx=0.02, sy=1.0, sz=1.0, z0=0.1)      # 1 m square, facing -y, from x = 0.5 to 1.5
    edge = 3.99 * INTRINSICS["cx"] / INTRINSICS["fx"]               # how far to each side the image reaches at the board's face
    def share(x, *others, thing=board, yaw=math.pi / 2):            # from a level camera at (x, 2), 3.99 m before that face
        camera = Camera(**INTRINSICS, T_world_camera=camera_on_robot(RobotPose(x, 2.0, yaw), 0.0, 0.6, 0.0))
        return gen._share_shown(camera, render(camera, [thing, *others])[1], [thing])
    assert share(1.0) == 1.0 and gen._share_shown(camera_for(), rendered_observation().depth, []) == 0.0
    assert share(1.25 - edge) == pytest.approx(0.75, abs=0.01)      # the image ends 0.75 m along the board
    assert share(0.75 - edge) == pytest.approx(0.25, abs=0.01)
    assert share(0.4 - edge) == 0.0 and share(1.0, yaw=-math.pi / 2) == 0.0      # beside the image; behind the camera
    screen = Box(cx=-0.04, cy=5.0, yaw=0.0, sx=1.92, sy=0.02, sz=2.0)                    # 1 m nearer, up to x = 0.92
    assert share(1.0, screen) == pytest.approx(0.6, abs=0.015)      # its edge is seen before x = 0.9: 0.4 m of the board is hidden
    assert share(1.0, dataclasses.replace(screen, cy=7.0)) == 1.0   # the same screen behind the board hides nothing
    wall = dataclasses.replace(board, sy=40.0)                      # a wall 40 m long: the frame shows 4.6 m of it, and the widest
    assert share(1.0, thing=wall) == pytest.approx(1 / 3, abs=0.01)     # picture reckoned with, three frames wide, 13.8 m

def queue(*ys):                                     # people one behind another before the desk, on the observation's line of sight
    def edit(gt):
        gt.update(people=[{"id": f"p{i}", "x": 1.0, "y": y, "yaw": math.pi / 2} for i, y in enumerate(ys)])
    return edit, tuple(Figure(1.0, y, math.pi / 2) for y in ys)

def test_a_frame_lists_a_place_whose_lettering_it_shows_though_a_line_hides_the_anchor():
    pillar = Box(cx=1.0, cy=6.3, yaw=-math.pi / 2, sx=0.5, sy=0.5, sz=1.9)             # a kiosk 0.5 m wide on scene_dict()'s footprint
    def sign(**lettering):                          # and the board over it, 0.6 m wide, whose front is 4 m from the observation
        return Box(cx=1.0, cy=6.05, yaw=-math.pi / 2, sx=0.1, sy=0.6, sz=0.35, z0=1.9, **lettering)
    def seen(board, *ys):
        edit, figures = queue(*ys)
        return desk_tour(edit=edit, boxes=(pillar, board), figures=figures, mount=HIGH)[3]
    assert seen(sign(label="INFO")) == [ALL]                        # nobody there
    line = (5.2, 4.6)                               # two people hide the anchor from frame 3, and the kiosk but for its board
    assert seen(sign(), *line) == [[0, 1, 2, 4, 5, 6, 7]]           # a board with nothing on it: under half of the place is in view
    assert seen(sign(label="INFO"), *line) == [ALL]                 # its name reads over their heads
    assert seen(sign(label="INFO", label_face="back"), *line) == [[0, 1, 2, 4, 5, 6, 7]]      # the name is on the far side
    assert seen(sign(label="INFO"), *line, 3.4) == [[0, 1, 2, 4, 5, 6, 7]]      # the head of a third, 1.2 m from the camera, is before it
    def reads(degrees, board=sign(label="INFO")):   # from 4 m, this far off the square view of the board's front
        a = math.radians(degrees)
        robot = RobotPose(1.0 + 4.0 * math.sin(a), 6.0 - 4.0 * math.cos(a), math.pi / 2 + a)
        camera = Camera(**INTRINSICS, T_world_camera=camera_on_robot(robot, *HIGH))
        return gen._reads(camera, render(camera, [pillar, board])[1], board)
    assert reads(0) and reads(55) and reads(-55) and not reads(70) and not reads(180)      # read from no more than 62 degrees off
    end = sign(label="INFO", label_face="left")     # the board's end towards +x carries the name: 20 and 70 degrees off its square view
    assert reads(70, end) and not reads(20, end) and not reads(-70, end)

def kept(tour, moved=()):                           # whether every entry but those at the indices given is the plain swing
    return all(at(entry, *pose) for k, (entry, pose) in enumerate(zip(tour, POSES, strict=True)) if k not in moved)

def test_tour_poses_stay_in_the_room_and_clear_of_walls_obstacles_and_people():
    def tour(edit):                                 # the tour of the desk scene with its ground truth edited
        return desk_tour(edit=edit)[1].tour
    walled = tour(lambda gt: gt.update(bounds={"polygon": [[-2.5, -8.0], [11.0, -8.0], [11.0, 12.0], [-2.5, 12.0]]}))
    assert at(walled[0], -1.7, 6.0, 0) and at(walled[1], -1.6847, 4.45, 30) and kept(walled, (0, 1))    # 0.8 m inside the wall at x = -2.5
    bench = [[4.6, 5.6], [5.4, 5.6], [5.4, 6.4], [4.6, 6.4]]                    # under the robot of frame 6
    furnished = tour(lambda gt: gt["obstacles"].append({"polygon": bench}))
    assert at(furnished[6], 3.8, 6.0, 180) and kept(furnished, (6,))             # 0.8 m short of it, on the same bearing
    crowded = tour(lambda gt: gt.update(people=[{"id": "p0", "x": 3.0, "y": 2.6, "yaw": 0.0}]))     # 0.06 m from the robot of frame 4
    assert at(crowded[4], 2.4, 3.5751, 120) and kept(crowded, (4,))              # 1.15 m from them: 1.05 m is the least, in 0.1 m steps
    fence = oriented_rectangle((1.0, 6.0), math.radians(30.0), 1.0, 6.0, 0.2)    # along the whole bearing of frame 7
    fenced = tour(lambda gt: gt["obstacles"].append({"polygon": fence}))
    assert at(fenced[7], 4.8637, 7.0353, 195) and kept(fenced, (7,))             # the first bearing that clears it: 15 degrees back
    narrow = [[0.1, 1.1], [1.9, 1.1], [1.9, 9.0], [0.1, 9.0]]                    # a room with no place to stand but the robot's own
    scene, toured, _, _ = desk_tour(edit=lambda gt: gt.update(bounds={"polygon": narrow}))
    assert [entry.T_world_camera for entry in toured.tour] == [scene.observation.T_world_camera] * 8
    for frame in (*walled, *furnished, *crowded, *fenced):                      # the mount of the observation's camera is kept
        T = np.array(frame.T_world_camera)
        assert T[2, 3] == pytest.approx(1.2) and T[2, 2] == pytest.approx(-math.sin(math.radians(20.0)))

def test_the_longer_arm_of_the_tour_leads_past_the_front_of_the_target():
    def tour(front):
        return desk_tour(edit=lambda gt: gt["targets"][0].update(front_yaw=front))[1].tour
    assert kept(tour(-math.pi / 2 + 0.5)) and kept(tour(None))                  # the robot right of the front, and no front at all
    mirrored = tour(-math.pi / 2 - 0.5)                                         # the robot left of the front: the other way round
    assert all(at(entry, *pose) for entry, pose in zip(mirrored[:7], POSES[6::-1])) and at(mirrored[7], -2.4641, 8.0, -30)

STAGES = ("plan", "locate", "context", "goal")
FIELD = {"subtasks": "plan", "target_point": "locate", "stop_elsewhere": "context", "goal_point": "goal"}
LETTER = {"plan": "p", "locate": "L", "context": "c", "goal": "g"}
ON_DESK, ON_FLOOR, GOAL = {"x": 500, "y": 147}, {"x": 935, "y": 450}, {"x": 500, "y": 565}
ACT = {"type": "act", "description": "Ask the way", "target": None, "tour_frame": None}
REFERENCE_PLAN = [{"type": "navigate", "target_ids": ["t0"]}, {"type": "act", "optional": True}]
REFUSED = BackendResult(text=None, finish="refused", usage=Usage())      # an answer without text: parse_failed, never retried

def nav(target, frame):
    return {"type": "navigate", "description": f"Go to the {target}", "target": target, "tour_frame": frame}

class Script:
    """Mock responder: reads the stage off the schema; a locate answer is looked up by the target named in the prompt."""
    def __init__(self, plan, locate):
        self.plan, self.seen = plan, []
        self.locate = {"info desk": ON_DESK, "front door": ON_FLOOR, "lift": None, **locate}
    def __call__(self, parts, json_schema):
        self.seen.append(next(s for field, s in FIELD.items() if field in json_schema["properties"]))
        if self.seen[-1] == "locate":
            point = next(p for target, p in self.locate.items() if f"Target: {target}" in parts[0].text)
            return point if isinstance(point, (Exception, BackendResult)) else {"target_point": point}
        return {"plan": {"subtasks": self.plan}, "context": {"relevant_people": "none", "stop_elsewhere": True},
                "goal": {"goal_point": GOAL}}[self.seen[-1]]

def episode(tmp_path, plan, n_frames=8, reference=(2, 3), locate=(), **run_settings):
    """Episode e0 of scene_dict() in tour mode. Tour frame i is a 64 x 48 image from a robot at (i, 0) facing +y."""
    doc = scene_dict()
    doc["ground_truth"]["targets"][0]["tour_frames"] = list(reference)
    doc["episodes"][0]["ground_truth"]["reference_plan"] = REFERENCE_PLAN
    inputs, truth = load_scene(write_scene(tmp_path, doc, rendered_observation(boxes=[DESK])))
    frames = tuple(TourFrameData(i, np.full((48, 64, 3), 10 * i, np.uint8), camera_for(RobotPose(float(i), 0.0, math.pi / 2)).T_world_camera)
                   for i in range(n_frames))
    inputs = dataclasses.replace(inputs, tour=frames)
    cfg = load_config(overrides={"stages": {s: {"strategy": "zero_shot"} for s in STAGES},
                                 "run": {"grounding_source": "tour", **run_settings}})
    script = Script(plan, dict(locate))
    backend = MockBackend(script)
    pipe = Pipeline(cfg, dict.fromkeys(STAGES, VLMClient(cfg.models["mock"], backend, None, sleep=lambda _: None)))
    def go(inputs=inputs):
        return run_episode(pipe, inputs, truth, prepare(inputs, truth, cfg)["e0"], inputs.episodes[0], cfg, 0)
    return SimpleNamespace(go=go, script=script, backend=backend, inputs=inputs)

def labels(parts):                                  # the plan request: one text, then a label and an image for each frame
    return [p.text for p in parts[1::2]]

def located(record):
    return [s["target_point"] for s in record["plan"]["subtasks"]]

def test_tour_run_reaches_a_goal_and_records_every_call(tmp_path):
    ep = episode(tmp_path, [nav("info desk", 2), ACT])
    record = ep.go()
    result, measures = GoalResult.model_validate(record["result"]), record["measures"]
    assert ep.script.seen == ["plan", "locate", "context", "goal"] == [c.stage for c in result.calls]
    assert set(result.timings_s) == {"plan", "locate", "context", "goal", "geometry"}
    assert (result.status, result.source, record["score"]["success"]) == ("ok", "direct_marking", True)
    assert result.location.target_point.model_dump() == ON_DESK == result.target.point.model_dump()
    assert result.pose_robot == pytest.approx((2.993, 0.0, 0.0), abs=0.005)
    assert (record["evaluated_subtask"], located(record), record["plan"]["image_sha256"]) == (0, [ON_DESK, None], None)
    assert record["tour"] == {"shown": list(range(8)), "chosen": 2, "reference_frames": [2, 3],
                              "coarse_goal_world": pytest.approx([2.0, 0.2, math.pi / 2])}
    assert (measures["tour_frame_correct"], measures["plan_exact"], summarise([record])[0]["tour_frame_accuracy"]) == (True, True, 1.0)
    parts = ep.backend.calls[0][1]                                 # eight different 64 x 48 frames and no 640 x 480 observation
    assert labels(parts) == [f"Frame {i}" for i in range(8)] and len({(p.png, p.width, p.height) for p in parts[2::2]}) == 8
    assert {(p.width, p.height) for p in parts[2::2]} == {(64, 48)}

@pytest.mark.parametrize("frame, status, correct", [(5, "ok", True), (6, "ok", False), (4, "parse_failed", False)])
def test_sixteen_of_twenty_frames_are_shown_under_their_original_indices(tmp_path, frame, status, correct):
    kept = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 19]
    ep = episode(tmp_path, [nav("info desk", frame)], n_frames=20, reference=(4, 9), max_tour_frames=16)
    record = ep.go()
    tour = record["tour"]
    assert labels(ep.backend.calls[0][1]) == [f"Frame {i}" for i in kept]
    assert (record["result"]["status"], tour["shown"], tour["reference_frames"]) == (status, kept, [4, 9])
    assert record["measures"]["tour_frame_correct"] is correct     # 4 and 9 are hidden: 3, 5, 8 and 10 are the nearest frames shown
    if status == "ok":
        assert (tour["chosen"], tour["coarse_goal_world"]) == (frame, pytest.approx([frame, 0.2, math.pi / 2]))
    else:                                                          # frame 4 was not shown: a validation failure, retried once
        assert ep.script.seen == ["plan", "plan"] and (record["plan"], tour["chosen"], tour["coarse_goal_world"]) == (None, None, None)
        assert [(c["stage"], c["validation_retries"]) for c in record["result"]["calls"]] == [("plan", 1)]
        measures = record["measures"]                              # an answer that cannot be read is a wrong plan, not an unmeasured one
        assert (measures["plan_exact"], measures["navigate_recall"], measures["later_subtask_hit_target"]) == (False, 0.0, False)

@pytest.mark.parametrize("plan, locate, sent, calls, status, points, evaluated, later", [
    ([nav("front door", 5), ACT, nav("info desk", 2), nav("lift", None)], {}, "pLLLcg", "pLLLcg", "ok", [ON_FLOOR, None, ON_DESK, None], 0, True),
    ([nav("lift", 7), nav("info desk", 3)], {}, "pLLcg", "pLLcg", "ok", [None, ON_DESK], 1, False),
    ([nav("lift", 2), ACT], {}, "pL", "pL", "no_target", [None, None], None, False),
    ([nav("front door", 2), nav("info desk", 2)], {"front door": "not a point"}, "pLL", "pL", "parse_failed", [None, None], None, False),
    ([nav("info desk", 2), nav("front door", 2), nav("lift", 2)], {"front door": REFUSED}, "pLL", "pLL", "parse_failed", [ON_DESK, None, None], 0, False),
    ([nav("info desk", 2), nav("front door", 2), nav("lift", 2)], {"front door": TransientError("503")},
     "pLLLL", "pLL", "vlm_error", [ON_DESK, None, None], 0, False),
])
def test_every_navigation_subtask_is_located_and_the_first_hit_is_evaluated(tmp_path, plan, locate, sent, calls, status, points, evaluated, later):
    ep = episode(tmp_path, plan, locate=locate)
    record = ep.go()
    result, measures = GoalResult.model_validate(record["result"]), record["measures"]
    assert "".join(LETTER[s] for s in ep.script.seen) == sent      # requests that reached the back end, retries included
    assert "".join(LETTER[c.stage] for c in result.calls) == calls
    assert (result.status, located(record), record["evaluated_subtask"]) == (status, points, evaluated)
    assert (record["tour"]["chosen"], measures["navigate_recall"]) == (plan[0]["tour_frame"], float(ON_DESK in points))
    assert (result.pose_robot is not None, measures["later_subtask_hit_target"]) == (status == "ok", later)    # every hit is matched
    if status == "ok":
        assert result.target.point.model_dump() == points[evaluated] == result.location.target_point.model_dump()
        assert record["score"]["target_correct"] is (points[evaluated] == ON_DESK)

def test_a_scene_without_a_tour_and_oracle_grounding_are_refused(tmp_path):
    ep = episode(tmp_path, [nav("info desk", 2)])
    with pytest.raises(SceneError, match="^s0: tour"):
        ep.go(dataclasses.replace(ep.inputs, tour=()))
    assert ep.script.seen == []                                    # refused before any model call
    with pytest.raises(ConfigError, match="tour"):
        load_config(overrides={"run": {"grounding_source": "tour", "grounding": "oracle"}})

def test_run_in_tour_mode_with_the_default_mock(tmp_path, monkeypatch, capsys):
    _, toured, frames, _ = desk_tour()
    obs = rendered_observation(boxes=[DESK])
    save_scene(tmp_path / "scenes" / "s0", toured, obs.rgb, obs.depth, tour=frames)
    cfg = load_config(overrides={"run": {"grounding_source": "tour", "use_cache": False}, "stages": {"goal": {"strategy": "zero_shot"}}})
    out = run(tmp_path / "scenes", cfg, "tour", tmp_path / "runs")
    (record,) = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    assert record["result"]["status"] == "ok" and "".join(LETTER[c["stage"]] for c in record["result"]["calls"]) == "pLcg"
    assert record["tour"] == {"shown": list(range(8)), "chosen": 0, "reference_frames": list(range(8)),
                              "coarse_goal_world": pytest.approx([-2.8, 6.0, 0.0], abs=1e-6)}
    assert record["measures"]["tour_frame_correct"] is True        # the mock picks frame 0, which shows the desk from one end
    write_scene(tmp_path / "bare", scene_dict(), obs)
    with pytest.raises(SceneError, match="^s0: tour"):
        run(tmp_path / "bare", cfg, "bare", tmp_path / "refused")
    assert not (tmp_path / "refused").exists()                     # stopped when the scene was loaded, before anything is written
    (tmp_path / "tour.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    monkeypatch.chdir(tmp_path)                                    # main writes under ./runs, next to the run above
    assert main(["--scenes", "bare", "--config", "tour.yaml", "--name", "bare"]) == 2
    err = capsys.readouterr().err                                  # exit code 2: one line on stderr, here opening with the scene id
    assert err.startswith("error: s0: tour") and err.count("\n") == 1 and list((tmp_path / "runs").iterdir()) == [out]
