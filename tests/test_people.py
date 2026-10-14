import itertools
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from sgi.config import ModelEntry, Thresholds
from sgi.geometry import Camera, RobotPose, bearing, camera_on_robot, in_image, lift, pixel_to_point, project, wrap_angle
from sgi.gt.derive import is_visible
from sgi.imaging import encode_png
from sgi.inputs import ObservationData
from sgi.prompts import build_people_request
from sgi.run import OraclePeople
from sgi.scene import Person, load_scene, save_scene
from sgi.schemas import FACINGS, people_schema
from sgi.stages.people import FoundPerson, PeopleFound, VlmPeople, number_people
from sgi.synth.generate import build_scene
from sgi.synth.recipes import recipe_waiting_line
from sgi.types import DetectedPerson, GroundedTarget, Point
from sgi.vlm.base import ImagePart, TextPart, TransientError, VLMClient
from sgi.vlm.cache import Cache
from sgi.vlm.mock import MockBackend

TH = Thresholds()
HERE = RobotPose(0.0, 0.0, 0.0)                      # at the origin, heading along +x
LENS = dict(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
WALL_M = 5.0                                         # the depth of the picture `facing_wall` gives: everything is 5 m ahead
MIDDLE = {"x": 500, "y": 500}                        # the middle of the picture: it lifts to (5, 0, 1.1), straight ahead at chest height


def facing_wall(robot=HERE, forward_m=0.0, patches=()):
    """What a level camera 1.1 m up sees of a wall 5 m ahead of it: a pixel (u, v) lifts to 5 m ahead of the
    camera, (320 - u) / 100 m to its left and 1.1 - (v - 240) / 100 m up. Each of `patches` is (u, v, depth):
    the depth of the 21 pixels square round (u, v), where something else is seen."""
    camera = Camera(**LENS, T_world_camera=camera_on_robot(robot, forward_m, 1.1, 0.0))
    depth = np.full((480, 640), WALL_M, np.float32)
    for u, v, value in patches:
        depth[v - 10:v + 11, u - 10:u + 11] = value
    rows, cols = np.mgrid[0:480, 0:640]              # a picture no two rows or columns of which are alike
    rgb = np.stack([rows % 256, cols % 256, (rows + cols) % 256], axis=-1).astype(np.uint8)
    return ObservationData(rgb, depth, camera, robot)


def entry(body=MIDDLE, facing="unclear", faced=None, appearance="in grey"):
    return {"appearance": appearance, "body_point": body, "faced_point": faced, "facing": facing}


def source(answer, th=TH, strategy="zero_shot", point_format="xy_1000", max_side=None, trial=0, cache=None):
    """The `vlm` people source on a mock that gives `answer`, and the back end that records what it was sent."""
    backend = MockBackend(lambda parts, json_schema: answer)
    entry_ = ModelEntry(provider="mock", model="mock", point_format=point_format)
    client = VLMClient(entry_, backend, cache, sleep=lambda s: None, clock=itertools.count(0, 60).__next__)   # 60 s per read
    return VlmPeople(client, strategy, th, max_image_side=max_side, trial=trial), backend


def find(*entries, observation=None, **given):
    people, _ = source({"people": list(entries)}, **given)
    return people(observation or facing_wall()).found


def turned(angle):                                   # an angle as a point of the unit circle: -pi and pi are one heading
    return (math.cos(angle), math.sin(angle))


# ------------------------------------------------------------------------------------------- one entry of the answer
def test_the_centre_is_the_lifted_body_point_moved_a_body_radius_farther_from_the_camera():
    straight, aside = find(entry(), entry({"x": 650, "y": 500}, appearance="red coat"))
    assert straight == FoundPerson(xy=pytest.approx((5.25, 0.0)), yaw=None, pixel=(320.0, 240.0), label="in grey")
    seen = np.array([5.0, -0.96])                    # the pixel (416, 240): 0.96 m to the right, on the side of the body that is seen
    assert aside.xy == pytest.approx(tuple(seen + 0.25 * seen / np.hypot(*seen))) and aside.xy == pytest.approx((5.2455, -1.0071), abs=1e-4)
    assert (aside.pixel, aside.label) == ((416.0, 240.0), "red coat")
    wide = find(entry(), th=Thresholds(person_radius_m=0.4))
    assert wide[0].xy == pytest.approx((5.4, 0.0))   # the body's radius is the threshold's


def test_the_centre_is_moved_along_the_line_from_the_camera_not_from_the_robot():
    robot = RobotPose(1.0, 2.0, math.pi / 2)         # looking along +y, its camera 0.2 m ahead of it: over (1, 2.2)
    person, = find(entry({"x": 650, "y": 500}), observation=facing_wall(robot, forward_m=0.2))
    seen, camera = np.array([1.96, 7.2]), np.array([1.0, 2.2])
    assert person.xy == pytest.approx(tuple(seen + 0.25 * (seen - camera) / np.hypot(*(seen - camera))), abs=1e-9)
    from_robot = seen + 0.25 * (seen - [1.0, 2.0]) / np.hypot(*(seen - [1.0, 2.0]))
    assert math.dist(person.xy, from_robot) > 1e-3   # 1.7 mm apart here: the two lines are not one


@pytest.mark.parametrize("facing, heading_deg", list(zip(FACINGS, (180, -135, -90, -45, 0, 45, 90, 135, None))))
def test_a_facing_turns_the_bearing_to_the_camera_anticlockwise_by_its_row_of_the_table(facing, heading_deg):
    person, = find(entry(facing=facing))             # at (5.25, 0), the camera at the origin: the bearing to it is 180 degrees
    if heading_deg is None:
        assert facing == "unclear" and person.yaw is None
        return
    assert -math.pi < person.yaw <= math.pi and turned(person.yaw) == pytest.approx(turned(math.radians(heading_deg)))
    turn = (0, 45, 90, 135, 180, -135, -90, -45)[FACINGS.index(facing)]                     # the design's row, as it is written
    assert turned(person.yaw) == pytest.approx(turned(math.pi + math.radians(turn)))
    # What the name says, read off the picture: a point a metre ahead of the person lies to that side of them in the
    # image, and nearer the camera or farther from it than they are.
    camera = facing_wall().camera
    ahead = (person.xy[0] + math.cos(person.yaw), person.xy[1] + math.sin(person.yaw), 1.1)
    (u, _, z), (u0, _, z0) = project(camera, ahead), project(camera, (*person.xy, 1.1))
    assert ("right" in facing) == (u > u0 + 1) and ("left" in facing) == (u < u0 - 1)
    assert facing.startswith("towards") == (z < z0 - 0.1) and facing.startswith("away") == (z > z0 + 0.1)


def test_a_facing_is_read_from_where_the_person_stands():
    aside, = find(entry({"x": 650, "y": 500}, facing="towards_camera"))
    assert aside.yaw == pytest.approx(math.atan2(-aside.xy[1], -aside.xy[0]))        # at the camera, from (5.25, -1.01)
    robot = RobotPose(1.0, 2.0, math.pi / 2)
    person, = find(entry(facing="right"), observation=facing_wall(robot, forward_m=0.2))
    assert person.xy == pytest.approx((1.0, 7.45)) and person.yaw == pytest.approx(0.0)   # the image's right is +x there


def test_a_faced_point_gives_the_heading_when_it_lifts_a_body_radius_or_more_from_the_centre():
    looked_at = {"x": 200, "y": 500}                 # the pixel (128, 240): 1.92 m to the left of the person, on the wall
    person, = find(entry(facing="left", faced=looked_at))
    assert person.yaw == pytest.approx(math.atan2(1.92, 5.0 - 5.25))                 # 97.4 degrees: not the 90 of `left`
    assert find(entry(facing="unclear", faced=looked_at))[0].yaw == pytest.approx(math.atan2(1.92, -0.25))
    nowhere = facing_wall(patches=[(128, 240, 0.0)])                                 # no depth where the faced point is
    assert find(entry(facing="left", faced=looked_at), observation=nowhere)[0].yaw == pytest.approx(math.pi / 2)
    assert find(entry(facing="unclear", faced=looked_at), observation=nowhere)[0].yaw is None
    own = facing_wall(patches=[(320, 400, 5.25), (352, 400, 5.25)])                  # 5.25 m ahead: the ground under the person
    on_them, beside = {"x": 500, "y": 833}, {"x": 550, "y": 833}                     # at their centre, and 0.336 m beside it
    assert find(entry(facing="left", faced=on_them), observation=own)[0].yaw == pytest.approx(math.pi / 2)        # too near: the facing
    assert find(entry(facing="left", faced=beside), observation=own)[0].yaw == pytest.approx(-math.pi / 2)        # far enough: it decides


def test_a_body_point_that_does_not_lift_or_lies_on_the_floor_is_no_person():
    holes = facing_wall(patches=[(160, 240, 0.0), (480, 240, float("nan"))])         # two places without depth
    found = find(entry({"x": 250, "y": 500}, appearance="no depth"), entry({"x": 750, "y": 500}, appearance="nor here"),
                 entry({"x": 400, "y": 700}, appearance="0.14 m up"), entry({"x": 600, "y": 730}, appearance="4 mm under"),
                 entry({"x": 300, "y": 690}, appearance="0.19 m up"), entry(appearance="chest"), observation=holes)
    assert [person.label for person in found] == ["0.19 m up", "chest"]              # in the order of the answer
    assert find() == [] and find(entry({"x": 400, "y": 700}), th=Thresholds(floor_tolerance_m=0.1))[0].xy[0] > 5.0


# ------------------------------------------------------------------------------------------------- the call
@pytest.mark.parametrize("strategy, point_format, max_side, size", [
    ("zero_shot", "xy_1000", None, (640, 480)), ("cot", "xy_pixel", 320, (320, 240))])
def test_the_people_are_asked_for_in_one_call_on_the_camera_image_as_it_is(strategy, point_format, max_side, size):
    observation, th = facing_wall(), Thresholds(max_people=7)
    middle = MIDDLE if point_format == "xy_1000" else {"x": size[0] // 2, "y": size[1] // 2}      # in pixels of the image sent
    answer = {"people": [entry(middle, "left")], **({"analysis": "one person"} if strategy == "cot" else {})}
    people, backend = source(answer, th, strategy, point_format, max_side)
    result = people(observation)
    assert isinstance(result, PeopleFound) and len(backend.calls) == 1 and [call.stage for call in result.calls] == ["people"]
    assert result.calls[0].status == "ok" and result.calls[0].image_size == size
    system, parts, schema = backend.calls[0]
    image = encode_png(observation.rgb, max_side)                    # no ring and no number on it: nothing of an episode
    assert (system, parts) == build_people_request(strategy, image, point_format, 7) and parts[-1] == image
    assert schema == people_schema(strategy, point_format, size).model_json_schema()
    assert "List at most 7 people." in parts[0].text and [type(part) for part in parts] == [TextPart, ImagePart]
    person, = result.found                                           # the point is read at the size sent, and lifted at the observation's
    assert person == FoundPerson(xy=pytest.approx((5.25, 0.0)), yaw=pytest.approx(math.pi / 2), pixel=(320.0, 240.0), label="in grey")
    assert result.call_time_s >= result.calls[0].latency_total_s >= 60.0 and 0.0 <= result.geometry_time_s < 60.0


def test_the_trial_goes_into_the_request_and_a_cached_answer_gives_the_same_people(tmp_path):
    asked = []

    def generate(parts, schema, system=None, trial=0):
        asked.append(trial)
        return real.client.generate(parts, schema, system=system, trial=trial)

    real, _ = source({"people": [entry()]})
    entry_ = ModelEntry(provider="mock", model="mock")
    VlmPeople(SimpleNamespace(entry=entry_, generate=generate), "zero_shot", TH, trial=3)(facing_wall())
    VlmPeople(SimpleNamespace(entry=entry_, generate=generate), "zero_shot", TH)(facing_wall())
    assert asked == [3, 0]
    cache, runs = Cache(tmp_path), []
    for trial in (0, 0, 1):
        people, _ = source({"people": [entry()]}, trial=trial, cache=cache)
        runs.append(people(facing_wall()))
    assert [run.calls[0].cached for run in runs] == [False, True, False]             # the trial is part of the cache key
    assert runs[0].found == runs[1].found == runs[2].found and len(runs[0].found) == 1
    stored = runs[0].calls[0].latency_total_s                                        # fake-clock time: 60 s or more
    assert stored >= 60.0 and all(stored <= run.call_time_s < stored + 60.0 for run in runs[:2])


@pytest.mark.parametrize("answer, status", [("not json", "parse_failed"), ({"people": [entry(facing="north")]}, "parse_failed"),
                                            (TransientError("down"), "vlm_error")])
def test_a_call_without_a_valid_answer_finds_nobody_and_carries_its_status(answer, status):
    people, backend = source(answer)
    result = people(facing_wall())
    assert result.found == [] and [(call.stage, call.status) for call in result.calls] == [("people", status)]
    assert result.geometry_time_s == 0.0 and result.call_time_s >= result.calls[0].latency_total_s >= 60.0


# ------------------------------------------------------------------------------------- merge, cut and number
def someone(x, y, u, v=200.0, yaw=None, label=None):
    return FoundPerson(xy=(x, y), yaw=yaw, pixel=(u, v), label=label)


def thing_at(x, y):                                  # a target whose point lifts to a metre over (x, y)
    return GroundedTarget(point=Point(x=500, y=500), pixel=(320.0, 240.0), world=(x, y, 1.0))


def numbered(found, target=(4.0, 0.0), max_people=40, th=TH, observation=None):
    people, cut = number_people(found, thing_at(*target), observation or facing_wall(), th, max_people)
    return people, cut


def spots(people):
    return [person.xy for person in people]


def test_people_are_numbered_from_the_left_of_the_image_and_keep_what_was_found_of_them():
    found = [someone(5.0, -1.0, 420.0, yaw=0.5, label="on the right"), someone(6.0, 2.0, 153.3, v=220.0, label="on the left"),
             someone(4.0, 0.0, 320.0, yaw=-1.0)]
    people, cut = numbered(found)
    assert cut is False and people == [
        DetectedPerson(number=1, xy=(6.0, 2.0), yaw=None, pixel=(153.3, 220.0), label="on the left"),
        DetectedPerson(number=2, xy=(4.0, 0.0), yaw=-1.0, pixel=(320.0, 200.0), label=None),
        DetectedPerson(number=3, xy=(5.0, -1.0), yaw=0.5, pixel=(420.0, 200.0), label="on the right")]
    assert numbered([]) == ([], False)


def test_in_one_column_of_the_image_the_nearer_the_camera_has_the_lower_number():
    column = [someone(8.0, 0.0, 320.0), someone(3.0, 0.0, 320.0), someone(5.0, 0.0, 320.0), someone(4.0, 1.0, 100.0)]
    assert spots(numbered(column)[0]) == [(4.0, 1.0), (3.0, 0.0), (5.0, 0.0), (8.0, 0.0)]
    elsewhere = facing_wall(RobotPose(9.0, 0.0, math.pi))            # the camera beyond them, looking back: now 8 is the nearest
    assert spots(numbered(column, observation=elsewhere)[0]) == [(4.0, 1.0), (8.0, 0.0), (5.0, 0.0), (3.0, 0.0)]
    ahead = facing_wall(RobotPose(0.0, 0.0, 0.0), forward_m=6.0)     # the robot at the origin, its camera at (6, 0): the camera counts
    assert spots(numbered(column, observation=ahead)[0]) == [(4.0, 1.0), (5.0, 0.0), (8.0, 0.0), (3.0, 0.0)]


def test_a_person_found_within_a_body_radius_of_one_already_kept_is_dropped():
    twice = [someone(5.0, 1.0, 100.0, label="first"), someone(5.1, 1.1, 90.0, label="again"), someone(4.0, -1.0, 400.0)]
    people, cut = numbered(twice)
    assert [(person.xy, person.label) for person in people] == [((5.0, 1.0), "first"), ((4.0, -1.0), None)] and cut is False
    # In the order found: the second goes, so the third, a radius from the second and two from the first, stays.
    chain = [someone(3.0, 0.0, 300.0), someone(3.2, 0.0, 310.0), someone(3.4, 0.0, 320.0)]
    assert spots(numbered(chain)[0]) == [(3.0, 0.0), (3.4, 0.0)]
    assert spots(numbered(chain[::-1])[0]) == [(3.0, 0.0), (3.4, 0.0)] and numbered(chain[::-1])[0][1].pixel == (320.0, 200.0)
    edge = [someone(3.0, 0.0, 300.0), someone(3.25, 0.0, 310.0), someone(3.0, 0.26, 320.0)]      # at the radius, and just beyond it
    assert spots(numbered(edge)[0]) == [(3.0, 0.0), (3.0, 0.26)]
    slim, broad = Thresholds(person_radius_m=0.2), Thresholds(person_radius_m=0.5)               # the radius is the threshold's
    assert len(numbered(edge, th=slim)[0]) == 3 and len(numbered(chain, th=broad)[0]) == 1


def test_the_people_nearest_the_ground_under_the_target_point_are_kept_and_a_cut_is_said():
    found = [someone(5.0, 1.0, 100.0), someone(5.1, 1.1, 90.0), someone(4.0, -1.0, 400.0), someone(9.0, 0.0, 300.0)]
    people, cut = numbered(found, target=(4.0, 0.0), max_people=2)   # the second is the first again; the last is the farthest
    assert spots(people) == [(5.0, 1.0), (4.0, -1.0)] and [person.number for person in people] == [1, 2] and cut is True
    assert numbered(found, max_people=3) == (numbered(found)[0], False) and len(numbered(found)[0]) == 3      # room for all: no cut
    people, cut = numbered(found, target=(9.0, 0.5), max_people=1)   # nearness is to the target, wherever the camera is
    assert spots(people) == [(9.0, 0.0)] and people[0].number == 1 and cut is True
    file = [someone(float(x), 0.0, 320.0) for x in (7, 3, 6, 4, 5)]  # a guard: the farthest from the target go, the end of a file
    people, cut = numbered(file, target=(2.0, 0.0), max_people=3)
    assert spots(people) == [(3.0, 0.0), (4.0, 0.0), (5.0, 0.0)] and cut is True
    pair = [someone(4.0, 1.0, 200.0, label="found first"), someone(4.0, -1.0, 400.0)]             # as near the target as each other
    assert [person.label for person in numbered(pair, max_people=1)[0]] == ["found first"]
    # A person found twice takes one place: merging comes before the cut, and the cut before the numbers.
    assert numbered([someone(5.0, 1.0, 100.0), someone(5.1, 1.1, 90.0), someone(4.0, -1.0, 400.0)], max_people=2)[1] is False


def test_what_a_source_finds_goes_through_the_one_numbering_whoever_found_it():
    answer = {"people": [entry({"x": 650, "y": 500}, appearance="right"), entry(appearance="middle"), entry(appearance="again")]}
    result = source(answer)[0](facing_wall())
    people, cut = number_people(result.found, thing_at(5.0, 0.0), facing_wall(), TH, TH.max_people)
    assert [(person.number, person.label) for person in people] == [(1, "middle"), (2, "right")] and cut is False
    assert all(isinstance(person, DetectedPerson) for person in people) and people[0].xy == pytest.approx((5.25, 0.0))


# ------------------------------------------------------------------------------------ the people of ground truth
@pytest.fixture(scope="module")
def generated(tmp_path_factory):                     # one generated scene with three people in view, loaded as a run loads it
    scene, rgb, depth = build_scene(recipe_waiting_line(np.random.default_rng(3), 3), "s")
    folder = tmp_path_factory.mktemp("people") / "s"
    save_scene(folder, scene, rgb, depth)
    return load_scene(folder)


def test_the_oracle_gives_the_visible_people_of_ground_truth_with_their_true_position_and_heading(generated):
    inputs, truth = generated
    observation = inputs.observation
    result = OraclePeople(truth, TH)(observation)
    assert isinstance(result, PeopleFound) and result.calls == [] and result.call_time_s == 0.0 and result.geometry_time_s >= 0.0
    assert len(truth.people) == 3 and all(is_visible(person, observation, TH) for person in truth.people)
    assert result.found == [                         # in the order of ground truth, each marked at chest height, none with a label
        FoundPerson(xy=(person.x, person.y), yaw=person.yaw, pixel=project(observation.camera, (person.x, person.y, 1.1))[:2], label=None)
        for person in truth.people]
    for person, found in zip(truth.people, result.found):
        seen = lift(observation.depth, observation.camera, *found.pixel, TH.lift_window)         # the picture shows their body there
        assert in_image(observation.camera, *found.pixel) and math.dist(seen[:2], (person.x, person.y)) < 0.3 and 0.9 < seen[2] < 1.3
    higher = OraclePeople(truth, Thresholds(person_mark_height_m=1.5))(observation).found     # the height of the mark is the threshold's
    assert all(up.pixel[1] < chest.pixel[1] - 20 and up.xy == chest.xy for up, chest in zip(higher, result.found))


def test_the_oracle_leaves_out_whoever_does_not_pass_the_visibility_rule(generated):
    inputs, truth = generated
    observation, robot = inputs.observation, inputs.observation.robot
    first = truth.people[0]
    away = math.hypot(first.x - robot.x, first.y - robot.y)
    behind_first = Person(id="q0", x=first.x + 0.6 * (first.x - robot.x) / away, y=first.y + 0.6 * (first.y - robot.y) / away, yaw=0.0)
    behind_the_robot = Person(id="q1", x=robot.x - 2.0 * math.cos(robot.yaw), y=robot.y - 2.0 * math.sin(robot.yaw), yaw=0.0)
    not_in_the_picture = Person(id="q2", x=robot.x + 3.0 * math.cos(robot.yaw), y=robot.y + 3.0 * math.sin(robot.yaw), yaw=0.0)
    more = replace(truth, people=(behind_first, *truth.people, behind_the_robot, not_in_the_picture))
    assert [is_visible(person, observation, TH) for person in more.people] == [False, True, True, True, False, False]
    found = OraclePeople(more, TH)(observation).found
    assert [person.xy for person in found] == [(person.x, person.y) for person in truth.people]
    strict = Thresholds(visibility_min_samples=10)   # more than the nine heights the rule samples: nobody passes
    assert OraclePeople(truth, strict)(observation).found == [] and OraclePeople(replace(truth, people=()), TH)(observation).found == []


def test_a_model_that_points_at_each_body_finds_the_people_of_ground_truth_and_both_get_the_same_numbers(generated):
    inputs, truth = generated
    observation, camera = inputs.observation, inputs.observation.camera
    camera_xy = camera.T_world_camera[:2, 3]
    oracle = OraclePeople(truth, TH)(observation)

    def careful(person):                             # a careful reader's answer: the body point, and the nearest of the eight facings
        x, y = pixel_to_point(*person.pixel, camera.width, camera.height)
        off = [abs(wrap_angle(bearing(person.xy, camera_xy) + math.radians(turn) - person.yaw))
               for turn in (0, 45, 90, 135, 180, -135, -90, -45)]
        return entry({"x": x, "y": y}, FACINGS[int(np.argmin(off))])

    people, backend = source({"people": [careful(person) for person in oracle.found]})
    asked = people(observation)
    assert len(backend.calls) == 1 and len(asked.found) == 3
    raw = [math.dist(lift(observation.depth, camera, *person.pixel, TH.lift_window)[:2], person.xy) for person in oracle.found]
    moved = [math.dist(found.xy, person.xy) for found, person in zip(asked.found, oracle.found)]
    assert max(moved) < TH.person_radius_m and sum(moved) < sum(raw)                 # the move brings the centres nearer the true ones
    turned_by = [abs(wrap_angle(found.yaw - person.yaw)) for found, person in zip(asked.found, oracle.found)]
    assert max(turned_by) < math.radians(30.0)       # half a step of the table is 22.5 degrees
    target = thing_at(*next(iter(truth.targets.values())).anchor[:2])
    numbers = [number_people(found, target, observation, TH, TH.max_people) for found in (oracle.found, asked.found)]
    (from_truth, cut), (from_answer, cut_too) = numbers
    assert cut is False and cut_too is False
    assert [person.number for person in from_truth] == [1, 2, 3] == [person.number for person in from_answer]
    assert all(math.dist(a.xy, b.xy) < TH.person_radius_m for a, b in zip(from_truth, from_answer))       # number by number, one person
    assert [person.pixel[0] for person in from_truth] == sorted(person.pixel[0] for person in oracle.found)
