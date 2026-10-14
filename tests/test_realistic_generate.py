"""The realistic generator through the stub renderer (realistic-scenes spec 5.6 and 6): what a run writes for
a few scenes of seed 0, and what it does when it is run again, on part of the set, or when something fails.

The scenes are one of each sort the set has: a stall with nobody before it and someone behind it, the parcel on
its table, someone alone, a line at a desk, people talking with others standing by, and a line of the examples
split. Each is held against its twin of the placeholder set, which the run tests generate once for the session.

The placeholder generator takes half a minute to draw the layouts of a seed, and every run of the realistic
generator asks it for all of them. Here they are drawn once, through the generator's own call, and handed out
again after that (Replay). The library is the one the repository's manifest describes, prepared or not: the
stub opens no asset.

A renderer is called four times for a scene (CALLS): for the observation's depth of its dressing in each of the
three variants, which have to pass the scene check together, and then for all its frames. A test that only asks
whether a scene would be drawn again hands in a renderer that fails at its first call (Failing(1)): the run
then stops where the drawing would begin, and costs nothing.
"""
import dataclasses
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import sgi.synth.generate as gen
import sgi.synth.realistic.generate as real
from sgi.check_scene import check_scene, has_errors
from sgi.config import REPO_ROOT, load_config
from sgi.freespace import static_grid
from sgi.geometry import Camera, RobotPose, camera_on_robot, in_image, project
from sgi.polygons import distance_to_boundary, distance_to_polygon, point_in_polygon
from sgi.scene import Person, load_scene, read_scene_file
from sgi.synth.realistic import blender
from sgi.synth.realistic.blender_renderer import PREPARE_COMMAND
from sgi.synth.realistic.constructions import Lettered
from sgi.synth.realistic.dress import VARIANTS, dress
from sgi.synth.realistic.library import load_library
from sgi.synth.realistic.spec import Frame, from_json, to_json
from sgi.synth.realistic.stub import StubRenderer
from sgi.synth.recipes import CAMERA_LENS, CAMERA_MOUNT
from sgi.synth.render import Box, render
from tests.test_run import placeholder  # noqa: F401

CFG = load_config()
TH = CFG.thresholds
PLACE = "eval_02_no_one_place"                      # a stall with nobody before it and someone behind it
PARCEL = "eval_04_no_one_object"                    # the parcel on its table: a target without a footprint
PERSON = "eval_05_no_one_person"                    # someone alone, who is asked for; the cheapest scene to draw
LINE = "eval_07_waiting_line_2"                     # two people waiting at a desk, and someone behind it
GROUP = "eval_11_conversation_group_3"              # three people talking, one of them asked for, and two more beside
EXAMPLE = "examples_03_waiting_line_3"              # of the other split: three at a ticket counter with glass round it
SCENES = (LINE, EXAMPLE, PLACE, GROUP, PARCEL, PERSON)        # not in the order of the set, which a run keeps
IN_ORDER = (PLACE, PARCEL, PERSON, LINE, GROUP, EXAMPLE)
STAFFED = (PLACE, LINE, EXAMPLE)
JUDGED = (PLACE, PARCEL, GROUP)                     # the scenes whose tour frames are judged again, rule by rule
SPLITS = ("eval", "examples")                       # a dressing's seed names its split by its place in this
ATTEMPTS = 5                                        # a scene is dressed this often at the most
CALLS = 4                                           # of the renderer for a scene: each variant's depth, then the frames
JOB_KEYS = ["attempt", "judged", "renderer", "seed", "spec", "thresholds", "variant"]
TOUR = [f"tour/{k:03d}.png" for k in range(8)]
WRITTEN = sorted(["scene.json", "rgb.png", "depth.npy", "render_job.json", *TOUR])    # a scene folder, whole
OBSERVATION = gen._TOUR_TURNS_DEG.index(0)          # the tour frame that is the observation itself
NONE = {"rendered": 0, "skipped": 0, "dressed_again": 0}
WHOLE_SET = sum(len(rows) for _, rows in gen._SPLITS)       # every row of both splits of the placeholder generator


class Replay:
    """accepted_layouts, drawn once for a seed and handed out again after that. A call is remembered with what
    it was asked for and how far it was read."""

    def __init__(self):
        self.kept, self.calls = {}, []

    def __call__(self, work_dir, cfg, seed=0):
        call = SimpleNamespace(work_dir=Path(work_dir), seed=seed, handed_out=0, ended=False)
        self.calls.append(call)
        assert cfg == CFG                           # the layouts are kept by their seed alone
        if seed not in self.kept:
            self.kept[seed] = list(gen.accepted_layouts(work_dir, cfg, seed))
        for row in self.kept[seed]:
            call.handed_out += 1
            yield row
        call.ended = True


REPLAY = Replay()
FIRST = {}                                          # of the run that the fixture `realistic` makes: counts and calls


class Counting(StubRenderer):
    """The stub, remembering each call: the description and the frames it was asked for."""

    def __init__(self, version=None, name="stub"):
        self.calls, self.version, self.name = [], version, name

    def render(self, spec, frames):
        self.calls.append((spec, list(frames)))
        return super().render(spec, frames)


class Failing(Counting):
    """The stub until its call number `at`, which fails as a renderer does that runs out of memory."""

    def __init__(self, at):
        super().__init__()
        self.at = at

    def render(self, spec, frames):
        if len(self.calls) + 1 == self.at:
            self.calls.append((spec, list(frames)))
            raise RuntimeError("the card ran out of memory")
        return super().render(spec, frames)


@pytest.fixture(autouse=True)
def replayed(monkeypatch):
    monkeypatch.setattr(real, "accepted_layouts", REPLAY)


def watching(judged):
    """real._visible, noting in `judged` what it is asked of each frame and target, and what it answers."""
    visible = real._visible

    def watched(camera, depth, target, lettering, alone, left, above, th):
        listed = visible(camera, depth, target, lettering, alone, left, above, th)
        judged.append(SimpleNamespace(camera=camera, depth=depth, target=target, lettering=lettering, alone=alone,
                                      left=left, above=above, listed=listed))
        return listed

    return watched


@pytest.fixture(scope="module")
def realistic(tmp_path_factory):
    """The six scenes in the default variant, generated once: the folder they are written under."""
    out, renderer, judged = tmp_path_factory.mktemp("realistic"), Counting(), []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(real, "accepted_layouts", REPLAY)
        patch.setattr(real, "_visible", watching(judged))
        FIRST["counts"] = real.generate(out, CFG, seed=0, variant="real", renderer=renderer, only=SCENES)
    FIRST["calls"], FIRST["judged"], FIRST["work_dir"] = renderer.calls, judged, REPLAY.calls[-1].work_dir
    return out


def run(out, variant="real", only=SCENES, renderer=None, **more):
    return real.generate(out, CFG, 0, variant, renderer if renderer is not None else StubRenderer(), only, **more)


def rows():
    """The accepted layouts of seed 0 by the placeholder's scene id: (split, index, layout, scene)."""
    return {scene.scene_id: (split, index, layout, scene) for split, index, layout, scene in REPLAY.kept[0]}


def split_of(name):
    return name.split("_")[0]


def folder_of(out, name, variant="real"):
    return Path(out) / split_of(name) / f"{variant}_{name}"


def twin_of(placeholder, name):                     # noqa: F811
    return placeholder / split_of(name) / name


def job_of(folder):
    return json.loads((folder / "render_job.json").read_text(encoding="utf-8"))


def files(folder):
    return {str(path.relative_to(folder)): path.read_bytes() for path in sorted(folder.rglob("*")) if path.is_file()}


def state(folder):
    """What tells whether a folder was touched: the folder and each thing under it, by its inode, the time it
    was last written and its bytes."""
    return [(str(path.relative_to(folder)), path.stat().st_ino, path.stat().st_mtime_ns,
             path.read_bytes() if path.is_file() else None) for path in (folder, *sorted(folder.rglob("*")))]


def entries(out):
    """Everything directly under the split folders of a set: the scene folders, and whatever a run left beside them."""
    return sorted(f"{split.name}/{entry.name}" for split in Path(out).iterdir() for entry in split.iterdir())


def set_of(names, variant="real"):
    return sorted(f"{split_of(name)}/{variant}_{name}" for name in names)


def warnings(results):
    return {result.number for result in results if result.level == "warning"}


def density(count):
    return "low" if count <= 2 else "medium" if count <= 5 else "high"


def camera_of(scene, T=None):
    obs, lens = scene.observation, scene.observation.intrinsics
    return Camera(lens.fx, lens.fy, lens.cx, lens.cy, obs.width, obs.height,
                  np.array(obs.T_world_camera if T is None else T, dtype=np.float64))


def same_view(a, b):                                # two cameras alike: Camera holds an array, which == cannot compare
    return ((a.fx, a.fy, a.cx, a.cy, a.width, a.height) == (b.fx, b.fy, b.cx, b.cy, b.width, b.height)
            and np.array_equal(a.T_world_camera, b.T_world_camera))


# --- What a run writes. ---

def test_every_scene_passes_the_scene_check_with_only_the_warning_its_twin_has(realistic, placeholder):  # noqa: F811
    assert FIRST["counts"] == {**NONE, "rendered": len(SCENES)}
    for name in SCENES:
        results = check_scene(folder_of(realistic, name), CFG, write_png=False)
        twin = check_scene(twin_of(placeholder, name), CFG, write_png=False)
        assert len(results) == 12 and not has_errors(results), (name, [r.message for r in results if r.level != "ok"])
        assert warnings(results) <= warnings(twin) & {9}, name


def test_staff_carry_the_role_and_stand_on_free_cells(realistic):
    for name in SCENES:
        _, truth = load_scene(folder_of(realistic, name))
        staff = [person for person in truth.people if person.role is not None]
        # Whoever serves at the target, where it is a staffed place, and then the vendor of the second stall,
        # which every one of these scenes has on a wall.
        assert [(person.id, person.role, person.group) for person in staff] == [
            (f"s{number}", "staff", None) for number in range(2 if name in STAFFED else 1)], name
        assert truth.people[len(truth.people) - len(staff):] == tuple(staff)      # after the people of the twin
        grid = static_grid(truth, TH.grid_resolution_m)
        assert all(grid.is_free(person.x, person.y) for person in staff), name
        (target,) = truth.targets.values()
        on_target = [target.footprint is not None and distance_to_polygon((person.x, person.y), target.footprint) == 0.0
                     for person in staff]
        # The first behind the front of the place they serve at, inside its ground; the vendor on no target's.
        assert on_target == ([True, False] if name in STAFFED else [False]), name


def test_a_scene_is_the_twin_of_its_placeholder(realistic, placeholder):  # noqa: F811
    for name in SCENES:
        ours, theirs = read_scene_file(folder_of(realistic, name)), read_scene_file(twin_of(placeholder, name))
        truth, twin = ours.ground_truth, theirs.ground_truth
        assert [person for person in truth.people if person.role is None] == twin.people, name
        untoured = [[target.model_copy(update={"tour_frames": None}) for target in each.targets]
                    for each in (truth, twin)]
        assert untoured[0] == untoured[1], name
        assert truth.bounds == twin.bounds and truth.occupancy == twin.occupancy is None, name
        assert ours.observation == theirs.observation, name     # the robot's pose, T_world_camera, the lens, the files
        # What blocks the floor is the twin's but under a staffed target place and under the second stall: one
        # rectangle of the twin for each gives way to what is drawn there, which lies on it.
        gone = [obstacle for obstacle in twin.obstacles if obstacle not in truth.obstacles]
        drawn = [obstacle for obstacle in truth.obstacles if obstacle not in twin.obstacles]
        assert len(gone) == (2 if name in STAFFED else 1) and len(drawn) > len(gone), name
        assert all(any(all(distance_to_polygon(tuple(corner), was.polygon) <= 1e-9 for corner in obstacle.polygon)
                       for was in gone) for obstacle in drawn), name
        assert [obstacle for obstacle in truth.obstacles if obstacle in twin.obstacles] == [
            obstacle for obstacle in twin.obstacles if obstacle not in gone], name
        # Rows pair by (twin scene id, episode id): an episode asks the same of the same people.
        assert [(e.episode_id, e.request, e.context, e.ground_truth.request_type) for e in ours.episodes] == [
            (e.episode_id, e.request, e.context, e.ground_truth.request_type) for e in theirs.episodes], name
        assert len(ours.episodes) == 2
        for episode, other in zip(ours.episodes, theirs.episodes):
            mine, twins = episode.ground_truth, other.ground_truth
            regionless = {"acceptable_regions": None, "forbidden_regions": None}
            if name in STAFFED and twins.situation == "no_one":
                # The one field in which a scene with staff may differ from its twin (dress.py): what the rationale
                # says of the people, where nobody waits at the target and a member of staff stands behind it.
                assert name == PLACE
                assert mine.rationale.people != twins.rationale.people, name
                assert "a member of staff stands behind it" in mine.rationale.people, name
                mine = mine.model_copy(update={"rationale": mine.rationale.model_copy(
                    update={"people": twins.rationale.people})})
            assert mine.model_copy(update=regionless) == twins.model_copy(update=regionless), name
            assert mine.acceptable_regions and mine.forbidden_regions is not None, name
            for region, twin_region in zip(mine.acceptable_regions, twins.acceptable_regions, strict=True):
                assert region.polygon == twin_region.polygon, name
                assert region.model_copy(update={"representative_point": None}) == twin_region.model_copy(
                    update={"representative_point": None}), name
                assert region.representative_point is not None, name
            assert [(region.kind, region.polygon, region.target_id) for region in mine.forbidden_regions] == [
                (region.kind, region.polygon, region.target_id) for region in twins.forbidden_regions], name


def test_ids_folders_and_source_follow_the_variant(realistic):
    assert entries(realistic) == set_of(SCENES) and sorted(p.name for p in realistic.iterdir()) == list(SPLITS)
    work_dir = FIRST["work_dir"]                    # where the layouts were drawn: not in the set, and gone again
    assert not work_dir.exists() and realistic not in work_dir.parents
    library = load_library()
    for name in SCENES:
        folder = folder_of(realistic, name)
        scene = read_scene_file(folder)
        assert scene.scene_id == folder.name == f"real_{name}"
        assert (scene.source.simulator, scene.source.version) == ("sgi-synth-stub/real", None)
        assert sorted(files(folder)) == WRITTEN, name           # and nothing else: no check.png, nothing half written
        job = job_of(folder)
        assert sorted(job) == JOB_KEYS, name
        assert (job["variant"], job["seed"], job["renderer"]) == ("real", 0, {"name": "stub", "version": None}), name
        split, index, layout, _ = rows()[name]
        dressed = dress(layout, VARIANTS["real"], library, (0, SPLITS.index(split), index, job["attempt"]))
        assert job["spec"] == to_json(dressed.spec) and from_json(job["spec"]) == dressed.spec, name
        assert scene.ground_truth.people == dressed.scene_gt.people, name
        assert scene.ground_truth.obstacles == dressed.scene_gt.obstacles, name
        # And what else decides the folder: the thresholds, and a digest of what each variant was judged from.
        assert job["thresholds"] == json.loads(TH.model_dump_json()), name
        assert list(job["judged"]) == list(VARIANTS) and len(set(job["judged"].values())) == 3, name
        assert all(len(digest) == 64 and set(digest) <= set("0123456789abcdef") for digest in job["judged"].values())


def test_density_counts_people_without_a_role(realistic, placeholder):  # noqa: F811
    tipped = []
    for name in SCENES:
        ours, theirs = read_scene_file(folder_of(realistic, name)), read_scene_file(twin_of(placeholder, name))
        people = ours.ground_truth.people
        label = density(sum(person.role is None for person in people))
        factors = [[episode.ground_truth.factors for episode in scene.episodes] for scene in (ours, theirs)]
        assert [entered.density for entered in factors[0]] == [label, label] and factors[0] == factors[1], name
        if density(len(people)) != label:
            tipped.append(name)
    # Two waiting are "low"; with whoever serves them and the vendor of the second stall counted, "medium". Five
    # about the person asked for are "medium"; with the vendor counted, "high".
    assert tipped == [LINE, GROUP]


def test_the_tour_has_eight_frames_with_the_observation_among_them(realistic, placeholder):  # noqa: F811
    for name in SCENES:
        folder = folder_of(realistic, name)
        scene, twin = read_scene_file(folder), read_scene_file(twin_of(placeholder, name))
        inputs, truth = load_scene(folder)
        obs = inputs.observation
        assert [frame.rgb for frame in scene.tour] == TOUR and [frame.index for frame in inputs.tour] == list(range(8))
        assert all(frame.rgb.shape == obs.rgb.shape and frame.rgb.dtype == np.uint8 for frame in inputs.tour), name
        here = inputs.tour[OBSERVATION]
        assert np.array_equal(here.T_world_camera, obs.camera.T_world_camera), name
        assert np.array_equal(here.rgb, obs.rgb), name
        assert (folder / TOUR[OBSERVATION]).read_bytes() == (folder / "rgb.png").read_bytes(), name
        assert obs.rgb.std() > 10.0, name           # a picture, and not the grey one the scene was judged with
        assert np.array_equal(np.load(folder / "depth.npy"), obs.depth) and obs.depth.dtype == np.float32
        for target in truth.targets.values():       # the observation passed check 5, so it shows the anchor
            frames = target.tour_frames
            assert OBSERVATION in frames and frames == sorted(set(frames)) and set(frames) <= set(range(8)), name
        # The placeholder's walk, on this scene's own ground truth: its staff and what its place blocks.
        pose, drawn = scene.observation.robot_pose_world, scene.ground_truth
        T_obs, robot = np.array(scene.observation.T_world_camera), RobotPose(pose.x, pose.y, pose.yaw)
        centre = np.mean([target.anchor[:2] for target in drawn.targets], axis=0)
        side = gen._tour_side(robot, centre, drawn)
        for frame, degrees in zip(inputs.tour, gen._TOUR_TURNS_DEG):
            asked = T_obs if degrees == 0 else gen._tour_pose(T_obs, robot, centre, side * degrees, drawn, TH)
            assert np.array_equal(frame.T_world_camera, asked), (name, frame.index)
            xy, where = tuple(frame.T_world_camera[:2, 3]), f"{name} frame {frame.index}"
            assert point_in_polygon(xy, truth.bounds) and distance_to_boundary(xy, truth.bounds) >= 0.8 - 1e-6, where
            assert all(distance_to_polygon(xy, obstacle) >= 0.8 - 1e-6 for obstacle in truth.obstacles), where
            assert all(math.hypot(xy[0] - q.x, xy[1] - q.y) >= 1.05 - 1e-6 for q in truth.people), where
        assert len({tuple(np.round(frame.T_world_camera[:2, 3], 2)) for frame in inputs.tour}) == 8, name
        # The walk is the twin's, frame for frame, in the scenes with staff too. What differs from the twin on the
        # ground is the staff, who stand behind a counter, and what a staffed place and the second stall block, which
        # lies on the rectangle the twin has there: none of it comes nearer to a frame of the walk than the twin's.
        assert all(np.array_equal(frame.T_world_camera, other.T_world_camera)
                   for frame, other in zip(scene.tour, twin.tour, strict=True)), name


def test_the_walk_keeps_clear_of_everybody_in_the_scene_whatever_their_role(realistic):
    """The tour is walked on the scene's own ground truth: someone with a role, where a frame would stand, moves
    that frame as anybody would."""
    scene = read_scene_file(folder_of(realistic, PERSON))
    poses = real._tour_poses(scene, TH)
    assert all(np.array_equal(pose, frame.T_world_camera) for pose, frame in zip(poses, scene.tour, strict=True))
    for role in ("staff", None):
        x, y = poses[1][:2, 3]
        in_the_way = Person(id="s9", x=float(x), y=float(y), yaw=0.0, role=role)
        truth = scene.ground_truth.model_copy(update={"people": [*scene.ground_truth.people, in_the_way]})
        moved = real._tour_poses(scene.model_copy(update={"ground_truth": truth}), TH)
        assert math.hypot(*(moved[1][:2, 3] - poses[1][:2, 3])) >= 1.05 - 1e-6, role
        assert np.array_equal(moved[OBSERVATION], poses[OBSERVATION]), role     # the observation stays where it was
        assert sum(np.array_equal(pose, other) for pose, other in zip(moved, poses)) >= 5, role


def anchored(camera, depth, anchor):                # check 5 on a frame: the anchor in view, on the surface it shows
    uvz = project(camera, anchor)
    return bool(uvz is not None and in_image(camera, uvz[0], uvz[1])
                and abs(float(depth[int(uvz[1]), int(uvz[0])]) - uvz[2]) <= TH.anchor_depth_tolerance_m)


def on_surface(camera, depth, point):               # the frame shows this point of a surface: nothing stands before it
    uvz = project(camera, point)
    return bool(uvz is not None and in_image(camera, uvz[0], uvz[1])
                and abs(float(depth[int(uvz[1]), int(uvz[0])]) - uvz[2]) <= 0.05)


def reads(camera, depth, lettered):
    """Lettering is read where the frame shows all of its rectangle, at points no more than 0.1 m apart, from no
    further than 62 degrees off the square view of it."""
    centre, normal, across = (np.array(vector) for vector in (lettered.centre, lettered.normal, lettered.across))
    to_camera = camera.T_world_camera[:3, 3] - centre
    if to_camera @ normal < math.cos(math.radians(62.0)) * np.linalg.norm(to_camera):
        return False
    n, m = (max(1, math.ceil(side / 0.1)) for side in (lettered.width, lettered.height))
    return all(on_surface(camera, depth, centre + (i / n - 0.5) * lettered.width * across
                          + (j / m - 0.5) * lettered.height * np.array([0.0, 0.0, 1.0]))
               for i in range(n + 1) for j in range(m + 1))


def widest(camera):
    """The camera with its image grown by its own width to either side and by its own height above and below:
    as far as a picture of a target alone is ever grown."""
    return Camera(camera.fx, camera.fy, camera.cx + camera.width, camera.cy + camera.height, 3 * camera.width,
                  3 * camera.height, camera.T_world_camera)


def share_shown(depth, alone):
    """Of the pixels a target fills in a picture of it alone, taken with widest(), the share at which the frame
    with this depth shows it too."""
    height, width = depth.shape
    theirs = alone > 0
    inside = np.s_[height:2 * height, width:2 * width]
    return float((theirs[inside] & (np.abs(depth - alone[inside]) < 1e-3)).sum() / max(theirs.sum(), 1))


def test_tour_frames_follow_the_three_rules(realistic):
    """A frame lists a target when it shows its anchor, when its lettering reads, or when half of it shows, and
    not otherwise: each frame of a place, of the parcel and of someone among others is drawn again here from the
    stored description and judged by rules written out for this test."""
    library, stub, said = load_library(), StubRenderer(), Counter()
    for name in JUDGED:
        folder = folder_of(realistic, name)
        split, index, layout, _ = rows()[name]
        scene, job = read_scene_file(folder), job_of(folder)
        dressed = dress(layout, VARIANTS["real"], library, (0, SPLITS.index(split), index, job["attempt"]))
        spec, targets = from_json(job["spec"]), scene.ground_truth.targets
        for k, frame in enumerate(scene.tour):
            camera = camera_of(scene, frame.T_world_camera)
            shots = [Frame(camera, beauty=False), *(Frame(widest(camera), beauty=False, only=(n,))
                                                    for n in range(1, len(targets) + 1))]
            (_, depth), *alone = stub.render(spec, shots)
            for target, (_, theirs) in zip(targets, alone, strict=True):
                anchor = anchored(camera, depth, target.anchor)
                name_read = any(reads(camera, depth, lettered) for lettered in dressed.lettering[target.id])
                share, listed = share_shown(depth, theirs), k in target.tour_frames
                if anchor or name_read or abs(share - 0.5) > 0.02:      # a share at the limit may fall either way
                    assert listed == (anchor or name_read or share > 0.5), (name, k, anchor, name_read, share)
                said["listed" if listed else "not listed"] += 1
                said["by half of it alone"] += share > 0.52 and not anchor and not name_read
    assert said["listed"] + said["not listed"] == 8 * len(JUDGED)
    assert said["not listed"] >= 1, said            # somewhere a frame does not show the target
    assert said["by half of it alone"] >= 1, said   # and somewhere one shows it with its anchor hidden


def test_each_of_the_three_rules_lists_a_frame_by_itself():
    """A board 1.6 m wide and 2 m high stands for a place, with its anchor low on its front and its name
    high up. What stands before it decides which of the rules still holds."""
    board = Box(0.0, 0.0, 0.0, 0.1, 1.6, 2.0)
    target = SimpleNamespace(anchor=(0.05, 0.0, 0.6))
    name = [Lettered((0.05, 0.0, 1.8), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.8, 0.2)]
    camera = Camera(**CAMERA_LENS, T_world_camera=camera_on_robot(RobotPose(4.0, 0.0, math.pi), **CAMERA_MOUNT))
    alone, ground = (render(camera, things, antialias=False)[1] for things in ([board], []))
    alone = np.where(alone != ground, alone, np.float32(0.0))   # the board with nothing else there, as a renderer does
    wall = Box(1.0, 0.0, 0.0, 0.1, 3.0, 1.5)        # before all of the board but its top quarter
    slot = [Box(1.0, side * 1.7, 0.0, 0.1, 3.0, 2.2) for side in (-1, 1)]       # two that leave a slit 0.4 m wide
    post = Box(1.0, 0.0, 0.0, 0.1, 0.2, 2.2)        # before the anchor and the middle of the name
    wide_post = Box(1.0, 0.0, 0.0, 0.1, 0.9, 2.2)   # before the anchor, all of the name and most of the board

    def judged(before, lettering, of=target):       # whether the frame lists the board, its name read, its share shown
        depth = render(camera, [board, *before], antialias=False)[1]
        return (real._visible(camera, depth, of, lettering, alone, 0, 0, TH),
                [real._reads(camera, depth, lettered) for lettered in lettering], real._share_shown(depth, alone, 0, 0))

    assert judged([], name) == (True, [True], 1.0) and judged([], []) == (True, [], 1.0)
    listed, read, share = judged([wall], name)      # its name reads over the wall, which hides the anchor
    assert listed and read == [True] and share < 0.3
    assert judged([wall], [])[0] is False           # and without a name nothing lists it
    listed, _, share = judged(slot, [])             # its anchor shows through the slit
    assert listed and share < 0.4
    # An anchor inside the thing, as a person's is, shows by the tolerance of scene check 5 and no further in.
    assert judged(slot, [], SimpleNamespace(anchor=(-0.2, 0.0, 0.6)))[0] is True
    assert judged(slot, [], SimpleNamespace(anchor=(-0.3, 0.0, 0.6)))[0] is False
    listed, read, share = judged([post], name)      # neither shows, and most of it does
    assert listed and read == [False] and 0.5 < share < 0.95
    assert judged([wide_post], name) == (False, [False], pytest.approx(0.25, abs=0.1))      # neither, and too little


def test_lettering_reads_by_the_placeholder_rule(realistic):
    """In the placeholder's own picture of a place, the rectangles that a dressing in labels mode lists read in
    exactly the frames in which the placeholder reads the boxes they were made from, one for one."""
    library, seen = load_library(), Counter()
    for name in (LINE, EXAMPLE):
        split, index, layout, scene = rows()[name]
        (target,) = layout.scene_gt.targets
        own = gen._own(layout, layout.scene_gt.people, target, TH)
        boxes = [part for part in own if isinstance(part, Box) and part.label and part.label_face != "top"]
        lettering = dress(layout, VARIANTS["plain"], library, (0, SPLITS.index(split), index, 0)).lettering[target.id]
        assert len(lettering) == len(boxes) >= 3, name          # the three boards of its sign at the least
        pose = scene.observation.robot_pose_world
        T_obs, robot = np.array(scene.observation.T_world_camera), RobotPose(pose.x, pose.y, pose.yaw)
        centre = np.array(target.anchor[:2])
        for degrees in range(-165, 181, 15):        # all the way round the place, where a tour frame would stand
            T = gen._tour_pose(T_obs, robot, centre, degrees, layout.scene_gt, TH) if degrees else T_obs
            camera = camera_of(scene, T)
            depth = render(camera, own, layout.figures, antialias=False)[1]     # the place and the people before it
            for box, lettered in zip(boxes, lettering):
                read = gen._reads(camera, depth, box)
                assert real._reads(camera, depth, lettered) == read, (name, degrees, box.label, box.label_face)
                assert reads(camera, depth, lettered) == read, (name, degrees, box.label, box.label_face)
                seen[read] += 1
    assert seen[True] >= 10 and seen[False] >= 10, seen


def test_lettering_does_not_read_edge_on_from_behind_or_with_something_before_part_of_it():
    board = Box(0.0, 0.0, 0.0, 0.1, 1.0, 0.4, z0=1.0)           # its front, at x = 0.05, is 1 m wide and 0.4 m high
    lettered = Lettered((0.05, 0.0, 1.2), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.8, 0.3)
    plated = dataclasses.replace(lettered, centre=(0.05 - 0.003, 0.0, 1.2))     # a plate is 3 mm before its rectangle

    def view(robot, extra=()):
        camera = Camera(**CAMERA_LENS, T_world_camera=camera_on_robot(robot, **CAMERA_MOUNT))
        return camera, render(camera, [board, *extra], antialias=False)[1]

    def from_bearing(degrees, extra=()):            # from 4 m off, looking at the middle of the board
        turn = math.radians(degrees)
        return view(RobotPose(4.0 * math.cos(turn), 4.0 * math.sin(turn), turn + math.pi), extra)

    assert real._reads(*from_bearing(0.0), lettered) and real._reads(*from_bearing(0.0), plated)
    assert real._reads(*from_bearing(60.0), lettered)           # 62 degrees off the square view is the limit
    assert not real._reads(*from_bearing(64.0), lettered) and not real._reads(*from_bearing(-64.0), lettered)
    assert not real._reads(*from_bearing(180.0), lettered)      # from behind
    post = Box(0.5, 0.35, 0.0, 0.1, 0.1, 2.0)       # before the right end of the lettering, seen from the front
    assert not real._reads(*from_bearing(0.0, [post]), lettered)
    aside = Box(0.5, 0.43, 0.0, 0.06, 0.06, 2.0)    # before the end of the board, beside the lettering
    camera, depth = from_bearing(0.0, [aside])
    assert real._reads(camera, depth, lettered) and not gen._shows(camera, depth, (0.05, 0.48, 1.2), 0.05)
    camera, depth = view(RobotPose(4.0, 2.2, math.pi))          # the side of the image cuts the lettering in two
    assert gen._shows(camera, depth, (0.05, 0.3, 1.2), 0.05)
    assert not gen._shows(camera, depth, (0.05, -0.3, 1.2), 0.05) and not real._reads(camera, depth, lettered)
    # It is looked for at points 0.1 m apart: a row 0.6 m long at 0, 0.1, 0.2 and 0.3 m either side of its middle.
    row = Lettered((0.05, 0.0, 1.2), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.6, 0.1)
    at_a_point = Box(0.5, 0.176, 0.0, 0.036, 0.036, 2.0)        # a rod before the row 0.18 to 0.22 m from its middle
    between = Box(0.5, 0.132, 0.0, 0.036, 0.036, 2.0)           # and one before it 0.13 to 0.17 m from its middle
    assert real._reads(*from_bearing(0.0), row) and not real._reads(*from_bearing(0.0, [at_a_point]), row)
    assert real._reads(*from_bearing(0.0, [between]), row)


def test_the_share_shown_is_of_the_pixels_a_target_fills_alone_those_the_frame_shows_at_the_same_depth():
    depth = np.full((4, 6), 3.0, dtype=np.float32)              # the frame: 6 wide and 4 high
    alone = np.zeros((6, 10), dtype=np.float32)                 # the target alone: grown by 2 to the left and 1 above
    alone[0, 0] = alone[5, 9] = alone[2, 1] = 3.0               # three of its pixels are beside the frame's image
    alone[1:3, 2:5] = 3.0                                       # six are in it, at its first two rows
    depth[0, 2] = 2.0                                           # before one of them the frame shows something nearer
    depth[1, 1] = 3.0005                                        # and one it shows as good as where it is
    alone[2, 3] = 3.0005
    assert real._share_shown(depth, alone, 2, 1) == pytest.approx(5 / 9)
    assert real._share_shown(depth, np.zeros_like(alone), 2, 1) == 0.0          # a target of which nothing is drawn
    assert real._share_shown(depth, depth.copy(), 0, 0) == 1.0


def test_all_frames_of_a_scene_are_one_call_of_the_renderer(realistic):
    """Four calls for a scene. Three for the observation's depth of its dressing in each variant, its own first,
    which the scene is judged by. Then everything of its own in one: the observation's depth alone once more,
    before any picture; the observation and seven more tour frames with pictures; and each target alone in every
    tour frame, depth only, through a camera grown by whole pixels until it holds the target's blocks."""
    library, calls, judged, grown = load_library(), FIRST["calls"], iter(FIRST["judged"]), Counter()
    assert len(calls) == CALLS * len(SCENES) and len(FIRST["judged"]) == 8 * len(SCENES)
    for number, name in enumerate(IN_ORDER):        # in the order of the set, whatever order they were named in
        *judgements, (again, frames) = calls[CALLS * number: CALLS * (number + 1)]
        folder = folder_of(realistic, name)
        split, index, layout, _ = rows()[name]
        scene, job = read_scene_file(folder), job_of(folder)
        seed = (0, SPLITS.index(split), index, job["attempt"])
        dressed = dress(layout, VARIANTS["real"], library, seed)
        observation = camera_of(scene)
        for (spec, asked), variant in zip(judgements, ("real", "plain", "staffed"), strict=True):
            assert spec == dress(layout, VARIANTS[variant], library, seed).spec, (name, variant)
            assert len(asked) == 1 and not asked[0].beauty and asked[0].only is None, (name, variant)
            assert same_view(asked[0].camera, observation), (name, variant)
        assert again == dressed.spec, name
        targets = scene.ground_truth.targets
        before, pictures, alone = frames[0], frames[1:9], frames[9:]
        assert not before.beauty and before.only is None and same_view(before.camera, observation), name
        assert all(frame.beauty and frame.only is None for frame in pictures), name
        assert same_view(pictures[0].camera, observation), name
        others = [camera_of(scene, frame.T_world_camera) for k, frame in enumerate(scene.tour) if k != OBSERVATION]
        assert all(same_view(frame.camera, other) for frame, other in zip(pictures[1:], others, strict=True)), name
        assert len(alone) == 8 * len(targets), name
        views = [OBSERVATION, *(k for k in range(8) if k != OBSERVATION)]      # in the order of the pictures
        for (k, target), frame in zip(((k, target) for k in views for target in targets), alone):
            camera = camera_of(scene, scene.tour[k].T_world_camera)
            wide, left, above = frame.camera, frame.camera.cx - camera.cx, frame.camera.cy - camera.cy
            assert not frame.beauty and frame.only == dressed.target_groups[target.id], (name, k)
            assert np.array_equal(wide.T_world_camera, camera.T_world_camera), (name, k)
            assert (wide.fx, wide.fy) == (camera.fx, camera.fy) and left == int(left) and above == int(above), (name, k)
            assert 0 <= left <= camera.width and 0 <= above <= camera.height, (name, k)
            right, below = wide.width - camera.width - left, wide.height - camera.height - above
            assert 0 <= right <= camera.width and 0 <= below <= camera.height, (name, k)
            blocks = dressed.target_blocks[target.id]
            assert blocks, name
            for block in blocks:                    # each corner of a block is in the picture, or the picture
                u, v, z = gen._corners(wide, *block)            # is grown that way as far as one ever is
                if (z <= 0).any():
                    assert (left, above, right, below) == (camera.width, camera.height) * 2, (name, k)
                    continue
                u, v = wide.fx * u / z + wide.cx, wide.fy * v / z + wide.cy
                assert u.min() >= -1e-6 or left == camera.width, (name, k)
                assert v.min() >= -1e-6 or above == camera.height, (name, k)
                assert u.max() <= wide.width + 1e-6 or right == camera.width, (name, k)
                assert v.max() <= wide.height + 1e-6 or below == camera.height, (name, k)
            # And it is grown no further than that: as the placeholder grows a picture for boxes of those sizes.
            as_boxes = [Box(x, y, yaw, 2 * half_x, 2 * half_y, top) for x, y, yaw, half_x, half_y, top in blocks]
            assert same_view(wide, gen._widened(camera, as_boxes)[0]), (name, k)
            grown[(left, above, right, below) != (0, 0, 0, 0)] += 1
            # The frame is judged through its own camera, with the target's lettering and the target alone as it
            # lies in the grown picture; and the target lists the frame when the three rules say so.
            asked = next(judged)
            assert same_view(asked.camera, camera) and asked.target.id == target.id, (name, k)
            assert asked.depth.shape == (camera.height, camera.width), (name, k)
            assert tuple(asked.lettering) == dressed.lettering[target.id], (name, k)
            assert (target.kind == "place") == bool(asked.lettering), name
            assert asked.alone.shape == (wide.height, wide.width), (name, k)
            assert (asked.left, asked.above) == (left, above), (name, k)
            assert asked.listed == (k in target.tour_frames), (name, k)
    assert grown[True] >= 8 and grown[False] >= 8, grown        # some targets are cut off by a frame, some are not


def test_two_runs_write_the_same_scene_json_and_depth(realistic, tmp_path, capsys, caplog):
    """The second run is the command's, with one id named as the set has it and one as the placeholder set does."""
    caplog.set_level("INFO", logger="sgi.synth.realistic.generate")
    assert real.main(["--out", str(tmp_path), "--renderer", "stub", "--only", f"real_{LINE}", PERSON]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert printed == [f"{tmp_path}: 2 scenes rendered, 0 skipped as unchanged, 0 dressed again"]
    said = [record.getMessage() for record in caplog.records if record.name == "sgi.synth.realistic.generate"]
    assert [line.split(" in ")[0] for line in said] == [f"real_{PERSON}: rendered", f"real_{LINE}: rendered"]
    assert entries(tmp_path) == set_of([LINE, PERSON])
    for name in (LINE, PERSON):
        first, second = files(folder_of(realistic, name)), files(folder_of(tmp_path, name))
        assert sorted(second) == WRITTEN and second["scene.json"] == first["scene.json"], name
        assert second["depth.npy"] == first["depth.npy"], name
        assert second["render_job.json"] == first["render_job.json"], name
        assert second == first, name                # with the stub, which draws the same picture every time, all of it


# --- Part of a set, and a set that is there already. ---

def test_only_rewrites_the_named_scenes(realistic, tmp_path):
    out = Path(shutil.copytree(realistic, tmp_path / "set"))
    (out / "eval" / "notes.txt").write_text("kept")             # a run leaves alone what it does not render
    before = {name: state(folder_of(out, name)) for name in SCENES}
    asked = len(REPLAY.calls)
    assert run(out, only=[f"real_{PERSON}"], force=True) == {**NONE, "rendered": 1}
    assert len(REPLAY.calls) == asked + 1 and REPLAY.calls[-1].ended and REPLAY.calls[-1].handed_out == WHOLE_SET
    for name in SCENES:
        after = state(folder_of(out, name))
        if name == PERSON:                          # a new folder in the old one's place, with the same in it
            assert after[0][1] != before[name][0][1], name
            assert files(folder_of(out, name)) == files(folder_of(realistic, name)), name
        else:
            assert after == before[name], name
    assert entries(out) == sorted([*set_of(SCENES), "eval/notes.txt"])
    # An id that names no scene of the set stops the run before anything is drawn, whatever else is named.
    renderer = Counting()
    for unknown in ("eval_99_nowhere", f"plain_{PERSON}", "real_real_" + PERSON):
        with pytest.raises(ValueError, match=unknown):
            run(out, only=[PARCEL, unknown], renderer=renderer, force=True)
    assert renderer.calls == [] and state(folder_of(out, PARCEL)) == before[PARCEL]


def test_a_run_without_only_renders_the_whole_set(tmp_path, monkeypatch):
    made = []
    monkeypatch.setattr(real, "_scene", lambda split_dir, twin, *args, **kwargs: made.append(
        (split_dir, twin.scene_id)) or ("rendered", 0))
    counts = real.generate(tmp_path, CFG, renderer=StubRenderer())
    assert counts == {**NONE, "rendered": WHOLE_SET}
    assert made == [(tmp_path / split, scene.scene_id) for split, _, _, scene in REPLAY.kept[0]]


def test_an_unchanged_job_is_skipped_unless_forced(realistic, tmp_path):
    out, renderer = Path(shutil.copytree(realistic, tmp_path / "set")), Counting()
    person, parcel = folder_of(out, PERSON), folder_of(out, PARCEL)
    before = [state(person), state(parcel)]
    assert run(out, only=[PERSON, PARCEL], renderer=renderer) == {**NONE, "skipped": 2}
    assert run(out, only=PERSON, renderer=renderer) == {**NONE, "skipped": 1}     # one id may be given as it is
    assert renderer.calls == [] and [state(person), state(parcel)] == before      # nothing drawn, nothing touched
    assert run(out, only=[PERSON], renderer=renderer, force=True) == {**NONE, "rendered": 1}
    assert len(renderer.calls) == CALLS and state(person)[0][1] != before[0][0][1] and state(parcel) == before[1]
    # A stored job that is not the one about to be run is not skipped: whichever of its entries says otherwise,
    # the scene is begun again. (That is as far as this renderer lets it get, and what was there stays.)
    job, kept = job_of(parcel), state(parcel)
    other = {"spec": {**job["spec"], "seed": 1 + job["spec"]["seed"]}, "renderer": {"name": "stub", "version": "0.9"},
             "thresholds": {**job["thresholds"], "person_radius_m": 0.4}, "judged": {**job["judged"], "real": "0" * 64},
             "seed": 1, "variant": "plain", "attempt": 1}
    assert sorted(other) == JOB_KEYS
    for key, value in other.items():
        (parcel / "render_job.json").write_text(json.dumps({**job, key: value}), encoding="utf-8")
        with pytest.raises(RuntimeError, match="out of memory"):
            run(out, only=[PARCEL], renderer=Failing(1))
    for without in ("thresholds", "judged"):        # as is a job of before these two were written
        older = {key: value for key, value in job.items() if key != without}
        (parcel / "render_job.json").write_text(json.dumps(older), encoding="utf-8")
        with pytest.raises(RuntimeError, match="out of memory"):
            run(out, only=[PARCEL], renderer=Failing(1))
    (parcel / "render_job.json").write_text("[]", encoding="utf-8")             # or one that is no job at all
    with pytest.raises(RuntimeError, match="out of memory"):
        run(out, only=[PARCEL], renderer=Failing(1))
    assert [each[3] for each in state(parcel) if each[0] != "render_job.json"] == [
        each[3] for each in kept if each[0] != "render_job.json"]
    # So is a folder without a job, and one that does not load as a scene. Rendered, each is as it was.
    (person / "render_job.json").unlink()
    (parcel / "render_job.json").write_text(json.dumps(job), encoding="utf-8")
    (parcel / "tour" / "005.png").unlink()
    assert run(out, only=[PERSON, PARCEL], renderer=renderer) == {**NONE, "rendered": 2}
    assert files(person) == files(folder_of(realistic, PERSON)) and files(parcel) == files(folder_of(realistic, PARCEL))
    assert run(out, only=[PERSON, PARCEL], renderer=renderer) == {**NONE, "skipped": 2}
    assert len(renderer.calls) == 3 * CALLS and entries(out) == set_of(SCENES)


def test_a_scene_made_from_something_else_is_not_skipped_though_its_description_is_the_same(realistic, tmp_path,
                                                                                            monkeypatch, caplog):
    """The description says what stands in the room, not what is asked there, from where it is seen or how it
    is judged. A request put in other words, a robot that stands elsewhere and other thresholds each leave the
    description as it was, and the scene is drawn again all the same: its scene.json or its pictures would be
    stale, and its rows would no longer pair with its twin's."""
    caplog.set_level("INFO", logger="sgi.synth.realistic.generate")
    out = Path(shutil.copytree(realistic, tmp_path / "set"))
    folder = folder_of(out, PERSON)
    before = state(folder)

    def reworded(layout, twin):
        return dataclasses.replace(layout, episodes=tuple(episode.model_copy(
            update={"request": episode.request + ", please"}) for episode in layout.episodes)), twin

    def moved(layout, twin):                        # the robot and its camera half a metre along x
        T, pose = np.array(twin.observation.T_world_camera), twin.observation.robot_pose_world
        T[0, 3] += 0.5
        observation = twin.observation.model_copy(update={
            "T_world_camera": T.tolist(), "robot_pose_world": pose.model_copy(update={"x": pose.x + 0.5})})
        return layout, twin.model_copy(update={"observation": observation})

    def layouts(change):                            # the layouts of the seed, whatever the config, with one changed
        def changed(work_dir, cfg, seed=0):
            for split, index, layout, twin in REPLAY.kept[seed]:
                yield (split, index, *(change(layout, twin) if twin.scene_id == PERSON else (layout, twin)))
        return changed

    monkeypatch.setattr(real, "accepted_layouts", layouts(lambda layout, twin: (layout, twin)))
    assert run(out, only=[PERSON], renderer=Failing(1)) == {**NONE, "skipped": 1}       # as it was, it is skipped
    other = CFG.model_copy(update={"thresholds": TH.model_copy(update={"person_radius_m": 0.4})})
    with pytest.raises(RuntimeError, match="out of memory"):        # begun again, which is as far as this one gets
        real.generate(out, other, 0, "real", Failing(1), [PERSON])
    for change in (moved, reworded):
        monkeypatch.setattr(real, "accepted_layouts", layouts(change))
        with pytest.raises(RuntimeError, match="out of memory"):
            run(out, only=[PERSON], renderer=Failing(1))
    assert state(folder) == before
    # Rendered, the scene asks what its twin now asks. Its description is as before, and so, with the stub, are
    # its pictures and its depth; its job says that all three variants were judged from something else.
    assert run(out, only=[PERSON]) == {**NONE, "rendered": 1}
    said = [record.getMessage() for record in caplog.records if record.name == "sgi.synth.realistic.generate"]
    assert said[-2] == (f"real_{PERSON}: drawn again, its stored job differs in judged.real, judged.plain, "
                        "judged.staffed") and said[-1].startswith(f"real_{PERSON}: rendered in ")
    old, new = read_scene_file(folder_of(realistic, PERSON)), read_scene_file(folder)
    assert [episode.request for episode in new.episodes] == [episode.request + ", please" for episode in old.episodes]
    was, now = files(folder_of(realistic, PERSON)), files(folder)
    assert set(was) == set(now)
    assert {name for name in was if was[name] != now[name]} == {"scene.json", "render_job.json"}
    stored, job = job_of(folder_of(realistic, PERSON)), job_of(folder)
    assert {key for key in job if job[key] != stored[key]} == {"judged"}
    assert all(job["judged"][variant] != stored["judged"][variant] for variant in VARIANTS)
    assert run(out, only=[PERSON], renderer=Failing(1)) == {**NONE, "skipped": 1}


# --- When something fails. ---

def bare(dressed):
    """The dressing with nothing standing in its room: no depth at any anchor, and nobody where the people are."""
    return dataclasses.replace(dressed, spec=dataclasses.replace(dressed.spec, parts=(), models=(), people=()))


def test_a_scene_that_fails_the_check_is_dressed_again_and_never_redrawn(tmp_path, monkeypatch):
    seeds, layouts, asked, renderer = [], [], len(REPLAY.calls), Counting()

    def at_first_bare(layout, variant, library, seed):
        seeds.append((variant.name, tuple(seed)))
        layouts.append(layout)
        dressed = dress(layout, variant, library, seed)
        return dressed if seed[-1] >= 2 else bare(dressed)

    monkeypatch.setattr(real, "dress", at_first_bare)
    assert run(tmp_path, only=[PERSON], renderer=renderer) == {**NONE, "rendered": 1, "dressed_again": 1}
    split, index, layout, _ = rows()[PERSON]
    assert (split, index) == ("eval", 5)
    # Each attempt dresses the scene in every variant, its own first: with the seed, the split, the row, the attempt.
    assert seeds == [(variant, (0, 0, 5, attempt)) for attempt in range(3) for variant in ("real", "plain", "staffed")]
    assert all(each is layout for each in layouts) and len(REPLAY.calls) == asked + 1      # the layouts are drawn once
    assert len(renderer.calls) == 2 + CALLS         # a dressing that fails in the variant made is judged no further
    folder = folder_of(tmp_path, PERSON)
    job, kept = job_of(folder), dress(layout, VARIANTS["real"], load_library(), (0, 0, 5, 2))
    assert job["attempt"] == 2 and job["spec"] == to_json(kept.spec)
    results = check_scene(folder, CFG, write_png=False)
    assert not has_errors(results) and warnings(results) == set()
    assert entries(tmp_path) == set_of([PERSON])                # nothing of the dressings that failed is left
    before = state(folder)

    # No dressing passes: the run stops, says which scene, how often and why, and leaves what was there.
    monkeypatch.setattr(real, "dress", lambda *args: seeds.append((args[1].name, tuple(args[3]))) or bare(dress(*args)))
    with pytest.raises(RuntimeError) as error:
        run(tmp_path, only=[PERSON, PARCEL], force=True)
    assert len(seeds) == 9 + 3 * ATTEMPTS           # it stops at the first such scene
    assert seeds[9::3] == [("real", (0, 0, 4, attempt)) for attempt in range(ATTEMPTS)]
    message = str(error.value)
    assert f"real_{PARCEL}" in message and f"{ATTEMPTS} " in message and "check_4=5" in message, message
    assert entries(tmp_path) == set_of([PERSON]) and state(folder) == before
    # A scene whose stored job is the one of its third dressing is found unchanged once the first two have failed.
    monkeypatch.setattr(real, "dress", at_first_bare)
    renderer = Counting()
    assert run(tmp_path, only=[PERSON], renderer=renderer) == {**NONE, "skipped": 1, "dressed_again": 1}
    assert len(renderer.calls) == 2 and state(folder) == before


def in_plain(change):
    """dress, with the plain variant's dressing of the first attempt changed and every other one as it is."""
    def dressing(layout, variant, library, seed):
        dressed = dress(layout, variant, library, seed)
        return change(dressed) if (variant.name, seed[-1]) == ("plain", 0) else dressed

    return dressing


def test_a_dressing_is_kept_only_when_it_passes_the_check_in_every_variant(tmp_path, monkeypatch):
    """An attempt seeds the cast, and the variants of a scene show the same people. So a dressing that fails the
    check in any variant is given up in all of them: here the plain one fails at first, and the default set,
    whose own scene passes, goes on to the next attempt, where the plain set will be too."""
    monkeypatch.setattr(real, "dress", in_plain(bare))
    monkeypatch.setattr(real, "_MAX_DRESSINGS", 1)
    renderer = Counting()
    with pytest.raises(RuntimeError, match=f"real_{PERSON}: no dressing passed the scene check in 1 attempts; "
                                           r"rejected: check_\d+ as plain=1$"):
        run(tmp_path, only=[PERSON], renderer=renderer)
    assert len(renderer.calls) == 2 and entries(tmp_path) == []     # its own depth, which passes, and the plain one's
    monkeypatch.setattr(real, "_MAX_DRESSINGS", ATTEMPTS)
    renderer = Counting()
    assert run(tmp_path, only=[PERSON], renderer=renderer) == {**NONE, "rendered": 1, "dressed_again": 1}
    assert len(renderer.calls) == 2 + CALLS
    library, (_, index, layout, _) = load_library(), rows()[PERSON]
    first, second = (dress(layout, VARIANTS["real"], library, (0, 0, index, attempt)) for attempt in (0, 1))
    job = job_of(folder_of(tmp_path, PERSON))
    assert job["attempt"] == 1 and job["spec"] == to_json(second.spec)
    assert first.spec.people != second.spec.people  # another attempt is another cast: why the variants keep one attempt
    results = check_scene(folder_of(tmp_path, PERSON), CFG, write_png=False)
    assert not has_errors(results) and warnings(results) == set()


def test_when_only_another_variant_is_dressed_otherwise_a_scene_is_judged_again_and_not_drawn_again(
        realistic, tmp_path, monkeypatch):
    """A scene's job says what each variant was judged from, since all three have to pass for an attempt to be
    kept. When only another variant's dressing has changed, that one is judged again. If it passes, the scene
    stands as it was drawn, pixel for pixel, and its job is brought up to date. If it does not, the scene goes
    on to the next attempt, as its siblings will."""
    out = Path(shutil.copytree(realistic, tmp_path / "set"))
    folder, library, (_, index, layout, _) = folder_of(out, PERSON), load_library(), rows()[PERSON]
    before, job, inode = files(folder), job_of(folder), folder_of(out, PERSON).stat().st_ino

    def refloored(dressed):                         # another floor: the same depth, another description
        floor = next(name for name in ("floor_tile", "floor_laminate") if name != dressed.spec.room.floor)
        room = dataclasses.replace(dressed.spec.room, floor=floor)
        return dataclasses.replace(dressed, spec=dataclasses.replace(dressed.spec, room=room))

    monkeypatch.setattr(real, "dress", in_plain(refloored))
    renderer = Counting()
    assert run(out, only=[PERSON], renderer=renderer) == {**NONE, "skipped": 1}
    ((spec, frames),) = renderer.calls              # one call: the depth of the plain dressing, alone
    assert spec == refloored(dress(layout, VARIANTS["plain"], library, (0, 0, index, 0))).spec
    assert len(frames) == 1 and not frames[0].beauty and frames[0].only is None
    after, brought = files(folder), job_of(folder)
    assert folder.stat().st_ino == inode and set(after) == set(before) and entries(out) == set_of(SCENES)
    assert {name for name in before if before[name] != after[name]} == {"render_job.json"}
    assert brought["judged"]["plain"] != job["judged"]["plain"]
    assert {**brought, "judged": {**brought["judged"], "plain": job["judged"]["plain"]}} == job
    assert run(out, only=[PERSON], renderer=renderer) == {**NONE, "skipped": 1} and len(renderer.calls) == 1
    # The plain dressing no longer passes: this attempt is given up here too, and the next one is drawn.
    monkeypatch.setattr(real, "dress", in_plain(bare))
    renderer = Counting()
    assert run(out, only=[PERSON], renderer=renderer) == {**NONE, "rendered": 1, "dressed_again": 1}
    assert len(renderer.calls) == 1 + CALLS and [len(frames) for _, frames in renderer.calls[:4]] == [1, 1, 1, 1]
    assert job_of(folder)["attempt"] == 1 and folder.stat().st_ino != inode and entries(out) == set_of(SCENES)


def test_a_failed_render_leaves_the_old_folder_in_place(realistic, tmp_path, monkeypatch):
    out = Path(shutil.copytree(realistic, tmp_path / "set"))
    folder = folder_of(out, PERSON)
    before, whole = state(folder), entries(out)
    for at in (1, CALLS):                           # the depth the scene is judged by, and then all its frames
        renderer = Failing(at)
        with pytest.raises(RuntimeError, match="out of memory"):
            run(out, only=[PERSON], renderer=renderer, force=True)
        assert len(renderer.calls) == at and state(folder) == before and entries(out) == whole

    # Nor does a scene that cannot be written, or cannot be put where the old one is, cost the old one.
    saved, save = [], real.save_scene

    def write(scene_dir, *args, **kwargs):          # the scenes to be judged are written; the finished one is not
        saved.append(scene_dir)
        if len(saved) == CALLS:
            raise OSError("no space left")
        return save(scene_dir, *args, **kwargs)

    monkeypatch.setattr(real, "save_scene", write)
    with pytest.raises(OSError, match="no space left"):
        run(out, only=[PERSON], force=True)
    assert len(saved) == CALLS and state(folder) == before and entries(out) == whole
    assert [Path(scene_dir).name for scene_dir in saved] == [f"{variant}_{PERSON}" for variant in (*VARIANTS, "real")]
    assert saved[0] == saved[-1] and all(folder.parent in Path(scene_dir).parents for scene_dir in saved)
    monkeypatch.setattr(real, "save_scene", save)
    moved, replace = [], os.replace

    def put(source, target):                        # the old folder is moved aside; the new one does not go in
        moved.append((Path(source), Path(target)))
        if [to for _, to in moved] == [moved[0][1], folder]:
            raise OSError("the new folder could not be put in place")
        return replace(source, target)

    monkeypatch.setattr(real.os, "replace", put)
    with pytest.raises(OSError, match="could not be put in place"):
        run(out, only=[PERSON], force=True)
    assert moved[0][0] == folder and moved[1][1] == folder and moved[2] == (moved[0][1], folder)   # aside, then back
    assert [each[1:] for each in state(folder)[1:]] == [each[1:] for each in before[1:]]    # the same files, no copies
    assert state(folder)[0][1] == before[0][1] and entries(out) == whole
    monkeypatch.setattr(real.os, "replace", replace)

    # A first render that fails, here at the depth of another variant's dressing, leaves no folder at all.
    with pytest.raises(RuntimeError, match="out of memory"):
        run(tmp_path / "new", only=[PERSON], renderer=Failing(2))
    assert entries(tmp_path / "new") == []

    # Should nothing go into the scene's place any more, not even the old folder, the old folder is not thrown away
    # with what was made: its pictures cannot be drawn again pixel for pixel. It stays where it was moved to.
    def shut(source, target):
        if Path(target) == folder:
            raise OSError("nothing can be put there")
        return replace(source, target)

    monkeypatch.setattr(real.os, "replace", shut)
    with pytest.raises(OSError, match="nothing can be put there"):
        run(out, only=[PERSON], force=True)
    (left,) = [entry for entry in entries(out) if entry not in whole]
    assert left.startswith(f"eval/.real_{PERSON}.") and not folder.exists()
    assert files(out / left / "replaced") == {name: data for name, _, _, data in before if data is not None}


class Flickering(Counting):
    """The stub, but that in a call `pixels` of every whole view, spread over the image, show something 2 m
    nearer from the call's first picture on: as Blender moves a few pixels of people's hair once a process has
    drawn a picture, and none before."""

    def __init__(self, pixels):
        super().__init__()
        self.pixels = pixels

    def render(self, spec, frames):
        pictured, moved = False, []
        for frame, (rgb, depth) in zip(frames, super().render(spec, frames)):
            pictured = pictured or frame.beauty
            if pictured and frame.only is None:
                depth, spread = depth.copy(), np.arange(self.pixels) * (depth.size // self.pixels)
                depth.flat[spread] = depth.flat[spread] - np.float32(2.0)
            moved.append((rgb, depth))
        return moved


class Forgetful(Counting):
    """The stub, whose calls for more than one frame draw another scene than the one asked for: without the
    model number `left_out`, or with the depth of `off` pixels of the call's first image a centimetre out."""

    def __init__(self, left_out=None, off=0):
        super().__init__()
        self.left_out, self.off = left_out, off

    def render(self, spec, frames):
        if len(frames) > 1 and self.left_out is not None:
            spec = dataclasses.replace(spec, models=spec.models[:self.left_out] + spec.models[self.left_out + 1:])
        drawn = super().render(spec, frames)
        if len(frames) > 1 and self.off:
            depth = drawn[0][1].copy()
            depth.flat[:self.off] += np.float32(0.01)
            drawn[0] = (drawn[0][0], depth)
        return drawn


def test_a_scene_keeps_the_depth_it_was_judged_with(realistic, tmp_path, monkeypatch):
    """The depth that comes with the observation's picture may differ from the depth drawn alone at single pixels:
    here at 28, the most that Blender was seen to move in a scene. The scene is written with the depth drawn
    alone, which the check read, and its tour reads the observation with that depth too."""
    first, judged = files(folder_of(realistic, PERSON)), []
    monkeypatch.setattr(real, "_visible", watching(judged))
    assert run(tmp_path, only=[PERSON], renderer=Flickering(28)) == {**NONE, "rendered": 1}
    written = files(folder_of(tmp_path, PERSON))
    assert written["depth.npy"] == first["depth.npy"] and written["rgb.png"] == first["rgb.png"]
    assert written["scene.json"] == first["scene.json"]
    depth = np.load(folder_of(tmp_path, PERSON) / "depth.npy")
    assert len(judged) == 8 and np.array_equal(judged[0].depth, depth)          # the observation is judged first


def test_the_call_for_the_pictures_must_draw_the_depth_the_scene_was_judged_with(realistic):
    """The call that draws a scene's pictures draws the observation's depth alone first, and that has to be the
    depth the scene was judged with, pixel for pixel: one pixel off, and the call drew another scene. The depth
    that comes with the picture is held to one pixel in a thousand. Here on someone alone, seen through an
    image a quarter the size each way, which is quick: 19,200 pixels."""
    _, index, layout, _ = rows()[PERSON]
    dressed, scene = dress(layout, VARIANTS["real"], load_library(), (0, 0, index, 0)), read_scene_file(
        folder_of(realistic, PERSON))
    obs = scene.observation
    lens = obs.intrinsics.model_copy(update={name: value / 4 for name, value in obs.intrinsics.model_dump().items()})
    scene = scene.model_copy(update={"observation": obs.model_copy(
        update={"width": obs.width // 4, "height": obs.height // 4, "intrinsics": lens})})
    ((_, depth),) = StubRenderer().render(dressed.spec, [Frame(camera_of(scene), beauty=False)])
    renderer = Flickering(19)
    toured, rgb, pictures = real._toured(scene, dressed, renderer, depth, TH)
    ((_, frames),) = renderer.calls                 # the depth alone comes first, before any picture
    assert not frames[0].beauty and frames[0].only is None and same_view(frames[0].camera, camera_of(scene))
    assert all(frame.beauty for frame in frames[1:9]) and not any(frame.beauty for frame in frames[9:])
    assert rgb.shape == (120, 160, 3) and len(pictures) == len(toured.tour) == 8
    with pytest.raises(RuntimeError, match="depth that comes with the observation's picture"):
        real._toured(scene, dressed, Flickering(20), depth, TH)
    with pytest.raises(RuntimeError, match="another depth image of the observation .* at 1 pixels"):
        real._toured(scene, dressed, Forgetful(off=1), depth, TH)
    with pytest.raises(RuntimeError, match="another depth image of the observation .* at 9600 pixels"):
        real._toured(scene, dressed, StubRenderer(), depth[:60], TH)            # nor one of another size: all of it


def test_a_second_call_that_leaves_a_small_thing_out_is_refused(realistic, tmp_path):
    """The monitor on the desk fills a few hundred pixels of the observation, fewer than one in a thousand. A
    renderer that draws the scene without it in its second call has not drawn the scene that was judged: its
    pictures and its tour would show another scene than the depth and the description say."""
    folder = folder_of(realistic, LINE)
    scene, spec, stub = read_scene_file(folder), from_json(job_of(folder)["spec"]), StubRenderer()
    whole, shots = np.load(folder / "depth.npy"), [Frame(camera_of(scene), beauty=False)]

    def shown(number):                              # the pixels of the observation that show a model
        without = dataclasses.replace(spec, models=spec.models[:number] + spec.models[number + 1:])
        return int((stub.render(without, shots)[0][1] != whole).sum())

    small = next(number for number in range(len(spec.models)) if 0 < shown(number) <= whole.size // 1000)
    renderer = Forgetful(left_out=small)
    with pytest.raises(RuntimeError, match=f"real_{LINE}: .* another depth image of the observation"):
        run(tmp_path, only=[LINE], renderer=renderer)
    assert len(renderer.calls) == CALLS and entries(tmp_path) == []


def test_generate_refuses_to_start_without_prepared_assets(tmp_path, monkeypatch):
    drawn, made = [], []
    monkeypatch.setattr(blender, "available", lambda: False)
    monkeypatch.setattr(real, "accepted_layouts", lambda *args, **kwargs: drawn.append(args) or iter(()))
    monkeypatch.setattr(real, "BlenderRenderer", lambda *args, **kwargs: made.append(args) or Counting(name="blender"))
    for renderer in (None, Counting(name="blender")):           # Blender by default, and Blender handed in
        with pytest.raises(RuntimeError) as error:
            real.generate(tmp_path / "out", CFG, renderer=renderer)
        assert blender.FETCH_COMMAND in str(error.value) and PREPARE_COMMAND in str(error.value)
    # It says which of the two is missing where that can be told, and names both commands either way.
    monkeypatch.setattr(blender, "binary_available", lambda: False)
    monkeypatch.setattr(real, "is_prepared", lambda library: True)
    with pytest.raises(RuntimeError, match=r"here: no Blender of the 4\.5 series answers\. Fetch Blender with"):
        real.generate(tmp_path / "out", CFG)
    monkeypatch.setattr(blender, "binary_available", lambda: True)
    monkeypatch.setattr(real, "is_prepared", lambda library: False)
    with pytest.raises(RuntimeError, match=r"here: the assets are not prepared\. Fetch Blender with") as error:
        real.generate(tmp_path / "out", CFG)
    assert blender.FETCH_COMMAND in str(error.value) and PREPARE_COMMAND in str(error.value)
    assert drawn == [] and made == [] and not (tmp_path / "out").exists()       # not a layout drawn, nothing made
    assert real.generate(tmp_path / "out", CFG, renderer=StubRenderer()) == NONE        # the stub needs neither
    assert len(drawn) == 1 and made == []


def test_without_a_renderer_the_scenes_are_drawn_by_blender(tmp_path, monkeypatch):
    made = []
    monkeypatch.setattr(blender, "available", lambda: True)
    monkeypatch.setattr(real, "BlenderRenderer", lambda: made.append(Counting("4.5.99", "blender")) or made[-1])
    assert real.generate(tmp_path, CFG, only=[PERSON]) == {**NONE, "rendered": 1}
    scene = read_scene_file(folder_of(tmp_path, PERSON))
    assert len(made) == 1 and len(made[0].calls) == CALLS
    assert (scene.source.simulator, scene.source.version) == ("sgi-synth-blender/real", "4.5.99")
    assert job_of(folder_of(tmp_path, PERSON))["renderer"] == {"name": "blender", "version": "4.5.99"}


def test_an_unknown_variant_is_refused_before_a_layout_is_drawn(tmp_path):
    asked = len(REPLAY.calls)
    with pytest.raises(ValueError, match="fancy.*real, plain, staffed"):
        real.generate(tmp_path, CFG, variant="fancy", renderer=StubRenderer())
    assert len(REPLAY.calls) == asked and list(tmp_path.iterdir()) == []


def test_a_scene_of_a_split_without_a_number_is_refused_before_it_is_drawn(realistic, tmp_path, monkeypatch):
    """A dressing's seed names the split by a number. Should the placeholder generator gain a split, it gets one
    here first: the scenes of the two there are must keep their seeds."""
    _, index, layout, scene = rows()[PERSON]
    monkeypatch.setattr(real, "accepted_layouts", lambda *asked: iter([("held_out", index, layout, scene)]))
    renderer = Counting()
    with pytest.raises(ValueError, match="held_out.*eval, examples"):
        real.generate(tmp_path, CFG, renderer=renderer)
    assert renderer.calls == [] and list(tmp_path.iterdir()) == []


# --- The variants and the command. ---

def test_the_three_variants_share_people_and_differ_in_staff(realistic, tmp_path):
    folders, library = {"real": folder_of(realistic, LINE)}, load_library()
    split, index, layout, _ = rows()[LINE]
    for variant in ("plain", "staffed"):
        renderer = Counting()
        assert run(tmp_path / variant, variant, only=[LINE], renderer=renderer) == {**NONE, "rendered": 1}
        folders[variant] = folder_of(tmp_path / variant, LINE, variant)
        assert entries(tmp_path / variant) == set_of([LINE], variant)
        # Whichever set is made, the scene is judged in all three variants, the one that is made first.
        judged = [spec for spec, _ in renderer.calls[:CALLS - 1]]
        assert judged == [dress(layout, VARIANTS[each], library, (0, SPLITS.index(split), index, 0)).spec
                          for each in (variant, *(other for other in VARIANTS if other != variant))], variant
        assert len(renderer.calls) == CALLS and renderer.calls[-1][0] == judged[0], variant
    scenes = {variant: read_scene_file(folder) for variant, folder in folders.items()}
    specs = {variant: from_json(job_of(folder)["spec"]) for variant, folder in folders.items()}
    for variant, scene in scenes.items():
        assert scene.scene_id == f"{variant}_{LINE}" and scene.source.simulator == f"sgi-synth-stub/{variant}"
        assert job_of(folders[variant])["variant"] == variant
        assert not has_errors(check_scene(folders[variant], CFG, write_png=False)), variant
    # The three were judged together: their jobs name the same seed, the same attempt and the same three dressings,
    # which is how three sets are known to belong together, scene by scene.
    jobs = {variant: job_of(folder) for variant, folder in folders.items()}
    assert jobs["real"]["judged"] == jobs["plain"]["judged"] == jobs["staffed"]["judged"]
    assert list(jobs["real"]["judged"]) == list(VARIANTS)
    assert {(job["seed"], job["attempt"]) for job in jobs.values()} == {(0, 0)}
    people = {variant: scene.ground_truth.people for variant, scene in scenes.items()}
    cast = {variant: {person.person_id: person for person in spec.people} for variant, spec in specs.items()}
    # The people of the twin are in all three, cast and posed alike.
    assert people["plain"] == [person for person in people["real"] if person.role is None] and len(people["plain"]) == 2
    assert cast["plain"] == {name: person for name, person in cast["real"].items() if name not in ("s0", "s1")}
    assert list(cast["plain"]) == [person.id for person in people["plain"]]
    # Staff are in the two that have them, the same people in the same spots, and in the picture as in the ground
    # truth: whoever serves at the desk, and the vendor of the second stall.
    assert people["staffed"] == people["real"] == [*people["plain"], *people["real"][-2:]]
    assert [(person.id, person.role, person.group) for person in people["real"][-2:]] == [
        ("s0", "staff", None), ("s1", "staff", None)]
    assert cast["staffed"] == cast["real"] and {"s0", "s1"} <= set(cast["real"])
    assert cast["real"]["s0"].avatar != cast["real"]["s1"].avatar
    assert not np.array_equal(np.load(folders["plain"] / "depth.npy"), np.load(folders["staffed"] / "depth.npy"))
    # The two control sets block the same floor: the desk as it is built, with the posts that carry its name. The
    # default set has no such sign and no posts. The episodes ask the same in all three, of the same regions.
    blocked = {variant: scene.ground_truth.obstacles for variant, scene in scenes.items()}
    assert blocked["plain"] == blocked["staffed"] and len(blocked["plain"]) == len(blocked["real"]) + 2
    assert all(obstacle in blocked["plain"] for obstacle in blocked["real"])
    def asked(scene):                               # what each episode asks, of whom, and where the robot may stop
        return [(e.episode_id, e.request, e.context, e.ground_truth.request_type, e.ground_truth.factors,
                 [region.polygon for region in e.ground_truth.acceptable_regions]) for e in scene.episodes]

    for variant in ("plain", "staffed"):
        assert scenes[variant].ground_truth.targets[0].model_copy(update={"tour_frames": None}) == scenes[
            "real"].ground_truth.targets[0].model_copy(update={"tour_frames": None})
        assert scenes[variant].observation == scenes["real"].observation
        assert asked(scenes[variant]) == asked(scenes["real"]), variant
    # Signage and small things are what tells "staffed" from "real": the two control sets spell the place out.
    styles = {variant: {sign.art.style for part in spec.parts for sign in part.signs}
              for variant, spec in specs.items()}
    assert "label" in styles["plain"] and styles["plain"] == styles["staffed"] and "label" not in styles["real"]
    assert specs["plain"].parts == specs["staffed"].parts != specs["real"].parts
    assert specs["plain"].models == specs["staffed"].models == specs["real"].models[:len(specs["plain"].models)]
    assert specs["plain"].room == specs["staffed"].room == specs["real"].room


def test_main_passes_on_what_it_is_asked_and_prints_the_counts_and_the_folder(monkeypatch, capsys, tmp_path):
    calls = []

    def generate(out_dir, cfg, seed=0, variant="real", renderer=None, only=(), force=False):
        calls.append((out_dir, cfg, seed, variant, renderer, tuple(only), force))
        return {"rendered": 3, "skipped": 2, "dressed_again": 1}

    monkeypatch.setattr(real, "generate", generate)
    assert real.main(["--out", str(tmp_path), "--variant", "plain", "--seed", "7", "--renderer", "stub", "--only", "a",
                      "b", "--force", "--config", str(REPO_ROOT / "configs" / "default.yaml")]) == 0
    out, cfg, seed, variant, renderer, only, force = calls[0]
    assert (out, cfg, seed, variant, only, force) == (tmp_path, CFG, 7, "plain", ("a", "b"), True)
    assert type(renderer) is StubRenderer
    printed = capsys.readouterr().out.splitlines()
    assert printed == [f"{tmp_path}: 3 scenes rendered, 2 skipped as unchanged, 1 dressed again"]
    # By default: seed 0, the whole set, Blender, which generate makes itself, and the variant's own folder.
    for arguments, variant, folder in (([], "real", "realistic"), (["--variant", "plain"], "plain", "realistic_plain"),
                                       (["--variant", "staffed"], "staffed", "realistic_staffed"),
                                       (["--renderer", "blender"], "real", "realistic")):
        assert real.main(arguments) == 0
        assert calls[-1] == (REPO_ROOT / "data" / folder, CFG, 0, variant, None, (), False), arguments
        assert str(REPO_ROOT / "data" / folder) in capsys.readouterr().out
    for wrong in (["--variant", "fancy"], ["--renderer", "isaac"], ["--only"]):
        with pytest.raises(SystemExit):
            real.main(wrong)
    assert len(calls) == 5 and real.RENDERERS == ("blender", "stub")
    assert [real.default_out(variant) for variant in VARIANTS] == [
        REPO_ROOT / "data" / folder for folder in ("realistic", "realistic_plain", "realistic_staffed")]
