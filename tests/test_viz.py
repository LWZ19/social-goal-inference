import dataclasses
import importlib
import math

import matplotlib
import numpy as np
import pytest
from matplotlib import pyplot
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrow

from sgi import solver, viz
from sgi.config import Thresholds
from sgi.evaluate import Score
from sgi.geometry import project, to_robot_xy
from sgi.polygons import disc
from sgi.scene import load_scene
from sgi.stages.goal import standard_goal
from sgi.types import Constraint, DetectedPerson, Flags, GoalResult, GroundedTarget, Point
from tests.helpers import floor_observation, scene_dict, write_scene

ANCHOR, GOAL = (1.0, 6.0, 1.0), (1.4, 3.9, 0.0)      # scene_dict's desk anchor; a floor point in the acceptable region

@pytest.fixture
def scene(tmp_path):                                  # robot at (1, 2) facing +Y; desk footprint x 0.5..1.5, y 6..7
    d = scene_dict()
    d["ground_truth"]["people"] = [{"id": "p0", "x": 1.4, "y": 5.1, "yaw": 0.0},       # p0 faces +X: right in the image
                                   {"id": "p1", "x": 2.0, "y": 0.5, "yaw": 0.0}]       # p1 is behind the camera
    gt = d["episodes"][0]["ground_truth"]
    gt["acceptable_regions"] = [{"polygon": [[1.0, 1.4], [1.8, 1.4], [1.8, 4.2], [1.0, 4.2]],
                                 "heading": {"face_point": [1.4, 5.1]}}]               # its near edge is behind the camera
    gt["forbidden_regions"] = [{"kind": "ahead_in_line", "polygon": [[0.8, 5.1], [2.0, 5.1], [2.0, 6.0], [0.8, 6.0]]}]
    inputs, truth = load_scene(write_scene(tmp_path, d, floor_observation()))
    return inputs.observation, truth

@pytest.fixture
def saved(monkeypatch):                               # every figure passed to savefig, in order
    figures, real = [], Figure.savefig
    monkeypatch.setattr(Figure, "savefig", lambda fig, *a, **k: figures.append(fig) or real(fig, *a, **k))
    return figures

def marked(fig, obs, world, r=12):                    # did the image panel change near this world point?
    u, v, _ = project(obs.camera, world)
    box = np.s_[int(v) - r:int(v) + r + 1, int(u) - r:int(u) + r + 1]
    return bool((np.asarray(fig.axes[0].images[0].get_array())[box] != obs.rgb[box]).any())

def check_file_and_topdown(out, fig):
    assert out.read_bytes()[:4] == b"\x89PNG" and out.stat().st_size > 5000
    _, top = fig.axes                                 # exactly two axes: image, top-down
    assert top.get_aspect() == 1.0
    assert top.get_xlim() == pytest.approx((-1.0, 3.5)) and top.get_ylim() == pytest.approx((-1.0, 8.5))
    assert pyplot.get_fignums() == []                 # closed after saving

def test_import_forces_the_agg_backend():
    matplotlib.use("svg")
    importlib.reload(viz)
    assert matplotlib.get_backend().lower() == "agg"

def test_episode_figure_marks_target_and_goal(scene, saved, tmp_path):
    obs, truth = scene
    (tu, tv, _), (gu, gv, _) = project(obs.camera, ANCHOR), project(obs.camera, GOAL)
    result = GoalResult(status="ok", source="direct_marking", pose_world=(1.4, 3.9, math.pi / 2), goal_pixel=(gu, gv),
                        target=GroundedTarget(point=Point(x=500, y=147), pixel=(tu, tv), world=ANCHOR))
    out = tmp_path / "viz" / "s0_e0.png"              # the folder does not exist yet
    viz.episode_figure(out, obs, truth, truth.episodes["e0"], result,
                       Score(success=True, target_correct=True), title="s0/e0")
    assert saved[-1].get_suptitle() == "s0/e0 | status=ok | success=True"
    assert marked(saved[-1], obs, ANCHOR) and marked(saved[-1], obs, GOAL)
    assert not marked(saved[-1], obs, (0.8, 5.1, 0.0))  # ground truth is not drawn on this image
    check_file_and_topdown(out, saved[-1])
    viz.episode_figure(out, obs, truth, truth.episodes["e0"], result.model_copy(update={"goal_pixel": None}), None)
    assert saved[-1].get_suptitle() == "status=ok"    # no title and no score: no separators
    assert marked(saved[-1], obs, GOAL)               # no marked pixel: the pose's ground point is projected instead

@pytest.mark.parametrize("result, score, suptitle", [
    (None, None, "s0/e0"),
    (GoalResult(status="no_target"), Score(success=False, target_correct=False),
     "s0/e0 | status=no_target | success=False"),
    (GoalResult(status="vlm_error"), None, "s0/e0 | status=vlm_error"),
])
def test_episode_figure_without_a_pose(scene, saved, tmp_path, result, score, suptitle):
    obs, truth = scene
    viz.episode_figure(tmp_path / "e0.png", obs, truth, truth.episodes["e0"], result, score, title="s0/e0")
    assert saved[-1].get_suptitle() == suptitle
    assert (np.asarray(saved[-1].axes[0].images[0].get_array()) == obs.rgb).all()     # nothing to mark
    check_file_and_topdown(tmp_path / "e0.png", saved[-1])

def test_scene_figure_projects_ground_truth(scene, saved, tmp_path):
    obs, truth = scene
    assert project(obs.camera, (2.0, 0.5, 1.0)) is None       # p1 at 1 m: must be skipped, not raise
    assert project(obs.camera, (1.0, 1.4, 0.0)) is None       # a corner of the acceptable region: likewise
    viz.scene_figure(tmp_path / "check.png", obs, truth)
    assert saved[-1].get_suptitle() == truth.scene_id
    for world in [ANCHOR, (1.4, 5.1, 1.0), (1.8, 4.2, 0.0), (0.8, 5.1, 0.0)]:   # anchor, p0 at 1 m, a corner per region
        assert marked(saved[-1], obs, world), world
    assert marked(saved[-1], obs, (1.8, 5.1, 1.0))            # p0's heading line, 0.4 m along its yaw (+X)
    check_file_and_topdown(tmp_path / "check.png", saved[-1])

def outline_marks(fig, obs, polygon, step=0.01):      # walking the outline on the floor: is the image marked at each step?
    marks = []
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        for t in np.arange(0.0, 1.0, step / max(math.dist(a, b), step)):
            marks.append(marked(fig, obs, (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), 0.0), r=1))
    return np.array(marks)

def longest_run(flags):                               # the longest run of True around a closed loop of flags
    best = run = 0
    for flag in np.tile(flags, 2):
        run = run + 1 if flag else 0
        best = max(best, run)
    return min(best, len(flags))

RING = disc((1.2, 5.0), 0.6)                          # the shape of conversation_group's group_space: 48 sides of 0.08 m

@pytest.mark.parametrize("polygon", [
    RING,                                             # every side shorter than one 0.1 m piece
    disc((1.2, 5.0), 1.0),                            # every side longer: 0.13 m
    RING + RING[:1],                                  # closed by repeating its first vertex: one side of no length
], ids=["short_sides", "long_sides", "repeated_vertex"])
def test_scene_figure_dashes_forbidden_outlines_only(saved, tmp_path, polygon):
    marks = {}
    for key, region in [("acceptable_regions", {"polygon": polygon, "heading": {"yaw": 0.0}}),
                        ("forbidden_regions", {"kind": "group_space", "polygon": polygon})]:
        d = scene_dict()
        d["episodes"][0]["ground_truth"][key] = [region]
        inputs, truth = load_scene(write_scene(tmp_path / key, d, floor_observation()))
        viz.scene_figure(tmp_path / key / "check.png", inputs.observation, truth)
        marks[key] = outline_marks(saved[-1], inputs.observation, polygon)
    solid, dashed = marks["acceptable_regions"], marks["forbidden_regions"]
    assert solid.all()                                # acceptable: no step along the outline is left unmarked
    assert 5 <= longest_run(~dashed) <= 12            # forbidden: gaps of one 0.1 m piece (a step is 0.01 m) ...
    assert longest_run(dashed) <= 30                  # ... between dashes of one piece, or two where the outline closes

def arrows(ax):                                       # (patch, tail, tip) per arrow on the plot; a FancyArrow lists its tip
    found = []                                        # first and the two ends of its base fourth and fifth
    for patch in ax.patches:
        if isinstance(patch, FancyArrow):
            xy = patch.get_xy()
            found.append((patch, (xy[3] + xy[4]) / 2, xy[0]))
    return found

def test_topdown_arrows_run_half_a_metre_from_each_centre(scene, saved, tmp_path):
    obs, truth = scene                                # yaws that are no multiple of a right angle; p1 is out of view
    truth = dataclasses.replace(truth, people=tuple(p.model_copy(update={"yaw": yaw})
                                                    for p, yaw in zip(truth.people, (0.7, 2.4))))
    result = GoalResult(status="ok", pose_world=(1.4, 3.9, 2.0))
    viz.episode_figure(tmp_path / "e0.png", obs, truth, truth.episodes["e0"], result, None)
    fig = saved[-1]
    top, found = fig.axes[1], arrows(fig.axes[1])
    poses = [(p.x, p.y, p.yaw) for p in truth.people] + [(obs.robot.x, obs.robot.y, obs.robot.yaw), result.pose_world]
    assert len(found) == len(poses) == 4              # two people, the robot and the goal
    for x, y, yaw in poses:
        tip = (x + 0.5 * math.cos(yaw), y + 0.5 * math.sin(yaw))
        assert any(np.allclose(tail, (x, y)) and np.allclose(end, tip) for _, tail, end in found), (x, y, yaw)

    canvas = FigureCanvasAgg(fig)                     # the arrow shows over its person's disc, not only beyond it
    canvas.draw()
    pixels = np.asarray(canvas.buffer_rgba())[..., :3].astype(int)

    def colour(x, y):
        col, row = top.transData.transform((x, y))
        return pixels[len(pixels) - 1 - int(row), int(col)]

    for person in truth.people:
        body = [c for c in top.patches if isinstance(c, Circle) and np.allclose(c.center, (person.x, person.y))]
        arrow = next(a for a, tail, _ in found if np.allclose(tail, (person.x, person.y)))
        assert len(body) == 1 and body[0].get_radius() == Thresholds().person_radius_m
        assert arrow.get_zorder() > body[0].get_zorder()
        along, across = (math.cos(person.yaw), math.sin(person.yaw)), (-math.sin(person.yaw), math.cos(person.yaw))
        on_shaft = colour(person.x + 0.12 * along[0], person.y + 0.12 * along[1])        # both points are on the disc
        beside = colour(person.x + 0.12 * across[0], person.y + 0.12 * across[1])
        assert np.abs(on_shaft - np.array(to_rgb(viz.PEOPLE)) * 255).max() <= 12          # the people colour, undimmed
        assert np.abs(on_shaft - beside).max() >= 60                                      # and the disc under it is not

# --------------------------------------------------------------------------------- a result of the constraint method
TH = Thresholds()
YELLOW = (255, 221, 0)                                # the disc a person's number stands in, as the model was shown it

def constraint_result(obs, stated=()):                # two people before the desk, numbered, and what the solver makes of `stated`
    spots = [((0.6, 5.0), -math.pi / 2), ((1.4, 5.1), None)]
    people = [DetectedPerson(number=n, xy=xy, yaw=yaw, pixel=project(obs.camera, (*xy, 1.1))[:2]) for n, (xy, yaw) in enumerate(spots, 1)]
    target = GroundedTarget(point=Point(x=500, y=147), pixel=project(obs.camera, ANCHOR)[:2], world=ANCHOR)
    solved = solver.solve(obs, target, people, list(stated), standard_goal(obs, target, TH), TH)
    u, v, _ = project(obs.camera, (*solved.pose_world[:2], 0.0))
    return GoalResult(status="ok", source="constraints", pose_world=solved.pose_world, pose_robot=solved.pose_robot, goal_pixel=(u, v),
                      target=target, flags=solved.flags, people=people, constraints=solved.uses, candidates=solved.candidates,
                      solver=solved.note)

def shaded(top):                                      # the image of the floor by rank on the top-down plot, and the frame it lies in
    floor, = [image for image in top.images[1:]]      # after the map of free space
    return np.ma.masked_invalid(floor.get_array()), floor.get_extent(), floor.get_transform() - top.transData

def cell_xy(extent, frame, row, col, n):              # where the middle of a cell of that image lies in the world
    step = (extent[1] - extent[0]) / n
    return frame.transform((extent[0] + (col + 0.5) * step, extent[2] + (row + 0.5) * step))

def test_episode_figure_of_a_constraint_result_draws_the_numbers_the_place_and_the_runners_up(scene, saved, tmp_path):
    obs, truth = scene
    after = [Constraint(because="They were there first.", kind="come_after", people=[1, 2])]
    result = constraint_result(obs, after)
    assert len(result.candidates) == TH.candidate_count and result.solver.keys.turn_m is not None
    viz.episode_figure(tmp_path / "e0.png", obs, truth, truth.episodes["e0"], result, Score(success=False, target_correct=True), "s0/e0")
    fig = saved[-1]
    picture = np.asarray(fig.axes[0].images[0].get_array())
    for person in result.people:                      # each number in its disc, on the body point
        u, v = (int(round(c)) for c in person.pixel)
        assert (picture[v - 9:v + 10, u - 9:u + 10] == YELLOW).all(axis=-1).sum() > 60, person.number
    assert marked(fig, obs, ANCHOR) and marked(fig, obs, (*result.pose_world[:2], 0.0))       # the target and the place
    in_view = [c for c in result.candidates if viz._pixel(obs, (*c.pose_world[:2], 0.0)) is not None]
    assert len(in_view) >= 2 and all(marked(fig, obs, (*c.pose_world[:2], 0.0), r=8) for c in in_view)      # and each runner-up in view
    top = fig.axes[1]
    labels = [text.get_text() for text in top.get_legend().get_texts()]
    assert labels[-4:] == ["goal", "runner-up", "person found, by number", "floor to stand on (darker: better)"]
    drawn = {line.get_label(): np.column_stack(line.get_data()) for line in top.lines}
    assert sorted(map(tuple, drawn["runner-up"])) == sorted(c.pose_world[:2] for c in result.candidates)
    assert sorted(map(tuple, drawn["person found, by number"])) == sorted(person.xy for person in result.people)
    numbers = {text.get_text(): text.xy for text in top.texts if text.get_text() in ("1", "2")}
    assert numbers == {str(person.number): person.xy for person in result.people}
    (x0, x1), (y0, y1) = top.get_xlim(), top.get_ylim()                # everything drawn is in view
    assert all(x0 < x < x1 and y0 < y < y1 for x, y in [person.xy for person in result.people] + [c.pose_world[:2] for c in result.candidates])
    assert fig.get_suptitle() == "s0/e0 | status=ok | success=False" and pyplot.get_fignums() == []

def test_the_floor_is_shaded_by_rank_as_the_solver_orders_it_from_what_the_record_holds(scene, saved, tmp_path):
    obs, truth = scene
    after = [Constraint(because="They were there first.", kind="come_after", people=[1, 2])]
    for stated in ([], after):
        result = GoalResult.model_validate_json(constraint_result(obs, stated).model_dump_json())       # as a record holds it
        viz.episode_figure(tmp_path / "e0.png", obs, truth, truth.episodes["e0"], result, None)
        shade, extent, frame = shaded(saved[-1].axes[1])
        grid = solver.floor(obs, result.people, TH)
        assert shade.shape == grid.selectable.shape and np.array_equal(~np.ma.getmaskarray(shade), grid.selectable)     # that floor, no more
        assert shade.min() == 0.0 and shade.max() == 1.0
        first = np.argwhere(shade == 1.0)             # the cells that come first: the place is one of them
        at = [cell_xy(extent, frame, row, col, len(shade)) for row, col in first]
        assert min(math.dist(xy, result.pose_world[:2]) for xy in at) < 1e-6 and len(first) <= 5
        ahead, left = to_robot_xy(obs.robot, *result.pose_world[:2])   # the grid lies in the robot's frame, as the solver's does
        row, col = grid.index(*result.pose_world[:2])
        assert shade[row, col] == 1.0 and grid.offsets[col] == pytest.approx(ahead) and grid.offsets[row] == pytest.approx(left)
        for candidate in result.candidates:           # a runner-up is the best of what is left: high in the order, and below the place
            row, col = grid.index(*candidate.pose_world[:2])
            assert 0.5 < shade[row, col] < 1.0
        own = [grid.index(*person.xy) for person in result.people]     # nobody stands where a person stands
        assert all(np.ma.getmaskarray(shade)[row, col] for row, col in own)
    cells, settled = solver._over_floor(obs, result.target, result.people, after, standard_goal(obs, result.target, TH), TH)
    values = shade[grid.selectable]                   # in the order of the cells; then, key by key as the solver compares them:
    broken, held = settled.keys["excluded_m"] > 0, settled.keys["excluded_m"] == 0
    assert broken.any() and values[broken].max() < values[held].min()                 # a place that breaks an exclusion comes after all the others
    near, far = held & (settled.keys["turn_m"] <= 0.5), held & (settled.keys["turn_m"] >= 1.0)
    assert near.any() and far.any() and values[near].min() > values[far].max()        # then the nearer the place in turn the better

def test_a_figure_of_a_baseline_result_is_as_it_was_and_a_failed_result_of_the_method_shows_the_numbers_alone(scene, saved, tmp_path):
    obs, truth = scene
    result = constraint_result(obs)
    baseline = GoalResult(status="ok", source="direct_marking", pose_world=(1.4, 3.9, math.pi / 2), target=result.target)
    viz.episode_figure(tmp_path / "e0.png", obs, truth, truth.episodes["e0"], baseline, None)
    top = saved[-1].axes[1]
    assert len(top.images) == 1 and [text.get_text() for text in top.get_legend().get_texts()][-1] == "goal"
    assert not (np.asarray(saved[-1].axes[0].images[0].get_array()) == YELLOW).all(axis=-1).any()
    check_file_and_topdown(tmp_path / "e0.png", saved[-1])
    failed = GoalResult(status="parse_failed", source="constraints", target=result.target, people=result.people, flags=Flags(people_cut=True))
    viz.episode_figure(tmp_path / "e0.png", obs, truth, truth.episodes["e0"], failed, None)        # the constraints call gave no answer
    fig = saved[-1]
    assert (np.asarray(fig.axes[0].images[0].get_array()) == YELLOW).all(axis=-1).sum() > 120 and len(fig.axes[1].images) == 1
    labels = [text.get_text() for text in fig.axes[1].get_legend().get_texts()]
    assert "person found, by number" in labels and "runner-up" not in labels and "goal" not in labels
    nobody = GoalResult(status="vlm_error", source="constraints", target=result.target)           # nor did the people call
    viz.episode_figure(tmp_path / "e0.png", obs, truth, truth.episodes["e0"], nobody, None)
    assert len(saved[-1].axes[1].images) == 1 and marked(saved[-1], obs, ANCHOR)
