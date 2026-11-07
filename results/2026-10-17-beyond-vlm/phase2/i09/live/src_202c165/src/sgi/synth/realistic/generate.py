"""The realistic scene generator (realistic-scenes spec 5.6): the placeholder generator's scenes, dressed and
drawn by a renderer.

    python -m sgi.synth.realistic.generate [--variant real|plain|staffed] [--out DIR] [--only ID ...] [--force]
                                           [--renderer blender|stub] [--config PATH] [--seed N]

It draws no layout of its own. For the seed it takes the layouts the placeholder generator keeps
(accepted_layouts), all of them and in its order, since they come from one stream of random numbers, and makes
of each one scene in scene format v0.1 under `<out>/<split>/<id>/`. The scene is the twin of the placeholder
scene of that layout: the same people in the same places, the same robot, camera, targets, bounds and
episodes, each episode with its twin's id, request, context and request type. Its id is the twin's with the
variant before it, as in `real_eval_07_waiting_line_2`, and its source names the renderer and the variant.
What dressing adds, the staff and what a staffed place blocks, is in the ground truth as it is drawn
(dress.py); the `density` label counts the people without a role, so it is the twin's. One text of an episode
may differ from the twin's: what its rationale says of the people, where nobody waits at a target that a member
of staff stands behind.

A scene is made in these steps:

1. It is dressed with the seed (seed, split, row, attempt), in every variant: the split is 0 for eval and 1 for
   examples, the attempt counts from 0. Each scene's dressing is its own, whatever happens to the scenes
   before it.
2. The renderer draws the observation's depth alone, the scene is written with a grey picture to a temporary
   folder beside its place, and it is judged as the placeholder generator judges a layout (its _rejection:
   the scene check without an error or a warning other than the expected one, and a standard goal that is
   reached where nobody is about). So is the scene's dressing in each of the other variants. A dressing that
   fails in any of them is given up in all for the next attempt, five in all; then the run stops and says
   why. The attempt seeds the cast, and the variants of a scene show the same people (spec 5.6): each run
   keeps the first attempt at which all three pass, whichever variant it makes. The layout is never drawn
   again.
3. The reference regions are derived from that folder, as for the placeholder set.
4. The tour is the placeholder's walk on this scene's own ground truth: the same turns about the mean of the
   target anchors, each pose clear of the walls, of what blocks the floor here and of everybody, staff
   included. The frame of the turn by 0 is the observation itself.
5. The renderer draws everything else in one call. First the observation's depth alone once more, before any
   picture: it must be the depth of step 2, pixel for pixel, or the call has not drawn the scene that was
   judged. Then the observation and the other tour frames with their pictures, and for every tour frame and
   target a depth image of what draws the target, alone, through the frame's camera grown until it holds the
   target (target_blocks, widened as the placeholder widens). The scene keeps the depth of step 2. The depth
   that comes with the observation's picture may differ from it at single pixels (_STRAY_SHARE says why).
6. A target's `tour_frames` are the frames it is visible in, by the placeholder's three rules read off depth:
   the frame shows its anchor, or lettering on it reads, or the frame shows half of it or more.
7. `render_job.json` is written beside `scene.json`: the variant, the seed, the attempt that was kept, the
   renderer's name and version, the description that was drawn, the thresholds, and for each variant a digest
   of what it was judged from (see _digest). Then the folder is put in its place. One that is there already
   is moved aside first, put back if the new one does not go in, and removed once the new one stands: at any
   failure the place holds the old folder or the new one, whole, and nothing else is left behind.

A scene whose folder loads and whose stored job is the one about to be run is left as it is, unless `force`
is set: a path tracer on a graphics card does not draw the same pixels twice, and a model's cached answers are
keyed on pixels. The job holds, or holds a digest of, the data the folder is made from: a request put in other
words, a camera that has moved or another threshold draws the scene again though its description is the same.
Where the stored job differs only in what another variant was judged from, that variant is judged again and,
if it passes, the scene stays as it was drawn and its job is brought up to date. Three sets belong together
when, scene for scene, their jobs hold the same seed, attempt and digests. What no job holds is the code that
derives from the data (the rules for regions and for the tour), the files of the assets and the renderer's
settings: after a change to those, `force` is the remedy.

`only` names the scenes to make, by their ids or their twins'. A run never empties a folder: it replaces the
scenes it renders and leaves everything else.

With the Blender renderer nothing is started before Blender and the prepared assets are known to be there.
The stub needs neither, and draws from the manifest alone.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import os
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from sgi.config import DEFAULT_CONFIG, REPO_ROOT, Config, Thresholds, load_config
from sgi.geometry import Camera, RobotPose
from sgi.scene import Factors, SceneError, SceneFile, Source, Target, TourFrameModel, load_scene, save_scene
from sgi.synth.generate import (_LETTERING_STEP_M, _ON_SURFACE_M, _SAME_DEPTH_M, _TOUR_SHARE, _TOUR_TURNS_DEG, _density,
                                _rejection, _shows, _tour_pose, _tour_side, _widened, _with_regions, accepted_layouts)
from sgi.synth.realistic import blender
from sgi.synth.realistic.blender_renderer import PREPARE_COMMAND, BlenderRenderer
from sgi.synth.realistic.constructions import Lettered
from sgi.synth.realistic.dress import VARIANTS, Block, Dressed, Variant, dress
from sgi.synth.realistic.library import Library, is_prepared, load_library
from sgi.synth.realistic.spec import Frame, Renderer, SceneSpec, to_json
from sgi.synth.realistic.stub import StubRenderer
from sgi.synth.recipes import _READABLE_COS, Layout
from sgi.synth.render import Box

log = logging.getLogger("sgi.synth.realistic.generate")    # by name: run as a command, this module is __main__

RENDERERS = (BlenderRenderer.name, StubRenderer.name)      # what --renderer takes: "blender", the default, or "stub"
_SPLITS = ("eval", "examples")                     # a dressing's seed names the split by its place here
_MAX_DRESSINGS = 5                                 # attempts at a dressing that passes the scene check
_JOB_FILE = "render_job.json"
_GREY = 128                                        # of the picture a scene is judged with: the check reads depth alone
# Of the observation's pixels, the most at which the depth that comes with its picture may differ from the depth
# drawn alone, which the scene is judged with and keeps. The stub draws the two alike. Blender's depth is the same
# from frame to frame and from run to run until a process draws its first picture; from then on a few pixels on
# people's hair show another strand of it or what is behind it. Over the 20 scenes seed 0 had when this was measured that is 0 to 25 of
# the 307,200 pixels, alike with staff and without. This is some twelve times that: enough to tell a picture
# taken from elsewhere, which differs at every edge, and not meant to tell more. That the call drew the scene
# that was judged does not rest on it: the depth drawn alone before the first picture must be equal (_toured).
_STRAY_SHARE = 1e-3
_FETCH_ASSETS_COMMAND = ".venv/bin/python -m sgi.synth.realistic.assets fetch"     # the action before PREPARE_COMMAND


def default_out(variant: str) -> Path:
    """Where a variant's set is written when no folder is named: data/realistic, and data/realistic_<variant>
    for the control sets. The folder data/ is not committed."""
    return REPO_ROOT / "data" / ("realistic" if variant == "real" else f"realistic_{variant}")


# --- What a tour frame shows, read off depth. ---

def _reads(camera: Camera, depth: np.ndarray, lettered: Lettered) -> bool:
    """Whether the frame with this depth image shows the lettered rectangle so that it can be read: all of it,
    from no further off the square view of it than lettering is read from.

    The placeholder's rule for a lettered box (its _reads), on a rectangle. A sign on a plate lies 3 mm before
    the rectangle, which is well within what counts as the same surface."""
    middle, normal, across = (np.asarray(vector, dtype=np.float64) for vector in
                              (lettered.centre, lettered.normal, lettered.across))
    to_camera = camera.T_world_camera[:3, 3] - middle
    if to_camera @ normal < _READABLE_COS * np.linalg.norm(to_camera) * np.linalg.norm(normal):
        return False
    up = np.array([0.0, 0.0, 1.0])
    along = [np.linspace(-0.5, 0.5, math.ceil(side / _LETTERING_STEP_M) + 1) * side
             for side in (lettered.width, lettered.height)]
    return all(_shows(camera, depth, middle + a * across + b * up, _ON_SURFACE_M) for a in along[0] for b in along[1])


def _wide(camera: Camera, blocks: Sequence[Block]) -> tuple[Camera, int, int]:
    """The camera with its image grown until it holds all of the blocks, and the column and the row that its
    first pixel has in the grown image: the placeholder's _widened, which reads of a thing only the upright
    block that holds it."""
    return _widened(camera, [Box(x, y, yaw, 2.0 * half_x, 2.0 * half_y, top)
                             for x, y, yaw, half_x, half_y, top in blocks])


def _share_shown(depth: np.ndarray, alone: np.ndarray, left: int, above: int) -> float:
    """The share of a target that the frame with this depth image shows.

    `alone` is the depth of what draws the target with nothing else there, 0.0 where it is not, taken along the
    frame's rays in an image grown to hold it; the frame's first pixel is at column `left` and row `above` of
    it. The whole is the pixels the target fills there. The frame shows those of them that lie in its image
    and at which it has the same depth: nothing stands before them. 0.0 for a target of which nothing is drawn."""
    theirs = alone > 0
    if not theirs.any():
        return 0.0
    height, width = depth.shape
    inside = np.s_[above:above + height, left:left + width]
    shown = theirs[inside] & (np.abs(depth - alone[inside]) < _SAME_DEPTH_M)
    return float(shown.sum() / theirs.sum())


def _visible(camera: Camera, depth: np.ndarray, target: Target, lettering: Sequence[Lettered], alone: np.ndarray,
             left: int, above: int, th: Thresholds) -> bool:
    """Whether the target is visible in the frame with this depth image, by the placeholder's three rules: the
    frame shows its anchor as scene check 5 asks, or lettering on it so that it can be read, or half of it or
    more (see _share_shown for `alone`, `left` and `above`)."""
    return (_shows(camera, depth, target.anchor, th.anchor_depth_tolerance_m)
            or any(_reads(camera, depth, lettered) for lettered in lettering)
            or _share_shown(depth, alone, left, above) >= _TOUR_SHARE)


# --- One scene. ---

def _camera(scene: SceneFile, T: np.ndarray | None = None) -> Camera:
    """The scene's camera at its observation, or with the same lens at the pose T."""
    obs, lens = scene.observation, scene.observation.intrinsics
    pose = np.array(obs.T_world_camera, dtype=np.float64) if T is None else T
    return Camera(lens.fx, lens.fy, lens.cx, lens.cy, obs.width, obs.height, pose)


def _described(twin: SceneFile, layout: Layout, dressed: Dressed, variant: Variant, renderer: Renderer) -> SceneFile:
    """The dressed layout as a scene file without regions or a tour: the twin's observation, the episodes as
    dressing gives them (the layout's, but for what a rationale says of the people where staff stand at the
    target: dress.py) with the density label as the placeholder generator adds it, and the ground truth as it is
    drawn. Only people without a role are counted, so the label is the twin's."""
    density = _density(sum(person.role is None for person in dressed.scene_gt.people))
    episodes = [episode.model_copy(update={"ground_truth": episode.ground_truth.model_copy(
        update={"factors": Factors(density=density)})}) for episode in dressed.episodes]
    source = Source(simulator=f"sgi-synth-{renderer.name}/{variant.name}", version=renderer.version)
    return SceneFile(format_version="0.1", scene_id=f"{variant.name}_{twin.scene_id}", source=source,
                     observation=twin.observation, episodes=episodes, ground_truth=dressed.scene_gt)


def _digest(dressed: Dressed, scene: SceneFile) -> str:
    """What a variant of a scene is judged from, in one word: the SHA-256 of all that dressing gave and of the
    scene as described from it, which holds its camera, its episodes and its ground truth before regions and
    tour. With the thresholds and the renderer it decides whether the scene passes the check, and, for the
    variant that is made, everything in its folder but the picture's own noise."""
    lettering = {target: [dataclasses.asdict(each) for each in lettered]
                 for target, lettered in dressed.lettering.items()}
    told = {"spec": to_json(dressed.spec), "scene": scene.model_dump(mode="json"), "groups": dressed.target_groups,
            "blocks": dressed.target_blocks, "lettering": lettering}
    return hashlib.sha256(json.dumps(told, sort_keys=True).encode("utf-8")).hexdigest()


def _job(variant: Variant, seed: int, attempt: int, renderer: Renderer, spec: SceneSpec, judged: Mapping[str, str],
         th: Thresholds) -> dict:
    """What render_job.json holds for a dressing, as it reads back from the file: stored jobs are compared as
    parsed JSON. `judged` is the digest of each variant at this attempt, by the variant's name."""
    job = {"variant": variant.name, "seed": int(seed), "attempt": int(attempt),
           "renderer": {"name": renderer.name, "version": renderer.version}, "spec": to_json(spec),
           "thresholds": th.model_dump(mode="json"), "judged": {name: judged[name] for name in VARIANTS}}
    return json.loads(json.dumps(job))


def _stored_job(scene_dir: Path) -> dict | None:
    """The job a scene folder says it was rendered from. None for a folder that is not there, that does not
    load as a scene with all its files, or whose job does not read as one."""
    if not (scene_dir / _JOB_FILE).is_file():
        return None
    try:
        load_scene(scene_dir)
        job = json.loads((scene_dir / _JOB_FILE).read_text(encoding="utf-8"))
    except (SceneError, OSError, ValueError):
        return None
    return job if isinstance(job, dict) else None


def _stands(stored: dict | None, job: dict) -> bool:
    """Whether the folder with this stored job is the one `job` would make, should the job be carried out: the
    two are the same but, at the most, in what the other variants were judged from."""
    if stored is None or not isinstance(stored.get("judged"), dict):
        return False
    own = job["variant"]
    return {**stored, "judged": None} == {**job, "judged": None} and stored["judged"].get(own) == job["judged"][own]


def _differences(stored: dict, job: dict) -> list[str]:
    """The entries in which a stored job differs from a job, by their names; `judged` variant by variant."""
    was = stored.get("judged") if isinstance(stored.get("judged"), dict) else {}
    judged = [f"judged.{name}" for name, digest in job["judged"].items() if was.get(name) != digest]
    return sorted(key for key in {*stored, *job} if key != "judged" and stored.get(key) != job.get(key)) + judged


def _tour_poses(scene: SceneFile, th: Thresholds) -> list[np.ndarray]:
    """T_world_camera of each tour frame, by the placeholder generator's rules (its _build_tour) on this scene's
    ground truth: frame k asks for the observation's robot turned about the mean of the target anchors by
    _TOUR_TURNS_DEG[k], counted the way _tour_side gives, and _tour_pose moves it until it stands clear. The
    turn by 0 is the observation's own pose, as is a turn that finds no pose at all."""
    obs, truth = scene.observation, scene.ground_truth
    T_obs = np.array(obs.T_world_camera, dtype=np.float64)
    robot = RobotPose(obs.robot_pose_world.x, obs.robot_pose_world.y, obs.robot_pose_world.yaw)
    centre = np.mean([target.anchor[:2] for target in truth.targets], axis=0)
    side = _tour_side(robot, centre, truth)
    return [T_obs if degrees == 0 else _tour_pose(T_obs, robot, centre, side * degrees, truth, th)
            for degrees in _TOUR_TURNS_DEG]


def _toured(scene: SceneFile, dressed: Dressed, renderer: Renderer, depth: np.ndarray,
            th: Thresholds) -> tuple[SceneFile, np.ndarray, list[np.ndarray]]:
    """The scene with its tour and every target's `tour_frames`, the observation's picture, and the tour's
    pictures in index order. `depth` is the observation's, as the scene was judged with it: it stays the
    scene's depth, and the observation is read with it here too, so a target whose anchor passed scene
    check 5 lists that frame.

    Everything is drawn in one call of the renderer. First comes the observation's depth alone, before any
    picture is drawn: what a renderer draws so is the same in every call, so it tells whether this call drew
    the scene that was judged. Then come the views. Tour frames that stand at one pose are one view and share
    one picture, so the frame of the turn by 0 is the observation's picture, pixel for pixel. After the views
    come, view by view and target by target, the depth images of each target alone.

    RuntimeError when the depth drawn alone is not `depth`, pixel for pixel, and when the depth that comes
    with the observation's picture differs from it at more than single pixels (see _STRAY_SHARE)."""
    targets, poses = scene.ground_truth.targets, _tour_poses(scene, th)
    views, view_of = [_camera(scene)], []          # the distinct views, the observation first; each tour frame's view
    for T in poses:
        same = [number for number, view in enumerate(views) if np.array_equal(view.T_world_camera, T)]
        if not same:
            views.append(_camera(scene, T))
        view_of.append(same[0] if same else len(views) - 1)
    frames, offsets = [Frame(views[0], beauty=False), *(Frame(view) for view in views)], []
    for view in views:
        for target in targets:
            wide, left, above = _wide(view, dressed.target_blocks[target.id])
            frames.append(Frame(wide, beauty=False, only=tuple(dressed.target_groups[target.id])))
            offsets.append((left, above))
    drawn = renderer.render(dressed.spec, frames)
    if len(drawn) != len(frames):
        raise RuntimeError(f"{scene.scene_id}: the renderer {renderer.name} gave {len(drawn)} images for "
                           f"{len(frames)} frames")
    (_, again), pictured, alone = drawn[0], drawn[1:1 + len(views)], iter(zip(offsets, drawn[1 + len(views):]))
    if not np.array_equal(again, depth):
        differing = int(np.sum(again != depth)) if again.shape == depth.shape else depth.size
        raise RuntimeError(f"{scene.scene_id}: the renderer {renderer.name} drew another depth image of the "
                           f"observation than the one the scene was judged with, at {differing} pixels: its "
                           "pictures would not be of the scene that was judged")
    stray = float(np.mean(np.abs(pictured[0][1] - depth) >= _SAME_DEPTH_M))
    if stray > _STRAY_SHARE:
        raise RuntimeError(f"{scene.scene_id}: the depth that comes with the observation's picture differs from "
                           f"the observation's depth at {stray:.2%} of the pixels: the renderer {renderer.name} "
                           "drew the picture from elsewhere than the scene was judged from")
    visible = []                                   # for each view: whether it shows each target, by the target's id
    for view, seen in zip(views, (depth, *(seen for _, seen in pictured[1:]))):
        shows = {}
        for target in targets:
            (left, above), (_, theirs) = next(alone)
            shows[target.id] = _visible(view, seen, target, dressed.lettering[target.id], theirs, left, above, th)
        visible.append(shows)
    listed = {target.id: [k for k, view in enumerate(view_of) if visible[view][target.id]] for target in targets}
    toured = [target.model_copy(update={"tour_frames": listed[target.id]}) for target in targets]
    entries = [TourFrameModel(rgb=f"tour/{k:03d}.png", T_world_camera=T.tolist()) for k, T in enumerate(poses)]
    truth = scene.ground_truth.model_copy(update={"targets": toured})
    return (scene.model_copy(update={"tour": entries, "ground_truth": truth}), pictured[0][0],
            [pictured[view][0] for view in view_of])


def _put(made: Path, place: Path, aside: Path) -> None:
    """Put a scene folder that is whole where its scene belongs. A folder that is there already is moved aside
    first, and is moved back when the new one cannot be put there: the place holds one of the two, whole."""
    replaced = os.path.lexists(place)
    if replaced:
        os.replace(place, aside)
    try:
        os.replace(made, place)
    except BaseException:
        if replaced:
            os.replace(aside, place)
        raise


def _judged(folder: Path, scene: SceneFile, dressed: Dressed, renderer: Renderer,
            cfg: Config) -> tuple[str | None, np.ndarray]:
    """Judge a dressing: the renderer draws the observation's depth alone, the scene is written to `folder` with
    it and a grey picture, and is judged there as the placeholder generator judges a layout. Returns why the
    dressing is not kept, or None when it is, and the depth."""
    camera = _camera(scene)
    ((_, depth),) = renderer.render(dressed.spec, [Frame(camera, beauty=False)])
    save_scene(folder, scene, np.full((camera.height, camera.width, 3), _GREY, dtype=np.uint8), depth)
    return _rejection(folder, cfg), depth


def _scene(split_dir: Path, twin: SceneFile, layout: Layout, key: tuple[int, int, int], variant: Variant,
           library: Library, renderer: Renderer, cfg: Config, force: bool) -> tuple[str, int]:
    """Make the scene of one layout under split_dir, as the module's steps say. `key` is the seed, the split and
    the row, which with the attempt seed its dressing.

    Returns what became of it, "rendered" or "skipped", and how many dressings were given up before the one
    that was kept. RuntimeError when no dressing passes; whatever fails, a folder that was there is still there
    and nothing else is. (Only where the old folder was moved aside and then nothing could be put in its place,
    not even itself, is it left in the hidden folder beside the scenes that it was moved to.)"""
    scene_id = f"{variant.name}_{twin.scene_id}"
    place = split_dir / scene_id
    stored = None if force else _stored_job(place)
    every = (variant, *(other for other in VARIANTS.values() if other.name != variant.name))   # its own first
    reasons: list[str] = []
    for attempt in range(_MAX_DRESSINGS):
        dressings = {each.name: dress(layout, each, library, (*key, attempt)) for each in every}
        scenes = {each.name: _described(twin, layout, dressings[each.name], each, renderer) for each in every}
        dressed = dressings[variant.name]
        digests = {each.name: _digest(dressings[each.name], scenes[each.name]) for each in every}
        job = _job(variant, key[0], attempt, renderer, dressed.spec, digests, cfg.thresholds)
        if stored == job:
            return "skipped", attempt
        # Where the scene itself is as it was drawn and passed, only what has changed beside it is judged again.
        stands = _stands(stored, job)
        to_judge = [each for each in every if not stands or stored["judged"].get(each.name) != digests[each.name]]
        split_dir.mkdir(parents=True, exist_ok=True)
        # The scene is made one folder down, so that what is beside the scenes holds no scene.json of its own.
        work = Path(tempfile.mkdtemp(prefix=f".{scene_id}.", dir=split_dir))
        making, aside = work / scene_id, work / "replaced"
        try:
            depth, failed = None, None
            for each in to_judge:
                reason, seen = _judged(work / scenes[each.name].scene_id, scenes[each.name], dressings[each.name],
                                       renderer, cfg)
                if reason is not None:             # in whichever variant: the attempt is given up in all of them
                    failed = reason if each.name == variant.name else f"{reason} as {each.name}"
                    break
                if each.name == variant.name:
                    depth = seen
            if failed is not None:
                reasons.append(failed)
                continue
            if stands:                             # nothing of the folder changes but what its job says was judged
                (work / _JOB_FILE).write_text(json.dumps(job, indent=1), encoding="utf-8")
                os.replace(work / _JOB_FILE, place / _JOB_FILE)
                return "skipped", attempt
            scene = _with_regions(making, scenes[variant.name], cfg)
            scene, rgb, pictures = _toured(scene, dressed, renderer, depth, cfg.thresholds)
            save_scene(making, scene, rgb, depth, tour=pictures)
            (making / _JOB_FILE).write_text(json.dumps(job, indent=1), encoding="utf-8")
            _put(making, place, aside)
            if stored is not None:                 # new pictures are new pixels: say what made them necessary
                log.info("%s: drawn again, its stored job differs in %s", scene_id,
                         ", ".join(_differences(stored, job)))
            return "rendered", attempt
        finally:
            # Should the old folder have been moved aside and neither the new one nor the old one have gone into
            # its place, the old one is all there is of the scene's pictures: it stays where it was moved to.
            if os.path.lexists(place) or not os.path.lexists(aside):
                shutil.rmtree(work, ignore_errors=True)
    rejected = ", ".join(f"{reason}={count}" for reason, count in Counter(reasons).items())
    raise RuntimeError(f"{scene_id}: no dressing passed the scene check in {_MAX_DRESSINGS} attempts; "
                       f"rejected: {rejected}")


# --- The set. ---

def _ready_for_blender() -> None:
    """RuntimeError, naming what provides it, unless Blender answers and the assets are prepared."""
    if blender.available():
        return
    missing = [] if blender.binary_available() else [f"no Blender of the {blender.BLENDER_SERIES} series answers"]
    try:
        prepared = is_prepared(load_library())
    except Exception:                              # a manifest that does not read: nothing is prepared from it
        prepared = False
    missing += [] if prepared else ["the assets are not prepared"]
    raise RuntimeError(
        "the realistic scenes are rendered with Blender from prepared assets, and that is not to be had here"
        f"{': ' if missing else ''}{' and '.join(missing)}. Fetch Blender with:\n    {blender.FETCH_COMMAND}\n"
        f"then fetch the assets and prepare them with:\n    {_FETCH_ASSETS_COMMAND}\n    {PREPARE_COMMAND}\n"
        "The stub draws the scenes as a sketch without either: --renderer stub.")


def _named(only: Sequence[str], variant: str, twins: Sequence[str]) -> set[str] | None:
    """The scenes that `only` names, by the ids of their twins, or None when it names none, which is all of
    them. An id may have the variant before it or not. ValueError for one that names no scene of the set."""
    names = (only,) if isinstance(only, str) else tuple(only)
    if not names:
        return None
    asked = {name: name.removeprefix(f"{variant}_") for name in names}
    unknown = [name for name, twin in asked.items() if twin not in twins]
    if unknown:
        raise ValueError(f"no scene of the set is named {', '.join(unknown)}: its scenes are "
                         + ", ".join(f"{variant}_{twin}" for twin in twins))
    return set(asked.values())


def generate(out_dir: Path, cfg: Config, seed: int = 0, variant: str = "real", renderer: Renderer | None = None,
             only: Sequence[str] = (), force: bool = False) -> dict[str, int]:
    """Write the variant's scenes for the seed under out_dir/eval and out_dir/examples: all of them, or those
    that `only` names. `renderer` is anything that honours the Renderer interface; None is Blender.

    Returns how many scenes were rendered, how many were skipped because their stored job is the one that
    would be run (never with `force`), and how many of both had a dressing given up before the one kept:
    {"rendered": n, "skipped": n, "dressed_again": n}. A dressing is kept when the scene passes the check in
    every variant, so that the sets of the three variants show the same cast scene for scene.

    ValueError for an unknown variant and for an id in `only` that names no scene, before anything is drawn;
    RuntimeError for Blender without its binary or its prepared assets, before a layout is drawn, and for a
    scene of which no dressing passes the scene check."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}: expected one of {', '.join(VARIANTS)}")
    if renderer is None or renderer.name == RENDERERS[0]:
        _ready_for_blender()
    if renderer is None:
        renderer = BlenderRenderer()
    out_dir, library = Path(out_dir), load_library()
    with tempfile.TemporaryDirectory(prefix="sgi_realistic_") as work_dir:
        kept = list(accepted_layouts(Path(work_dir), cfg, seed))    # to the end, whatever is named: one random stream
    named = _named(only, variant, [twin.scene_id for _, _, _, twin in kept])
    counts = {"rendered": 0, "skipped": 0, "dressed_again": 0}
    for split, index, layout, twin in kept:
        if named is not None and twin.scene_id not in named:
            continue
        if split not in _SPLITS:
            raise ValueError(f"{twin.scene_id} is of the split {split!r}, which has no number for a dressing's seed: "
                             f"expected one of {', '.join(_SPLITS)}")
        begun, key = time.perf_counter(), (seed, _SPLITS.index(split), index)
        outcome, given_up = _scene(out_dir / split, twin, layout, key, VARIANTS[variant], library, renderer, cfg, force)
        counts[outcome] += 1
        counts["dressed_again"] += given_up > 0
        log.info("%s_%s: %s in %.1f s, after %d dressings given up", variant, twin.scene_id, outcome,
                 time.perf_counter() - begun, given_up)
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sgi.synth.realistic.generate",
                                     description="Render the realistic scenes: the placeholder set's twins.")
    parser.add_argument("--out", type=Path, default=None,
                        help="the set's folder; by default data/realistic, or data/realistic_<variant> for a control")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="real")
    parser.add_argument("--renderer", choices=RENDERERS, default=RENDERERS[0])
    parser.add_argument("--only", nargs="+", default=[], metavar="ID",
                        help="the scenes to make, by their ids, with the variant before them or without")
    parser.add_argument("--force", action="store_true", help="render a scene again though its stored job is unchanged")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)        # a set takes minutes: each scene is said on stderr as it is done
    out = args.out if args.out is not None else default_out(args.variant)
    # Blender is made by generate, once it knows that the binary and the prepared assets are there.
    renderer = StubRenderer() if args.renderer == StubRenderer.name else None
    counts = generate(out, load_config(args.config), args.seed, args.variant, renderer, args.only, args.force)
    print(f"{out}: {counts['rendered']} scenes rendered, {counts['skipped']} skipped as unchanged, "
          f"{counts['dressed_again']} dressed again")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
