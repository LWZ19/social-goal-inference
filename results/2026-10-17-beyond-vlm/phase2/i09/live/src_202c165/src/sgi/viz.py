"""Pictures for the harness: the episode figure and the scene-check overlay (harness side).

Both figures are an annotated copy of the observation image beside a top-down plot in world metres.
The image is annotated with pillow so that the array shown can be compared with the input image.

For a result of the constraint method the episode figure also shows what that method worked with: the number
of every person found, the runners-up beside the place, and the floor the robot may stand on, shaded by the
order the solver puts its cells in. That floor is worked out again from what the result records, with the
default thresholds as everything here."""
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgba  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyArrow, Patch, Polygon as PolygonPatch  # noqa: E402
from matplotlib.transforms import Affine2D  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from sgi import solver  # noqa: E402
from sgi.config import Thresholds  # noqa: E402
from sgi.evaluate import Score  # noqa: E402
from sgi.freespace import static_grid  # noqa: E402
from sgi.geometry import in_image, project  # noqa: E402
from sgi.imaging import draw_people_numbers  # noqa: E402
from sgi.inputs import ObservationData  # noqa: E402
from sgi.scene import EpisodeGT, ForbiddenRegion, GroundTruth, Region  # noqa: E402
from sgi.stages.goal import standard_goal  # noqa: E402
from sgi.types import GoalResult  # noqa: E402

PEOPLE, TARGET, GOAL = "#2a78d6", "#eda100", "#4a3aa7"
ACCEPTABLE, FORBIDDEN, ROBOT = "#008300", "#e34948", "#0b0b0b"
NUMBER = "#ffdd00"       # the disc a found person's number stands in: the yellow it has on the image a model is shown
# The floor by rank: one hue from light to dark, the darker the earlier a cell comes in the solver's order.
RANK = LinearSegmentedColormap.from_list("rank", [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b"])

_RING_PX = 10            # target ring radius on the image
_DISC_PX = 6             # goal and person disc radius on the image
_RUNNER_UP_PX = 4        # and the radius of a runner-up's ring there
_ARROW_M = 0.5           # heading arrow length, top-down and on the image
_ARROW_SHAFT_M = 0.05    # top-down arrow: width of the shaft
_ARROW_HEAD_M = 0.16     # top-down arrow: length and width of the head
_PERSON_FILL = 0.3       # opacity of a person's disc on the top-down plot, so that the arrow over it shows
_MARGIN_M = 1.5          # view margin around the content
_PIECE_M = 0.1           # longest piece of a projected line (spec 6, check.png)
_LANDMARK_Z = 1.0        # height of the landmark points projected on the image (spec 6, check 4)
_DPI = 110
_FREE_GREY = 0.85        # shade of non-free cells on the top-down plot


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    return tuple(int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))


def _label(draw: ImageDraw.ImageDraw, u: float, v: float, text: str) -> None:
    draw.text((u + 12, v - 6), text, fill=(255, 255, 255), stroke_width=1, stroke_fill=(20, 20, 20))


def _ring(draw: ImageDraw.ImageDraw, u: float, v: float, text: str) -> None:
    draw.ellipse((u - _RING_PX, v - _RING_PX, u + _RING_PX, v + _RING_PX), outline=_rgb(TARGET), width=3)
    _label(draw, u, v, text)


def _disc(draw: ImageDraw.ImageDraw, u: float, v: float, colour: str, text: str) -> None:
    draw.ellipse((u - _DISC_PX, v - _DISC_PX, u + _DISC_PX, v + _DISC_PX), fill=_rgb(colour),
                 outline=(255, 255, 255), width=2)
    _label(draw, u, v, text)


def _pixel(obs: ObservationData, world) -> tuple[float, float] | None:
    """Pixel of a world point, or None when it is behind the camera or outside the image."""
    hit = project(obs.camera, world)
    if hit is None or not in_image(obs.camera, hit[0], hit[1]):
        return None
    return hit[0], hit[1]


def _line(draw: ImageDraw.ImageDraw, obs: ObservationData, a, b, colour: str, dashed: bool = False,
          width: int = 2, start: float = 0.0) -> float:
    """A world segment cut into pieces of at most _PIECE_M, a piece drawn only when both ends are in the image.

    The segment may be one side of an outline: `start` is the length of outline before `a`, and the length up to
    `b` is returned. The cuts fall at every _PIECE_M of that whole length, and a dashed line leaves out every
    other _PIECE_M of it, so the dashes run on round the corners whatever the length of the sides."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    length = float(np.linalg.norm(b - a))
    end = start + length
    if length == 0.0:        # a repeated vertex
        return end
    cuts = [start, *(k * _PIECE_M for k in range(math.floor(start / _PIECE_M) + 1, math.ceil(end / _PIECE_M))), end]
    for s, t in zip(cuts, cuts[1:]):
        if dashed and math.floor((s + t) / 2 / _PIECE_M) % 2:
            continue
        p, q = (_pixel(obs, a + (b - a) * (along - start) / length) for along in (s, t))
        if p is not None and q is not None:
            draw.line([p, q], fill=_rgb(colour), width=width)
    return end


def _outline(draw: ImageDraw.ImageDraw, obs: ObservationData, polygon, colour: str, dashed: bool) -> None:
    done = 0.0               # length of outline behind the current side: one dash pattern for the whole outline
    for i, a in enumerate(polygon):
        b = polygon[(i + 1) % len(polygon)]
        done = _line(draw, obs, (a[0], a[1], 0.0), (b[0], b[1], 0.0), colour, dashed, start=done)


def _mark_scene(draw: ImageDraw.ImageDraw, obs: ObservationData, truth: GroundTruth) -> None:
    for ep in truth.episodes.values():
        for region in ep.acceptable_regions or []:
            _outline(draw, obs, region.polygon, ACCEPTABLE, dashed=False)
        for region in ep.forbidden_regions or []:
            _outline(draw, obs, region.polygon, FORBIDDEN, dashed=True)
    for target in truth.targets.values():
        pixel = _pixel(obs, target.anchor)
        if pixel is not None:
            _ring(draw, *pixel, target.label)
    for person in truth.people:
        pixel = _pixel(obs, (person.x, person.y, _LANDMARK_Z))
        if pixel is not None:
            _disc(draw, *pixel, PEOPLE, person.id)
        tip = (person.x + _ARROW_M * math.cos(person.yaw), person.y + _ARROW_M * math.sin(person.yaw))
        _line(draw, obs, (person.x, person.y, _LANDMARK_Z), (*tip, _LANDMARK_Z), PEOPLE, width=3)


def _mark_result(draw: ImageDraw.ImageDraw, obs: ObservationData, result: GoalResult) -> None:
    if result.target is not None:
        _ring(draw, *result.target.pixel, "TARGET")
    for place, candidate in enumerate(result.candidates or [], 2):     # the runners-up, by their place in the order
        pixel = _pixel(obs, (*candidate.pose_world[:2], 0.0))
        if pixel is not None:
            u, v = pixel
            draw.ellipse((u - _RUNNER_UP_PX, v - _RUNNER_UP_PX, u + _RUNNER_UP_PX, v + _RUNNER_UP_PX),
                         fill=(255, 255, 255), outline=_rgb(GOAL), width=2)
            _label(draw, u - 4, v, str(place))
    goal = result.goal_pixel
    if goal is None and result.pose_world is not None:
        goal = _pixel(obs, (result.pose_world[0], result.pose_world[1], 0.0))
    if goal is not None:
        _disc(draw, *goal, GOAL, "GOAL")


def _arrow(ax, x: float, y: float, yaw: float, colour: str) -> None:
    """A heading arrow sized in world metres: its tail at (x, y), its tip _ARROW_M along yaw."""
    ax.add_patch(FancyArrow(x, y, _ARROW_M * math.cos(yaw), _ARROW_M * math.sin(yaw), width=_ARROW_SHAFT_M,
                            head_width=_ARROW_HEAD_M, head_length=_ARROW_HEAD_M, length_includes_head=True,
                            facecolor=colour, edgecolor=colour, lw=0.8, joinstyle="round", zorder=6))


def _ranks(keys: dict[str, np.ndarray], band_m: float) -> np.ndarray:
    """The rank of each cell in the order the solver compares them in (the design of the constraint method,
    section 7), 0 for the cells that come first; cells that no key tells apart share a rank.

    Key by key, as the solver chooses its place: the cells of one rank are told apart by the next key, and at
    a key the solver takes with a band, those within `band_m` of the least among them stay together."""
    rank = np.zeros(len(next(iter(keys.values()))), dtype=np.int64)
    for name, values in keys.items():
        if name in solver._BANDED:
            least = np.full(rank.max() + 1, np.inf)
            np.minimum.at(least, rank, values)     # the least of the key within each rank so far
            values = np.maximum(0.0, np.ceil(np.round((values - least[rank]) / band_m, 6)) - 1.0)
        _, rank = np.unique(np.stack([rank, values], axis=1), axis=0, return_inverse=True)
        rank = rank.reshape(-1)
    return rank


def _shade_floor(ax, obs: ObservationData, result: GoalResult, th: Thresholds) -> None:
    """The floor the robot may stand on, each cell shaded by its rank: solved again from what the result
    records of the constraint method, the people and the constraints as they were stated. The grid lies in the
    robot's frame, as the solver's does."""
    stated = [use.constraint for use in result.constraints]
    standard = standard_goal(obs, result.target, th)
    grid = solver.floor(obs, result.people, th)
    # The solver's own working over those cells, in the order of the grid: the keys it compares them by.
    _, settled = solver._over_floor(obs, result.target, result.people, stated, standard, th)
    rank = _ranks(settled.keys, th.grid_resolution_m)
    shade = np.full(grid.selectable.shape, np.nan)
    shade[grid.selectable] = 1.0 - rank / max(rank.max(), 1)           # the first cells darkest
    reach = (len(grid.offsets) / 2) * th.grid_resolution_m             # the grid's edge, from the robot
    frame = Affine2D().rotate(obs.robot.yaw).translate(obs.robot.x, obs.robot.y) + ax.transData
    ax.imshow(np.ma.masked_invalid(shade), cmap=RANK, vmin=0.0, vmax=1.0, origin="lower", interpolation="nearest",
              extent=(-reach, reach, -reach, reach), transform=frame, zorder=0.5)


def _topdown(ax, truth: GroundTruth, obs: ObservationData, acceptable: list[Region],
             forbidden: list[ForbiddenRegion], result: GoalResult | None) -> None:
    th = Thresholds()
    grid = static_grid(truth, th.grid_resolution_m)
    ny, nx = grid.free.shape
    shade = np.where(grid.free, 1.0, _FREE_GREY)
    ax.imshow(shade, cmap="gray", vmin=0.0, vmax=1.0, origin="lower", zorder=0,
              extent=(grid.x0, grid.x0 + nx * grid.resolution, grid.y0, grid.y0 + ny * grid.resolution))
    points: list[tuple[float, float]] = []
    handles: list = []

    def outline(polygon, **style):
        ax.add_patch(PolygonPatch(polygon, closed=True, fill=False, **style))

    if truth.bounds:
        outline(truth.bounds, edgecolor="grey", lw=1.0, zorder=1)
    for obstacle in truth.obstacles:
        outline(obstacle, edgecolor="grey", lw=1.0, zorder=1)
    for region in forbidden:
        outline(region.polygon, edgecolor=FORBIDDEN, hatch="///", lw=1.2, zorder=2)
        points += list(region.polygon)
    for region in acceptable:
        outline(region.polygon, edgecolor=ACCEPTABLE, lw=1.8, zorder=2)
        points += list(region.polygon)
    if forbidden:
        handles.append(Patch(fill=False, hatch="///", edgecolor=FORBIDDEN, label="forbidden region"))
    if acceptable:
        handles.append(Patch(fill=False, edgecolor=ACCEPTABLE, lw=1.8, label="acceptable region"))

    for target in truth.targets.values():
        if target.footprint:
            outline(target.footprint, edgecolor=TARGET, lw=1.5, zorder=3)
            points += list(target.footprint)
        ax.plot(target.anchor[0], target.anchor[1], marker="*", ms=13, color=TARGET, mec="black", mew=0.5, zorder=4)
        ax.annotate(target.label, target.anchor[:2], xytext=(5, 5), textcoords="offset points", color="black",
                    fontsize=8, zorder=7)
        points.append(target.anchor[:2])
    if truth.targets:
        handles.append(Line2D([], [], marker="*", ls="", color=TARGET, mec="black", ms=11, label="target"))

    for person in truth.people:
        ax.add_patch(Circle((person.x, person.y), th.person_radius_m, facecolor=to_rgba(PEOPLE, _PERSON_FILL),
                            edgecolor=PEOPLE, lw=1.5, zorder=4))
        _arrow(ax, person.x, person.y, person.yaw, PEOPLE)
        ax.annotate(person.id, (person.x, person.y), xytext=(5, -12), textcoords="offset points",
                    color="black", fontsize=8, zorder=7)
        points.append((person.x, person.y))
    if truth.people:
        handles.append(Line2D([], [], marker="o", ls="", mfc=to_rgba(PEOPLE, _PERSON_FILL), mec=PEOPLE, mew=1.5,
                              ms=9, label="person"))

    robot = obs.robot
    ax.add_patch(Circle((robot.x, robot.y), th.robot_radius_m, fill=False, edgecolor=ROBOT, lw=1.8, zorder=4))
    _arrow(ax, robot.x, robot.y, robot.yaw, ROBOT)
    points.append((robot.x, robot.y))
    handles.append(Line2D([], [], marker="o", ls="", mfc="none", mec=ROBOT, ms=9, label="robot"))

    solved = result is not None and None not in (result.pose_world, result.target, result.people, result.constraints)
    if solved:                                     # a result of the constraint method: the floor it chose on
        _shade_floor(ax, obs, result, th)
    if result is not None and result.pose_world is not None:
        gx, gy, gtheta = result.pose_world
        ring = {"mec": "white", "mew": 1.5} if solved else {}          # on shaded floor the place keeps a rim
        ax.plot(gx, gy, marker="o", ms=8, color=GOAL, zorder=5, **ring)
        _arrow(ax, gx, gy, gtheta, GOAL)
        points.append((gx, gy))
        handles.append(Line2D([], [], marker="o", ls="", color=GOAL, ms=8, label="goal"))
    if result is not None and result.candidates:
        spots = [candidate.pose_world[:2] for candidate in result.candidates]
        style = dict(marker="o", ls="", ms=7, mfc="white", mec=GOAL, mew=1.5, label="runner-up")
        ax.plot(*zip(*spots), zorder=5, **style)
        for place, spot in enumerate(spots, 2):
            ax.annotate(str(place), spot, xytext=(6, 4), textcoords="offset points", color="black", fontsize=7,
                        zorder=7)
        points += spots
        handles.append(Line2D([], [], **style))
    if result is not None and result.people:       # the people the method found, each with the number it gave them
        spots = [person.xy for person in result.people]
        style = dict(marker="o", ls="", ms=11, mfc=NUMBER, mec="black", mew=1.0, label="person found, by number")
        ax.plot(*zip(*spots), zorder=6.5, **style)
        for person in result.people:
            ax.annotate(str(person.number), person.xy, ha="center", va="center", color="black", fontsize=7, zorder=7)
        points += spots
        handles.append(Line2D([], [], **style))
    if solved:
        handles.append(Patch(facecolor=RANK(0.6), edgecolor="none", label="floor to stand on (darker: better)"))

    xs, ys = zip(*points)
    ax.set_xlim(min(xs) - _MARGIN_M, max(xs) + _MARGIN_M)
    ax.set_ylim(min(ys) - _MARGIN_M, max(ys) + _MARGIN_M)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    # With the rows of the constraint method the legend is tall enough to hide the robot or a person: it then
    # goes where it covers least of what is drawn.
    method = result is not None and result.people is not None
    ax.legend(handles=handles, loc="best" if method else "upper right", fontsize=8)


def _build(path: Path, obs: ObservationData, truth: GroundTruth, acceptable: list[Region],
           forbidden: list[ForbiddenRegion], result: GoalResult | None, suptitle: str, mark) -> None:
    shown = obs.rgb.copy()
    if result is not None and result.people:       # the numbers, as the constraints step was shown them
        shown = draw_people_numbers(shown, result.people, obs.camera.T_world_camera[:2, 3])
    annotated = Image.fromarray(shown)
    mark(ImageDraw.Draw(annotated))
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    try:
        left.imshow(np.asarray(annotated, dtype=np.uint8))
        left.axis("off")
        _topdown(right, truth, obs, acceptable, forbidden, result)
        fig.suptitle(suptitle)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, format="png", dpi=_DPI)
    finally:
        plt.close(fig)


def episode_figure(path: Path, observation: ObservationData, truth: GroundTruth, ep: EpisodeGT,
                   result: GoalResult | None, score: Score | None, title: str = "") -> None:
    """The image with the target and goal marked beside a top-down plot of the episode's ground truth.

    A result of the constraint method adds, on the image, the number of each person found and the runners-up
    and, on the plot, those two and the floor the robot may stand on, shaded by the solver's order: the darker
    a cell, the earlier it comes, and the place is among the darkest."""
    parts = [title] if title else []
    if result is not None:
        parts.append(f"status={result.status}")
        if score is not None:
            parts.append(f"success={score.success}")
    _build(path, observation, truth, list(ep.acceptable_regions or []), list(ep.forbidden_regions or []), result,
           " | ".join(parts), lambda draw: result is not None and _mark_result(draw, observation, result))


def scene_figure(path: Path, observation: ObservationData, truth: GroundTruth) -> None:
    """The scene-check overlay: people, headings, anchors and the stored regions projected into the image."""
    acceptable = [r for ep in truth.episodes.values() for r in ep.acceptable_regions or []]
    forbidden = [r for ep in truth.episodes.values() for r in ep.forbidden_regions or []]
    _build(path, observation, truth, acceptable, forbidden, None, truth.scene_id,
           lambda draw: _mark_scene(draw, observation, truth))
