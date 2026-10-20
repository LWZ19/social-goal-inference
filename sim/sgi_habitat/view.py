"""What the window shows: the watching camera, the robot's RGB and depth, a map from above, and a status line."""
from __future__ import annotations

import math

import cv2
import numpy as np

from sgi_habitat.frames import to_hab
from sgi_habitat.inference import CANDIDATE_COLOR, TARGET_COLOR, text_panel

MAP_RES = 0.04                     # metres per map pixel
PANEL_H = 480
INFERENCE_H = 360                  # the row under the panels that shows the last answer
STATUS_H = 56
ROBOT_COLOR, GOAL_COLOR, PATH_COLOR = (255, 160, 0), (0, 0, 255), (0, 200, 255)    # BGR
PERSON_COLOR, QUEUE_COLOR, SLOT_COLOR = (60, 200, 60), (60, 60, 230), (160, 160, 160)
STANDING_COLOR = (200, 80, 200)    # someone at a place of their own, such as a barista


class Map:
    """The navmesh from above, in Habitat's x (right) and z (down)."""

    def __init__(self, pathfinder, floor_y: float, area=None):
        """`area`, (x0, x1, y0, y1) in the pipeline's world, crops the map to that part of the scene."""
        self.floor_y = floor_y
        lo, _ = pathfinder.get_bounds()
        self.x0, self.z0 = float(lo[0]), float(lo[2])
        free = pathfinder.get_topdown_view(MAP_RES, floor_y + 0.15)
        if area is not None:
            x0, x1, y0, y1 = area                  # Habitat's z runs from -y1 to -y0
            cols = slice(max(0, int((x0 - self.x0) / MAP_RES)), int((x1 - self.x0) / MAP_RES))
            rows = slice(max(0, int((-y1 - self.z0) / MAP_RES)), int((-y0 - self.z0) / MAP_RES))
            free = free[rows, cols]
            self.x0, self.z0 = self.x0 + cols.start * MAP_RES, self.z0 + rows.start * MAP_RES
        self.base = np.where(free[..., None], np.uint8(235), np.uint8(70)).repeat(3, axis=2)

    def px(self, p_hab) -> tuple:
        return int((p_hab[0] - self.x0) / MAP_RES), int((p_hab[2] - self.z0) / MAP_RES)

    def world_px(self, x: float, y: float) -> tuple:
        return self.px(to_hab(x, y, self.floor_y))

    def world_at(self, u: float, v: float) -> tuple:
        """The world (x, y) at a pixel of the base map: the inverse of world_px."""
        return self.x0 + (u + 0.5) * MAP_RES, -(self.z0 + (v + 0.5) * MAP_RES)


def _arrow(img, origin, yaw: float, length: int, color, thickness: int = 2) -> None:
    # On the map, Habitat's x is right and z is down, so the pipeline's +y (Habitat's -z) is up.
    tip = (int(origin[0] + length * math.cos(yaw)), int(origin[1] - length * math.sin(yaw)))
    cv2.arrowedLine(img, origin, tip, color, thickness, tipLength=0.35)


def draw_map(demo) -> np.ndarray:
    m = demo.map
    img = m.base.copy()
    for line in demo.queues.values():
        for slot in line.slots:
            cv2.circle(img, m.world_px(slot.x, slot.y), 4, SLOT_COLOR, 1)
    robot = demo.robot
    if robot.path:
        points = [m.px(robot.position_hab)] + [m.px(p) for p in robot.path]
        cv2.polylines(img, [np.array(points, np.int32)], False, PATH_COLOR, 2)
    if robot.goal is not None and robot.state != "idle":
        g = m.world_px(robot.goal.x, robot.goal.y)
        cv2.circle(img, g, 6, GOAL_COLOR, 2)
        _arrow(img, g, robot.goal.yaw, 16, GOAL_COLOR)
    if demo.drag is not None:                      # a click on the map being dragged to a heading
        start, end = (m.world_px(*xy) for xy in demo.drag)
        cv2.circle(img, start, 6, GOAL_COLOR, 1)
        if start != end:
            cv2.arrowedLine(img, start, end, GOAL_COLOR, 1, tipLength=0.3)
    shown = (demo.answer or {}).get("inference") or {}
    for candidate in shown.get("candidates") or []:            # the pipeline's runners-up
        cv2.circle(img, m.world_px(*candidate["pose_world"][:2]), 3, CANDIDATE_COLOR, -1)
    for person in shown.get("people") or []:                    # where the model put the people it found
        c = m.world_px(*person["xy"])
        cv2.circle(img, c, 9, (0, 200, 230), 1, cv2.LINE_AA)
        cv2.putText(img, str(person["number"]), (c[0] + 8, c[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 150, 190), 1)
    if shown.get("target"):
        cv2.drawMarker(img, m.world_px(*shown["target"]["world"][:2]), TARGET_COLOR, cv2.MARKER_TILTED_CROSS, 12, 2)
    for human in demo.humans:
        x, y = human.world_xy
        color = QUEUE_COLOR if human.queue is not None and human in human.queue.members else PERSON_COLOR
        if human.home is not None:
            color = STANDING_COLOR
        c = m.world_px(x, y)
        cv2.circle(img, c, 6, color, -1)
        _arrow(img, c, human.yaw, 13, color, 2)
    x, y, yaw = robot.world_pose
    c = m.world_px(x, y)
    cv2.circle(img, c, 8, ROBOT_COLOR, -1)
    _arrow(img, c, yaw, 18, ROBOT_COLOR, 3)
    scale = PANEL_H / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * scale), PANEL_H), interpolation=cv2.INTER_NEAREST), scale


def _depth_image(depth: np.ndarray, size: tuple) -> np.ndarray:
    shown = np.clip(depth / 6.0, 0, 1)
    shown = cv2.applyColorMap((255 * (1 - shown)).astype(np.uint8), cv2.COLORMAP_TURBO)
    shown[depth <= 0] = 0
    return cv2.resize(shown, size)


def _label(img, text: str) -> None:
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def compose(frames: dict, demo) -> tuple:
    """The window's picture, BGR, and where its clickable panels lie in it: {"rgb": (x, y, w, h),
    "map": (x, y, w, h, scale)}, scale being window pixels per base-map pixel."""
    view = cv2.cvtColor(np.ascontiguousarray(frames["view"][..., :3]), cv2.COLOR_RGB2BGR)
    view = cv2.resize(view, (int(view.shape[1] * PANEL_H / view.shape[0]), PANEL_H))
    _label(view, "watching camera")
    half = (PANEL_H * 4 // 6, PANEL_H // 2)
    rgb = cv2.resize(cv2.cvtColor(np.ascontiguousarray(frames["rgb"][..., :3]), cv2.COLOR_RGB2BGR), half)
    _label(rgb, "robot RGB (sent to sgi)")
    depth = _depth_image(frames["depth"], half)
    _label(depth, "robot depth")
    robot_column = np.vstack([rgb, depth])
    plan, scale = draw_map(demo)
    top = np.hstack([view, robot_column, plan])
    layout = {"rgb": (view.shape[1], 0, half[0], half[1]),
              "map": (view.shape[1] + half[0], 0, plan.shape[1], plan.shape[0], scale)}
    status = np.full((STATUS_H, top.shape[1], 3), 30, np.uint8)
    line1 = f"t={demo.t:6.1f}s  robot: {demo.robot.state}   {demo.status}"
    hint = "click the map (drag for heading) or the robot RGB to send Spot there"
    cv2.putText(status, hint, (top.shape[1] - 9 * len(hint) + 40, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(status, line1[:160], (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if demo.last_request:
        line2 = f"request: {demo.last_request}" + ("   (waiting for the models...)" if demo.busy else "")
        cv2.putText(status, line2[:160], (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1, cv2.LINE_AA)
    rows = [top]
    if demo.show_inference:                        # kept in place before the first answer, so the window keeps its size
        cam = demo.camera
        width = int(cam.width * INFERENCE_H / cam.height)
        if demo.inference_image is None:
            shot = np.full((INFERENCE_H, width, 3), 40, np.uint8)
            lines = ["No answer yet. Type a request in the terminal: what the pipeline makes of it shows here.",
                     "", "Clicks on the map or the robot's image move Spot without asking the pipeline."]
        else:
            shot, lines = cv2.resize(demo.inference_image, (width, INFERENCE_H)), demo.inference_lines
        _label(shot, "what the pipeline saw (i hides)")
        rows.append(np.hstack([shot, text_panel(lines, top.shape[1] - width, INFERENCE_H)]))
    return np.vstack(rows + [status]), layout
