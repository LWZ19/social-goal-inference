"""What the pipeline made of a request, drawn for a person to read: the image it was sent with the target, the
people it found and the goal on it, and the reasoning of each stage as text (src/sgi/serve.py, `inference`)."""
from __future__ import annotations

import math
import textwrap
from typing import List, Optional

import cv2
import numpy as np

TARGET_COLOR, PERSON_COLOR, GOAL_COLOR, CANDIDATE_COLOR = (255, 0, 255), (0, 220, 255), (0, 0, 255), (0, 140, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def project(camera: dict, x: float, y: float, z: float = 0.0) -> Optional[tuple]:
    """The pixel of a world point in the camera the request was sent with; None behind it."""
    p = np.linalg.inv(np.array(camera["T_world_camera"])) @ np.array([x, y, z, 1.0])
    if p[2] <= 0.05:
        return None
    return camera["fx"] * p[0] / p[2] + camera["cx"], camera["fy"] * p[1] / p[2] + camera["cy"]


def _pt(pixel) -> tuple:
    return int(round(pixel[0])), int(round(pixel[1]))


def _tag(img, text: str, at: tuple, color) -> None:
    cv2.putText(img, text, at, FONT, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, at, FONT, 0.5, color, 1, cv2.LINE_AA)


def annotate(rgb: np.ndarray, camera: dict, answer: dict) -> np.ndarray:
    """The image that was sent (BGR) with what the pipeline saw in it and where it put the goal."""
    img = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]), cv2.COLOR_RGB2BGR)
    shown = answer.get("inference") or {}
    for candidate in shown.get("candidates") or []:
        x, y, _ = candidate["pose_world"]
        pixel = project(camera, x, y)
        if pixel is not None:
            cv2.circle(img, _pt(pixel), 5, CANDIDATE_COLOR, 1, cv2.LINE_AA)
    for person in shown.get("people") or []:
        c = _pt(person["pixel"])
        cv2.circle(img, c, 13, PERSON_COLOR, -1, cv2.LINE_AA)
        number = str(person["number"])
        (w, h), _ = cv2.getTextSize(number, FONT, 0.55, 2)
        cv2.putText(img, number, (c[0] - w // 2, c[1] + h // 2), FONT, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    target = shown.get("target")
    if target is not None:
        c = _pt(target["pixel"])
        cv2.circle(img, c, 18, TARGET_COLOR, 2, cv2.LINE_AA)
        cv2.drawMarker(img, c, TARGET_COLOR, cv2.MARKER_CROSS, 12, 2)
        subtasks = (shown.get("plan") or {}).get("subtasks") or []
        name = next((s.get("target") for s in subtasks if s.get("type") == "navigate" and s.get("target")), "target")
        _tag(img, name, (c[0] + 20, c[1] - 12), TARGET_COLOR)
    pose = answer.get("pose_world")
    if pose is not None:
        x, y, theta = pose
        base, tip = project(camera, x, y), project(camera, x + 0.5 * math.cos(theta), y + 0.5 * math.sin(theta))
        if base is None and shown.get("goal_pixel"):
            base = shown["goal_pixel"]
        if base is not None:
            cv2.circle(img, _pt(base), 9, GOAL_COLOR, -1, cv2.LINE_AA)
            if tip is not None:
                cv2.arrowedLine(img, _pt(base), _pt(tip), GOAL_COLOR, 3, cv2.LINE_AA, tipLength=0.3)
            _tag(img, "GOAL", (_pt(base)[0] + 12, _pt(base)[1] + 20), GOAL_COLOR)
    return img


def _wrap(text: str, width: int, indent: str = "    ") -> List[str]:
    text = str(text)
    lead = " " * (len(text) - len(text.lstrip()))           # kept: it sets an item under its heading
    return textwrap.wrap(" ".join(text.split()), width, initial_indent=lead, subsequent_indent=indent) or [""]


def describe(answer: dict, request: str, width: int = 70) -> List[str]:
    """The answer as lines of text, stage by stage."""
    shown = answer.get("inference") or {}
    lines = _wrap(f'REQUEST "{request}"  ->  {answer.get("status")}'
                  + (f' ({answer.get("source")})' if answer.get("source") else "")
                  + (f' in {answer["wall_s"]:.1f} s' if "wall_s" in answer else ""), width)
    if "error" in answer:
        return lines + _wrap(f"error: {answer['error']}", width)
    plan = shown.get("plan")
    if plan is not None:
        if plan.get("reasoning"):
            lines += _wrap(f"PLAN: {plan['reasoning']}", width)
        for i, s in enumerate(plan.get("subtasks") or [], 1):
            target = f' [target: {s["target"]}]' if s.get("target") else ""
            lines += _wrap(f"  {i}. {s['type']}: {s['description']}{target}", width, "       ")
    target = shown.get("target")
    if target is not None:
        x, y, z = target["world"]
        lines.append(f"TARGET at ({x:.2f}, {y:.2f}, {z:.2f}) m")
    check = shown.get("context_check")
    if check is not None:
        lines += _wrap(f"CONTEXT: people matter: {'YES' if check['people_matter'] else 'no'}"
                       f" - relevant: {check.get('relevant_people') or '-'}", width)
        if check.get("reasoning"):
            lines += _wrap(f"  why: {check['reasoning']}", width, "       ")
    people = shown.get("people")
    if people is not None:
        lines.append(f"PEOPLE FOUND: {len(people)}")
        for p in people:
            far = ""
            if target is not None:
                far = f", {math.hypot(p['xy'][0] - target['world'][0], p['xy'][1] - target['world'][1]):.1f} m from target"
            facing = "" if p.get("yaw") is None else f", facing {math.degrees(p['yaw']):.0f} deg"
            lines += _wrap(f"  #{p['number']} {p.get('label') or ''} at ({p['xy'][0]:.2f}, {p['xy'][1]:.2f}){far}{facing}",
                           width, "       ")
    constraints = shown.get("constraints")
    if constraints is not None:
        lines.append(f"CONSTRAINTS: {len(constraints)}")
        for use in constraints:
            c = use["constraint"]
            about = []
            if c.get("people"):
                about.append("people " + ",".join(str(n) for n in c["people"]))
            if c.get("target"):
                about.append("target")
            if c.get("range_m") and c["kind"] == "within_distance":
                about.append(f"{c['range_m']['at_least']:.1f}-{c['range_m']['at_most']:.1f} m")
            state = use["status"] + ("" if use["status"] == "used" else f": {use.get('reason')}")
            lines += _wrap(f"  [{state}] {c['kind']} ({'; '.join(about) or '-'}): {c['because']}", width, "       ")
    mark = shown.get("goal_mark")
    if mark is not None and mark.get("reasoning"):
        r = mark["reasoning"]
        lines += _wrap(f"GOAL STAGE: people: {r['people']} | expectations: {r['expectations']} | "
                       f"placement: {r['placement']}", width)
    pose = answer.get("pose_world")
    if pose is not None:
        lines.append(f"GOAL ({pose[0]:.2f}, {pose[1]:.2f}, {math.degrees(pose[2]):.0f} deg)"
                     f", {len(shown.get('candidates') or [])} runners-up")
    if shown.get("flags"):
        lines += _wrap("flags: " + ", ".join(shown["flags"]), width)
    calls = shown.get("calls") or []
    if calls:
        lines += _wrap("CALLS: " + ", ".join(f"{c['stage']} {c['status']} {c['latency_s']:.1f}s"
                                             + (" (cached)" if c.get("cached") else "") for c in calls), width)
    return lines


def text_panel(lines: List[str], width: int, height: int) -> np.ndarray:
    panel = np.full((height, width, 3), 24, np.uint8)
    y = 18
    for line in lines:
        if y > height - 6:
            cv2.putText(panel, "...", (8, height - 6), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            break
        heading = line[:1].isupper() and line.split(" ")[0].rstrip(":").isupper()
        color = (120, 220, 255) if heading else (225, 225, 225)
        cv2.putText(panel, line, (8, y), FONT, 0.45, color, 1, cv2.LINE_AA)
        y += 17
    return panel
