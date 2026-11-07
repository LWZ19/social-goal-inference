"""Marks probe M2 library: the candidate marks of a scene, the marked image, and the rule that turns an answer (a mark
number and a point) into a grounded target. No truth is read here; no model is called.

Candidates: the I09 A1 detector's stored people for the scene (phase1c/i09/a1/eval_people.json, by score), then the
largest depth components (phase1b/i11 G1, object or structure) whose mark point lies below 2.5 m and more than 0.5 m
from every detected person, up to MAX_MARKS in all. People come first so that touching people are separate marks (M1c, Q13).
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import m1_coverage as M
import paths  # noqa: E402  (results/_lib/paths.py, on sys.path through m1_coverage)

X = M.X
EVAL_PEOPLE = paths.STILL / 'people-finder/early-detector/phase1c/a1/eval_people.json'
MAX_MARKS = 8
MAX_PEOPLE = 6
DEDUP_M = 0.5
CEILING_Z = 2.5
NEAR_PX = 15.0
FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 15)
_PEOPLE = None


@dataclass
class Cand:
    kind: str                 # 'person' or 'component'
    mark: tuple[float, float]  # (u, v) pixel the label sits on and the mark point lifts from
    comp: int | None = None   # component label (components)
    box: tuple | None = None  # (u0, v0, u1, v1) (people)
    xy: tuple | None = None   # detector world (x, y) (people)


def _people(folder: str):
    global _PEOPLE
    if _PEOPLE is None:
        _PEOPLE = json.loads(EVAL_PEOPLE.read_text())['scenes']
    return _PEOPLE[folder]['people']


def candidates(folder: str, g: 'X.Geom') -> list[Cand]:
    out = []
    for p in sorted(_people(folder), key=lambda p: -p['score'])[:MAX_PEOPLE]:
        if p.get('torso_pixel') is None:
            continue
        out.append(Cand('person', tuple(p['torso_pixel']), box=tuple(p['box']), xy=(p['x'], p['y'])))
    for c, cls, size, q, tgt in M.proposer(g):
        if len(out) >= MAX_MARKS:
            break
        if tgt.world[2] > CEILING_Z:               # a mark on the ceiling names nothing a robot goes to
            continue
        if any(math.hypot(tgt.world[0] - o.xy[0], tgt.world[1] - o.xy[1]) <= DEDUP_M for o in out if o.kind == 'person'):
            continue
        out.append(Cand('component', (q[1] + 0.5, q[0] + 0.5), comp=c))
    return out


def numbering(n: int, scene_id: str, trial: int) -> list[int]:
    """The number shown on each candidate: 1..n shuffled, seeded by scene and trial (binding check, Q12)."""
    nums = list(range(1, n + 1))
    random.Random(f'm2|{scene_id}|{trial}').shuffle(nums)
    return nums


def render(rgb: np.ndarray, cands: list[Cand], nums: list[int]) -> np.ndarray:
    """The view with a small white label carrying each candidate's number, centred on its mark pixel."""
    im = Image.fromarray(np.asarray(rgb).astype(np.uint8)).convert('RGB')
    d = ImageDraw.Draw(im)
    for cand, n in zip(cands, nums):
        u, v = cand.mark
        text = str(n)
        l, t, r, b = d.textbbox((0, 0), text, font=FONT)
        w, h = r - l + 8, b - t + 6
        x0, y0 = int(round(u - w / 2)), int(round(v - h / 2))
        d.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=4, fill=(255, 255, 255), outline=(0, 0, 0), width=2)
        d.text((x0 + 4 - l, y0 + 3 - t), text, fill=(0, 0, 0), font=FONT)
    return np.asarray(im)


def _lift_click(g, u, v):
    from sgi.geometry import lift
    w = lift(g.depth, g.cam, u, v, window=5)
    if w is None:
        return None
    return X.make_target(g.cam, int(v), int(u), w)


def _person_target(g, cand: Cand):
    u, v = cand.mark
    return _lift_click(g, u, v)


def _dist_to_cand(g, cand: Cand, u: float, v: float) -> float:
    if cand.kind == 'person':
        u0, v0, u1, v1 = cand.box
        du = max(u0 - u, 0.0, u - u1)
        dv = max(v0 - v, 0.0, v - v1)
        return math.hypot(du, dv)
    cp = g.comps('G1')
    ii, jj = np.nonzero(cp.label == cand.comp)
    return float(np.sqrt(((ii + 0.5 - v) ** 2 + (jj + 0.5 - u) ** 2).min()))


def resolve(g, cands: list[Cand], nums: list[int], mark: int | None, point_px: tuple | None, rule: str = 'both'):
    """The grounded target of an answer, by rule:
    - 'both' (the M2 design): with a number, keep the click when it lies within NEAR_PX of that candidate (moved
      into a component when just outside it; a click in a person's box whose lift lands more than 0.6 m from the
      detected person is replaced by the person's mark); otherwise the candidate's own mark point. Without a
      number, the click as now.
    - 'number': the numbered candidate's mark point; the click only without a number.
    - 'point': the click alone, as the plan answers now.
    Returns (target or None, how)."""
    by_num = {n: c for c, n in zip(cands, nums)}
    cand = by_num.get(mark) if mark is not None else None
    if rule == 'point' or cand is None:
        if point_px is None:
            return None, 'no point'
        return _lift_click(g, *point_px), 'click'
    if rule == 'number' or point_px is None:
        return (_person_target(g, cand) if cand.kind == 'person'
                else X.comp_lift(g, g.comps('G1'), cand.comp, (int(cand.mark[1]), int(cand.mark[0])))), 'mark'
    u, v = point_px
    if _dist_to_cand(g, cand, u, v) <= NEAR_PX:
        if cand.kind == 'person':
            t = _lift_click(g, u, v)
            if t is not None and math.hypot(t.world[0] - cand.xy[0], t.world[1] - cand.xy[1]) <= 0.6:
                return t, 'click in person'
            return _person_target(g, cand), 'person mark (click off body)'
        cp = g.comps('G1')
        ui, vi = int(u), int(v)
        if 0 <= vi < g.h and 0 <= ui < g.w and cp.label[vi, ui] == cand.comp:
            return X.comp_lift(g, cp, cand.comp, (vi, ui)), 'click in component'
        ii, jj = np.nonzero(cp.label == cand.comp)
        k = int(np.argmin((ii + 0.5 - v) ** 2 + (jj + 0.5 - u) ** 2))
        return X.comp_lift(g, cp, cand.comp, (int(ii[k]), int(jj[k]))), 'click snapped into component'
    return (_person_target(g, cand) if cand.kind == 'person'
            else X.comp_lift(g, g.comps('G1'), cand.comp, (int(cand.mark[1]), int(cand.mark[0])))), 'mark (click elsewhere)'
