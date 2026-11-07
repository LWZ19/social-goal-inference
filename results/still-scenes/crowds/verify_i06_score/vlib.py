"""Independent verifier of i06_score: shared helpers (FROZEN.md sha256 a83c0ffe...693e).

Written without importing the compute agent's score/ code. Imports the sgi package, the step-0 harness and earlier
phases' code only. No model call, no network, no torch, no .env.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

WT = Path('.-scenes')
RES = WT / 'results/2026-10-17-beyond-vlm'
I06 = RES / 'phase2/i06'
OUT = I06 / 'verify_i06_score/out'
FROZEN_SHA = 'a83c0ffe0cf5fafd9f007fee478d5899c9f858d2eb77629f01885cbeccef693e'
assert hashlib.sha256((I06 / 'FROZEN.md').read_bytes()).hexdigest() == FROZEN_SHA, 'FROZEN.md changed'

sys.path.insert(0, str(WT / 'docs/model_notes/tools'))
import research_harness as H  # noqa: E402

from sgi.config import load_config  # noqa: E402
from sgi.evaluate import score as pkg_score  # noqa: E402
from sgi.freespace import connected, robot_grid  # noqa: E402
from sgi.geometry import bearing, in_image, project, to_world_xy, wrap_angle  # noqa: E402
from sgi.gt.derive import is_visible, oracle_target, reference_people_matter  # noqa: E402
from sgi.polygons import grid_points_in  # noqa: E402
from sgi.run import OraclePeople  # noqa: E402
from sgi.scene import load_scene  # noqa: E402
from sgi import solver as SOLVER  # noqa: E402
from sgi.stages.goal import standard_goal  # noqa: E402
from sgi.stages.people import number_people  # noqa: E402
from sgi.types import Constraint, GoalResult  # noqa: E402

T0 = 't0'
TOP = (('A', 6), ('B', 1), ('C', 1), ('D', 5), ('E', 2))


# ------------------------------------------------------------------------------------------------- the crowd scenes
def manifest():
    return [json.loads(line) for line in (I06 / 'build/manifest.jsonl').read_text().splitlines() if line.strip()]


def kept():
    """Every accepted entry (30 bases + accepted variants), sorted by seed, row, family, level."""
    rows = [e for e in manifest() if e['accepted']]
    rows.sort(key=lambda e: (e['seed'], e['row'], e['kind'] != 'base', e['family'], e['level']))
    return rows


def load_entry(e):
    folder = I06 / e['folder']
    H.guard_scene(folder)
    inputs, truth = load_scene(folder)
    assert inputs.scene_id == e['scene_id'], (inputs.scene_id, e['scene_id'])
    return inputs, truth


# ------------------------------------------------------------------------------------ reference constraints (I01 R5)
def _ask(kind, people, target=False):
    return {'because': 'reference constraint', 'kind': kind, 'people': list(people), 'range_m': None, 'target': target}


def reference(situation, members, needed):
    """phase1/i01_repairs/repairs.py reference_constraints, re-typed (it copies tests/test_reference_constraints)."""
    if situation == 'waiting_line':
        return [_ask('come_after', members)]
    if situation == 'conversation_group':
        out = [_ask('keep_out_of_shared_space', members)]
        if needed is not None:
            out += [_ask('visible_to', [needed]), _ask('face', [needed])]
        return out
    if situation == 'audience':
        return [_ask('keep_out_of_view', members, target=True), _ask('face', [], target=True)]
    if situation == 'no_one':
        return []
    raise ValueError(situation)


# --------------------------------------------------------------------------------------------------------- labels
def goal_of(result, obs):
    if result.status != 'ok' or result.pose_robot is None:
        return None
    return to_world_xy(obs.robot, result.pose_robot[0], result.pose_robot[1])


def l2_pass(xy, truth, group, th):
    """FROZEN 5: body gap (centre - person r - robot r) >= personal_gap_m to every non-member, and outside every other
    group's circle (centre = members' mean, radius = largest member distance), groups of >= 2."""
    if xy is None:
        return False
    for p in truth.people:
        member = group is not None and p.group == group
        if member:
            continue
        if math.dist(xy, (p.x, p.y)) - th.person_radius_m - th.robot_radius_m < th.personal_gap_m:
            return False
    others = collections.defaultdict(list)
    for p in truth.people:
        if p.group is not None and p.group != group:
            others[p.group].append((p.x, p.y))
    for pts in others.values():
        if len(pts) < 2:
            continue
        P = np.array(pts)
        c = P.mean(axis=0)
        r = float(np.max(np.hypot(*(P - c).T)))
        if math.dist(xy, c) < r:
            return False
    return True


def l2_reasons(xy, truth, group, th):
    out = []
    for p in truth.people:
        if group is not None and p.group == group:
            continue
        gap = math.dist(xy, (p.x, p.y)) - th.person_radius_m - th.robot_radius_m
        if gap < th.personal_gap_m:
            out.append((p.id, round(gap, 3)))
    return out


def lattice_l1(truth, obs, regions, th):
    """L1 cells: 0.05 m lattice points inside the acceptable polygons, free on the scorer's robot grid and joined."""
    g = robot_grid(truth, obs.robot, th)
    j = connected(g, (obs.robot.x, obs.robot.y))
    pts = {}
    for reg in regions:
        for x, y in grid_points_in(reg.polygon, th.grid_resolution_m):
            ix = g.index(float(x), float(y))
            if ix is not None and j[ix]:
                pts[(round(float(x), 6), round(float(y), 6))] = None
    return np.array(list(pts), dtype=float).reshape(-1, 2)


def l2_cells(cells, truth, group, th):
    return int(sum(l2_pass((float(x), float(y)), truth, group, th) for x, y in cells))


# ------------------------------------------------------------------------------------------------------------ arm 3
def oracle_c(inputs, truth, et, th, found=None, with_constraints=True):
    """Arm 3 (found None: OraclePeople; else [(FoundPerson, source id or None)] for arm 4). Returns a dict."""
    obs = inputs.observation
    t = truth.targets[T0]
    target = oracle_target(obs, t, th)
    if target is None:
        raise RuntimeError(f'{inputs.scene_id}: oracle target does not lift')
    out = {'target': target}
    if not reference_people_matter(et, T0):
        r = standard_goal(obs, target, th).model_copy(update={'target': target})
        out.update(result=r, matter=False, people=[], src={}, cons=[], note=None, uses=[])
        return out
    if found is None:
        idx = {(p.x, p.y): p.id for p in truth.people}
        found = [(f, idx[(f.xy[0], f.xy[1])]) for f in OraclePeople(truth, th)(obs).found]
    key = {}
    for f, s in found:
        key[(float(f.xy[0]), float(f.xy[1]), float(f.pixel[0]), float(f.pixel[1]))] = s
    people, cut = number_people([f for f, _ in found], target, obs, th, th.max_people)
    src = {d.number: key[(float(d.xy[0]), float(d.xy[1]), float(d.pixel[0]), float(d.pixel[1]))] for d in people}
    num_of = {}
    for n, s in src.items():
        if s is not None and s not in num_of:
            num_of[s] = n
    g = truth.group_for(et.episode_id, T0)
    members = [num_of[p.id] for p in truth.people if g is not None and p.group == g and p.id in num_of]
    needed = num_of.get(t.person_id) if t.person_id is not None else None
    cons = reference(et.gt.situation, members, needed) if with_constraints else []
    C = [Constraint.model_validate(c) for c in cons]
    s = SOLVER.solve(obs, target, people, C, standard_goal(obs, target, th), th)
    pix = project(obs.camera, (*s.pose_world[:2], 0.0))
    shown = pix is not None and in_image(obs.camera, *pix[:2])
    r = GoalResult(status='ok', source='constraints', pose_world=s.pose_world, pose_robot=s.pose_robot,
                   goal_pixel=pix[:2] if shown else None, flags=s.flags.model_copy(update={'people_cut': cut}),
                   people=people, constraints=s.uses, candidates=s.candidates, solver=s.note, target=target)
    out.update(result=r, matter=True, people=people, src=src, cons=cons, note=s.note, uses=s.uses,
               members=members, needed=needed)
    return out


# --------------------------------------------------------------------------------------------------------- detectors
REASON_IV = ('no cell is level with or behind', 'every cell level with or behind')


def detect(inputs, truth, et, th, c, cells):
    """Raw detector values for one solved result (FROZEN 7), not yet compared with the base."""
    obs = inputs.observation
    r = c['result']
    xy = goal_of(r, obs)
    sc = pkg_score(r, obs, truth, et.gt, th)
    g = truth.group_for(et.episode_id, T0)
    members = {p.id for p in truth.people if g is not None and p.group == g}
    note = c.get('note')
    uses = c.get('uses') or []
    src = c.get('src') or {}
    d = {'l1': bool(sc.success), 'in_region': bool(sc.in_region), 'heading_ok': bool(sc.heading_ok)}
    # (i)
    jw = [] if note is None else list(note.joined_waiting)
    d['joined'] = jw
    d['joined_src'] = [src.get(n) for n in jw]
    d['i'] = any(s is not None and s not in members for s in d['joined_src'])
    d['ca_used'] = any(u.constraint.kind == 'come_after' and u.status == 'used' for u in uses)
    # (ii)
    vis = [p for p in truth.people if is_visible(p, obs, th)]
    if len(cells):
        ok = np.ones(len(cells), bool)
        for p in vis:
            ok &= np.hypot(cells[:, 0] - p.x, cells[:, 1] - p.y) >= th.standoff_m
        d['F'] = int(ok.sum())
    else:
        d['F'] = 0
    exm = None if note is None else float(note.keys.excluded_m)
    d['excluded_m'] = exm
    d['ii'] = d['F'] == 0 and (not sc.in_region) and exm is not None and exm == 0.0
    # (iii)
    t = truth.targets[T0]
    tp = None
    if t.person_id is not None:
        tp = next((p for p in truth.people if p.id == t.person_id), None)
    grp = tp is not None and tp.group is not None and sum(q.group == tp.group for q in truth.people) >= 2
    d['iii_pre'] = bool(grp)
    if grp and xy is not None:
        off = abs(math.degrees(wrap_angle(bearing((tp.x, tp.y), xy) - tp.yaw)))
        d['off_deg'] = off
        d['iii'] = off >= 110.0
    else:
        d['off_deg'] = None
        d['iii'] = False
    # (iv)
    kov = any(u.constraint.kind == 'keep_out_of_view' and u.constraint.target and u.status == 'used' for u in uses)
    d['iv_pre'] = kov
    d['not_ahead_of'] = [] if note is None else list(note.not_ahead_of)
    d['not_ahead_reason'] = None if note is None else note.not_ahead_reason
    reason = d['not_ahead_reason'] or ''
    d['iv'] = bool(kov and not d['not_ahead_of'] and reason.startswith(REASON_IV))
    # (v)
    fl = SOLVER.floor(obs, c.get('people') or [], th)
    n_stand = 0
    for x, y in cells:
        ix = fl.index(float(x), float(y))
        if ix is not None and bool(fl.standable[ix]):
            n_stand += 1
    d['standable'] = n_stand
    d['v'] = bool(len(cells) > 0 and n_stand == 0 and not sc.in_region)
    d['far'] = bool(r.flags.far_from_target)
    d['exbr'] = bool(r.flags.exclusion_broken)
    d['tdist'] = None if note is None else float(note.target_distance_m)
    return d
