"""I06 scoring library (FROZEN.md sha256 a83c0ffe...693e, sections 5-9). Research code: imports the sgi package and
changes nothing in it. No model, no network, no torch. Recorded runs and whitelisted scenes are read only through
docs/model_notes/tools/research_harness.py; the crowd scenes are this probe's own (phase2/i06/scenes), guarded with
H.guard_scene before loading.

Arms (FROZEN section 6), per scene and episode:
  arm1  full oracle: run.run_episode, grounding oracle, context_check always_yes, goal oracle, mock entries, no cache.
  arm2  solver alone: run.run_episode, grounding oracle, context_check oracle, goal constraints, people oracle,
        constraints none (no call).
  arm3  oracle C: derive.oracle_target, reference_people_matter(et, "t0"), run.OraclePeople + number_people, the
        reference constraints (I01 R5's copy of tests/test_reference_constraints.reference_constraints), solver.solve,
        a GoalResult built as ConstraintInferrer.infer builds it, scored by sgi.evaluate.score.
  arm3e arm3 with an empty constraint list (the equivalence check against arm2, to 1e-9 m).
  arm4  noisy C: arm3 with simulated people (noise.py).
Labels (section 5): L1 = sgi.evaluate.score success; L2 = L1 and the goal has body gap >= personal_gap_m to every
non-member and lies outside every other group's circle.
"""
from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

import numpy as np

WT = Path('.-scenes')
RES = WT / 'results/2026-10-17-beyond-vlm'
I06 = RES / 'phase2/i06'
BUILD = I06 / 'build'
SCORE = I06 / 'score'
FROZEN_SHA = 'a83c0ffe0cf5fafd9f007fee478d5899c9f858d2eb77629f01885cbeccef693e'
sys.path.insert(0, str(WT / 'docs/model_notes/tools'))
import research_harness as H  # noqa: E402

from sgi.config import load_config  # noqa: E402
from sgi.evaluate import score  # noqa: E402
from sgi.freespace import connected, robot_grid  # noqa: E402
from sgi.geometry import bearing, in_image, project, to_world_xy, wrap_angle  # noqa: E402
from sgi.gt.derive import is_visible, oracle_target, prepare, reference_people_matter  # noqa: E402
from sgi.polygons import grid_points_in  # noqa: E402
from sgi.run import OraclePeople  # noqa: E402
from sgi.scene import load_scene  # noqa: E402
from sgi import solver as S  # noqa: E402
from sgi.stages.goal import standard_goal  # noqa: E402
from sgi.stages.people import number_people  # noqa: E402
from sgi.types import Constraint, Flags, GoalResult  # noqa: E402

TARGET_ID = 't0'
REASON_IV = ('no cell is level with or behind', 'every cell level with or behind')
OFF_FACING_DEG = 110.0


# ---------------------------------------------------------------------------------------------- reference constraints
def ask(kind, *people, target=False):
    # phase1/i01_repairs/repairs.py ask (tests/solver_layouts.ask, copied): one constraint of an answer.
    return dict(because="reference constraint", kind=kind, people=list(people), range_m=None, target=target)


def reference_constraints(situation, members, needed):
    """phase1/i01_repairs/repairs.py reference_constraints (tests/test_reference_constraints.py, copied); `needed`
    None -> the two that name the needed person are omitted. Circular: reads the situation label and the true group
    (REPORT section 8). It is the oracle answer, not a method."""
    if situation == "waiting_line":
        return [ask("come_after", *members)]
    if situation == "conversation_group":
        return [ask("keep_out_of_shared_space", *members)] + (
            [] if needed is None else [ask("visible_to", needed), ask("face", needed)])
    if situation == "audience":
        return [ask("keep_out_of_view", *members, target=True), ask("face", target=True)]
    assert situation == "no_one", situation
    return []


# ------------------------------------------------------------------------------------------------------------ scenes
def manifest_entries():
    rows = [json.loads(l) for l in (BUILD / 'manifest.jsonl').read_text().splitlines() if l]
    return rows


def kept_scenes():
    """(entry) for every base and every accepted variant, bases first per seed."""
    out = [e for e in manifest_entries() if e['accepted']]
    out.sort(key=lambda e: (e['seed'], e['kind'] != 'base', e['row'], e['family'], e['level']))
    return out


def load(entry):
    folder = I06 / entry['folder']
    H.guard_scene(folder)
    inputs, truth = load_scene(folder)
    assert inputs.scene_id == entry['scene_id'], (inputs.scene_id, entry['scene_id'])
    return inputs, truth


def config_variants():
    cfg = load_config()
    run = cfg.run
    c1 = cfg.model_copy(update={'run': run.model_copy(update={'grounding': 'oracle', 'context_check': 'always_yes',
                                                             'goal': 'oracle', 'use_cache': False, 'trials': 1})})
    c2 = cfg.model_copy(update={'run': run.model_copy(update={'grounding': 'oracle', 'context_check': 'oracle',
                                                             'goal': 'constraints', 'people': 'oracle',
                                                             'constraints': 'none', 'use_cache': False, 'trials': 1})})
    return cfg, c1, c2


# ---------------------------------------------------------------------------------------------------- label helpers
def goal_xy(result, observation):
    if result.status != 'ok' or result.pose_robot is None:
        return None
    return to_world_xy(observation.robot, result.pose_robot[0], result.pose_robot[1])


def l2_ok(xy, truth, group_id, th):
    """FROZEN section 5: body gap >= personal_gap_m to every non-member and outside every other group's circle."""
    if xy is None:
        return False
    for p in truth.people:
        if p.group != group_id or group_id is None:
            if math.hypot(xy[0] - p.x, xy[1] - p.y) - th.person_radius_m - th.robot_radius_m < th.personal_gap_m:
                return False
    groups = collections.defaultdict(list)
    for p in truth.people:
        if p.group is not None and p.group != group_id:
            groups[p.group].append(p)
    for gid, mem in groups.items():
        if len(mem) >= 2:
            cx, cy = float(np.mean([p.x for p in mem])), float(np.mean([p.y for p in mem]))
            radius = max(math.hypot(p.x - cx, p.y - cy) for p in mem)
            if math.hypot(xy[0] - cx, xy[1] - cy) < radius:
                return False
    return True


def l2_fail_reason(xy, truth, group_id, th):
    out = []
    if xy is None:
        return ['no pose']
    for p in truth.people:
        if p.group != group_id or group_id is None:
            gap = math.hypot(xy[0] - p.x, xy[1] - p.y) - th.person_radius_m - th.robot_radius_m
            if gap < th.personal_gap_m:
                out.append(f'gap {gap:.3f} m to {p.id}')
    groups = collections.defaultdict(list)
    for p in truth.people:
        if p.group is not None and p.group != group_id:
            groups[p.group].append(p)
    for gid, mem in groups.items():
        if len(mem) >= 2:
            cx, cy = float(np.mean([p.x for p in mem])), float(np.mean([p.y for p in mem]))
            radius = max(math.hypot(p.x - cx, p.y - cy) for p in mem)
            if math.hypot(xy[0] - cx, xy[1] - cy) < radius:
                out.append(f'inside circle of {gid}')
    return out


def l1_cells(truth, observation, regions, th) -> np.ndarray:
    """build/label_checks.py l1_cells, copied: 0.05 m lattice points in the polygons, free on the scorer's robot grid,
    joined to the robot."""
    grid = robot_grid(truth, observation.robot, th)
    joined = connected(grid, (observation.robot.x, observation.robot.y))
    cells = set()
    for region in regions:
        for x, y in grid_points_in(region.polygon, th.grid_resolution_m):
            index = grid.index(float(x), float(y))
            if index is not None and joined[index]:
                cells.add((round(float(x), 9), round(float(y), 9)))
    return np.array(sorted(cells), dtype=float).reshape(-1, 2)


# ------------------------------------------------------------------------------------------------------------- arm 3
def solve_c(inputs, truth, et, th, people_found=None, constraints='reference', members_of=None, needed_of=None,
            target=None):
    """Arm 3 (or arm 4 with `people_found` simulated): returns dict with result, people, uses, note, flags, inputs.

    people_found: list of (FoundPerson, source id or None); None -> OraclePeople (true people, source by exact xy).
    constraints: 'reference' or 'empty'."""
    obs = inputs.observation
    t = truth.targets[TARGET_ID]
    if target is None:
        target = oracle_target(obs, t, th)
    assert target is not None, inputs.scene_id
    matter = reference_people_matter(et, TARGET_ID)
    out = {'people_matter': bool(matter), 'target': target}
    if not matter:
        res = standard_goal(obs, target, th)
        out.update(result=res.model_copy(update={'target': target}), people=None, uses=None, note=None,
                   flags=res.flags, source_of={}, constraints=[])
        return out
    if people_found is None:
        found = OraclePeople(truth, th)(obs).found
        by_xy = {(p.x, p.y): p.id for p in truth.people}
        tagged = [(f, by_xy[tuple(f.xy)]) for f in found]
    else:
        tagged = people_found
    src = {}
    for f, sid in tagged:
        src[(tuple(map(float, f.xy)), tuple(map(float, f.pixel)))] = sid
    people, cut = number_people([f for f, _ in tagged], target, obs, th, th.max_people)
    source_of = {d.number: src[(tuple(map(float, d.xy)), tuple(map(float, d.pixel)))] for d in people}
    group = truth.group_for(et.episode_id, TARGET_ID)
    member_ids = set() if group is None else {p.id for p in truth.people if p.group == group}
    members = [n for n, sid in sorted(source_of.items()) if sid in member_ids]
    needed = next((n for n, sid in sorted(source_of.items()) if t.person_id is not None and sid == t.person_id), None)
    if constraints == 'reference':
        cons = [Constraint.model_validate(c) for c in reference_constraints(et.gt.situation, members, needed)]
    else:
        cons = []
    solved = S.solve(obs, target, people, cons, standard_goal(obs, target, th), th)
    shown = project(obs.camera, (*solved.pose_world[:2], 0.0))
    in_picture = shown is not None and in_image(obs.camera, *shown[:2])
    res = GoalResult(status='ok', source='constraints', pose_world=solved.pose_world, pose_robot=solved.pose_robot,
                     goal_pixel=shown[:2] if in_picture else None,
                     flags=solved.flags.model_copy(update={'people_cut': cut}), people=people,
                     constraints=solved.uses, candidates=solved.candidates, solver=solved.note, target=target)
    out.update(result=res, people=people, uses=solved.uses, note=solved.note, flags=res.flags, source_of=source_of,
               constraints=[c.model_dump() for c in cons], members=members, needed=needed)
    return out


# --------------------------------------------------------------------------------------------------------- detectors
def ahead_of_hindmost(xy, truth, group_id, target_xy, visible_ids):
    """How far the place is ahead of the hindmost visible member, along the members' middle -> target direction
    (the solver's own measure, here with all visible members; FROZEN section 7 (iv) 'also recorded')."""
    mem = [p for p in truth.people if p.group == group_id and p.id in visible_ids]
    if not mem or xy is None:
        return None
    P = np.array([(p.x, p.y) for p in mem])
    to_t = np.asarray(target_xy) - P.mean(axis=0)
    g = float(np.hypot(*to_t))
    if g < 1e-9:
        return None
    along = to_t / g
    before = (np.asarray(target_xy) - P) @ along
    cell_before = float((np.asarray(target_xy) - np.asarray(xy)) @ along)
    return round(max(0.0, float(before.max()) - cell_before), 3)


def detectors(inputs, truth, et, th, c, cells_e0):
    """Per-scene detector values (not yet differenced against the base). `c` is solve_c's output (arm 3 or 4) or a
    dict with result/note for arm 2."""
    obs = inputs.observation
    res = c['result']
    xy = goal_xy(res, obs)
    note = c.get('note')
    group = truth.group_for(et.episode_id, TARGET_ID)
    vis = [p for p in truth.people if is_visible(p, obs, th)]
    vis_ids = {p.id for p in vis}
    member_ids = {p.id for p in truth.people if group is not None and p.group == group}
    sc = score(res, obs, truth, et.gt, th)
    d = {'in_region': bool(sc.in_region), 'success_l1': bool(sc.success)}
    # (i) joined_waiting holds a non-member
    joined = [] if note is None else list(note.joined_waiting)
    src = c.get('source_of') or {}
    joined_src = [src.get(n) for n in joined]
    d['joined_waiting'] = joined
    d['joined_sources'] = joined_src
    d['det_i'] = any(s is not None and s not in member_ids for s in joined_src)
    d['det_i_false_person'] = any(s is None for s in joined_src) if joined else False
    d['come_after_used'] = bool(c.get('uses')) and any(
        u.constraint.kind == 'come_after' and u.status == 'used' for u in c['uses'])
    # (ii) a crowd keeps the robot out
    if len(cells_e0):
        far = np.ones(len(cells_e0), dtype=bool)
        cover = []
        for p in vis:
            dist = np.hypot(cells_e0[:, 0] - p.x, cells_e0[:, 1] - p.y)
            far &= dist >= th.standoff_m
            if (dist < th.standoff_m).any():
                cover.append(p)
        d['F_cells'] = int(far.sum())
        pair = [math.hypot(a.x - b.x, a.y - b.y) for i, a in enumerate(cover) for b in cover[i + 1:]]
        d['min_pair_covering_m'] = round(min(pair), 3) if pair else None
    else:
        d['F_cells'] = 0
        d['min_pair_covering_m'] = None
    d['excluded_m'] = None if note is None else float(note.keys.excluded_m)
    d['target_distance_m'] = None if note is None else round(float(note.target_distance_m), 3)
    d['det_ii_core'] = (not sc.in_region) and d['excluded_m'] is not None and d['excluded_m'] == 0.0
    # (iii) large-circle off-facing
    t = truth.targets[TARGET_ID]
    tp = next((p for p in truth.people if t.person_id is not None and p.id == t.person_id), None)
    in_group = tp is not None and tp.group is not None and sum(p.group == tp.group for p in truth.people) >= 2
    d['iii_applies'] = bool(in_group)
    if in_group and xy is not None:
        off = abs(math.degrees(wrap_angle(bearing((tp.x, tp.y), xy) - tp.yaw)))
        d['off_facing_deg'] = round(off, 2)
        d['det_iii'] = off >= OFF_FACING_DEG
    else:
        d['off_facing_deg'] = None
        d['det_iii'] = False
    # (iv) the hindmost watcher rule dropped silently
    kov = bool(c.get('uses')) and any(u.constraint.kind == 'keep_out_of_view' and u.constraint.target
                                      and u.status == 'used' for u in c['uses'])
    d['iv_applies'] = kov
    d['not_ahead_of'] = [] if note is None else list(note.not_ahead_of)
    d['not_ahead_reason'] = None if note is None else note.not_ahead_reason
    d['det_iv_core'] = kov and not d['not_ahead_of'] and bool(d['not_ahead_reason']) and \
        d['not_ahead_reason'].startswith(REASON_IV)
    target_xy = c['target'].world[:2] if c.get('target') is not None else t.anchor[:2]
    d['ahead_of_hindmost_m'] = ahead_of_hindmost(xy, truth, group, target_xy, vis_ids) if kov else None
    # (v) the region's floor is not standable on the solver's own floor
    people = c.get('people') or []
    fl = S.floor(obs, people, th)
    stand = 0
    for x, y in cells_e0:
        ix = fl.index(float(x), float(y))
        if ix is not None and fl.standable[ix]:
            stand += 1
    d['l1_cells_standable'] = stand
    d['det_v'] = (len(cells_e0) > 0 and stand == 0) and not sc.in_region
    # flags
    fl_ = res.flags
    d['far_from_target'] = bool(fl_.far_from_target)
    d['exclusion_broken'] = bool(fl_.exclusion_broken)
    return d


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)
