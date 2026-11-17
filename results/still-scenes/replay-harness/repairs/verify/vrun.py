"""Independent verifier for i01_repairs (written before reading the first agent's scripts).

Replays every recorded C solver step through H.resolve with one input repaired at a time and writes one JSON row per
record to vrows_<wording>_<model>.jsonl. Definitions (from REPORT #### I01 and the verifier task):
  matching  sgi.run._matched_people(detected, visible true people, th) (0.5 m); visible = sgi.gt.derive.is_visible.
  R0  constraints = []
  R1  matched detections moved to the true xy (number, yaw, pixel, label kept)
  R2  matched detections given the true yaw
  R3  unmatched visible true people appended as new people (numbers max+1.., truth order, pixel = projection of the
      point person_mark_height_m above them, skipped when it does not project, as OraclePeople does); constraints kept
  R4  unmatched detections dropped; their numbers removed from every constraint's people list (constraint kept even
      if its list empties). R4s: the same with the numbers left in.
  P   R1+R2+R3+R4.  Ps: with numbers left in.
  R5  recorded people; constraints = tests/test_reference_constraints.reference_constraints, members = numbers of
      detections matched to visible members of the group at the chosen target, needed = number matched to the chosen
      target's person (visible_to / face left out when None).
  R6  oracle_target of the chosen target, recorded people and constraints.
  chosen = score.matched_target_id if target_correct, else the episode's first target id.
  P1m/R3_1m/R4_1m: the same with matching width 1.0 m.
Records that are not solver steps keep their recorded verdict. R6x additionally gives non-solver ok records the
standard goal of the oracle target.
"""
import collections
import json
import math
import sys
from multiprocessing import Pool

sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H  # noqa: E402

OUT = 'results/still-scenes/replay-harness/repairs/verify'
ARMS = ('real', 'staffed', 'plain')
CELLS = [('old', 's19', 'qwen38_low'), ('old', 's19', 'glm53'), ('old', 's20', 'dsv41'),
         ('new', 's21', 'qwen38_low'), ('new', 's23', 'glm53'), ('new', 's23', 'dsv41'),
         ('new', 's23', 'haiku55'), ('new', 's23', 'sonnet55_low'), ('new', 's23', 'gemini38_low')]


def ask(kind, *people, target=False):
    return dict(because="reference", kind=kind, people=list(people), range_m=None, target=target)


def reference_constraints(situation, members, needed):
    if situation == "waiting_line":
        return [ask("come_after", *members)]
    if situation == "conversation_group":
        out = [ask("keep_out_of_shared_space", *members)]
        if needed is not None:
            out += [ask("visible_to", needed), ask("face", needed)]
        return out
    if situation == "audience":
        return [ask("keep_out_of_view", *members, target=True), ask("face", target=True)]
    assert situation == "no_one", situation
    return []


def cell(args):
    wording, stage, model = args
    from sgi.gt.derive import is_visible, oracle_target
    from sgi.geometry import project
    from sgi.run import _matched_people
    from sgi.stages.goal import standard_goal
    from sgi.types import DetectedPerson
    rows = []
    for arm in ARMS:
        refs = H.runs(stage=stage, arm=arm, model=model, method='con')
        assert len(refs) == 1, (stage, arm, model, refs)
        ref = refs[0]
        th = H.recorded_config(ref)[1].thresholds
        th1 = th.model_copy(update={'person_radius_m': 0.5})
        for rec in H.records(ref):
            row = dict(wording=wording, model=model, arm=arm, scene=rec.scene_id, episode=rec.episode_id,
                       trial=rec.trial, situation=rec.situation, C=bool(rec.score.get('success')),
                       target_correct=bool(rec.score.get('target_correct')), status=rec.result.get('status'),
                       solver_step=rec.solver_step, in_region=rec.score.get('in_region'),
                       reachable=rec.score.get('reachable'))
            sc = H.scene(ref, rec.scene_id)
            obs, truth = sc.observation, sc.truth
            ep = truth.episodes[rec.episode_id]
            chosen = rec.score.get('matched_target_id') if row['target_correct'] else None
            chosen = chosen or ep.target_ids[0]
            ot = oracle_target(obs, truth.targets[chosen], th)
            row['oracle_none'] = ot is None
            if not rec.solver_step:
                # R6x: standard goal of the oracle target on non-solver ok records
                r6x = row['C']
                if rec.result.get('status') == 'ok' and ot is not None:
                    g = standard_goal(obs, ot, th)
                    if g.pose_world is not None:
                        r6x = bool(H.score_pose(rec, g.pose_world, target=ot).success)
                row['R6x'] = r6x
                rows.append(row)
                continue
            vis = [p for p in truth.people if is_visible(p, obs, th)]
            det = [DetectedPerson.model_validate(p) for p in rec.people]
            cons = [u['constraint'] for u in rec.constraints]
            row['come_after'] = sorted({n for c in cons if c['kind'] == 'come_after' for n in c['people']})
            row['n_det'] = len(det)
            row['n_vis'] = len(vis)
            base = H.resolve(rec)
            row['reproduced'] = base.reproduced
            row['moved_m'] = base.moved_m
            row['replayC'] = bool(base.score.success)
            row['R0'] = bool(H.resolve(rec, constraints=[]).score.success)

            def build(width_th, tag):
                matched = _matched_people(det, vis, width_th)       # number -> true id
                byid = {p.id: p for p in vis}
                found = set(matched.values())
                missed = [p for p in vis if p.id not in found]
                spurious = [d.number for d in det if d.number not in matched]
                nxt = max([d.number for d in det], default=0) + 1
                added = []
                for p in missed:
                    m = project(obs.camera, (p.x, p.y, th.person_mark_height_m))
                    if m is None:
                        continue
                    added.append(DetectedPerson(number=nxt, xy=(p.x, p.y), yaw=p.yaw, pixel=tuple(m[:2]), label=None))
                    nxt += 1

                def mod(d, pos, yaw):
                    t = byid.get(matched.get(d.number))
                    if t is None:
                        return d
                    upd = {}
                    if pos:
                        upd['xy'] = (t.x, t.y)
                    if yaw:
                        upd['yaw'] = t.yaw
                    return d.model_copy(update=upd)

                drop = set(spurious)
                cons_dropped = [dict(c, people=[n for n in c['people'] if n not in drop]) for c in cons]
                kept = [d for d in det if d.number not in drop]
                arms = {
                    'R1': ([mod(d, True, False) for d in det], cons),
                    'R2': ([mod(d, False, True) for d in det], cons),
                    'R3': (det + added, cons),
                    'R4': (kept, cons_dropped),
                    'R4s': (kept, cons),
                    'P': ([mod(d, True, True) for d in kept] + added, cons_dropped),
                    'Ps': ([mod(d, True, True) for d in kept] + added, cons),
                }
                if tag:
                    arms = {k + tag: v for k, v in arms.items() if k in ('R3', 'R4', 'P')}
                out = {}
                for k, (people, cc) in arms.items():
                    out[k] = bool(H.resolve(rec, people=people, constraints=cc).score.success)
                info = dict(matched=matched, missed=[p.id for p in missed], spurious=spurious,
                            added=len(added), unprojectable=len(missed) - len(added))
                return out, info

            res, info = build(th, '')
            row.update(res)
            row['n_matched'] = len(info['matched'])
            row['missed'] = info['missed']
            row['spurious'] = info['spurious']
            row['unprojectable'] = info['unprojectable']
            # distance of each spurious detection to nearest visible true person
            row['spur_dist'] = {str(d.number): min([math.dist(d.xy, (p.x, p.y)) for p in vis], default=None)
                                for d in det if d.number in set(info['spurious'])}
            res1, info1 = build(th1, '_1m')
            row.update(res1)
            # R5
            matched = info['matched']
            group = truth.group_for(rec.episode_id, chosen)
            mem_ids = {p.id for p in vis if group is not None and p.group == group}
            members = sorted(n for n, i in matched.items() if i in mem_ids)
            pid = truth.targets[chosen].person_id
            needed = next((n for n, i in matched.items() if pid is not None and i == pid), None)
            row['R5_members'] = members
            row['R5_needed'] = needed
            r5 = reference_constraints(rec.situation, members, needed)
            row['R5'] = bool(H.resolve(rec, constraints=r5).score.success)
            # R6
            if ot is None:
                row['R6'] = row['C']
            else:
                r6 = H.resolve(rec, target=ot)
                row['R6'] = bool(r6.score.success)
                row['R6_tc'] = bool(r6.score.target_correct)
            row['R6x'] = row['R6']
            rows.append(row)
    path = f'{OUT}/vrows_{wording}_{model}.jsonl'
    with open(path, 'w') as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + '\n')
    return path, len(rows)


if __name__ == '__main__':
    which = sys.argv[1:] or [c[0] + ':' + c[2] for c in CELLS]
    todo = [c for c in CELLS if c[0] + ':' + c[2] in which]
    with Pool(2) as pool:
        for path, n in pool.imap_unordered(cell, todo):
            print(path, n, flush=True)
