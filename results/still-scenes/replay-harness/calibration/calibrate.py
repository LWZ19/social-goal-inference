"""I01 Step 2: the s8 calibration of the perception repairs (offline, no model call).

For each placeholder-panel model, every s8_convlm record (VLM people) is paired with the s8_conorc record (true
people, same plan/context/target answers from the cache) of the same (scene, episode, trial). On every convlm goal
step the recorded constraints are kept and the people list is repaired, then the unchanged solver is run again
(H.resolve) and the place scored by the package scorer:

  R1  true positions: each detection matched to a true person gets that person's xy (number, yaw kept).
  R2  true facing: each matched detection gets the true person's yaw (reported; part of P).
  R3  missed people added as unlisted bodies: each visible true person that no detection is matched to is added
      with a new number (max + 1, ...), true xy and yaw, pixel = projection of person_mark_height_m (as OraclePeople),
      so no constraint names them.
  R4  spurious people dropped: each detection matched to nobody is removed (a constraint naming it loses that number).
  P   R1 + R2 + R3 + R4 together.

Matching is the package's one-to-one rule (sgi.run._matched_people, copied): pairs nearest first within
2 * person_radius_m (0.5 m), against the true people that pass the visibility rule (sgi.gt.derive.is_visible).
A record without a goal step keeps its verdict in every repair.

Writes rows.jsonl (one row per convlm record) and run.log beside this file.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H  # noqa: E402

from sgi.geometry import project  # noqa: E402
from sgi.gt.derive import is_visible  # noqa: E402

HERE = Path(__file__).resolve().parent
MODELS = ['glm53', 'ornith15', 'qwen36_35', 'qwen38', 'qwen3vl']
REPAIRS = ('R1', 'R2', 'R3', 'R4', 'P')


def matched_people(detected, visible, th):
    """sgi.run._matched_people, copied: number -> index into visible."""
    width = 2 * th.person_radius_m
    pairs = sorted((math.dist(found['xy'], (person.x, person.y)), found['number'], index)
                   for found in detected for index, person in enumerate(visible))
    matched, taken = {}, set()
    for gap, number, index in pairs:
        if gap <= width and number not in matched and index not in taken:
            matched[number] = index
            taken.add(index)
    return matched


def repaired(detected, visible, matched, obs, th, *, r1, r2, r3, r4):
    out = []
    for d in detected:
        d = dict(d)
        index = matched.get(d['number'])
        if index is None:
            if r4:
                continue
        else:
            person = visible[index]
            if r1:
                d['xy'] = [person.x, person.y]
            if r2:
                d['yaw'] = person.yaw
        out.append(d)
    if r3:
        taken = set(matched.values())
        number = max([d['number'] for d in detected], default=0)
        for index, person in enumerate(visible):
            if index in taken:
                continue
            marked = project(obs.camera, (person.x, person.y, th.person_mark_height_m))
            if marked is None:                     # OraclePeople leaves such a person out too
                continue
            number += 1
            out.append({'number': number, 'xy': [person.x, person.y], 'yaw': person.yaw,
                        'pixel': [marked[0], marked[1]], 'label': None})
    return out


def ambiguity(detected, visible, matched, th, people_all):
    width = 2 * th.person_radius_m + 1e-9
    near_true = {d['number']: sum(math.dist(d['xy'], (p.x, p.y)) <= width for p in visible) for d in detected}
    near_true_all = {d['number']: sum(math.dist(d['xy'], (p.x, p.y)) <= width for p in people_all) for d in detected}
    near_det = [sum(math.dist(d['xy'], (p.x, p.y)) <= width for d in detected) for p in visible]
    pairs = []
    for number, index in matched.items():
        amb = near_true[number] >= 2 or near_det[index] >= 2
        amb_all = near_true_all[number] >= 2 or near_det[index] >= 2
        pairs.append({'number': number, 'person': visible[index].id, 'ambiguous': bool(amb),
                      'ambiguous_all_people': bool(amb_all), 'det_near_true': near_true[number],
                      'true_near_det': near_det[index]})
    return pairs


def main():
    log = open(HERE / 'run.log', 'w')
    rows = []
    started = time.monotonic()
    for model in MODELS:
        lvm = H.runs(stage='s8', model=model, method='convlm')
        orc = H.runs(stage='s8', model=model, method='conorc')
        assert len(lvm) == 1 and len(orc) == 1, model
        lvm, orc = lvm[0], orc[0]
        th = H.recorded_config(lvm)[1].thresholds
        assert th == H.recorded_config(orc)[1].thresholds
        o_by_key = {(r.scene_id, r.episode_id, r.trial): r for r in H.records(orc)}
        for rec in H.records(lvm):
            key = (rec.scene_id, rec.episode_id, rec.trial)
            ko = o_by_key[key]
            row = {'model': model, 'scene': rec.scene_id, 'episode': rec.episode_id, 'trial': rec.trial,
                   'situation': rec.situation, 'goal_step_lvm': rec.solver_step, 'goal_step_orc': ko.solver_step,
                   'same_target': json.dumps(rec.target, sort_keys=True) == json.dumps(ko.target, sort_keys=True),
                   'target_correct': rec.score.get('target_correct'),
                   'lvm_rec': bool(rec.score.get('success')), 'orc_rec': bool(ko.score.get('success')),
                   'lvm_rescore': bool(H.rescore(rec).success), 'orc_rescore': bool(H.rescore(ko).success)}
            if ko.solver_step:
                again = H.resolve(ko)
                row['orc_now'], row['orc_moved_m'] = bool(again.score.success), again.moved_m
            else:
                row['orc_now'], row['orc_moved_m'] = row['orc_rescore'], None
            if rec.solver_step:
                sc = H.scene(lvm, rec.scene_id)
                obs, truth = sc.observation, sc.truth
                visible = [p for p in truth.people if is_visible(p, obs, th)]
                detected = rec.people
                matched = matched_people(detected, visible, th)
                again = H.resolve(rec)
                row['lvm_now'], row['lvm_moved_m'] = bool(again.score.success), again.moved_m
                row['n_detected'], row['n_visible'], row['n_true'] = len(detected), len(visible), len(truth.people)
                row['n_matched'] = len(matched)
                row['n_missed'] = len(visible) - len(matched)
                row['n_spurious'] = len(detected) - len(matched)
                row['found_share_recorded'] = rec.measures.get('people_found_share')
                row['unmatched_recorded'] = rec.measures.get('people_unmatched')
                row['xy_err_matched'] = [math.dist(d['xy'], (visible[matched[d['number']]].x, visible[matched[d['number']]].y))
                                         for d in detected if d['number'] in matched]
                row['pairs'] = ambiguity(detected, visible, matched, th, truth.people)
                row['n_orc_people'] = len(ko.people or [])
                flags = {'R1': dict(r1=True, r2=False, r3=False, r4=False),
                         'R2': dict(r1=False, r2=True, r3=False, r4=False),
                         'R3': dict(r1=False, r2=False, r3=True, r4=False),
                         'R4': dict(r1=False, r2=False, r3=False, r4=True),
                         'P': dict(r1=True, r2=True, r3=True, r4=True)}
                for name, f in flags.items():
                    people = repaired(detected, visible, matched, obs, th, **f)
                    res = H.resolve(rec, people=people)
                    row[name] = bool(res.score.success)
                    row[name + '_moved_from_rec_m'] = res.moved_m
                # the oracle people list of conorc vs P's list: same set of true positions?
                p_people = repaired(detected, visible, matched, obs, th, **flags['P'])
                orc_xy = sorted((round(p['xy'][0], 6), round(p['xy'][1], 6)) for p in (ko.people or []))
                p_xy = sorted((round(p['xy'][0], 6), round(p['xy'][1], 6)) for p in p_people)
                row['P_people_equal_orc'] = orc_xy == p_xy
            else:
                row['lvm_now'] = row['lvm_rescore']
                row['lvm_moved_m'] = None
                for name in REPAIRS:
                    row[name] = row['lvm_rescore']
            rows.append(row)
        print(f'{model}: done, {time.monotonic() - started:.0f} s', file=log, flush=True)
        print(f'{model}: done, {time.monotonic() - started:.0f} s', flush=True)
    with open(HERE / 'rows.jsonl', 'w') as fh:
        for row in rows:
            fh.write(json.dumps(row) + '\n')
    print(f'{len(rows)} rows, {time.monotonic() - started:.0f} s', file=log)


if __name__ == '__main__':
    main()
