"""Verifier: noise-model validation against recorded C (FROZEN 8, kill line K4), with the R5 diagnostic.

    v_validate.py MODEL  -> out/validate_<MODEL>.jsonl
"""
from __future__ import annotations

import json
import sys
import time

import vlib as V
import v_noise as N
from vlib import H
from sgi.gt.derive import is_visible
from sgi.run import _matched_people
from sgi.stages.people import number_people
from sgi.types import DetectedPerson, GroundedTarget

DRAWS = 20


def main(key):
    nz = N.load_noise('models')[key]
    t_start = time.perf_counter()
    n = 0
    with open(V.OUT / f'validate_{key}.jsonl', 'w') as fh:
        for ref in N.fit_runs(key):
            th = H.recorded_config(ref)[1].thresholds
            for rec in H.records(ref):
                now = H.rescore(rec)
                row = {'run': ref.folder, 'arm': ref.arm, 'dom': 'realistic' if ref.arm in N.REAL else 'placeholder',
                       'scene': rec.scene_id, 'episode': rec.episode_id, 'trial': rec.trial, 'situation': rec.situation,
                       'step': rec.solver_step, 'recorded': bool(rec.score.get('success')), 'rescore': bool(now.success)}
                if not rec.solver_step:
                    row['sim'] = float(now.success)
                    row['r5'] = bool(now.success)
                    fh.write(json.dumps(row) + '\n')
                    continue
                sc = H.scene(ref, rec.scene_id)
                obs, truth = sc.observation, sc.truth
                gt = sc.truths[rec.episode_id].gt
                vis = [p for p in truth.people if is_visible(p, obs, th)]
                rg = truth.group_for(rec.episode_id, gt.target_ids[0])
                roles = {p.id: N.role(p, rg) for p in truth.people}
                tid = (rec.score.get('matched_target_id') if rec.score.get('target_correct') else None) or gt.target_ids[0]
                want = truth.targets[tid]
                grp = truth.group_for(rec.episode_id, tid)
                target = GroundedTarget.model_validate(rec.target)
                skey = f'{ref.folder}/{rec.scene_id}/{rec.trial}'
                wins = []
                for d in range(DRAWS):
                    found = N.simulate(nz, N.rng_for(key, skey, d), truth, obs, th, roles)
                    kmap = {(f.xy[0], f.xy[1], f.pixel[0], f.pixel[1]): s for f, s in found}
                    people, _ = number_people([f for f, _ in found], target, obs, th, th.max_people)
                    src = {p.number: kmap[(p.xy[0], p.xy[1], p.pixel[0], p.pixel[1])] for p in people}
                    num_of = {}
                    for nn, s in src.items():
                        if s is not None and s not in num_of:
                            num_of[s] = nn
                    members = [num_of[p.id] for p in truth.people if grp is not None and p.group == grp and p.id in num_of]
                    needed = num_of.get(want.person_id) if want.person_id else None
                    res = H.resolve(rec, people=[p.model_dump() for p in people],
                                    constraints=V.reference(rec.situation, members, needed))
                    wins.append(bool(res.score.success))
                det = [DetectedPerson.model_validate(p) for p in rec.people]
                m = _matched_people(det, vis, th)
                num5 = {v: k for k, v in m.items()}
                vis_ids = {p.id for p in vis}
                mem5 = [num5[p.id] for p in truth.people if grp is not None and p.group == grp and p.id in vis_ids
                        and p.id in num5]
                need5 = num5.get(want.person_id) if want.person_id else None
                r5 = H.resolve(rec, constraints=V.reference(rec.situation, mem5, need5))
                row.update(draws=wins, sim=sum(wins) / DRAWS, r5=bool(r5.score.success))
                fh.write(json.dumps(row) + '\n')
                fh.flush()
                n += 1
                if n % 25 == 0:
                    print(f'{key}: {n} solver steps [{time.perf_counter() - t_start:.0f} s]', flush=True)
    print(f'{key}: done, {n} solver steps [{time.perf_counter() - t_start:.0f} s]', flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
