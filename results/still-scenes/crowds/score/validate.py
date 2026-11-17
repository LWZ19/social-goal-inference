"""I06 noise-model validation (FROZEN.md section 8, the kill line K4), and the R5 diagnostic.

On every allowed record of a model's seven fit runs (development scenes, realistic and placeholder), through the harness:
- a solver-step record is re-solved 20 times with H.resolve(rec, people=simulated, constraints=reference), keeping
  the recorded target; its simulated success is the mean over draws. The scene key is "run folder/scene id/trial", so
  both episodes of one trial share their draws;
- every other record keeps its verdict (the current scorer's, H.rescore);
- diagnostic (outside the rule): R5 = the record's actual people with the reference constraints (I01's construction).
Reference constraints: target id = score.matched_target_id when the recorded target is right, else the episode's first
(as I01); members = the numbers whose person is a visible member of the group at that target; needed = the number of
the target person; visible_to and face omitted when the target person is not listed.

    validate.py MODEL   -> score/noise/validation_<model>.jsonl
"""
from __future__ import annotations

import json
import sys
import time

import lib as L
import noise as N
from lib import H
from sgi.gt.derive import is_visible
from sgi.run import _matched_people
from sgi.stages.people import number_people
from sgi.types import DetectedPerson, GroundedTarget

DRAWS = 20


def main(key: str):
    import hashlib
    assert hashlib.sha256((L.I06 / 'FROZEN.md').read_bytes()).hexdigest() == L.FROZEN_SHA
    doc = json.loads((L.SCORE / 'noise_model.json').read_text())
    assert doc['check_i03'], 'the I03 cross-check failed: the noisy arm is stopped until explained'
    model = N.Model(doc['models'][key])
    _, stage, name, label = next(m for m in N.MODELS if m[0] == key)
    out = N.NOISE / f'validation_{key}.jsonl'
    started = time.perf_counter()
    n = 0
    with open(out, 'w') as fh:
        for ref in N.runs_of(stage, name):
            th = H.recorded_config(ref)[1].thresholds
            recs = sorted(H.records(ref), key=lambda r: (r.scene_id, str(r.trial), r.episode_id))
            for rec in recs:
                rs = H.rescore(rec)
                row = {'model': key, 'run': ref.folder, 'arm': ref.arm,
                       'domain': 'realistic' if ref.arm in N.REALISTIC else 'placeholder', 'scene': rec.scene_id,
                       'episode': rec.episode_id, 'trial': rec.trial, 'situation': rec.situation,
                       'solver_step': rec.solver_step, 'recorded': bool(rec.score.get('success')),
                       'rescore': bool(rs.success)}
                if not rec.solver_step:
                    row.update(sim_mean=float(row['rescore']), r5=row['rescore'])
                    fh.write(json.dumps(row) + '\n')
                    continue
                sc = H.scene(ref, rec.scene_id)
                obs, truth = sc.observation, sc.truth
                gt = sc.truths[rec.episode_id].gt
                vis = [p for p in truth.people if is_visible(p, obs, th)]
                role_group = truth.group_for(rec.episode_id, gt.target_ids[0])
                roles = {p.id: N.role_of(p, role_group) for p in truth.people}
                tid = rec.score.get('matched_target_id') if rec.score.get('target_correct') else None
                tid = tid or gt.target_ids[0]
                wanted = truth.targets[tid]
                group = truth.group_for(rec.episode_id, tid)
                member_ids = set() if group is None else {p.id for p in truth.people if p.group == group}
                target = GroundedTarget.model_validate(rec.target)
                skey = f'{ref.folder}/{rec.scene_id}/{rec.trial}'
                draws = []
                for d in range(DRAWS):
                    rng = N.scene_rng(key, skey, d)
                    found = N.simulate(model, rng, truth, obs, th, roles)
                    src = {(tuple(map(float, f.xy)), tuple(map(float, f.pixel))): s for f, s in found}
                    people, cut = number_people([f for f, _ in found], target, obs, th, th.max_people)
                    source_of = {p.number: src[(tuple(map(float, p.xy)), tuple(map(float, p.pixel)))] for p in people}
                    members = [nn for nn, s in sorted(source_of.items()) if s in member_ids]
                    needed = next((nn for nn, s in sorted(source_of.items())
                                   if wanted.person_id is not None and s == wanted.person_id), None)
                    refs = L.reference_constraints(rec.situation, members, needed)
                    res = H.resolve(rec, people=[p.model_dump() for p in people], constraints=refs)
                    draws.append(bool(res.score.success))
                # R5 (I01): recorded people, reference constraints from the matched members
                det = [DetectedPerson.model_validate(p) for p in rec.people]
                matched = _matched_people(det, vis, th)
                number_of = {t: nn for nn, t in matched.items()}
                vmem = [p.id for p in truth.people if p.group == group and p.id in {v.id for v in vis}] if group else []
                members5 = [number_of[m] for m in vmem if m in number_of]
                needed5 = number_of.get(wanted.person_id) if wanted.person_id else None
                r5 = H.resolve(rec, constraints=L.reference_constraints(rec.situation, members5, needed5))
                row.update(sim_draws=draws, sim_mean=sum(draws) / len(draws), r5=bool(r5.score.success),
                           n_visible=len(vis), n_det=len(det))
                fh.write(json.dumps(row) + '\n')
                fh.flush()
                n += 1
                if n % 20 == 0:
                    print(f'{key}: {n} solver steps [{time.perf_counter() - started:.0f} s]', flush=True)
    print(f'{key}: done, {n} solver steps [{time.perf_counter() - started:.0f} s]', flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
