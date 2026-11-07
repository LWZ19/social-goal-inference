"""I01 Step 2 with the shared repair code: phase1/i01_repairs/repairs.py solver_row (R0..R6, R4s, Ps, P1m, ...)
applied to every s8_convlm goal step, paired with s8_conorc on (scene, episode, trial). Adds the ambiguity counts
and the conorc verdicts (recorded, rescored, solved again). calibrate.py is my own earlier implementation (written
before i01_repairs existed); its R1/R2/R3 must equal these, and its R4/P equal R4s/Ps here.

Writes rows2.jsonl beside this file.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, 'docs/model_notes/tools')
sys.path.insert(0, 'results/2026-10-17-beyond-vlm/phase1/i01_repairs')
import research_harness as H  # noqa: E402
import repairs as R  # noqa: E402

HERE = Path(__file__).resolve().parent
MODELS = ['glm53', 'ornith15', 'qwen36_35', 'qwen38', 'qwen3vl']


def ambiguity(det, vis, matched_idx, width, people_all):
    """Per matched pair: ambiguous when the detection is within `width` of >= 2 visible true people, or its true
    person is within `width` of >= 2 detections. Also the variant against all true people (visible or not)."""
    w = width + 1e-9
    out = []
    for number, index in matched_idx.items():
        d = next(x for x in det if x['number'] == number)
        p = vis[index]
        near_true = sum(math.dist(d['xy'], (q.x, q.y)) <= w for q in vis)
        near_true_all = sum(math.dist(d['xy'], (q.x, q.y)) <= w for q in people_all)
        near_det = sum(math.dist(x['xy'], (p.x, p.y)) <= w for x in det)
        out.append({'number': number, 'person': p.id, 'gap': math.dist(d['xy'], (p.x, p.y)),
                    'det_near_true': near_true, 'det_near_true_all': near_true_all, 'true_near_det': near_det,
                    'ambiguous': near_true >= 2 or near_det >= 2,
                    'ambiguous_all': near_true_all >= 2 or near_det >= 2})
    return out


def main():
    rows = []
    started = time.monotonic()
    for model in MODELS:
        lvm = H.runs(stage='s8', model=model, method='convlm')[0]
        orc = H.runs(stage='s8', model=model, method='conorc')[0]
        o_by_key = {(r.scene_id, r.episode_id, r.trial): r for r in H.records(orc)}
        for rec in sorted(H.records(lvm), key=lambda r: (r.scene_id, r.episode_id, r.trial)):
            key = (rec.scene_id, rec.episode_id, rec.trial)
            ko = o_by_key[key]
            s = rec.score
            row = dict(model=model, run=lvm.folder, scene=rec.scene_id, episode=rec.episode_id, trial=rec.trial,
                       situation=rec.situation, source=rec.result.get('source'), solver_step=rec.solver_step,
                       orc_solver_step=ko.solver_step, C_rec=bool(s.get('success')),
                       target_correct=bool(s.get('target_correct')),
                       same_target=json.dumps(rec.target, sort_keys=True) == json.dumps(ko.target, sort_keys=True),
                       orc_rec=bool(ko.score.get('success')), orc_rescore=bool(H.rescore(ko).success),
                       C_rescore=bool(H.rescore(rec).success))
            if ko.solver_step:
                again = H.resolve(ko)
                row.update(orc_now=bool(again.score.success), orc_moved_m=again.moved_m,
                           orc_n_people=len(ko.people or []))
            else:
                row.update(orc_now=row['orc_rescore'], orc_moved_m=None)
            if rec.solver_step:
                R.solver_row(rec, row)
                sc, th, vis, oracle = R.scene_info(rec)
                det = rec.people
                from sgi.run import _matched_people
                from sgi.types import DetectedPerson
                m = _matched_people([DetectedPerson.model_validate(p) for p in det], vis, th)
                idx = {n: next(i for i, p in enumerate(vis) if p.id == tid) for n, tid in m.items()}
                row['pairs'] = ambiguity(det, vis, idx, 2 * th.person_radius_m, sc.truth.people)
                row['n_true_all'] = len(sc.truth.people)
                # does P's people set equal conorc's people set (positions)?
                orc_xy = sorted((round(p['xy'][0], 6), round(p['xy'][1], 6)) for p in (ko.people or []))
                vis_xy = sorted((round(p.x, 6), round(p.y, 6)) for p in vis if oracle[p.id] is not None)
                row['P_people_equal_orc'] = orc_xy == vis_xy
            else:
                for k in R.REPAIRS:
                    row[k] = row['C_rec']
                row['C_replay_ok'] = row['C_rec']
            rows.append(row)
        print(f'{model}: done, {time.monotonic() - started:.0f} s', flush=True)
    with open(HERE / 'rows2.jsonl', 'w') as fh:
        for row in rows:
            fh.write(json.dumps(row) + '\n')
    print(f'{len(rows)} rows, {time.monotonic() - started:.0f} s')


if __name__ == '__main__':
    main()
