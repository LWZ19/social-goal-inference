"""I06 repair, protocol author's format check before FROZEN.md is hashed. R0 ONLY: the package's own sgi.solver._turn,
wrapped in a pass-through counter to prove that a function installed as sgi.solver._turn at run time is the one the
solver calls (both through I06's lib.solve_c and through research_harness.resolve). No repair rule is run; no repair
outcome is computed. Zero model or network calls; scenes and runs only through I06's lib.load (H.guard_scene) and the
step-0 harness.

Checks:
  1. sgi is imported from the worktree's src/, and src/sgi/solver.py's sha256.
  2. I06 oracle C (arm 3) on two scenes, R0 through the counter: e0 L1, L2, xy and joined_waiting equal I06's
     score/work/arms_seed0.jsonl; the counter shows how many _turn calls were made.
  3. I06 noisy arm, R0, model glm, draws 0-1 of one line base and one A1 variant (seed 0): e0 L1, L2, xy (4 dp) equal
     I06's score/work/noisy_models_glm_seed0.jsonl (is I06's seeding reproducible?).
  4. One recorded goal step with a come_after (s19 glm53 real): H.resolve(rec) through the counter is reproduced and
     its verdict equals the recorded one.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

RD = Path('results/2026-10-17-beyond-vlm')
sys.path.insert(0, str(RD / 'phase2/i06/score'))
import lib as L  # noqa: E402  (I06's, unchanged)
import noise as N  # noqa: E402  (I06's, unchanged)
from lib import H  # noqa: E402
import sgi  # noqa: E402
from sgi import solver as S  # noqa: E402
from sgi.evaluate import score  # noqa: E402
from sgi.gt.derive import prepare  # noqa: E402

ORIG = S._turn
CALLS = {'n': 0}


def counting(*a, **k):
    CALLS['n'] += 1
    return ORIG(*a, **k)


def main():
    print('sgi from', sgi.__file__)
    print('solver.py sha256', hashlib.sha256(Path(S.__file__).read_bytes()).hexdigest())
    S._turn = counting
    cfg = L.load_config()
    th = cfg.thresholds
    arms = {r['scene_id']: r for r in map(json.loads, open(RD / 'phase2/i06/score/work/arms_seed0.jsonl'))}
    entries = {e['scene_id']: e for e in L.kept_scenes() if e['seed'] == 0}
    for sid in ('eval_06_waiting_line_1', 'i06_s0_A1_eval_06_waiting_line_1'):
        e = entries[sid]
        inputs, truth = L.load(e)
        ets = prepare(inputs, truth, cfg)
        et = ets['e0']
        g = truth.group_for('e0', L.TARGET_ID)
        before = CALLS['n']
        c = L.solve_c(inputs, truth, et, th)
        res = c['result']
        xy = L.goal_xy(res, inputs.observation)
        sc = score(res, inputs.observation, truth, et.gt, th)
        l1, l2 = bool(sc.success), bool(sc.success and L.l2_ok(xy, truth, g, th))
        rec = arms[sid]['episodes']['e0']['arm3']
        same = (l1 == rec['l1'] and l2 == rec['l2'] and max(abs(a - b) for a, b in zip(xy, rec['xy'])) <= 1e-9
                and list(c['note'].joined_waiting) == rec['note']['joined_waiting'])
        print(f'oracle C R0 {sid}: _turn calls {CALLS["n"] - before}; L1 {l1} L2 {l2} joined '
              f'{list(c["note"].joined_waiting)}; equal to I06 arm 3: {same}')
    doc = json.loads((L.SCORE / 'noise_model.json').read_text())
    model = N.Model(doc['models']['glm'])
    noisy = {r['scene_id']: r for r in map(json.loads, open(RD / 'phase2/i06/score/work/noisy_models_glm_seed0.jsonl'))}
    for sid in ('eval_06_waiting_line_1', 'i06_s0_A1_eval_06_waiting_line_1'):
        e = entries[sid]
        inputs, truth = L.load(e)
        ets = prepare(inputs, truth, cfg)
        g0 = truth.group_for('e0', L.TARGET_ID)
        roles = {p.id: N.role_of(p, g0) for p in truth.people}
        for d in (0, 1):
            rng = N.scene_rng('glm', sid, d)
            found = N.simulate(model, rng, truth, inputs.observation, th, roles)
            c = L.solve_c(inputs, truth, ets['e0'], th, people_found=found)
            res = c['result']
            xy = L.goal_xy(res, inputs.observation)
            sc = score(res, inputs.observation, truth, ets['e0'].gt, th)
            got = {'l1': bool(sc.success), 'l2': bool(sc.success and L.l2_ok(xy, truth, g0, th)),
                   'xy': None if xy is None else [round(float(xy[0]), 4), round(float(xy[1]), 4)]}
            want = noisy[sid]['draws'][d]['e0']
            print(f'noisy glm R0 {sid} draw {d}: {got} recorded {want} equal {got == want}')
    refs = H.runs(stage='s19', arm='real', model='glm53', method='con')
    assert len(refs) == 1
    for rec in sorted(H.records(refs[0]), key=lambda r: (r.scene_id, r.episode_id, r.trial)):
        if rec.solver_step and any(u['constraint']['kind'] == 'come_after' for u in rec.constraints):
            before = CALLS['n']
            rr = H.resolve(rec)
            print(f'record R0 {refs[0].folder} {rec.scene_id} {rec.episode_id} {rec.trial}: _turn calls '
                  f'{CALLS["n"] - before}; reproduced {rr.reproduced} moved {rr.moved_m}; success {rr.score.success} '
                  f'recorded {rec.score.get("success")}')
            break
    S._turn = ORIG


if __name__ == '__main__':
    main()
