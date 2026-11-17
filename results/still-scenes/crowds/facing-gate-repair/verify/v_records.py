"""Verifier arm 2: development records, settings (a) record's own inputs, (b) I01's P people + stripped constraints,
(c) A2's D (frozen I09 detector people), per rule, with my own _turn (vturn.py). Runs and scenes only through the
step-0 harness; repairs.py and replay_d.py imported unchanged (FROZEN section 5).

    v_records.py CELL [CELL ...]  -> verify/out/records_<cell>.jsonl
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402
RD = paths.STILL
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RD / 'people-finder/early-detector/phase1c/a2'))
import replay_d as A2  # noqa: E402  (unchanged; it imports repairs as R and the harness as H)
R = A2.R
H = A2.H
import vturn as V  # noqa: E402

RULES = ("R0", "RF90", "RF60", "RN")
REAL = ("real", "staffed", "plain")
DEV = ("dev_seed0", "dev_seed1", "dev_seed2")
CELLS = {
    'old_qwen38_low': ('s19', 'qwen38_low', REAL, 'old'), 'old_glm53': ('s19', 'glm53', REAL, 'old'),
    'old_dsv41': ('s20', 'dsv41', REAL, 'old'),
    'watch_qwen38_low': ('s19', 'qwen38_low', DEV, 'watch'), 'watch_glm53': ('s19', 'glm53', DEV, 'watch'),
    'watch_dsv41': ('s20', 'dsv41', DEV, 'watch'),
    'new_glm53': ('s23', 'glm53', REAL, 'new'), 'new_dsv41': ('s23', 'dsv41', REAL, 'new'),
    'new_haiku55': ('s23', 'haiku55', REAL, 'new'), 'new_sonnet55_low': ('s23', 'sonnet55_low', REAL, 'new'),
    'new_gemini38_low': ('s23', 'gemini38_low', REAL, 'new'),
}
FROZEN_SHA = '02432f62320c0ebb0924c0f0e5b4b2089d508eb4a85fd28cf56641c25e27560f'
SOLVER_SHA = '10ece70f521eff2b1224267fa0b91bfac546d5740f85622073989a16c6c5d800'


def layout_of(scene_id):
    for p in ('real_', 'staffed_', 'plain_'):
        if scene_id.startswith(p):
            return scene_id[len(p):]
    return scene_id


def people_b(rec):
    """I01's P inputs, written from repairs.solver_row's p_p and c_strip lines."""
    from sgi.run import _matched_people
    from sgi.types import DetectedPerson
    sc, th, vis, oracle = R.scene_info(rec)
    det = [DetectedPerson.model_validate(p) for p in rec.people]
    cons = [u["constraint"] for u in rec.constraints]
    m = _matched_people(det, vis, th)
    true_of = {p.id: p for p in vis}
    have = set(m.values())
    dropped = {d.number for d in det if d.number not in m}
    extra = [p for p in vis if p.id not in have and oracle[p.id] is not None]
    start = max((d.number for d in det), default=0) + 1
    out = []
    for d in det:
        if d.number in m:
            tp = true_of[m[d.number]]
            out.append(d.model_copy(update={"xy": (tp.x, tp.y), "yaw": tp.yaw}).model_dump())
    for i, p in enumerate(extra):
        out.append(DetectedPerson(number=start + i, label=None, **oracle[p.id]).model_dump())
    return out, R.strip(cons, dropped)[0]


def people_c(rec, dets):
    from sgi.types import DetectedPerson
    sc, th, vis, _ = R.scene_info(rec)
    det = [DetectedPerson.model_validate(p) for p in rec.people]
    cons = [u["constraint"] for u in rec.constraints]
    nums = [d.number for d in det]
    cnums = [n for c in cons for n in (c.get("people") or [])]
    M = max(nums + cnums, default=0)
    pairs = A2.match_xy(det, dets)
    people, out_cons, dropped, add_nums, info = A2.build(det, cons, dets, pairs, False, M, sc.observation, th)
    return [p.model_dump() for p in people], out_cons


def solve(rec, rule, people=None, constraints=None):
    V.take_log()
    try:
        with V.rule_on(rule):
            if people is None:
                rr = H.resolve(rec)
            else:
                rr = H.resolve(rec, people=people, constraints=constraints)
    except AssertionError:
        raise
    except Exception:
        if rule == 'R0':
            raise
        return {'ok': False, 'error': traceback.format_exc(limit=2), 'log': V.take_log()}
    log = V.take_log()
    return {'ok': bool(rr.score.success), 'moved': rr.moved_m, 'reproduced': bool(rr.reproduced),
            'joined': [] if rr.note is None else [int(n) for n in rr.note.joined_waiting], 'log': log,
            'xy': [float(rr.pose_world[0]), float(rr.pose_world[1])]}


def main(cells):
    assert hashlib.sha256((RD / 'crowds/facing-gate-repair/FROZEN.md').read_bytes()).hexdigest() == FROZEN_SHA
    assert hashlib.sha256(Path(V.S.__file__).read_bytes()).hexdigest() == SOLVER_SHA
    hashes = A2.verify_hashes()
    print('replay_d hashes ok', hashes, flush=True)
    det_of = A2.load_detector_people()
    bad_keys = [k for k in det_of if H.forbidden(k[1]) is not None or H.forbidden(k[0]) is not None]
    print('detector-people keys', len(det_of), 'forbidden keys', bad_keys, flush=True)
    for cell in cells:
        stage, model, arms, kind = CELLS[cell]
        t0 = time.time()
        with open(HERE / 'out' / f'records_{cell}.jsonl', 'w') as fh:
            for arm in arms:
                refs = H.runs(stage=stage, arm=arm, model=model, method='con')
                assert len(refs) == 1, (stage, arm, model, refs)
                ref = refs[0]
                recs = sorted(H.records(ref), key=lambda r: (r.scene_id, r.episode_id, r.trial))
                for rec in recs:
                    if not rec.solver_step:
                        continue
                    if kind == 'watch' and rec.situation != 'audience':
                        continue
                    row = {'cell': cell, 'run': ref.folder, 'arm': ref.arm, 'scene': rec.scene_id,
                           'layout': layout_of(rec.scene_id), 'episode': rec.episode_id, 'trial': rec.trial,
                           'situation': rec.situation, 'C_rec': bool(rec.score.get('success'))}
                    pb, cb = people_b(rec)
                    row['a'] = {r: solve(rec, r) for r in RULES}
                    row['b'] = {r: solve(rec, r, pb, cb) for r in RULES}
                    if kind != 'watch':
                        folder, dets = det_of[(str(ref.scenes_dir), rec.scene_id)]
                        pc, cc = people_c(rec, dets)
                        row['c'] = {r: solve(rec, r, pc, cc) for r in RULES}
                    fh.write(json.dumps(row) + '\n')
                    fh.flush()
        print(f"{cell}: done in {time.time() - t0:.0f} s; shadow {V.STATS}", flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
