"""C5 flow pilot step 2b (FROZEN.md sections 2, 5, 6(ii)-(iii)): development-record arrays for settings (b) and (a).

Records: goal-step records (H.Record.solver_step) of the primary cells, read only through research_harness:
old-wording realistic (arms real, staffed, plain) s19 qwen38_low, s19 glm53, s20 dsv41; watching s19 dev_seed0-2
qwen38_low and glm53, s20 dev_seed0-2 dsv41, situation audience. 549 in all.

(b) each model's recorded people: the observation, the recorded people (recorded yaws), result.target.world, the
    recorded constraints; F(rec) and keys from sgi.solver._over_floor on the recorded target, people and constraints
    with standard_goal; picks[0] is asserted to reproduce the recorded place within 1e-6 m.
(a) true people (I01's P, built with phase1/i01_repairs/repairs.py's own code: scene_info, strip and the P lines of
    solver_row): the visible true people at their true positions and yaws (model numbers where matched, oracle-made
    bodies for the missed), the recorded constraints with dropped numbers removed, the recorded target. P's place is
    H.resolve's; _over_floor's picks[0] is asserted equal to it within 1e-6 m. On realistic records P's verdict is
    compared with I01's rows_old_*.jsonl.

Raster per setting (common.paint): origin result.target.world (x, y); floor and people from the setting's people
under the run's recorded thresholds; the seen face from _seen_face on the record's GroundedTarget.
rho_room per setting: share of F cells inside the matched target's acceptable regions (package point_in_polygon),
0 for a wrong target; for (b) asserted equal to I04's rho_room within 1e-9.

Painting and _over_floor are timed (setting b) and reported apart from the sidecar latency (FROZEN.md 6(ii)).
Writes data/dev.npz and data/dev.json. No model, no network, no torch.
"""
from __future__ import annotations

import collections
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, 'results/2026-10-17-beyond-vlm/phase1/i01_repairs')
import common as C  # noqa: E402
import research_harness as H  # noqa: E402
import repairs  # noqa: E402
from sgi import solver  # noqa: E402
from sgi.geometry import to_world_xy  # noqa: E402
from sgi.run import _matched_people  # noqa: E402
from sgi.stages.goal import standard_goal  # noqa: E402
from sgi.types import Constraint, DetectedPerson, GroundedTarget  # noqa: E402

I04 = Path(C.ROOT) / 'phase1d' / 'i04'
I01 = Path(C.ROOT) / 'phase1' / 'i01_repairs'
OLD = [('s19', 'qwen38_low', 'qwen'), ('s19', 'glm53', 'glm'), ('s20', 'dsv41', 'deepseek')]
REAL_ARMS = ('real', 'staffed', 'plain')
DEV_ARMS = ('dev_seed0', 'dev_seed1', 'dev_seed2')


def plan():
    out = []
    for stage, model, short in OLD:
        for arm in REAL_ARMS:
            (ref,) = H.runs(stage=stage, arm=arm, model=model, method='con')
            out.append(('old', f'old_{short}', ref))
    for stage, model, short in OLD:
        for arm in DEV_ARMS:
            (ref,) = H.runs(stage=stage, arm=arm, model=model, method='con')
            out.append(('watch', f'watch_{short}', ref))
    return out


def i04_rows():
    rows = {}
    for f in sorted(I04.glob('rows_old_*.jsonl')) + sorted(I04.glob('rows_watch_*.jsonl')):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            rows[r['key']] = r
    return rows


def i01_rows():
    rows = {}
    for f in sorted(I01.glob('rows_old_*.jsonl')):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            rows[f"{r['run']}/{r['scene']}/{r['episode']}/{r['trial']}"] = r
    return rows


def p_inputs(rec):
    """I01 P's people and constraints, with repairs.py's own code (the P lines of solver_row, verbatim in effect)."""
    sc, th, vis, oracle = repairs.scene_info(rec)
    det = [DetectedPerson.model_validate(p) for p in rec.people]
    cons = [u["constraint"] for u in rec.constraints]
    matched = _matched_people(det, vis, th)                  # number -> true id
    by_id = {p.id: p for p in vis}
    number_of = {tid: n for n, tid in matched.items()}
    dropped = {d.number for d in det if d.number not in matched}
    missed = [p for p in vis if p.id not in number_of]
    added_ok = [p for p in missed if oracle[p.id] is not None]
    nxt = max((d.number for d in det), default=0) + 1
    added = [DetectedPerson(number=nxt + i, label=None, **oracle[p.id]) for i, p in enumerate(added_ok)]

    def move(d, pos=False, yaw=False):
        if d.number not in matched:
            return d
        tp = by_id[matched[d.number]]
        upd = {}
        if pos:
            upd["xy"] = (tp.x, tp.y)
        if yaw:
            upd["yaw"] = tp.yaw
        return d.model_copy(update=upd)

    kept = [d for d in det if d.number in matched]
    p_p = [p.model_dump() for p in [move(d, pos=True, yaw=True) for d in kept] + added]
    c_strip, _ = repairs.strip(cons, dropped)
    return p_p, c_strip


def rho_room(F_world, gt, matched):
    if matched is None:
        return 0.0
    regions = [g for g in gt.acceptable_regions or [] if g.target_id in (None, matched)]
    return float(C.in_union(regions, F_world[:, 0], F_world[:, 1]).mean()) if len(F_world) else 0.0


def setting(rec, sc, th, target, standard, people, cons, frame, face, gt, matched):
    obs = sc.observation
    t0 = time.perf_counter()
    cells, st = solver._over_floor(obs, target, people, cons, standard, th)
    t1 = time.perf_counter()
    fl = solver.floor(obs, people, th)
    raster = C.paint(obs, people, fl, frame, face)
    t2 = time.perf_counter()
    assert len(cells) == int(fl.selectable.sum())
    F = C.floor_world_cells(cells, obs.robot)
    ex = np.asarray(st.keys['excluded_m'], dtype=np.float64)
    p0 = to_world_xy(obs.robot, *cells[st.picks[0]])
    return dict(raster=raster, F=F, ex=ex, place_ex=float(ex[st.picks[0]]), pick0=p0, picks=list(st.picks),
                rho=rho_room(F, gt, matched), t_over=t1 - t0, t_paint=t2 - t1)


def main():
    rows04, rows01 = i04_rows(), i01_rows()
    out = collections.defaultdict(list)
    meta, stats = [], collections.Counter()
    t_start = time.monotonic()
    for label, cell, ref in plan():
        th = H.recorded_config(ref)[1].thresholds
        n = std = 0
        for rec in H.records(ref):
            if label == 'watch' and rec.situation != 'audience':
                continue
            if not rec.solver_step:
                std += 1
                continue
            key = f"{rec.ref.folder}/{rec.scene_id}/{rec.episode_id}/{rec.trial}"
            sc = H.scene(rec.ref, rec.scene_id)
            obs = sc.observation
            gt = sc.truths[rec.episode_id].gt
            target = GroundedTarget.model_validate(rec.target)
            standard = standard_goal(obs, target, th)
            frame = C.Frame(target.world[:2], (obs.robot.x, obs.robot.y))
            face = solver._seen_face(obs, target, th)
            rec_places = [tuple(rec.pose_world)] + [tuple(c['pose_world']) for c in rec.candidates]
            assert len(rec_places) == 5, (key, len(rec_places))
            pkg = H.score_pose(rec, rec.pose_world)
            matched = pkg.matched_target_id
            assert bool(pkg.success) == bool(rec.score.get('success')), key
            # (b)
            people_b = [DetectedPerson.model_validate(p) for p in rec.people]
            cons_b = [Constraint.model_validate(u['constraint']) for u in rec.constraints]
            b = setting(rec, sc, th, target, standard, people_b, cons_b, frame, face, gt, matched)
            moved_b = math.dist(b['pick0'], rec.pose_world[:2])
            assert moved_b <= 1e-6, (key, moved_b)
            r04 = rows04[key]
            assert abs(r04['rho_room'] - b['rho']) <= 1e-9, (key, r04['rho_room'], b['rho'])
            assert r04['n_cells'] == len(b['F']), (key, r04['n_cells'], len(b['F']))
            # (a)
            p_p, c_strip = p_inputs(rec)
            resP = H.resolve(rec, people=p_p, constraints=c_strip)
            people_a = [DetectedPerson.model_validate(p) for p in p_p]
            cons_a = [Constraint.model_validate(c) for c in c_strip]
            a = setting(rec, sc, th, target, standard, people_a, cons_a, frame, face, gt, matched)
            moved_a = math.dist(a['pick0'], resP.pose_world[:2])
            assert moved_a <= 1e-6, (key, moved_a)
            p_places = [tuple(resP.pose_world)] + [tuple(c.pose_world) for c in resP.candidates]
            assert len(p_places) == 5, (key, len(p_places))
            P_top1 = bool(resP.score.success)
            i01 = rows01.get(key)
            i01_P = None if i01 is None else bool(i01['P'])
            if label == 'old':
                assert i01 is not None, key
                stats['P_compared'] += 1
                stats['P_mismatch'] += int(i01_P != P_top1)
            for s, d in (('b', b), ('a', a)):
                out[f'raster_{s}'].append(d['raster'])
                out[f'F_{s}'].append(d['F'])
                out[f'ex_{s}'].append(d['ex'])
                out[f'place_ex_{s}'].append(d['place_ex'])
            out['places_b'].append(np.array(rec_places, dtype=np.float64))
            out['places_a'].append(np.array(p_places, dtype=np.float64))
            out['frame'].append([frame.ox, frame.oy, frame.yaw])
            out['keys'].append(key)
            out['situation'].append(rec.situation)
            out['robot_radius'].append(th.robot_radius_m)
            meta.append(dict(key=key, set=label, cell=cell, model=rec.model, arm=rec.arm, run=rec.run,
                             scene=rec.scene_id, ep=rec.episode_id, trial=rec.trial, sit=rec.situation,
                             layout=r04['layout'], matched=matched, target_ok=bool(pkg.target_correct),
                             C_top1=bool(rec.score.get('success')), P_top1=P_top1, I01_P=i01_P,
                             rho_b=b['rho'], rho_a=a['rho'], n_F_b=len(b['F']), n_F_a=len(a['F']),
                             place_ex_b=b['place_ex'], place_ex_a=a['place_ex'],
                             n_people_b=len(people_b), n_people_a=len(people_a), face_none=face is None,
                             frame_dist=frame.dist, t_over_b=b['t_over'], t_paint_b=b['t_paint'],
                             t_over_a=a['t_over'], t_paint_a=a['t_paint']))
            n += 1
        stats[f'n_{cell}'] += n
        stats[f'std_{cell}'] += std
        print(f"{time.monotonic() - t_start:7.1f} s  {cell:15s} {ref.folder}: goal steps {n}, standard {std}", flush=True)
    off_b = np.concatenate([[0], np.cumsum([len(f) for f in out['F_b']])])
    off_a = np.concatenate([[0], np.cumsum([len(f) for f in out['F_a']])])
    np.savez(Path(C.OUT) / 'data' / 'dev.npz',
             keys=np.array(out['keys']), situation=np.array(out['situation']),
             raster_b=np.stack(out['raster_b']), raster_a=np.stack(out['raster_a']),
             F_b=np.concatenate(out['F_b']), ex_b=np.concatenate(out['ex_b']), off_b=off_b,
             F_a=np.concatenate(out['F_a']), ex_a=np.concatenate(out['ex_a']), off_a=off_a,
             place_ex_b=np.array(out['place_ex_b']), place_ex_a=np.array(out['place_ex_a']),
             places_b=np.stack(out['places_b']), places_a=np.stack(out['places_a']),
             frame=np.array(out['frame']), robot_radius=np.array(out['robot_radius']))
    tb = np.array([m['t_paint_b'] for m in meta])
    to = np.array([m['t_over_b'] for m in meta])
    summary = dict(n=len(meta), stats=dict(stats), seconds=round(time.monotonic() - t_start, 1),
                   paint_b_s=dict(median=float(np.median(tb)), p90=float(np.percentile(tb, 90))),
                   over_floor_b_s=dict(median=float(np.median(to)), p90=float(np.percentile(to, 90))))
    (Path(C.OUT) / 'data' / 'dev.json').write_text(json.dumps(dict(summary=summary, records=meta)))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
