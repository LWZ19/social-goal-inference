"""I06 repair probe, arm 2 (FROZEN.md section 5): every goal step of the development records re-solved under each
rule in settings (a) the record's own inputs (H.resolve(rec)), (b) I01's P (true people, recorded constraints
stripped of dropped numbers; the lines of repairs.solver_row that build p_p and c_strip copied verbatim) and (c) A2's D
(the frozen I09 detector people; replay_d.build, unchanged), realistic cells only. Zero calls; runs and scenes only
through research_harness; I01's repairs.py and A2's replay_d.py imported unchanged.

    records.py r0              -> work/records_r0_<cell>.jsonl; out/r0_records_{a,b,c}.txt
    records.py repair          -> rows_records_<cell>.jsonl (RF90, RF60, RN in each setting whose R0 file is full,
                                  merged with R0)
"""
from __future__ import annotations

import json
import sys
import time
import traceback

import common as C

C.start()
sys.path.insert(0, str(C.A2))
sys.path.insert(0, str(C.I01))
sys.path.insert(0, str(C.WT / 'docs/model_notes/tools'))
import research_harness as H  # noqa: E402
import repairs as REP  # noqa: E402  (phase1/i01_repairs/repairs.py, unchanged)
import replay_d as A2D  # noqa: E402  (phase1c/i09/a2/replay_d.py, unchanged)
import rules as RU  # noqa: E402

REAL = ('real', 'staffed', 'plain')
DEV = ('dev_seed0', 'dev_seed1', 'dev_seed2')
# cell -> (stage, model, arms, wording of the reference rows, primary)
CELLS = {
    'old_qwen38_low': ('s19', 'qwen38_low', REAL, 'old', True),
    'old_glm53': ('s19', 'glm53', REAL, 'old', True),
    'old_dsv41': ('s20', 'dsv41', REAL, 'old', True),
    'watch_qwen38_low': ('s19', 'qwen38_low', DEV, 'ph', True),
    'watch_glm53': ('s19', 'glm53', DEV, 'ph', True),
    'watch_dsv41': ('s20', 'dsv41', DEV, 'ph', True),
    'new_glm53': ('s23', 'glm53', REAL, 'new', False),
    'new_dsv41': ('s23', 'dsv41', REAL, 'new', False),
    'new_haiku55': ('s23', 'haiku55', REAL, 'new', False),
    'new_sonnet55_low': ('s23', 'sonnet55_low', REAL, 'new', False),
    'new_gemini38_low': ('s23', 'gemini38_low', REAL, 'new', False),
}
SETTINGS = ('a', 'b', 'c')
TABLE_5_1 = {  # FROZEN Table 5.1: goal steps, recorded success, P, D
    'old_qwen38_low': (143, 134, 136, 136), 'old_glm53': (141, 96, 112, 112), 'old_dsv41': (157, 109, 120, 126),
    'watch_qwen38_low': (36, 32, 33, None), 'watch_glm53': (36, 23, 29, None), 'watch_dsv41': (36, 23, 31, None),
    'new_glm53': (141, 86, 107, 106), 'new_dsv41': (151, 106, 116, 119), 'new_haiku55': (146, 134, 135, 134),
    'new_sonnet55_low': (143, 134, 134, 134), 'new_gemini38_low': (135, 109, 109, 109)}


def goal_steps(cell):
    stage, model, arms, wording, _ = CELLS[cell]
    out = []
    for arm in arms:
        refs = H.runs(stage=stage, arm=arm, model=model, method='con')
        assert len(refs) == 1, (stage, arm, model, refs)
        for rec in sorted(H.records(refs[0]), key=lambda r: (r.scene_id, r.episode_id, r.trial)):
            if not rec.solver_step:
                continue
            if wording == 'ph' and rec.situation != 'audience':
                continue
            out.append(rec)
    return out


def key_of(rec):
    return (rec.arm, rec.scene_id, rec.episode_id, rec.trial)


def inputs_b(rec):
    """I01's P: the lines of repairs.solver_row that build p_p and c_strip, copied verbatim."""
    from sgi.run import _matched_people
    from sgi.types import DetectedPerson
    scene_info, strip = REP.scene_info, REP.strip
    sc, th, vis, oracle = scene_info(rec)
    gt = sc.truths[rec.episode_id].gt
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

    dump = lambda people: [p.model_dump() for p in people]  # noqa: E731
    kept = [d for d in det if d.number in matched]
    p_p = dump([move(d, pos=True, yaw=True) for d in kept] + added)
    c_strip, n_changed = strip(cons, dropped)
    del gt, n_changed
    return p_p, c_strip


DET_OF = None


def inputs_c(rec):
    """A2's D: replay_d.build with the frozen I09 detector people, M as in arms_step."""
    global DET_OF
    from sgi.types import DetectedPerson
    if DET_OF is None:
        DET_OF = A2D.load_detector_people()
    sc, th, vis, _ = REP.scene_info(rec)
    obs = sc.observation
    folder, dets = DET_OF[(str(rec.ref.scenes_dir), rec.scene_id)]
    det = [DetectedPerson.model_validate(p) for p in rec.people]
    nums = [d.number for d in det]
    cons = [u["constraint"] for u in rec.constraints]
    cnums = [n for c in cons for n in (c.get("people") or [])]
    M = max(nums + cnums, default=0)
    people, out_cons, dropped, add_nums, info = A2D.build(det, cons, dets, A2D.match_xy(det, dets), False, M, obs, th)
    return [p.model_dump() for p in people], out_cons


def solve(rec, setting, trace):
    """One solve under the installed rule; a dict of what is kept. An exception is caught and recorded."""
    trace.clear()
    try:
        if setting == 'a':
            rr = H.resolve(rec)
        elif setting == 'b':
            people, cons = inputs_b(rec)
            rr = H.resolve(rec, people=people, constraints=cons)
        else:
            people, cons = inputs_c(rec)
            rr = H.resolve(rec, people=people, constraints=cons)
    except RU.CopyCheckError:
        raise
    except Exception as exc:  # noqa: BLE001
        return {'success': False, 'error': f'{type(exc).__name__}: {exc}', 'tb': traceback.format_exc()[-800:]}
    tr = [dict(t) for t in trace]
    assert len(tr) <= 1, len(tr)
    joined = [] if rr.note is None else [int(n) for n in rr.note.joined_waiting]
    assert (tr[0]['joined'] if tr else []) == joined
    s = rr.score
    return {'success': bool(s.success), 'tc': bool(s.target_correct), 'in_region': bool(s.in_region),
            'reachable': bool(s.reachable), 'xy': [float(rr.pose_world[0]), float(rr.pose_world[1])],
            'moved_m': rr.moved_m, 'reproduced': rr.reproduced, 'joined': joined, 'turn_called': bool(tr),
            'absorbed': C.absorbed(tr[0]) if tr else [], 'gated': C.gated(tr[0]) if tr else []}


def settings_of(cell):
    return ('a', 'b') if CELLS[cell][3] == 'ph' else SETTINGS


def ref_rows(cell):
    stage, model, arms, wording, _ = CELLS[cell]
    load = lambda p: {(r['arm'], r['scene'], r['episode'], r['trial']): r for r in C.read_jsonl(p)}  # noqa: E731
    if wording == 'ph':
        return {'i01': None, 'base_ph': load(C.A2 / f'base_ph_{model}.jsonl'), 'a2': None}
    return {'i01': load(C.I01 / f'rows_{wording}_{model}.jsonl'), 'base_ph': None,
            'a2': load(C.A2 / f'rows_{wording}_{model}.jsonl')}


def mode_r0():
    C.wait_memory()
    hashes = A2D.verify_hashes()
    print(f'{C.now()} replay_d.verify_hashes passed: {hashes}; repairs.py {C.sha256(C.I01 / "repairs.py")}, '
          f'replay_d.py {C.sha256(C.A2 / "replay_d.py")}', flush=True)
    trace = []
    report = {s: [] for s in SETTINGS}
    totals = {s: {'n': 0, 'bad': 0} for s in SETTINGS}
    counts = {}
    started = time.monotonic()
    with RU.installed('R0', check=True, trace=trace):
        for cell in CELLS:
            refs = ref_rows(cell)
            recs = goal_steps(cell)
            rows = []
            t0 = time.monotonic()
            for rec in recs:
                k = key_of(rec)
                row = {'cell': cell, 'run': rec.run, 'arm': rec.arm, 'scene': rec.scene_id,
                       'layout': A2D.SB.layout_of(rec.scene_id), 'episode': rec.episode_id, 'trial': rec.trial,
                       'situation': rec.situation, 'C_rec': bool(rec.score.get('success')), 'R0': {}, 'mismatch': {}}
                for setting in settings_of(cell):
                    got = solve(rec, setting, trace)
                    row['R0'][setting] = got
                    mism = []
                    if 'error' in got:
                        mism.append('raised')
                    elif setting == 'a':
                        if not got['reproduced']:
                            mism.append('not reproduced')
                        if got['success'] != row['C_rec']:
                            mism.append('success != recorded')
                        if refs['i01'] is not None and got['success'] != refs['i01'][k]['C']:
                            mism.append("success != I01's C")
                    elif setting == 'b':
                        want = (refs['i01'][k]['P'] if refs['i01'] is not None else refs['base_ph'][k]['P'])
                        if got['success'] != want:
                            mism.append("success != P")
                    else:
                        a2 = refs['a2'][k]
                        if got['success'] != a2['D']:
                            mism.append("success != A2's D")
                        if got['joined'] != a2['D_d']['joined']:
                            mism.append("joined != A2's D_d.joined")
                    row['mismatch'][setting] = mism
                    totals[setting]['n'] += 1
                    if mism:
                        totals[setting]['bad'] += 1
                        report[setting].append((cell, *k, mism))
                rows.append(row)
            C.write_jsonl(C.WORK / f'records_r0_{cell}.jsonl', rows)
            cnt = {'goal_steps': len(rows), 'recorded_success': sum(r['C_rec'] for r in rows)}
            for s in settings_of(cell):
                cnt[f'R0_{s}'] = sum(r['R0'][s]['success'] for r in rows)
                cnt[f'absorbed_{s}'] = sum(bool(r['R0'][s].get('joined')) for r in rows)
            counts[cell] = cnt
            print(f'{C.now()} r0 {cell}: {cnt}; mismatches '
                  f'{ {s: sum(bool(r["mismatch"][s]) for r in rows) for s in settings_of(cell)} } '
                  f'[{time.monotonic() - t0:.0f} s, {time.monotonic() - started:.0f} s]', flush=True)
    for s in SETTINGS:
        cells = [c for c in CELLS if s in settings_of(c)]
        ok = totals[s]['bad'] == 0
        t51 = []
        for c in cells:
            want = TABLE_5_1[c]
            got = (counts[c]['goal_steps'], counts[c]['recorded_success'], counts[c].get('R0_b'), counts[c].get('R0_c'))
            idx = {'a': 1, 'b': 2, 'c': 3}[s]
            t51.append(f'  {c}: goal steps {got[0]} (Table 5.1 {want[0]}), recorded success {got[1]} ({want[1]}), '
                       f'R0 success in ({s}) {counts[c][f"R0_{s}"]} (Table 5.1 target {want[idx]}), goal steps where R0 '
                       f'absorbed someone {counts[c][f"absorbed_{s}"]}')
            ok &= got[0] == want[0] and got[1] == want[1] and counts[c][f'R0_{s}'] == want[idx]
        what = {'a': "H.resolve(rec): reproduced (<= 1e-6 m), success = recorded success, and = I01's C (realistic)",
                'b': "I01's P construction: success = I01's P (realistic) or A2 base_ph P (watching)",
                'c': "A2's D construction: success = A2's D and joined_waiting = A2's D_d.joined (realistic only)"}[s]
        text = [f'I06 repair probe: R0 reproduction, arm 2 setting ({s}). Written {C.now()}. Copy check active '
                f'(copy-check calls so far {RU.CHECKS["calls"]}).', f'compared: {what}',
                f'goal steps {totals[s]["n"]} over {len(cells)} cells; mismatching {totals[s]["bad"]}',
                *[f'  mismatch {m}' for m in report[s]], *t51,
                f'FULL REPRODUCTION: {"yes" if ok else "NO"}']
        (C.OUT / f'r0_records_{s}.txt').write_text('\n'.join(text) + '\n')
        print('\n'.join(text), flush=True)


def mode_repair():
    C.wait_memory()
    A2D.verify_hashes()
    full = {s: 'FULL REPRODUCTION: yes' in (C.OUT / f'r0_records_{s}.txt').read_text() for s in SETTINGS}
    print(f'{C.now()} settings with full R0 reproduction: {full}', flush=True)
    trace = []
    started = time.monotonic()
    for cell in CELLS:
        t0 = time.monotonic()
        r0 = {(r['arm'], r['scene'], r['episode'], r['trial']): r for r in C.read_jsonl(C.WORK / f'records_r0_{cell}.jsonl')}
        recs = goal_steps(cell)
        assert len(recs) == len(r0)
        rows = []
        res = {}
        for rec in recs:
            k = key_of(rec)
            for rule in ('RF90', 'RF60', 'RN'):
                for s in settings_of(cell):
                    if not full[s]:
                        continue
                    with RU.installed(rule, trace=trace):
                        got = solve(rec, s, trace)
                    b = r0[k]['R0'][s]
                    if 'error' not in got and 'error' not in b and not b['joined']:     # consequence 1
                        assert got['xy'] == b['xy'] and got['success'] == b['success'] and not got['joined'], \
                            (cell, k, s, rule, got['xy'], b['xy'])
                    if rule == 'RN' and 'error' not in got:                              # consequence 2
                        assert not got['joined'], (cell, k, s)
                    res[(k, s, rule)] = got
        for rec in recs:
            k = key_of(rec)
            base = r0[k]
            row = {x: base[x] for x in ('cell', 'run', 'arm', 'scene', 'layout', 'episode', 'trial', 'situation',
                                        'C_rec')}
            row['r0_mismatch'] = base['mismatch']
            row['settings'] = {}
            for s in settings_of(cell):
                row['settings'][s] = {'R0': base['R0'][s]}
                if full[s]:
                    for rule in ('RF90', 'RF60', 'RN'):
                        row['settings'][s][rule] = res[(k, s, rule)]
            rows.append(row)
        C.write_jsonl(C.HERE / f'rows_records_{cell}.jsonl', rows)
        summ = {s: {rule: sum(r['settings'][s][rule]['success'] for r in rows) for rule in RU.RULES
                    if rule in rows[0]['settings'][s]} for s in settings_of(cell)}
        print(f'{C.now()} repair {cell}: {len(rows)} goal steps; successes {summ} '
              f'[{time.monotonic() - t0:.0f} s, {time.monotonic() - started:.0f} s]', flush=True)


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'r0':
        mode_r0()
    elif mode == 'repair':
        mode_repair()
    else:
        raise SystemExit(f'unknown mode {mode}')
