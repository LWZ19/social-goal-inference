"""I06 repair probe, arm 1b (FROZEN.md section 4; DEVIATIONS D2): I06's noisy arm on the 168 kept line scenes, 20
draws per scene, models qwen, glm, dsv, under R0 (copy check active; compared with I06's recorded draws inline) or
RF90. Exactly as I06's noisy.py, except that only det_i is computed (the floor-based detectors are not run).

    crowd_noisy.py r0 MODEL SEED      -> work/noisy_r0_<MODEL>_seed<SEED>.jsonl
    crowd_noisy.py check_r0 MODEL     -> out/r0_noisy_<MODEL>.txt
    crowd_noisy.py rf90 MODEL SEED    -> work/noisy_rf90_<MODEL>_seed<SEED>.jsonl (needs work/noisy_r0_*; asserts found)
    crowd_noisy.py merge MODEL        -> rows_noisy_<MODEL>.jsonl (one row per scene and rule)
"""
from __future__ import annotations

import hashlib
import json
import sys
import time

import common as C

C.start()
import lib as L  # noqa: E402  (I06's, unchanged)
import noise as N  # noqa: E402  (I06's, unchanged)
import rules as R  # noqa: E402
from sgi.evaluate import score  # noqa: E402
from sgi.gt.derive import prepare  # noqa: E402

DRAWS = 20
LINE_ROWS = (6, 7, 8, 9, 13)
KEYS = ('scene_id', 'kind', 'seed', 'row', 'family', 'level', 'base_scene_id')


def line_entries(seed):
    out = [e for e in L.kept_scenes() if e['seed'] == seed and e['row'] in LINE_ROWS]
    return out


def found_sha(found):
    items = [((float(f.xy[0]), float(f.xy[1])), None if f.yaw is None else float(f.yaw),
              (float(f.pixel[0]), float(f.pixel[1])), f.label, sid) for f, sid in found]
    return hashlib.sha256(repr(items).encode()).hexdigest()


def run(mode, key, seed):
    rule = {'r0': 'R0', 'rf90': 'RF90'}[mode]
    C.wait_memory()
    doc = json.loads((L.SCORE / 'noise_model.json').read_text())
    assert doc['check_i03']
    model = N.Model(doc['models'][key])
    cfg = L.load_config()
    th = cfg.thresholds
    entries = line_entries(seed)
    if mode == 'r0':
        rec = {r['scene_id']: r for r in map(json.loads, open(L.SCORE / 'work' / f'noisy_models_{key}_seed{seed}.jsonl'))}
    else:
        assert f'FULL REPRODUCTION' in (C.OUT / f'r0_noisy_{key}.txt').read_text(), 'no R0 file: rf90 refused'
        rec = {r['scene_id']: r for r in C.read_jsonl(C.WORK / f'noisy_r0_{key}_seed{seed}.jsonl')}
    out = C.WORK / f'noisy_{mode}_{key}_seed{seed}.jsonl'
    started = time.perf_counter()
    trace = []
    n_bad = 0
    with R.installed(rule, check=(rule == 'R0'), trace=trace), open(out, 'w') as fh:
        for e in entries:
            t0 = time.perf_counter()
            inputs, truth = L.load(e)
            obs = inputs.observation
            ets = prepare(inputs, truth, cfg)
            assert ets['e0'].gt.situation == 'waiting_line', e['scene_id']
            g0 = truth.group_for('e0', L.TARGET_ID)
            member_ids = {p.id for p in truth.people if g0 is not None and p.group == g0}
            roles = {p.id: N.role_of(p, g0) for p in truth.people}
            draws = []
            for d in range(DRAWS):
                rng = N.scene_rng(key, e['scene_id'], d)
                found = N.simulate(model, rng, truth, obs, th, roles)
                fsha = found_sha(found)
                per_ep, c_prev, same_prev, tr_e0, joined_e0 = {}, None, None, None, None
                for episode in inputs.episodes:
                    eid = episode.episode_id
                    et = ets[eid]
                    g = truth.group_for(eid, L.TARGET_ID)
                    assert g == g0, (e['scene_id'], eid)
                    same = (et.gt.situation, g)
                    if c_prev is not None and same_prev == same:
                        c = c_prev
                    else:
                        trace.clear()
                        c = L.solve_c(inputs, truth, et, th, people_found=found)
                        tr = [dict(t) for t in trace]
                        assert len(tr) <= 1
                        if eid == 'e0':
                            tr_e0 = tr[0] if tr else None
                            joined_e0 = [] if c['note'] is None else [int(n) for n in c['note'].joined_waiting]
                            assert (tr_e0['joined'] if tr_e0 else []) == joined_e0
                    c_prev, same_prev = c, same
                    res = c['result']
                    xy = L.goal_xy(res, obs)
                    sc = score(res, obs, truth, et.gt, th)
                    per_ep[eid] = {'l1': bool(sc.success), 'l2': bool(sc.success and L.l2_ok(xy, truth, g, th)),
                                   'xy': None if xy is None else [round(float(xy[0]), 4), round(float(xy[1]), 4)]}
                # det_i, lib.detectors' own formula, on the last solve (as noisy.py: c_prev with e0's group)
                note = c_prev.get('note')
                joined = [] if note is None else [int(n) for n in note.joined_waiting]
                src = c_prev.get('source_of') or {}
                jsrc = [src.get(n) for n in joined]
                det_i = any(s is not None and s not in member_ids for s in jsrc)
                if rule == 'RN':
                    assert not joined_e0
                draw = {'d': d, 'e0': per_ep['e0'], 'agree': all(v == per_ep['e0'] for v in per_ep.values()),
                        'det_i': det_i, 'joined': joined, 'joined_sources': jsrc, 'joined_e0': joined_e0,
                        'found_sha': fsha,
                        'absorbed': C.absorbed(tr_e0) if tr_e0 else [], 'gated': C.gated(tr_e0) if tr_e0 else [],
                        'yaw_unknown_found': sum(f.yaw is None for f, _ in found)}
                if mode == 'r0':
                    want = rec[e['scene_id']]['draws'][d]
                    assert want['d'] == d
                    mism = []
                    if draw['e0'] != want['e0']:
                        mism.append('e0')
                    if det_i != want['det']['det_i']:
                        mism.append('det_i')
                    if not draw['agree'] or not want['agree']:
                        mism.append('agree')
                    draw['r0_mismatch'] = mism
                    n_bad += bool(mism)
                else:
                    base = rec[e['scene_id']]['draws'][d]
                    assert base['found_sha'] == fsha, ('found differs', e['scene_id'], d)
                    if not base['joined_e0']:           # consequence 1 (e0's solve)
                        assert draw['e0'] == base['e0'] and not joined_e0, (e['scene_id'], d)
                draws.append(draw)
            row = {k: e[k] for k in KEYS}
            row.update(model=key, rule=rule, draws=draws, l1_mean=sum(x['e0']['l1'] for x in draws) / DRAWS,
                       l2_mean=sum(x['e0']['l2'] for x in draws) / DRAWS,
                       agree_all=all(x['agree'] for x in draws), secs=round(time.perf_counter() - t0, 2))
            fh.write(json.dumps(row) + '\n')
            fh.flush()
            print(f"{C.now()} {mode} {key} seed {seed} {e['scene_id']}: L1 {row['l1_mean']:.2f} "
                  f"L2 {row['l2_mean']:.2f} agree {row['agree_all']} "
                  f"{'mismatch ' + str(sum(bool(x.get('r0_mismatch')) for x in draws)) if mode == 'r0' else ''} "
                  f"[{row['secs']} s, {time.perf_counter() - started:.0f} s]", flush=True)
    print(f'{C.now()} done {mode} {key} seed {seed}: {len(entries)} scenes, mismatching draws {n_bad}, '
          f'copy-check calls {R.CHECKS["calls"]}', flush=True)


def check_r0(key):
    rows = []
    for s in (0, 1, 2):
        rows += C.read_jsonl(C.WORK / f'noisy_r0_{key}_seed{s}.jsonl')
    n_draws = sum(len(r['draws']) for r in rows)
    bad = [(r['seed'], r['scene_id'], x['d'], x['r0_mismatch']) for r in rows for x in r['draws'] if x['r0_mismatch']]
    ok = len(rows) == 168 and n_draws == 168 * DRAWS and not bad
    text = [f'I06 repair probe: R0 reproduction, arm 1b (noisy), model {key}. Written {C.now()}.',
            f'scenes {len(rows)} (bases {sum(r["kind"] == "base" for r in rows)}), draws {n_draws}; each draw\'s e0 '
            f'{{l1, l2, xy 4 dp}}, det_i and agree compared with I06 score/work/noisy_models_{key}_seed*.jsonl',
            f'draws with any mismatch: {len(bad)}', *[f'  mismatch {b}' for b in bad[:200]],
            f'FULL REPRODUCTION: {"yes" if ok else "NO"}' + ('' if ok else
                ' (I06\'s draws not reproduced: RF90 is compared with this probe\'s own R0, FROZEN section 4)')]
    (C.OUT / f'r0_noisy_{key}.txt').write_text('\n'.join(text) + '\n')
    print('\n'.join(text))


def merge(key):
    rows = []
    for s in (0, 1, 2):
        a = C.read_jsonl(C.WORK / f'noisy_r0_{key}_seed{s}.jsonl')
        b = C.read_jsonl(C.WORK / f'noisy_rf90_{key}_seed{s}.jsonl')
        assert [r['scene_id'] for r in a] == [r['scene_id'] for r in b]
        rows += a + b
    rows.sort(key=lambda r: (r['seed'], r['kind'] != 'base', r['row'], r['family'], r['level'], r['scene_id'],
                             R.RULES.index(r['rule'])))
    C.write_jsonl(C.HERE / f'rows_noisy_{key}.jsonl', rows)
    print(f'{C.now()} merged {len(rows)} rows -> rows_noisy_{key}.jsonl')


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode in ('r0', 'rf90'):
        run(mode, sys.argv[2], int(sys.argv[3]))
    elif mode == 'check_r0':
        check_r0(sys.argv[2])
    elif mode == 'merge':
        merge(sys.argv[2])
    else:
        raise SystemExit(f'unknown mode {mode}')
