"""I06 repair probe, arm 1 (FROZEN.md section 3): I06's arm 3 (oracle C) on the 229 kept scenes, both episodes,
under R0 (with the copy check) or under the repair rules. Zero model calls; scenes through I06's lib.load
(H.guard_scene); I06's lib.py imported unchanged.

    crowd.py r0 SEED      -> work/crowd_r0_seed<SEED>.jsonl   (R0 with the copy check; compared with I06 inline)
    crowd.py check_r0     -> out/r0_crowd.txt                 (reproduction counts over all seeds; exit 1 on mismatch)
    crowd.py repair SEED  -> work/crowd_rep_seed<SEED>.jsonl  (RF90, RF60, RN; refuses unless out/r0_crowd.txt is full)
    crowd.py merge        -> rows_crowd.jsonl                 (one row per scene and rule)
"""
from __future__ import annotations

import json
import sys
import time

import common as C

C.start()
import lib as L  # noqa: E402  (I06's, unchanged)
import rules as R  # noqa: E402
from sgi.evaluate import score  # noqa: E402
from sgi.gt.derive import prepare  # noqa: E402

KEYS = ('scene_id', 'kind', 'seed', 'row', 'family', 'level', 'base_scene_id')
DET_KEEP = ('det_i', 'det_i_false_person', 'joined_waiting', 'joined_sources', 'come_after_used', 'in_region',
            'success_l1', 'F_cells', 'min_pair_covering_m', 'excluded_m', 'target_distance_m', 'det_ii_core',
            'iii_applies', 'off_facing_deg', 'det_iii', 'iv_applies', 'not_ahead_of', 'not_ahead_reason',
            'det_iv_core', 'ahead_of_hindmost_m', 'l1_cells_standable', 'det_v', 'far_from_target',
            'exclusion_broken')


def solve_scene(e, rules, cfg, th):
    """One row per rule for one kept scene: both episodes' L1, L2, xy, joined and trace; detectors on e0."""
    inputs, truth = L.load(e)
    obs = inputs.observation
    ets = prepare(inputs, truth, cfg)
    cells = L.l1_cells(truth, obs, ets['e0'].gt.acceptable_regions, th)
    rows = []
    for rule in rules:
        t0 = time.perf_counter()
        trace = []
        eps, det = {}, None
        with R.installed(rule, check=(rule == 'R0'), trace=trace):
            for episode in inputs.episodes:
                eid = episode.episode_id
                et = ets[eid]
                g = truth.group_for(eid, L.TARGET_ID)
                trace.clear()
                c = L.solve_c(inputs, truth, et, th)
                tr = [dict(t) for t in trace]
                res = c['result']
                xy = L.goal_xy(res, obs)
                sc = score(res, obs, truth, et.gt, th)
                if c['people'] is not None:          # consequence 3: oracle people have known yaw
                    assert all(p.yaw is not None for p in c['people']), (e['scene_id'], eid)
                joined = [] if c['note'] is None else [int(n) for n in c['note'].joined_waiting]
                if c['note'] is None:                # people do not matter: the standard goal, no solve
                    assert not tr
                if tr:
                    assert len(tr) == 1, (e['scene_id'], eid, len(tr))
                    assert tr[0]['joined'] == joined, (tr, joined)
                else:
                    assert not joined
                if rule == 'RN':                     # consequence 2
                    assert not joined, (e['scene_id'], eid, joined)
                eps[eid] = {'l1': bool(sc.success), 'l2': bool(sc.success and L.l2_ok(xy, truth, g, th)),
                            'xy': C.xy_list(xy), 'situation': et.gt.situation, 'in_region': sc.in_region,
                            'people_matter': c['people_matter'], 'joined': joined,
                            'joined_sources': [c['source_of'].get(n) for n in joined],
                            'target_distance_m': None if c['note'] is None else float(c['note'].target_distance_m),
                            'flags': {k: v for k, v in res.flags.model_dump().items() if v},
                            'absorbed': C.absorbed(tr[0]) if tr else [], 'gated': C.gated(tr[0]) if tr else [],
                            'turn_called': bool(tr)}
                if eid == 'e0':
                    d = L.detectors(inputs, truth, et, th, c, cells)
                    det = {k: d[k] for k in DET_KEEP}
        row = {k: e[k] for k in KEYS}
        row.update(rule=rule, situation=eps['e0']['situation'],
                   situations=sorted({ep['situation'] for ep in eps.values()}), episodes=eps, det=det,
                   agree=all((v['l1'], v['l2'], v['xy']) == (eps['e0']['l1'], eps['e0']['l2'], eps['e0']['xy'])
                             for v in eps.values()),
                   l1=eps['e0']['l1'], l2=eps['e0']['l2'], secs=round(time.perf_counter() - t0, 2))
        assert row['agree'], (e['scene_id'], rule)
        rows.append(row)
    return rows


def mode_r0(seed):
    C.wait_memory()
    print(f'{C.now()} I06 file hashes {C.i06_hashes()}', flush=True)
    cfg = L.load_config()
    th = cfg.thresholds
    arms = {r['scene_id']: r for r in map(json.loads, open(L.SCORE / 'work' / f'arms_seed{seed}.jsonl'))}
    oracle = {r['key']: r for r in map(json.loads, open(L.SCORE / 'rows_oracle.jsonl'))}
    entries = [e for e in L.kept_scenes() if e['seed'] == seed]
    out = C.WORK / f'crowd_r0_seed{seed}.jsonl'
    started = time.perf_counter()
    n_bad = 0
    with open(out, 'w') as fh:
        for e in entries:
            (row,) = solve_scene(e, ['R0'], cfg, th)
            rec = arms[e['scene_id']]['episodes']
            mism = []
            for eid, ep in row['episodes'].items():
                a3 = rec[eid]['arm3']
                if ep['l1'] != a3['l1']:
                    mism.append(f'{eid} l1')
                if ep['l2'] != a3['l2']:
                    mism.append(f'{eid} l2')
                if not C.xy_close(ep['xy'], a3['xy']):
                    mism.append(f'{eid} xy')
                want = None if a3['note'] is None else a3['note']['joined_waiting']
                if (want or []) != ep['joined']:
                    mism.append(f'{eid} joined')
            o = oracle[f"{seed}/{e['scene_id']}"]
            if row['det']['det_i'] != o['det']['i']:
                mism.append('det_i')
            row['r0_mismatch'] = mism
            n_bad += bool(mism)
            fh.write(json.dumps(row) + '\n')
            fh.flush()
            print(f"{C.now()} r0 seed {seed} {e['scene_id']}: L1 {row['l1']} L2 {row['l2']} joined "
                  f"{row['episodes']['e0']['joined']} mismatch {mism} [{row['secs']} s, "
                  f"{time.perf_counter() - started:.0f} s]", flush=True)
    print(f'{C.now()} done r0 seed {seed}: {len(entries)} scenes, {n_bad} mismatching, copy-check calls '
          f'{R.CHECKS["calls"]}', flush=True)


def mode_check_r0():
    rows = []
    for s in (0, 1, 2):
        rows += C.read_jsonl(C.WORK / f'crowd_r0_seed{s}.jsonl')
    bad = [(r['seed'], r['scene_id'], r['r0_mismatch']) for r in rows if r['r0_mismatch']]
    fails = {}
    for r in rows:
        if r['kind'] != 'base' and not r['l1']:
            k = (r['family'], r['level'], r['situation'])
            fails[k] = fails.get(k, 0) + 1
    line = {f'{f}{l}': n for (f, l, s), n in sorted(fails.items()) if s == 'waiting_line'}
    other = {f'{f}{l} {s}': n for (f, l, s), n in sorted(fails.items()) if s != 'waiting_line'}
    want = {'A1': 12, 'A2': 12, 'A3': 15, 'A4': 15, 'A5': 15, 'A6': 15, 'B1': 8, 'C1': 3, 'D2': 4, 'E2': 4}
    n_joined = sum(bool(r['episodes']['e0']['joined']) for r in rows)
    joined_fail = sum(bool(r['episodes']['e0']['joined']) and not r['l1'] for r in rows)
    ok = (len(rows) == 229 and not bad and line == want and not other)
    text = [f'I06 repair probe: R0 reproduction, arm 1 (oracle C). Written {C.now()}.',
            f'scenes {len(rows)} (bases {sum(r["kind"] == "base" for r in rows)}, variants '
            f'{sum(r["kind"] != "base" for r in rows)}); both episodes compared with I06 score/work/arms_seed*.jsonl '
            f'arm3 (L1, L2, xy within 1e-9 m, note.joined_waiting) and rows_oracle.jsonl det[i]',
            f'scenes with any mismatch: {len(bad)}',
            *[f'  mismatch {b}' for b in bad],
            f'R0 L1 failures at lines by family and level: {line} (sum {sum(line.values())}); expected {want}',
            f'R0 L1 failures at groups or audiences: {other or "none"}',
            f'scenes with a non-empty joined_waiting (e0): {n_joined}; of them R0 failures: {joined_fail}',
            f'every episode pair agreed (asserted in the pass): yes',
            f'FULL REPRODUCTION: {"yes" if ok else "NO"}']
    (C.OUT / 'r0_crowd.txt').write_text('\n'.join(text) + '\n')
    print('\n'.join(text))
    sys.exit(0 if ok else 1)


def mode_repair(seed):
    assert 'FULL REPRODUCTION: yes' in (C.OUT / 'r0_crowd.txt').read_text(), 'R0 not reproduced: repair refused'
    C.wait_memory()
    cfg = L.load_config()
    th = cfg.thresholds
    r0 = {r['scene_id']: r for r in C.read_jsonl(C.WORK / f'crowd_r0_seed{seed}.jsonl')}
    entries = [e for e in L.kept_scenes() if e['seed'] == seed]
    out = C.WORK / f'crowd_rep_seed{seed}.jsonl'
    started = time.perf_counter()
    with open(out, 'w') as fh:
        for e in entries:
            base = r0[e['scene_id']]
            rows = solve_scene(e, ['RF90', 'RF60', 'RN'], cfg, th)
            for row in rows:
                for eid, ep in row['episodes'].items():
                    b = base['episodes'][eid]
                    if not b['joined']:              # consequence 1: R0 absorbed nobody -> R0's place and verdict
                        assert ep['xy'] == b['xy'] and ep['l1'] == b['l1'] and ep['l2'] == b['l2'], \
                            (e['scene_id'], row['rule'], eid, ep['xy'], b['xy'])
                        assert not ep['joined']
                fh.write(json.dumps(row) + '\n')
            fh.flush()
            print(f"{C.now()} repair seed {seed} {e['scene_id']}: R0 {base['l1']} "
                  + ' '.join(f"{r['rule']} {r['l1']}" for r in rows)
                  + f" [{time.perf_counter() - started:.0f} s]", flush=True)
    print(f'{C.now()} done repair seed {seed}: {len(entries)} scenes', flush=True)


def mode_merge():
    rows = []
    for s in (0, 1, 2):
        r0 = C.read_jsonl(C.WORK / f'crowd_r0_seed{s}.jsonl')
        rep = C.read_jsonl(C.WORK / f'crowd_rep_seed{s}.jsonl')
        assert len(rep) == 3 * len(r0)
        rows += r0 + rep
    rows.sort(key=lambda r: (r['seed'], r['kind'] != 'base', r['row'], r['family'], r['level'], r['scene_id'],
                             R.RULES.index(r['rule'])))
    C.write_jsonl(C.HERE / 'rows_crowd.jsonl', rows)
    print(f'{C.now()} merged {len(rows)} rows -> rows_crowd.jsonl')


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'r0':
        mode_r0(int(sys.argv[2]))
    elif mode == 'check_r0':
        mode_check_r0()
    elif mode == 'repair':
        mode_repair(int(sys.argv[2]))
    elif mode == 'merge':
        mode_merge()
    else:
        raise SystemExit(f'unknown mode {mode}')
