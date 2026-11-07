"""Verifier: arm 1b analysis from verify/out/noisy_<model>_seed*.jsonl; R0 reproduction against I06's work files;
means, per-draw flips and breaks, det_i rates; comparison with the agent's rows_noisy_<model>.jsonl.
"""
from __future__ import annotations

import collections
import json
import statistics
from pathlib import Path

RD = Path('results/2026-10-17-beyond-vlm')
HERE = Path(__file__).resolve().parent
I06 = RD / 'phase2/i06/score'
AG = RD / 'phase2/i06_repair'
out = {}
import sys
for m in (sys.argv[1:] or ['qwen', 'glm', 'dsv']):
    rows = []
    for s in (0, 1, 2):
        rows += [json.loads(l) for l in open(HERE / 'out' / f'noisy_{m}_seed{s}.jsonl')]
    ref = {}
    for s in (0, 1, 2):
        for l in open(I06 / 'work' / f'noisy_models_{m}_seed{s}.jsonl'):
            r = json.loads(l)
            ref[(r['seed'], r['scene_id'])] = r
    ag = {}
    for l in open(AG / f'rows_noisy_{m}.jsonl'):
        a = json.loads(l)
        ag[(a['seed'], a['scene_id'], a['rule'])] = a
    assert len(rows) == 168, len(rows)
    mism, n_draws, diff = [], 0, []
    flips = breaks = 0
    det = collections.Counter()
    fam_rows = collections.defaultdict(list)
    for r in rows:
        k = (r['seed'], r['scene_id'])
        fam = 'base' if r['kind'] == 'base' else f"{r['family']}{r['level']}"
        fam_rows[fam].append(r)
        for x in r['draws']:
            n_draws += 1
            o = ref[k]['draws'][x['d']]
            if x['R0']['e0'] != o['e0'] or x['R0']['det_i'] != o['det']['det_i'] or not (x['R0']['agree'] and o['agree']):
                mism.append((k, x['d']))
            a0, a1 = x['R0']['e0']['l1'], x['RF90']['e0']['l1']
            flips += (not a0) and a1
            breaks += a0 and not a1
            det['R0'] += x['R0']['det_i']
            det['RF90'] += x['RF90']['det_i']
            if not x['R0']['joined'] and x['RF90']['e0'] != x['R0']['e0']:
                diff.append((k, x['d'], 'consequence 1'))
            for rule in ('R0', 'RF90'):
                y = ag[(k[0], k[1], rule)]['draws'][x['d']]
                if y['e0'] != x[rule]['e0'] or y['det_i'] != x[rule]['det_i']:
                    diff.append((k, x['d'], rule, x[rule]['e0'], y['e0']))
    mean = lambda rs, rule, lab='l1': statistics.mean(statistics.mean(x[rule]['e0'][lab] for x in r['draws']) for r in rs)  # noqa: E731
    print(f"{m}: draws {n_draws}; R0 mismatches vs I06 {len(mism)} {mism[:5]}; mean L1 R0 {mean(rows, 'R0'):.3f} RF90 "
          f"{mean(rows, 'RF90'):.3f}; L2 R0 {mean(rows, 'R0', 'l2'):.3f} RF90 {mean(rows, 'RF90', 'l2'):.3f}; flips {flips} "
          f"breaks {breaks}; det_i rate R0 {det['R0'] / n_draws:.3f} RF90 {det['RF90'] / n_draws:.3f}; "
          f"differences vs agent rows {len(diff)} {diff[:3]}")
    for fam in sorted(fam_rows):
        rs = fam_rows[fam]
        fl = sum((not x['R0']['e0']['l1']) and x['RF90']['e0']['l1'] for r in rs for x in r['draws'])
        br = sum(x['R0']['e0']['l1'] and not x['RF90']['e0']['l1'] for r in rs for x in r['draws'])
        print(f"   {fam:5s} n {len(rs):2d} L1 {mean(rs, 'R0'):.3f} -> {mean(rs, 'RF90'):.3f} | L2 {mean(rs, 'R0', 'l2'):.3f} -> "
              f"{mean(rs, 'RF90', 'l2'):.3f} | flips {fl} breaks {br}")
    out[m] = {'draws': n_draws, 'mism': len(mism), 'l1': [mean(rows, 'R0'), mean(rows, 'RF90')], 'flips': flips,
              'breaks': breaks, 'det': [det['R0'] / n_draws, det['RF90'] / n_draws], 'diff_agent': len(diff)}
json.dump(out, open(HERE / 'out' / 'arm1b_v.json', 'w'), indent=1)
