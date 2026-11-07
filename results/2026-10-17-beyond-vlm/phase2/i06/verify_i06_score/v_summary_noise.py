"""Verifier: noise validation (K4), noisy separations (criterion e, K1's second half), from my outputs.

    v_summary_noise.py  -> out/summary_noise.txt, out/summary_noise.json
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np

import vlib as V

L = []


def say(s=''):
    print(s, flush=True)
    L.append(s)


def boot(byl, n=2000):
    keys = sorted(byl)
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(n):
        pick = rng.integers(len(keys), size=len(keys))
        xs = [v for i in pick for v in byl[keys[i]]]
        vals.append(np.mean(xs) if xs else np.nan)
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def main():
    out = {'valid': {}, 'sep': {}}
    say('Validation (simulated C vs recorded C = rescore), n records, pooled over the seven runs')
    for key in ('qwen', 'glm', 'dsv'):
        p = V.OUT / f'validate_{key}.jsonl'
        if not p.exists():
            say(f'  {key}: missing')
            continue
        R = [json.loads(x) for x in p.read_text().splitlines() if x]
        n = len(R)
        rec = 100 * np.mean([r['rescore'] for r in R])
        sim = 100 * np.mean([r['sim'] for r in R])
        r5 = 100 * np.mean([r['r5'] for r in R])
        diffs = sum(r['recorded'] != r['rescore'] for r in R)
        gap = sim - rec
        out['valid'][key] = {'n': n, 'recorded': rec, 'sim': sim, 'r5': r5, 'gap': gap, 'valid': abs(gap) <= 10,
                             'steps': sum(r['step'] for r in R)}
        say(f'  {key}: n {n} (solver steps {sum(r["step"] for r in R)}); recorded {rec:.1f}; simulated {sim:.1f}; '
            f'gap {gap:+.1f}; R5 {r5:.1f} (sim-R5 {sim - r5:+.1f}); recorded != rescore {diffs} -> '
            f'{"valid" if abs(gap) <= 10 else "INVALID"}')
        for grp, f in (('dom', 'dom'), ('situation', 'situation')):
            parts = []
            for v in sorted({r[f] for r in R}):
                q = [r for r in R if r[f] == v]
                parts.append(f'{v} n {len(q)} rec {100 * np.mean([r["rescore"] for r in q]):.1f} sim '
                             f'{100 * np.mean([r["sim"] for r in q]):.1f} gap '
                             f'{100 * (np.mean([r["sim"] for r in q]) - np.mean([r["rescore"] for r in q])):+.1f}')
            say('     ' + '; '.join(parts))
    valid = [k for k, v in out['valid'].items() if v['valid']]
    say(f'K4: valid models {valid}; fires only if every model misses -> {"FIRES" if not valid else "does not fire"}')
    # arm 3 from my arms output
    arms = {}
    for s in (0, 1, 2):
        for x in (V.OUT / f'arms_seed{s}.jsonl').read_text().splitlines():
            r = json.loads(x)
            arms[(r['seed'], r['scene_id'])] = r
    say('\nNoisy C (arm 4), 20 draws per scene; separation = arm 3 - arm 4, x 100')
    for which in ('models', 'placeholder_only'):
        for key in ('qwen', 'glm', 'dsv'):
            rows = {}
            for f in sorted(V.OUT.glob(f'noisy_{which}_{key}_s*_*.jsonl')):
                for x in f.read_text().splitlines():
                    r = json.loads(x)
                    rows[(r['seed'], r['scene_id'])] = r
            if not rows:
                continue
            scope = 'all' if all(any(f'_s{s}_all' in str(f) for f in V.OUT.glob(f'noisy_{which}_{key}_s*_all.jsonl'))
                                 for s in (0, 1, 2)) else 'top'
            top = [k for k, r in rows.items() if r['kind'] == 'variant' and (r['family'], r['level']) in V.TOP]
            bas = [k for k, r in rows.items() if r['kind'] == 'base']
            a3t = 100 * np.mean([arms[k]['eps']['e0']['arm3']['l1'] for k in top])
            a4t = 100 * np.mean([rows[k]['l1'] for k in top])
            a3b = 100 * np.mean([arms[k]['eps']['e0']['arm3']['l1'] for k in bas])
            a4b = 100 * np.mean([rows[k]['l1'] for k in bas])
            byl = collections.defaultdict(list)
            for k in top:
                byl[(k[0], rows[k]['row'])].append(100 * (arms[k]['eps']['e0']['arm3']['l1'] - rows[k]['l1']))
            ci = boot(byl)
            no6 = [k for k in top if rows[k]['family'] != 'A']
            sep6 = 100 * (np.mean([arms[k]['eps']['e0']['arm3']['l1'] for k in no6]) - np.mean([rows[k]['l1'] for k in no6]))
            agree = all(r['agree'] for r in rows.values())
            say(f'  [{which}] {key} ({scope}, scenes {len(rows)}): top n {len(top)}: arm3 {a3t:.1f} arm4 {a4t:.1f} sep '
                f'{a3t - a4t:+.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}]; bases n {len(bas)}: arm3 {a3b:.1f} arm4 {a4b:.1f} sep '
                f'{a3b - a4b:+.1f}; top without A6 n {len(no6)} sep {sep6:+.1f}; episodes agree {agree}')
            if which == 'models':
                out['sep'][key] = {'top': a3t - a4t, 'bases': a3b - a4b, 'n_top': len(top), 'scope': scope,
                                   'no_A6': sep6}
                # per family top level separations (K1 second half) and per (family, level) if full
                fam_top = {}
                for fam, lv in V.TOP:
                    ks = [k for k in top if rows[k]['family'] == fam]
                    fam_top[fam] = 100 * (np.mean([arms[k]['eps']['e0']['arm3']['l1'] for k in ks]) -
                                          np.mean([rows[k]['l1'] for k in ks]))
                out['sep'][key]['family_top'] = fam_top
                say('     family top-level separations: ' + ', '.join(f'{f} {v:+.1f}' for f, v in fam_top.items()))
                if scope == 'all':
                    parts = []
                    for fam, lvls in (('A', [1, 2, 3, 4, 5, 6]), ('B', [1]), ('C', [1]), ('D', [2, 5]), ('E', [1, 2])):
                        for lv in lvls:
                            ks = [k for k, r in rows.items() if r['kind'] == 'variant' and r['family'] == fam and r['level'] == lv]
                            parts.append(f'{fam}{lv} n {len(ks)} a4 {100 * np.mean([rows[k]["l1"] for k in ks]):.1f}')
                    say('     ' + '; '.join(parts))
                    var = [k for k, r in rows.items() if r['kind'] == 'variant']
                    dis = np.mean([r_l1 - r_l2 for r_l1, r_l2 in ((rows[k]['l1'], rows[k]['l2']) for k in var)])
                    say(f'     arm 4 L1/L2 disagreement over variants {100 * dis:.1f}% (mean over draws)')
                    deti = np.mean([d['det']['i'] for k in var for d in rows[k]['draws']])
                    say(f'     arm 4 detector (i) true in {100 * deti:.1f}% of variant draws')
    ok_e = [k for k in valid if k in out['sep'] and out['sep'][k]['top'] >= 10]
    say(f'\n(e): valid models >= 10 points at the top level: {ok_e} -> {"MET" if len(ok_e) >= 2 else "NOT MET"}')
    (V.OUT / 'summary_noise.txt').write_text('\n'.join(L) + '\n')
    (V.OUT / 'summary_noise.json').write_text(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
