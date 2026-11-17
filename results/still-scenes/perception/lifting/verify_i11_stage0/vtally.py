"""Independent verifier: tallies and criterion verdicts from vrows_<set>.jsonl (FROZEN.md sections 6-8)."""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
LINES = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    LINES.append(s)


def load(name):
    return [json.loads(l) for l in open(OUT / f'vrows_{name}.jsonl')]


def clus(rs):
    return len({r['layout'] for r in rs}), len({(r['model'], r['layout']) for r in rs})


def fix_stats(rows, f):
    pt = [r for r in rows if r['has_point']]
    gd = [r for r in pt if r['model'] in ('glm53', 'dsv41')]
    w46 = [r for r in gd if not r['right']]
    r454 = [r for r in gd if r['right']]
    w53 = [r for r in pt if not r['right']]
    r699 = [r for r in pt if r['right']]
    fixes46 = [r for r in w46 if r['fx'][f]['right_after']]
    brk454 = [r for r in r454 if not r['fx'][f]['right_after']]
    fixes53 = [r for r in w53 if r['fx'][f]['right_after']]
    brk699 = [r for r in r699 if not r['fx'][f]['right_after']]
    s46 = [r for r in fixes46 if r['fx'][f]['succ_after']]
    s53 = [r for r in fixes53 if r['fx'][f]['succ_after']]

    def net(rs):
        g = sum((not r['rec_success']) and r['fx'][f]['succ_after'] for r in rs)
        lo = sum(r['rec_success'] and not r['fx'][f]['succ_after'] for r in rs)
        return g, lo
    gGD, lGD = net(gd)
    gA, lA = net(pt)
    return dict(n46=len(w46), n454=len(r454), n53=len(w53), n699=len(r699),
                f46=len(fixes46), b454=len(brk454), f53=len(fixes53), b699=len(brk699), s46=len(s46), s53=len(s53),
                netGD=gGD - lGD, gGD=(gGD, lGD), netAll=gA - lA, gAll=(gA, lA),
                mvR=sum(r['fx'][f]['moved'] for r in r454), mvW=sum(r['fx'][f]['moved'] for r in w46),
                mvR699=sum(r['fx'][f]['moved'] for r in r699), mvW53=sum(r['fx'][f]['moved'] for r in w53),
                fix_cls46=dict(collections.Counter(r['lclass'] for r in fixes46)),
                fix_cls53=dict(collections.Counter(r['lclass'] for r in fixes53)),
                fix_kind53=dict(collections.Counter(r['kind'] for r in fixes53)),
                brk_kind699=dict(collections.Counter(r['kind'] for r in brk699)),
                clus46=clus(fixes46), fixes46=fixes46, brk699=brk699)


def primary():
    rows = load('primary')
    say('=' * 90)
    say('PRIMARY (9 runs)')
    say(f'records {len(rows)}; by model', dict(collections.Counter(r['model'] for r in rows)))
    say('with a point by model', dict(collections.Counter(r['model'] for r in rows if r['has_point'])))
    pt = [r for r in rows if r['has_point']]
    say('wrong (recorded target_correct False) by model',
        dict(collections.Counter(r['model'] for r in rows if not r['rec_tc'])))
    say('wrong with point by model', dict(collections.Counter(r['model'] for r in pt if not r['right'])))
    say('right with point by model', dict(collections.Counter(r['model'] for r in pt if r['right'])))
    say('tc disagreements', sum(r['tc_disagree'] for r in pt), 'max lift err',
        max(r['lift_err'] for r in pt), 'vec mismatches', sum(r.get('vec_bad', 0) for r in pt), 'of',
        sum(r.get('vec_n', 0) for r in pt), 'ntargets', set(r['ntargets'] for r in rows))
    gd = [r for r in pt if r['model'] in ('glm53', 'dsv41')]
    w46 = [r for r in gd if not r['right']]
    w53 = [r for r in pt if not r['right']]
    c46 = collections.Counter(r['lclass'] for r in w46)
    c53 = collections.Counter(r['lclass'] for r in w53)
    lift46 = sum(c46[k] for k in ('occluder', 'background', 'floor slip'))
    say(f'classes on 46: {dict(c46)}; lift classes {lift46}/46 = {lift46 / 46:.3f}; over 50 {lift46}/50')
    say(f'classes on 53: {dict(c53)}')
    for k in ('person', 'parcel', 'place'):
        say(f'  {k}: n={sum(r["kind"] == k for r in w53)} {dict(collections.Counter(r["lclass"] for r in w53 if r["kind"] == k))}')
    liftrs = [r for r in w46 if r['lclass'] in ('occluder', 'background', 'floor slip')]
    say('lift classes on 46 clustering (layouts, model x layout):', clus(liftrs))
    nopt = [r for r in rows if not r['has_point']]
    say('no point:', len(nopt), dict(collections.Counter((r['model'], r['rec_tc']) for r in nopt)))
    res = {}
    for f in ('oracle', 'V1', 'V2', 'V3', 'V4', 'front'):
        res[f] = fix_stats(rows, f)
    og, og3, ogn = res['oracle']['f46'] - res['oracle']['b454'], res['oracle']['f53'] - res['oracle']['b699'], \
        res['oracle']['netGD']
    for f, s in res.items():
        gain = s['f46'] - s['b454']
        gain3 = s['f53'] - s['b699']
        s['share'] = gain / og if og > 0 else None
        s['share3'] = gain3 / og3 if og3 > 0 else None
        s['shareNet'] = s['netGD'] / ogn if ogn > 0 else None
        say(f"{f:6s} fixes {s['f46']}/{s['n46']} breaks {s['b454']}/{s['n454']} | fixes {s['f53']}/{s['n53']} "
            f"breaks {s['b699']}/{s['n699']} | fixed->success {s['s46']}/{s['f46']} (all {s['s53']}/{s['f53']}) | "
            f"net GD {s['netGD']:+d} {s['gGD']} all {s['netAll']:+d} {s['gAll']} | moved R {s['mvR']}/454 W {s['mvW']}/46 "
            f"(R699 {s['mvR699']}, W53 {s['mvW53']}) | share {s['share']} share3 {s['share3']} shareNet {s['shareNet']}")
        say(f"        fix classes 46 {s['fix_cls46']} 53 {s['fix_cls53']} kinds53 {s['fix_kind53']} breaks by kind "
            f"{s['brk_kind699']} clustering of 46-fixes {s['clus46']}")
    # losses baselines
    for f in ('oracle', 'V1'):
        lo = [r for r in pt if r['rec_success'] and not r['fx'][f]['succ_after']]
        say(f'{f} losses {len(lo)}: baseline replays reproduce success: {[r.get(f + "_loss_baseline") for r in lo]}',
            [(r['model'], r['scene'], r['episode'], r['trial'], r['right'], r['fx'][f]['right_after']) for r in lo])
    # I33
    p15 = [r for r in w53 if r['kind'] == 'person']
    d1 = [r for r in pt if r['right'] and r['kind'] == 'person']
    d2 = [r for r in pt if r['right']]
    for g in ('G1', 'G2'):
        t1c = sum(r[f'T1_{g}'] for r in p15)
        t2c = sum(r[f'T2_{g}'] for r in p15)
        say(f'I33 {g}: person wrong n={len(p15)} by model {dict(collections.Counter(r["model"] for r in p15))}; '
            f'T1 catches {t1c}, T2 {t2c}; D1 n={len(d1)} T1 {sum(r[f"T1_{g}"] for r in d1)} '
            f'({sum(r[f"T1_{g}"] for r in d1) / len(d1):.3f}) T2 {sum(r[f"T2_{g}"] for r in d1)}; '
            f'D2 n={len(d2)} T1 {sum(r[f"T1_{g}"] for r in d2)} T2 {sum(r[f"T2_{g}"] for r in d2)}')
        say(f'   T1 by class (person wrong): {dict(collections.Counter((r["lclass"], r[f"T1_{g}"]) for r in p15))}')
        say(f'   T1 on all 53 by class: {dict(collections.Counter((r["lclass"], r[f"T1_{g}"]) for r in w53))}; '
            f'flagged {sum(r[f"T1_{g}"] for r in w53)}/53')
        fl = [r for r in pt if r[f'T1_{g}']]
        say(f'   T1 flags {len(fl)}/{len(pt)} with point; wrong among flagged {sum(not r["right"] for r in fl)}')
    say('person wrong clustering', clus(p15), dict(collections.Counter(r['layout'] for r in p15)))
    # criteria
    o, v1 = res['oracle'], res['V1']
    S1 = lift46 >= 19
    S2 = o['f46'] >= 15 and o['b454'] <= 2
    S3 = o['s46'] >= 8
    S4 = v1['share'] is not None and v1['share'] >= 0.70
    K = [lift46 <= 9, o['f46'] < 6, o['b454'] > 5]
    verdict = 'KILL' if any(K) else ('SUCCESS' if all([S1, S2, S3, S4]) else 'NEITHER')
    v1kill = v1['f46'] < 6 or v1['b454'] > 5
    t1c = sum(r['T1_G1'] for r in p15)
    t1fp = sum(r['T1_G1'] for r in d1)
    i33 = 'KILL' if t1c < 10 else ('SUCCESS' if (t1c >= 12 and t1fp / len(d1) <= 0.10) else 'NEITHER')
    say(f'CRITERIA: S1 {S1} ({lift46}/46) S2 {S2} ({o["f46"]}/46, {o["b454"]}/454) S3 {S3} ({o["s46"]}) '
        f'S4 {S4} (share {v1["share"]}) kill {K} -> verdict {verdict}; V1 kill {v1kill}; I33 {i33} '
        f'(T1 {t1c}/15, D1 {t1fp}/{len(d1)}); choice share {c46["choice"]}/46 = {c46["choice"] / 46:.3f}')
    summ = dict(lift46=lift46, choice46=c46['choice'], cls46=dict(c46), cls53=dict(c53),
                verdict=verdict, v1kill=v1kill, i33=i33, t1c=t1c, t1fp=t1fp, nd1=len(d1),
                res={f: {k: v for k, v in s.items() if k not in ('fixes46', 'brk699')} for f, s in res.items()})
    (OUT / 'vsummary_primary.json').write_text(json.dumps(summ, indent=1, default=str))
    return rows, res


def extras(name):
    try:
        rows = load(name)
    except FileNotFoundError:
        return
    say('=' * 90)
    say(f'EXTRA {name}: {len(rows)} records, runs {len({r["run"] for r in rows})}')
    pt = [r for r in rows if r['has_point']]
    bad_lift = [r for r in pt if r['lift_err'] is None or r['lift_err'] > 1e-9]
    say(f'  with point {len(pt)}; lift repro failures {len(bad_lift)}; tc disagreements {sum(r["tc_disagree"] for r in pt)}'
        f'; multi-target records {sum(r["ntargets"] > 1 for r in rows)}')
    pt = [r for r in pt if not (r['lift_err'] is None or r['lift_err'] > 1e-9)]
    models = sorted({r['model'] for r in rows})
    pool = collections.Counter()
    for m in models:
        rs = [r for r in pt if r['model'] == m]
        w = [r for r in rs if not r['right']]
        rt = [r for r in rs if r['right']]
        p = [r for r in w if r['kind'] == 'person']
        d1 = [r for r in rt if r['kind'] == 'person']
        line = (f'  {m:14s} wrong {len(w):3d} lift {sum(r["lclass"] in ("occluder", "background", "floor slip") for r in w):3d} '
                f'cls {dict(collections.Counter(r["lclass"] for r in w))}')
        for f in ('oracle', 'V1', 'V3'):
            line += (f' | {f} {sum(r["fx"][f]["right_after"] for r in w)}/{len(w)} br '
                     f'{sum(not r["fx"][f]["right_after"] for r in rt)}/{len(rt)}')
        say(line)
        t1 = sum(r['T1_G1'] for r in p)
        fp = sum(r['T1_G1'] for r in d1)
        say(f'      I33: person wrong {len(p)} T1 {t1}; D1 {fp}/{len(d1)} = {fp / max(len(d1), 1):.3f}; '
            f'D2 {sum(r["T1_G1"] for r in rt)}/{len(rt)}; person wrong by class '
            f'{dict(collections.Counter((r["lclass"], r["T1_G1"]) for r in p))}; layouts {clus(p)}')
        pool['p'] += len(p)
        pool['t1'] += t1
        pool['d1'] += len(d1)
        pool['fp'] += fp
        pool['d2'] += len(rt)
        pool['fp2'] += sum(r['T1_G1'] for r in rt)
    say(f'  pooled (post hoc): T1 {pool["t1"]}/{pool["p"]}; D1 {pool["fp"]}/{pool["d1"]}; D2 {pool["fp2"]}/{pool["d2"]}')
    allp = [r for r in pt if not r['right'] and r['kind'] == 'person']
    say(f'  pooled person wrong by class: {dict(collections.Counter((r["lclass"], r["T1_G1"]) for r in allp))}; '
        f'clustering {clus(allp)}')


if __name__ == '__main__':
    primary()
    for n in sys.argv[1:]:
        extras(n)
    (OUT / 'vtally_out.txt').write_text('\n'.join(LINES) + '\n')
