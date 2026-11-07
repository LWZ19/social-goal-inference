"""Tables from vrows_*.jsonl (independent verifier). Prints to stdout."""
import collections
import glob
import json
import os
import sys

OUT = 'results/2026-10-17-beyond-vlm/phase1/i01_repairs/verify'
ARMS = ['replayC', 'R0', 'R1', 'R2', 'R3', 'R4', 'R4s', 'P', 'Ps', 'R5', 'R6', 'R3_1m', 'R4_1m', 'P_1m']
SIT = [('no_one', 'nobody'), ('waiting_line', 'lines'), ('conversation_group', 'groups')]
ORDER = ['old_qwen38_low', 'old_glm53', 'old_dsv41', 'new_qwen38_low', 'new_glm53', 'new_dsv41', 'new_haiku55',
         'new_sonnet55_low', 'new_gemini38_low']


def val(r, arm):
    if not r['solver_step']:
        return r['R6x'] if arm == 'R6x' else r['C']
    return r[arm]


def line_failure(r):
    return (r['solver_step'] and r['situation'] == 'waiting_line' and r['target_correct'] and not r['C']
            and (not r['in_region'] or not r['reachable']))


for key in ORDER:
    path = f'{OUT}/vrows_{key}.jsonl'
    if not os.path.exists(path):
        continue
    rows = [json.loads(x) for x in open(path)]
    ss = [r for r in rows if r['solver_step']]
    C = sum(r['C'] for r in rows)
    print(f'=== {key}: records {len(rows)}, solver steps {len(ss)}, reproduced {sum(r["reproduced"] for r in ss)}, '
          f'C {C}, replayC-on-all {sum(val(r, "replayC") for r in rows)}')
    for sit, name in SIT + [(None, 'all')]:
        rs = [r for r in rows if sit is None or r['situation'] == sit]
        parts = [f'C {sum(r["C"] for r in rs)}']
        for a in ARMS + ['R6x']:
            s = sum(val(r, a) for r in rs)
            fl = sum(val(r, a) and not r['C'] for r in rs)
            br = sum(r['C'] and not val(r, a) for r in rs)
            parts.append(f'{a} {s}(+{fl}/-{br})')
        print(f'  {name:7s} n={len(rs):3d} ' + ' '.join(parts))
    lf = [r for r in rows if line_failure(r)]
    print(f'  line failures n={len(lf)}: ' + ' '.join(f'{a} {sum(r[a] for r in lf)}' for a in ARMS))
    # net pp
    print('  net pp of 252: ' + ' '.join(f'{a} {100 * (sum(val(r, a) for r in rows) - C) / len(rows):+.1f}'
                                         for a in ARMS + ['R6x']))
    # R4/P breaks detail
    for a in ('P', 'R3', 'R4', 'R2', 'R1', 'R5', 'R6'):
        br = [r for r in ss if r['C'] and not r[a]]
        if br:
            print(f'  breaks {a}: ' + '; '.join(f'{r["arm"]} {r["scene"]} {r["episode"]} t{r["trial"]} {r["situation"]} '
                                                 f'miss{len(r["missed"])} spur{len(r["spurious"])}' for r in br))
    # wrong-target among goal steps, R6
    wt = [r for r in ss if not r['target_correct']]
    wt_all = [r for r in rows if not r['target_correct']]
    print(f'  wrong target: goal steps {len(wt)}, R6 fixes {sum(r["R6"] for r in wt)}; all records {len(wt_all)} '
          f'(status not ok {sum(r["status"] != "ok" for r in wt_all)}), R6x fixes {sum(val(r, "R6x") for r in wt_all)}')
    print(f'  R6 target_correct false on goal steps: {sum(1 for r in ss if r.get("R6_tc") is False)}; oracle None: '
          f'{sum(r["oracle_none"] for r in rows)}; unprojectable missed: {sum(r.get("unprojectable", 0) for r in ss)}')
    if lf:
        # mechanism: P flips via R4
        pf = [r for r in lf if r['P']]
        print(f'  P flips {len(pf)}, of which R4 flips {sum(r["R4"] for r in pf)}, R3 {sum(r["R3"] for r in pf)}, '
              f'R1 {sum(r["R1"] for r in pf)}; R4 flips total {sum(r["R4"] for r in lf)}; R4 flips not P: '
              f'{sum(r["R4"] and not r["P"] for r in lf)}')
        dists = []
        for r in lf:
            if r['R4']:
                listed = [float(r['spur_dist'][str(n)]) for n in r['come_after'] if str(n) in r['spur_dist']]
                dists.append(min(listed) if listed else None)
        print('  R4 flips: min dist of come_after-listed spurious detection to a visible true person: ',
              sorted(round(d, 2) for d in dists if d is not None), 'none:', sum(d is None for d in dists))
        far = [r for r in lf if any(float(r['spur_dist'][str(n)]) > 3.0 for n in r['come_after'] if str(n) in r['spur_dist'])]
        print(f'  line failures whose come_after lists a spurious detection >3 m from every visible true person: {len(far)}')
