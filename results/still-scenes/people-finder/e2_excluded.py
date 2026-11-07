"""E2, GLM's left-out units (E2_DEVIATIONS item 44): written and hashed before any GLM score was read, after the
controller flagged that the 11 left-out units are uneven (8 conversation group, 3 waiting line). Reported, not a bar.
Zero calls.

    PYTHONPATH= .venv/bin/python e2_excluded.py --people e2_people.json \
        --scenes run_scenes runs/e2_soclaas_glm53

1. Each left-out unit (latest row has a vlm_error): its row, whether the model's check said people matter, and which
   arm's which call failed.
2. Worst case for D: every left-out unit counted as A0 right and D wrong (harm + k, the floor rw unchanged), and as D
   slower than A0 in time. Bars A, G and S2, and H2, are recomputed under it. If a bar holds in the worst case, the
   left-out units cannot have turned it.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict

from e2_score import all_ok, detector, goal_seconds, layout_row, load, ok, sign_p, signflip_p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--people', required=True)
    ap.add_argument('--scenes', default=None)
    ap.add_argument('folders', nargs='+')
    a = ap.parse_args(argv)
    det_s = detector(a.people, a.scenes)[0]
    rows = load(a.folders)
    print('# E2: left-out units and the worst case for D')
    for model in sorted({r['model'] for r in rows}):
        rs = [r for r in rows if r['model'] == model]
        out = [r for r in rs if r['vlm_error']]
        pm = [r for r in rs if not r['vlm_error'] and r['people_matter']]
        print(f'\n## {model}: {len(out)} left-out units')
        print(f"- rows: {dict(Counter(layout_row(r['scene'])[1] for r in out))}")
        print(f"- people-matter by the model's check: {sum(r['people_matter'] for r in out)} of {len(out)}")
        fails = Counter((arm, c['stage']) for r in out for arm, v in r['arms'].items() for c in v['calls']
                        if c['status'] == 'vlm_error')
        print(f'- failed calls (arm, stage): {dict(fails)}')
        for r in out:
            bad = sorted({arm for arm, v in r['arms'].items() if any(c['status'] == 'vlm_error' for c in v['calls'])})
            print(f"  - {r['scene']} {r['episode']} t{r['t']}: failed arms {bad}")
        k = len(out)
        rw = sum(1 for r in pm if ok(r['arms']['A0']) and not ok(r['arms']['A0b']))
        gain = sum(1 for r in pm if not ok(r['arms']['A0']) and ok(r['arms']['D']))
        harm = sum(1 for r in pm if ok(r['arms']['A0']) and not ok(r['arms']['D']))
        lay = defaultdict(int)
        for r in pm:
            lay[r['scene']] += int(ok(r['arms']['D'])) - int(ok(r['arms']['A0']))
        worst = defaultdict(int, lay)
        for r in out:
            worst[r['scene']] -= 1
        net_w = sum(worst.values())
        p_w = signflip_p(list(worst.values()))
        tm = [r for r in pm if all_ok(r)]
        f = sum(1 for r in tm if goal_seconds(r['arms']['D'], 'D', det_s) < goal_seconds(r['arms']['A0'], 'A0', det_s))
        s = sum(1 for r in tm if goal_seconds(r['arms']['D'], 'D', det_s) > goal_seconds(r['arms']['A0'], 'A0', det_s))
        print(f'- as scored: harm {harm}, gain {gain}, rw {rw}; time faster {f}, slower {s}')
        print(f'- worst case (+{k} harm, +{k} slower): A {(harm + k) - gain <= rw} ((harm + k) - gain = '
              f'{harm + k - gain} vs rw {rw}); G {not (p_w <= 0.05 and net_w < 0)} (net {net_w:+d}, p {p_w:.4f}); '
              f'S2 {f > s + k and sign_p(f, s + k) <= 0.05} (p {sign_p(f, s + k):.4f}); H2 {net_w > 0 and p_w <= 0.05}')


if __name__ == '__main__':
    main()
