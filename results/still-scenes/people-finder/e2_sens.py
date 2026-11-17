"""E2 addendum 1 (E2_ADDENDUM_1.md): S2 and its placebo on the fresh people-matter units (no cached goal-step call in any
arm), beside the frozen S2 on all units. Reported, not a bar. Zero calls.

    PYTHONPATH= .venv/bin/python e2_sens.py --people e2_people.json \
        --scenes run_scenes runs/e2_soclaas_qwen38_low runs/e2_soclaas_dsv41 [...]
"""
from __future__ import annotations

import argparse
import statistics as st

from e2_score import GOAL_STAGES, all_ok, detector, goal_seconds, load, sign_p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--people', required=True)
    ap.add_argument('--scenes', default=None)
    ap.add_argument('folders', nargs='+')
    a = ap.parse_args(argv)
    det_s = detector(a.people, a.scenes)[0]
    rows = load(a.folders)
    print('# E2 addendum 1: S2 on all people-matter units (frozen) and on the fresh ones (no cached goal-step call)')
    for model in sorted({r['model'] for r in rows}):
        pm = [r for r in rows if r['model'] == model and not r['vlm_error'] and r['people_matter'] and all_ok(r)]
        cached = {n: sum(1 for r in pm for c in r['arms'][n]['calls'] if c['stage'] in GOAL_STAGES and c.get('cached'))
                  for n in ('A0', 'A0b', 'D')}
        fresh = [r for r in pm if not any(c.get('cached') for n in ('A0', 'A0b', 'D') for c in r['arms'][n]['calls']
                                          if c['stage'] in GOAL_STAGES)]
        print(f'\n## {model}: cached goal-step calls {cached}')
        for name, sub in (('all people-matter units, every call ok', pm), ('fresh units', fresh)):
            g = {n: [goal_seconds(r['arms'][n], n, det_s) for r in sub] for n in ('A0', 'A0b', 'D')}
            f = sum(1 for x, y in zip(g['A0'], g['D']) if y < x)
            s = sum(1 for x, y in zip(g['A0'], g['D']) if y > x)
            bl = sum(1 for x, y in zip(g['A0'], g['A0b']) if y < x)
            bh = sum(1 for x, y in zip(g['A0'], g['A0b']) if y > x)
            pf, pp = sign_p(f, s), sign_p(bl, bh)
            s2 = pp > 0.05 and f > s and pf <= 0.05
            med = (f"median A0 {st.median(g['A0']):.1f}, A0b {st.median(g['A0b']):.1f}, D {st.median(g['D']):.1f}"
                   if sub else "no units")
            print(f"- {name}: n {len(sub)}; {med}; D faster {f}, slower {s}, p {pf:.4f}; placebo {bl}/{bh}, "
                  f"p {pp:.4f}; S2 {s2 if pp > 0.05 else 'not judgeable'}")


if __name__ == '__main__':
    main()
