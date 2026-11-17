"""E1b scorer (E1_FROZEN.md), zero calls.   e1_score.py runs/e1_<model> [more runs...]
Per model, on the people-matter records: success of the recorded C, of A0 (fresh VLM people) and of D (detector
people); the redraw floor = flips between the recorded C and A0 (same plan, context and target; a fresh draw of the
people and constraints calls); D against A0 (gain, harm); a layout-level sign-flip test; and the latency per record.
"""
from __future__ import annotations

import collections
import itertools
import json
import statistics as st
import sys
from pathlib import Path

DETECTOR_S = 0.0596          # A1 warm detector + pose time per image on the RTX 4060 (phase1c/README.md)


def main(paths):
    rows = [json.loads(l) for p in paths for l in (Path(p) / 'records.jsonl').read_text().splitlines() if l.strip()]
    pairs = collections.defaultdict(dict)
    for r in rows:
        pairs[(r['model'], r['run'], r['scene'], r['episode'], r['rec_trial'])][r['arm']] = r
    out = []
    say = out.append
    for model in sorted({k[0] for k in pairs}):
        pp = [v for k, v in pairs.items() if k[0] == model and 'A0' in v and 'D' in v]
        ok = lambda r: bool(r.get('score') and r['score']['success'])
        rec = sum(v['A0']['recorded_success'] for v in pp)
        a0 = sum(ok(v['A0']) for v in pp)
        d = sum(ok(v['D']) for v in pp)
        rw = sum(1 for v in pp if v['A0']['recorded_success'] and not ok(v['A0']))
        wr = sum(1 for v in pp if not v['A0']['recorded_success'] and ok(v['A0']))
        harm = sum(1 for v in pp if ok(v['A0']) and not ok(v['D']))
        gain = sum(1 for v in pp if not ok(v['A0']) and ok(v['D']))
        net = collections.Counter()
        for k, v in pairs.items():
            if k[0] == model and 'A0' in v and 'D' in v:
                net[k[2].split('_', 1)[1]] += int(ok(v['D'])) - int(ok(v['A0']))
        nz = [x for x in net.values() if x]
        obs = sum(nz)
        p = (sum(1 for s in itertools.product((1, -1), repeat=len(nz)) if abs(sum(a * b for a, b in zip(s, nz))) >= abs(obs))
             / 2 ** len(nz)) if nz else 1.0
        lat = lambda r, stage: sum(c['latency_s'] or 0.0 for c in r['calls'] if c['stage'] == stage)
        a0_people = [lat(v['A0'], 'people') for v in pp]
        a0_cons = [lat(v['A0'], 'constraints') for v in pp]
        d_cons = [lat(v['D'], 'constraints') for v in pp]
        noninf = (harm - gain) <= rw
        say(f'== {model}: {len(pp)} people-matter records')
        say(f'  success: recorded {rec}, A0 {a0}, D {d}')
        say(f'  redraw floor (recorded -> A0): right->wrong {rw}, wrong->right {wr}')
        say(f'  D vs A0: gain {gain}, harm {harm}, net {gain - harm:+d}; layout sign-flip p = {p:.3f} over {len(nz)} layouts')
        say(f'  latency per record (s): A0 people mean {st.mean(a0_people):.1f} (median {st.median(a0_people):.1f}), '
            f'A0 constraints mean {st.mean(a0_cons):.1f}, D constraints mean {st.mean(d_cons):.1f}, '
            f'detector {DETECTOR_S:.2f}; saved per record about {st.mean(a0_people) - DETECTOR_S:.1f} s; calls A0 2, D 1')
        say(f'  JUDGED: D non-inferior to A0 within the redraw floor (harm - gain <= {rw}): {noninf}')
    text = '\n'.join(out)
    print(text)
    Path(paths[0], 'e1_out.txt').write_text(text + '\n')


if __name__ == '__main__':
    main(sys.argv[1:])
