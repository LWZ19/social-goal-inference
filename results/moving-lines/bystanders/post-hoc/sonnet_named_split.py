"""POST HOC (after the Sonnet addendum's judgement). Zero calls. Is the Sonnet capture where the model named a
bystander? Per cell: named bystanders (CT2's answer), capture by arm, and for CT3 whether each captured second's `last`
was a named bystander or an un-named one (gate open)."""
import sys
from collections import defaultdict
sys.path.insert(0, 'results/2026-11-01-push/moving')
from bystanders_score import load, cells, capture, named_bystanders, measures, BYS
by = cells([e for e in load(['runs/sonnet55_low'], set(BYS), set(range(10, 20))) if e['label'].startswith('claude_sonnet')])
groups = defaultdict(lambda: defaultdict(float))
cause = defaultdict(float)
for k, v in sorted(by.items()):
    named = set(named_bystanders(v['CT2']) or [])
    g = 'named' if named else 'not named'
    groups[g]['cells'] += 1
    for a in ('CT2', 'CT3', 'Gl'):
        c = capture(v[a])
        groups[g][a + ' s'] += c[0] if c else 0
        groups[g][a + ' cells'] += bool(c and c[0] >= 1.0)
    for a in ('CT2', 'CT3'):
        groups[g][a + ' share'] += measures(v[a])['share_strict']
    e = v['CT3']; rs = e['resolves']; bys = set(e['bystanders'])
    for r, nxt in zip(rs, rs[1:] + [None]):
        last = r.get('last')
        if last is not None and int(last) in bys:
            dt = (e['samples'][-1]['t'] if nxt is None else nxt['delivered_t']) - r['delivered_t']
            cause['named' if int(last) in named else 'not named (gate open)'] += dt
for g, d in groups.items():
    n = d['cells']
    print(f"{g}: {int(n)} cells | capture s CT2 {d['CT2 s']:.1f} ({int(d['CT2 cells'])} cells), CT3 {d['CT3 s']:.1f} "
          f"({int(d['CT3 cells'])}), Gl {d['Gl s']:.1f} ({int(d['Gl cells'])}) | mean share CT2 {d['CT2 share'] / n:.3f}, CT3 {d['CT3 share'] / n:.3f}")
print('CT3 captured seconds by whether the capturing bystander was named:', {k: round(s, 1) for k, s in cause.items()})
