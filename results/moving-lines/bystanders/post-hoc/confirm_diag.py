"""POST HOC diagnosis of the bystander confirmation (after its frozen judgement). Zero calls.
1. cells whose first answer source is None; 2. false-drop cells (CT3, CT3b); 3. where CT3's residual capture comes
from; 4. are CT3's wins over CT2 in the cells where CT2 was captured?"""
import sys
from collections import Counter, defaultdict
sys.path.insert(0, 'results/2026-11-01-push/moving')
from bystanders_score import load, cells, capture, false_drops, named_bystanders, measures
SC = {"cafe_line2_bystanders", "sweet_spot_line2_bystanders"}
by = cells(load(['runs/dsv41'], SC, set(range(15, 45))))
print('1. first answers with source None:')
for k, v in sorted(by.items()):
    a = (v['CT2'].get('answers') or [{}])
    if a[0].get('source') is None:
        print(f"   {k[1]} s{k[2]}: answers {[(x.get('status'), x.get('delivered_t'), x.get('error')) for x in v['CT2'].get('answers', [])]}; "
              f"resolves CT2 {len(v['CT2'].get('resolves', []))}, Gl {len(v['Gl'].get('resolves', []))}; "
              f"shares " + ', '.join(f"{arm} {measures(v[arm])['share_strict']:.2f}" for arm in ('CT2', 'CT3', 'CT3b', 'Gl')))
print('2. false drops:')
for arm in ('CT3', 'CT3b'):
    for k, v in sorted(by.items()):
        bad, logged = false_drops(v[arm])
        if bad:
            nf = {}
            for r in v[arm]['resolves']:
                nf.update({int(i): t for i, t in (r.get('non_followers') or {}).items()})
            print(f"   {arm} {k[1]} s{k[2]}: false drop {bad} at {[round(nf[i], 1) for i in bad]} (queue_ids logged {logged})")
print('3. CT3 residual capture, by cause:')
causes = Counter()
for k, v in sorted(by.items()):
    c = capture(v['CT3'])
    if not c or c[0] < 1.0:
        continue
    e = v['CT3']
    bys = set(e['bystanders'])
    named = set(named_bystanders(e) or [])
    rs = e['resolves']
    per = defaultdict(float)
    for r, nxt in zip(rs, rs[1:] + [None]):
        last = r.get('last')
        if last is None or int(last) not in bys:
            continue
        dt = (e['samples'][-1]['t'] if nxt is None else nxt['delivered_t']) - r['delivered_t']
        cause = 'named by the model' if int(last) in named else ('gate open (not exempt)' if int(last) not in set(r.get('exempt') or []) else 'exempt but last?!')
        per[cause] += dt
    for cz, s in per.items():
        causes[cz] += s
    print(f"   {k[1]} s{k[2]}: CT3 {c[0]:.1f} s (CT3b {capture(v['CT3b'])[0]:.1f}, CT2 {capture(v['CT2'])[0]:.1f}); named {sorted(named)}; " +
          ', '.join(f'{cz} {s:.1f} s' for cz, s in per.items()))
print('   totals:', {k: round(s, 1) for k, s in causes.items()})
print('4. CT3 vs CT2 strict share, split by whether CT2 was captured (>= 1 s):')
for label, pick in (('CT2 captured', lambda c: c >= 1.0), ('CT2 not captured', lambda c: c < 1.0)):
    rows = [(measures(v['CT3'])['share_strict'], measures(v['CT2'])['share_strict']) for k, v in by.items()
            if capture(v['CT2']) is not None and pick(capture(v['CT2'])[0])]
    w = sum(a > b + 1e-9 for a, b in rows); l = sum(a < b - 1e-9 for a, b in rows)
    print(f"   {label}: {len(rows)} cells, {w}/{l}/{len(rows) - w - l}, mean {sum(a - b for a, b in rows) / max(len(rows), 1):+.3f}")
for arm in ('CT2', 'CT3', 'CT3b', 'Gl'):
    m = [measures(v[arm])['share_strict'] for v in by.values()]
    print(f'   mean strict share {arm}: {sum(m) / len(m):.3f}')
