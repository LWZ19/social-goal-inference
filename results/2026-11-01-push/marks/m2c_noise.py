"""POST HOC (questioner's M2 notes 1 and 5): the noise floor of right/wrong flips between two fresh draws of the
same unmarked arm (A0 trial t against A0 trial t', same episodes), set beside A0 against A2 at the same trial;
and centre-nearest picks against chance, by the number of marks shown. Zero calls. m2c_noise.py runs/m2_<model>"""
import collections, itertools, json, math, sys
from pathlib import Path
run = Path(sys.argv[1])
rows = [json.loads(l) for l in (run / 'm2_rows.jsonl').read_text().splitlines() if l.strip()]
by = {(r['scene'], r['episode'], r['trial']): r for r in rows}
eps = sorted({(r['scene'], r['episode']) for r in rows})
out = []; say = out.append
def flips(get_a, get_b, pairs):
    rw = wr = n = 0
    for a, b in pairs:
        if a is None or b is None: continue
        x, y = get_a(a), get_b(b)
        if x is None or y is None: continue
        n += 1; rw += bool(x) and not y; wr += (not x) and bool(y)
    return n, rw, wr
for field, label in (('right', 'target right'), ('replay', 'replayed success')):
    a0 = f'A0_{field}'; a2 = f'A2_both_{field}'
    if not any(r.get(a0) is not None for r in rows): continue
    pairs = [(by.get((s, e, t)), by.get((s, e, u))) for s, e in eps for t, u in itertools.combinations((3, 4, 5), 2)]
    n, rw, wr = flips(lambda r: r.get(a0), lambda r: r.get(a0), pairs)
    say(f'{label}: A0 vs A0 (two fresh unmarked draws, {n} comparisons): right->wrong {rw}, wrong->right {wr}, '
        f'per 252: {rw * 252 / n:.1f} / {wr * 252 / n:.1f}')
    same = [(by.get((s, e, t)), by.get((s, e, t))) for s, e in eps for t in (3, 4, 5)]
    n2, rw2, wr2 = flips(lambda r: r.get(a0), lambda r: r.get(a2), same)
    say(f'{label}: A0 vs A2 (same trial, {n2} comparisons): right->wrong {rw2}, wrong->right {wr2}')
# sign test on A0 vs A2 discordant pairs
def sign_p(k, n):
    from math import comb
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)
say(f"two-sided sign test on target-right discordant pairs: p = {sign_p(min(rw2, wr2), rw2 + wr2):.3f}" if False else '')
# salience by number of marks shown
by_k = collections.defaultdict(lambda: [0, 0])
for r in rows:
    if r.get('mark') is None: continue
    k = r['n_marks']; by_k[k][0] += 1; by_k[k][1] += bool(r.get('mark_is_centre'))
say('centre-nearest picks by marks shown: ' + ', '.join(f'K={k}: {c}/{n} (chance {n / k:.1f})' for k, (n, c) in sorted(by_k.items())))
exp = sum(n / k for k, (n, c) in by_k.items()); obs = sum(c for n, c in by_k.values())
say(f'centre-nearest overall: observed {obs}, expected by chance {exp:.1f}')
text = '\n'.join(x for x in out if x); print(text); (run / 'm2c_out.txt').write_text(text + '\n')
