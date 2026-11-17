"""Tables from the verifier's rows (verify/rows_*.jsonl). Prints and writes verify/v_tables.txt."""
import json, glob, collections, sys
import numpy as np

V = 'results/still-scenes/metrics-and-statistics/metric-v0/verify'
rows = []
for g in ('old', 'watch', 'new_a', 'new_b'):
    try:
        rows += [json.loads(l) for l in open(f'{V}/rows_{g}.jsonl')]
    except FileNotFoundError:
        pass
out = []
P = lambda *a: out.append(' '.join(str(x) for x in a))

ORDER = [('old', 'qwen38_low'), ('old', 'glm53'), ('old', 'dsv41'), ('watch', 'qwen38_low'), ('watch', 'glm53'),
         ('watch', 'dsv41'), ('new', 'qwen38_low'), ('new', 'glm53'), ('new', 'dsv41'), ('new', 'haiku55'),
         ('new', 'sonnet55_low'), ('new', 'gemini38_low')]
cells = collections.defaultdict(list)
for r in rows:
    cells[(r['tag'], r['model'])].append(r)

P('== checks')
for c in ORDER:
    R = cells.get(c, [])
    if not R:
        continue
    P(c, 'n', len(R), 'ncand4', sum(r['ncand'] == 4 for r in R), 'on_sel all5', sum(all(r['on_sel']) for r in R),
      'rec==pkg', sum(r['rec_success'] == r['succ'][0] for r in R),
      'tau45==pkg', sum(r['succ_tau45'] == r['succ'] for r in R), 'no disc', sum(r['N_disc'] == 0 for r in R),
      'disc<4', sum(r['N_disc'] < 4 for r in R))

P('\n== hit@k, P@5')
for c in ORDER:
    R = cells.get(c, [])
    if not R:
        continue
    hk = [sum(any(r['succ'][:k]) for r in R) for k in range(1, 6)]
    p5 = sum(np.mean(r['succ'][:5]) for r in R)
    P(c, 'n', len(R), 'hit@k', hk, 'P@5 %.1f (%.3f)' % (p5, p5 / len(R)))

P('\n== baselines (disc, exact R5/AR, AF deterministic, F5 MC 200)')
for c in ORDER:
    R = cells.get(c, [])
    if not R:
        continue
    top = np.array([r['succ'][0] for r in R], float)
    h5 = np.array([any(r['succ']) for r in R], float)
    R5 = sum(r['R5_disc'] for r in R)
    R1 = sum(r['R1_disc'] for r in R)
    AR = np.array([r['AR_disc'] for r in R])
    has_af = [r['AF5'] is not None for r in R]
    AF = np.array([float(r['AF5']) if r['AF5'] is not None else t for r, t in zip(R, top)])
    F5 = sum(r['F5_disc'] or 0 for r in R)
    ARr = np.array([r['AR_room'] for r in R])
    R5r = sum(r['R5_room'] for r in R)
    gain = h5.sum() - top.sum()
    rec = lambda x: ((x - top).sum() / gain) if gain > 0 else float('nan')
    P(c, 'top1 %d any5 %d gain %d | R5@1 %.1f R5@5 %.1f F5@5 %.1f | AR@5 %.1f (rec %.2f) AF@5 %.1f (rec %.2f) | '
      'room: R5@5 %.1f AR@5 %.1f (rec %.2f)' % (top.sum(), h5.sum(), gain, R1, R5, F5, AR.sum(), rec(AR), AF.sum(),
                                              rec(AF), R5r, ARr.sum(), rec(ARr)),
      'AF missing', len(R) - sum(has_af))

P('\n== lift')
for c in ORDER:
    R = cells.get(c, [])
    if not R:
        continue
    top = sum(r['succ'][0] for r in R)
    rd = sum(r['I_disc'] / r['N_disc'] if r['N_disc'] else 0 for r in R)
    rr = sum(r['I_room'] / r['N_room'] if r['N_room'] else 0 for r in R)
    sd = sum(r['S_disc'] / r['N_disc'] if r['N_disc'] else 0 for r in R)
    sr = sum(r['S_room'] / r['N_room'] if r['N_room'] else 0 for r in R)
    P(c, 'Lift_disc %.2f Lift_room %.2f ratio %.2f | rho_disc mean %.3f rho_room mean %.3f | success-share lift '
      'disc %.2f room %.2f' % (top / rd, top / rr, rd / rr, rd / len(R), rr / len(R), top / sd, top / sr))

P('\n== lift by situation, pooled per tag group (old+watch rows: old,watch; new)')
for tagset in (('old', 'watch'), ('new',)):
    bys = collections.defaultdict(list)
    for r in rows:
        if r['tag'] in tagset:
            bys[r['situation']].append(r)
    for s, R in sorted(bys.items()):
        top = sum(r['succ'][0] for r in R)
        rd = sum(r['I_disc'] / r['N_disc'] if r['N_disc'] else 0 for r in R)
        rr = sum(r['I_room'] / r['N_room'] if r['N_room'] else 0 for r in R)
        P(tagset, s, 'n', len(R), 'Lift_disc %.2f Lift_room %.2f ratio %.2f' % (top / rd, top / rr, rd / rr))

P('\n== forbidden mass (records whose episode has the kind)')
for c in ORDER:
    R = cells.get(c, [])
    if not R:
        continue
    parts = []
    for k in ('ahead_in_line', 'group_space', 'audience_view'):
        RR = [r for r in R if k in r['kinds']]
        if not RR:
            continue
        setm = np.mean([np.mean([fp.get(k, False) for fp in r['forb_places']]) for r in RR])
        topm = np.mean([r['forb_places'][0].get(k, False) for r in RR])
        r5 = np.mean([r['forb_disc_share'][k] for r in RR if r['forb_disc_share'].get(k) is not None])
        parts.append('%s n%d set %.4f top %.3f R5disc %.4f' % (k, len(RR), setm, topm, r5))
    P(c, ' | '.join(parts))

P('\n== jitter')
for c in ORDER:
    R = cells.get(c, [])
    if not R:
        continue
    flips = sum(any(j[0] != r['succ'][0] for j in r['jit']) for r in R)
    kept = sum(r['succ'][0] and all(j[0] for j in r['jit']) for r in R)
    tops = sum(r['succ'][0] for r in R)
    worst = sum(min(any(j) for j in r['jit']) for r in R)
    P(c, 'flips', flips, 'of', len(R), 'kept all 8', kept, '/', tops, 'any5 worst', worst, 'vs', sum(any(r['succ']) for r in R))

P('\n== heading curve 30/45/60: top1 | any5 | P@5')
for c in ORDER:
    R = cells.get(c, [])
    if not R:
        continue
    t = [sum(r[f'succ_tau{x}'][0] for r in R) for x in (30, 45, 60)]
    a = [sum(any(r[f'succ_tau{x}']) for r in R) for x in (30, 45, 60)]
    p = ['%.1f' % sum(np.mean(r[f'succ_tau{x}']) for r in R) for x in (30, 45, 60)]
    P(c, t, a, p)

P('\n== skeptic: R5 in-region share among right-target conversation_group records (disc)')
for c in ORDER:
    R = [r for r in cells.get(c, []) if r['situation'] == 'conversation_group' and r['matched'] is not None]
    if not R:
        continue
    P(c, 'n', len(R), 'R5 in-region %.1f (%.3f)' % (sum(r['R5_disc_inreg'] for r in R),
                                                   sum(r['R5_disc_inreg'] for r in R) / len(R)),
      'R5 success %.1f' % sum(r['R5_disc'] for r in R))

P('\n== disc coverage')
bys = collections.defaultdict(lambda: [0, 0])
for r in rows:
    bys[r['situation']][0] += 1
    bys[r['situation']][1] += (not r['top_in_disc'])
P('top place outside disc by situation (all rows):', dict(bys))
for c in ORDER:
    R = [r for r in cells.get(c, []) if r['matched'] is not None and r['I_room'] > 0]
    if not R:
        continue
    P(c, 'region cells outside disc %.3f' % (1 - sum(r['I_disc'] for r in R) / sum(r['I_room'] for r in R)))

P('\n== pooled chance ratio, old realistic cells together (alternative reading of S4)')
R = [r for r in rows if r['tag'] == 'old']
if R:
    rd = sum(r['I_disc'] / r['N_disc'] if r['N_disc'] else 0 for r in R)
    rr = sum(r['I_room'] / r['N_room'] for r in R)
    top = sum(r['succ'][0] for r in R)
    P('old pooled n', len(R), 'Lift_disc %.2f chance ratio %.3f' % (top / rd, rd / rr))
R = [r for r in rows if r['tag'] in ('old', 'watch')]
if R:
    rd = sum(r['I_disc'] / r['N_disc'] if r['N_disc'] else 0 for r in R)
    rr = sum(r['I_room'] / r['N_room'] for r in R)
    top = sum(r['succ'][0] for r in R)
    P('old+watch pooled n', len(R), 'Lift_disc %.2f chance ratio %.3f' % (top / rd, rd / rr))

P('\n== layout bootstrap of AR/AF recovery (B 2000, seed 7)')
def layout(r):
    s = r['scene']
    if r['arm'].startswith('dev'):
        return r['arm'] + '/' + s                   # the seeds reuse scene ids for different layouts
    for a in ('real_', 'staffed_', 'plain_'):
        if s.startswith(a):
            return s[len(a):]
    return s
for c in (('old', 'dsv41'), ('watch', 'dsv41'), ('new', 'dsv41'), ('old', 'glm53')):
    R = cells.get(c, [])
    if not R:
        continue
    by = collections.defaultdict(lambda: np.zeros(4))
    for r in R:
        t = float(r['succ'][0])
        af = float(r['AF5']) if r['AF5'] is not None else t
        by[layout(r)] += [float(any(r['succ'])) - t, r['AR_disc'] - t, af - t, 1]
    L = np.array(list(by.values()))
    rng = np.random.default_rng(7)
    ars, afs = [], []
    for _ in range(2000):
        s = L[rng.integers(len(L), size=len(L))].sum(0)
        if s[0] > 0:
            ars.append(s[1] / s[0]); afs.append(s[2] / s[0])
    P(c, 'layouts', len(L), 'AR %.2f [%.2f, %.2f]' % (L[:, 1].sum() / L[:, 0].sum(), *np.percentile(ars, [2.5, 97.5])),
      'AF %.2f [%.2f, %.2f]' % (L[:, 2].sum() / L[:, 0].sum(), *np.percentile(afs, [2.5, 97.5])))

text = '\n'.join(out)
print(text)
open(f'{V}/v_tables.txt', 'w').write(text + '\n')
