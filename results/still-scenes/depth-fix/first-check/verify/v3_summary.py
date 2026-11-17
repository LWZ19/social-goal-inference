import json, collections, statistics as st, io
V = 'results/still-scenes/depth-fix/first-check/verify/'
rows = [json.loads(l) for l in open(V + 'v2_snap_records.jsonl')]
OUT = io.StringIO()
def p(*a):
    print(*a); print(*a, file=OUT)
def klass(d, inreg):
    if d is None: return 'none'
    if inreg: return 'inreg'
    if d <= 0.3: return 'near'
    if d <= 1.0: return 'mid'
    return 'far'
MODELS = ['qwen38_low', 'glm53', 'dsv41', 'haiku55', 'sonnet55_low', 'gemini38_low']
SITS = ['no_one', 'waiting_line', 'conversation_group']
p('records', len(rows), collections.Counter((r['model'], r['method']) for r in rows))
med = lambda xs: (f'{st.median(xs):.3f}' if xs else '-')

p('\n=== T1 per model: failures with right target, snapD/snapP fixes, breaks, successes ===')
for m in MODELS:
    R = [r for r in rows if r['model'] == m]
    F = [r for r in R if not r['ok'] and r['tc']]
    std = [r for r in F if r['source'] != 'direct_marking']
    mk = [r for r in F if r['source'] == 'direct_marking' and 'okD' in r]
    p(f'\n{m}: records {len(R)} failures(tc) {len(F)} standard {len(std)} {dict(collections.Counter(r["sit"] for r in std))} marked {len(mk)}'
      f' {dict(collections.Counter(r["sit"] for r in mk))}; other (no pose etc) {len(F)-len(std)-len(mk)}')
    for s in SITS:
        x = [r for r in mk if r['sit'] == s]
        if not x: continue
        fD = sum(r['okD'] for r in x); fP = sum(r['okP'] for r in x)
        still = [r for r in x if not r['okD']]
        cls = collections.Counter(klass(r['dD'], r['inregD']) for r in still)
        body0 = sum(1 for r in still if r['gapD'] is not None and r['gapD'] < 0)
        body3 = sum(1 for r in still if r['gapD'] is not None and r['gapD'] < 0.3)
        extra = ''
        if s == 'waiting_line':
            beh = sum(1 for r in still if r.get('not_aheadD'))
            side = sum(1 for r in still if r.get('lat_camside', 0) > 0.3)
            extra = f' behind(not_ahead,snapped) {beh}; >0.3m camera side of axis {side}'
        p(f'   {s:19s} marked fails {len(x):3d} (R {sum(r["method"]=="cot" for r in x)}, E {sum(r["method"]=="fsc" for r in x)})'
          f' fixedD {fD} (R {sum(r["okD"] for r in x if r["method"]=="cot")}, E {sum(r["okD"] for r in x if r["method"]=="fsc")}) fixedP {fP};'
          f' still {len(still)} {dict(cls)} body(gap<0) {body0} gap<0.3 {body3}{extra}')
    succ = [r for r in R if r['source'] == 'direct_marking' and r['ok'] and 'okD' in r]
    brk = sum(1 for r in succ if not r['okD']); moved = sum(1 for r in succ if r['movedD'] > 0)
    p(f'   marked successes {len(succ)} moved {moved} broken by snapD {brk}')
    for meth, name in [('cot', 'R'), ('fsc', 'E')]:
        Rm = [r for r in R if r['method'] == meth]
        base = sum(r['ok'] for r in Rm)
        aD = sum((r['okD'] if 'okD' in r else r['ok']) for r in Rm)
        aP = sum((r['okP'] if ('okP' in r and not r['ok']) else (r['okD'] if 'okD' in r else r['ok'])) for r in Rm)
        p(f'   {name}: {base} -> snapD {aD} snapP {aP} of {len(Rm)}')
    cs = {}
    for r in R:
        if r['method'] == 'cot': cs[(r['arm'], r['scene'], r['ep'], r['trial'])] = r['c_ok']
    p(f'   C successes (paired keys) {sum(1 for v in cs.values() if v)} of {len(cs)}')

p('\n=== SNAP-gate, pooled over six models (marked failures with right target) ===')
mk = [r for r in rows if not r['ok'] and r['tc'] and r['source'] == 'direct_marking' and 'okD' in r]
for meth in ['cot', 'fsc']:
    for s in ['waiting_line', 'conversation_group']:
        x = [r for r in mk if r['method'] == meth and r['sit'] == s]
        near = [r for r in x if r['d'] is not None and r['d'] <= 0.3]
        far = [r for r in x if r['d'] is not None and r['d'] > 1.0]
        p(f'{meth} {s}: n {len(x)} near {len(near)} fixedD {sum(r["okD"] for r in near)} ({sum(r["okD"] for r in near)/max(1,len(near)):.0%})'
          f' fixedP {sum(r["okP"] for r in near)} ({sum(r["okP"] for r in near)/max(1,len(near)):.0%}); far {len(far)} ({len(far)/max(1,len(x)):.0%})')
allF = [r for r in rows if not r['ok'] and r['tc'] and r['sit'] == 'waiting_line']
for meth in ['cot', 'fsc']:
    x = [r for r in allF if r['method'] == meth]
    p(f'  far share at lines incl standard-goal: {meth} {sum(1 for r in x if r["d"] is not None and r["d"]>1)}/{len(x)}')
for m in ['glm53', 'dsv41']:
    for s in ['waiting_line', 'conversation_group']:
        x = [r for r in mk if r['model'] == m and r['method'] == 'fsc' and r['sit'] == s]
        near = [r for r in x if r['d'] is not None and r['d'] <= 0.3]
        far = [r for r in x if r['d'] is not None and r['d'] > 1]
        p(f'  {m} E {s}: n {len(x)} near {len(near)} fixedD {sum(r["okD"] for r in near)} fixedP {sum(r["okP"] for r in near)} far {len(far)}')
fixes = [r for r in mk if r['okD']]
p(f'snapD fixes {len(fixes)}; within 0.55 m of a true person centre {sum(1 for r in fixes if r["d_person"] is not None and r["d_person"] < 0.55)};'
  f' in 0.50-0.55 strip {sum(1 for r in fixes if r["d_person"] is not None and 0.5 <= r["d_person"] < 0.55)};'
  f' >1 m from region {sum(1 for r in fixes if r["d"] > 1)}; by model {dict(collections.Counter(r["model"] for r in fixes))}')
gfix = [r for r in fixes if r['model'] == 'gemini38_low']
p(f'  gemini snapD fixes in strip {sum(1 for r in gfix if 0.5 <= r["d_person"] < 0.55)} '
  f'{dict(collections.Counter(r["sit"] for r in gfix if 0.5 <= r["d_person"] < 0.55))}')

p('\n=== N1-s23: E failures at groups, right target, near & gap<0.3 ===')
tot = collections.Counter()
for meth in ['fsc', 'cot']:
    for m in MODELS:
        x = [r for r in rows if r['model'] == m and r['method'] == meth and r['sit'] == 'conversation_group' and not r['ok'] and r['tc'] and r['d'] is not None]
        near = [r for r in x if r['d'] <= 0.3]
        nb = [r for r in near if r['gap'] is not None and r['gap'] < 0.3]
        fD = sum(r.get('okD', False) for r in nb); fP = sum(r.get('okP', False) for r in nb)
        p(f'  {meth} {m}: n {len(x)} near {len(near)} near&body {len(nb)} fixedD {fD} fixedP {fP}')
        tot[(meth, 'n')] += len(x); tot[(meth, 'near')] += len(near); tot[(meth, 'nb')] += len(nb); tot[(meth, 'fD')] += fD; tot[(meth, 'fP')] += fP
p('  pooled', dict(tot))

p('\n=== MECH: lines, marked, right target ===')
for m in MODELS:
    for meth in ['cot', 'fsc']:
        x = [r for r in rows if r['model'] == m and r['method'] == meth and r['sit'] == 'waiting_line' and r['tc'] and r['source'] == 'direct_marking' and 'dv' in r]
        f = [r for r in x if not r['ok']]
        p(f'  {m:12s} {meth}: n {len(x)} dv med {med([r["dv"] for r in x])} drange med {med([r["drange"] for r in x])} |'
          f' fails n {len(f)} dv {med([r["dv"] for r in f])} drange {med([r["drange"] for r in f])}'
          f' lat>0.3 {sum(1 for r in f if r["lat_camside"]>0.3)} lat med {med([r["lat_camside"] for r in f])} t med {med([r["t"] for r in f])}')
x = [r for r in rows if r['model'] == 'sonnet55_low' and r['method'] == 'cot' and r['sit'] == 'waiting_line' and r['tc'] and 'drange' in r]
dr = sorted(r['drange'] for r in x)
q = st.quantiles(dr, n=4)
p(f'  sonnet R drange IQR {q[0]:.2f} .. {q[2]:.2f}')
p(' Haiku:')
for meth in ['cot', 'fsc']:
    f = [r for r in rows if r['model'] == 'haiku55' and r['method'] == meth and r['sit'] == 'waiting_line' and r['tc'] and not r['ok'] and r['source'] == 'direct_marking']
    counter = sum(1 for r in f if r['t'] < r['t_first'])
    fronth = sum(1 for r in f if r['t_first'] <= r['t'] < 0.5)
    p(f'  {meth}: fails n {len(f)} counter(t<t_first) {counter} front half {fronth} t med {med([r["t"] for r in f])} d_anchor med {med([r["d_anchor"] for r in f])}'
      f' lat>0.3 {sum(1 for r in f if r["lat_camside"]>0.3)} far {sum(1 for r in f if r["d"]>1)} nearer anchor than first {sum(1 for r in f if r["d_anchor"]<r["d_first"])}'
      f' fixedD {sum(r["okD"] for r in f)}')
p(' Gemini:')
for meth in ['cot', 'fsc']:
    x = [r for r in rows if r['model'] == 'gemini38_low' and r['method'] == meth and r['sit'] == 'waiting_line' and r['tc'] and r['source'] == 'direct_marking']
    s_ = [r for r in x if r['ok']]; f = [r for r in x if not r['ok']]
    p(f'  {meth}: n {len(x)} d_last med {med([r["d_last"] for r in x])}; succ {len(s_)} {med([r["d_last"] for r in s_])}; fails {len(f)} {med([r["d_last"] for r in f])}')
f = [r for r in rows if r['model'] == 'gemini38_low' and r['sit'] == 'waiting_line' and r['tc'] and not r['ok'] and r['source'] == 'direct_marking']
beh = [r for r in f if r['not_ahead']]
p(f'  R+E fails {len(f)} behind(not_ahead) {len(beh)} behind&unreach {sum(1 for r in beh if not r["reach"])}'
  f' d_person<0.5 {sum(1 for r in beh if not r["reach"] and r["d_person"]<0.5)} strip {sum(1 for r in beh if not r["reach"] and 0.5<=r["d_person"]<0.55)}'
  f' fixedD {sum(r["okD"] for r in f)} fixedP {sum(r["okP"] for r in f)}')
open(V + 'v3_summary.txt', 'w').write(OUT.getvalue())
