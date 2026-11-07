"""Independent N1/N2 recompute (verifier). s16b qwen/glm and s20 dsv41, cot and fsc, realistic arms."""
import sys, collections, statistics as st, io
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
OUT = io.StringIO()
def p(*a):
    print(*a); print(*a, file=OUT)
ARMS = ['real', 'staffed', 'plain']
SETS = [('qwen38_low', 's16b'), ('glm53', 's16b'), ('dsv41', 's20')]
mism = 0; total = 0
data = {}
for model, stage in SETS:
    for meth in ['cot', 'fsc']:
        rows = []
        for arm in ARMS:
            rs = H.runs(stage=stage, arm=arm, model=model, method=meth)
            assert len(rs) == 1, (stage, arm, model, meth, rs)
            tally = collections.Counter()
            for rec in H.records(rs[0], tally=tally):
                now = H.rescore(rec)
                total += 1
                if bool(now.success) != bool(rec.score.get('success')) or now.distance_to_region_m != rec.score.get('distance_to_region_m'):
                    mism += 1
                rows.append(rec)
            p(rs[0].folder, dict(tally))
        data[(model, meth)] = rows
p('records', total, 'recorded vs now mismatches (success or distance):', mism)

p('\n== N2: median distance_to_region_m of failures with right target at waiting_line ==')
for model, stage in SETS:
    for meth in ['cot', 'fsc']:
        allf = [r for r in data[(model, meth)] if r.situation == 'waiting_line' and not r.score.get('success')
                and r.score.get('target_correct') and r.score.get('distance_to_region_m') is not None]
        mk = [r for r in allf if r.result.get('source') == 'direct_marking']
        d_all = [r.score['distance_to_region_m'] for r in allf]
        d_mk = [r.score['distance_to_region_m'] for r in mk]
        p(f'{model:11s} {meth}: all sources n {len(d_all)} median {st.median(d_all):.3f} | marked n {len(d_mk)} median {st.median(d_mk):.3f}'
          f' | sources {dict(collections.Counter(r.result.get("source") for r in allf))}')
    pooled = [r.score['distance_to_region_m'] for meth in ['cot','fsc'] for r in data[(model, meth)] if r.situation == 'waiting_line' and not r.score.get('success') and r.score.get('target_correct') and r.score.get('distance_to_region_m') is not None]
    p(f'{model:11s} R+E pooled n {len(pooled)} median {st.median(pooled):.3f}')

p('\n== N1: E (fsc) failures at conversation_group with right target ==')
tot = collections.Counter()
for model, stage in SETS:
    for meth in ['fsc', 'cot']:
        f = [r for r in data[(model, meth)] if r.situation == 'conversation_group' and not r.score.get('success')
             and r.score.get('target_correct') and r.score.get('distance_to_region_m') is not None]
        n = len(f)
        near = [r for r in f if r.score['distance_to_region_m'] <= 0.3]
        g = lambda r: r.score.get('min_body_gap_m')
        nb = [r for r in near if g(r) is not None and g(r) < 0.3]
        nb0 = [r for r in near if g(r) is not None and g(r) < 0]
        nub = [r for r in nb if not r.score.get('reachable')]
        src = collections.Counter(r.result.get('source') for r in f)
        p(f'{model:11s} {meth}: n {n} near {len(near)} near&gap<0.3 {len(nb)} near&gap<0 {len(nb0)} near&unreach&gap<0.3 {len(nub)} sources {dict(src)}')
        if meth == 'fsc':
            tot.update(n=n, near=len(near), nb=len(nb), nb0=len(nb0), nub=len(nub))
p('fsc pooled', dict(tot))

p('\n== near.py part-2 style table (fsc, sit != no_one) ==')
for model, stage in SETS:
    c = collections.Counter()
    for r in data[(model, 'fsc')]:
        s = r.score
        if r.situation == 'no_one' or s.get('success') or not s.get('target_correct'): continue
        dd = s.get('distance_to_region_m')
        if dd is None: continue
        gg = s.get('min_body_gap_m')
        c[(r.situation, 'near' if dd <= 0.3 else 'far', 'unreach' if not s.get('reachable') else 'reach', 'gap<0.3' if (gg is not None and gg < 0.3) else 'gap>=0.3')] += 1
    for k, v in sorted(c.items()): p(model, k, v)
open('results/2026-10-17-beyond-vlm/rerun/snap/verify/v1_n1n2.txt', 'w').write(OUT.getvalue())
