import sys, collections
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
for st, mods, meths in (('s16b',('qwen38_low','glm53'),('fsc','cot')),('s19',('qwen38_low','glm53'),('con',)),('s20',('dsv41',),('cot','fsc','con')),('s21',('qwen38_low',),('cot','fsc','con')),('s23',('glm53',),('cot','fsc','con'))):
    for m in mods:
        for k in meths:
            for arm in ('real','staffed','plain'):
                rs = H.runs(stage=st, arm=arm, model=m, method=k)
                assert len(rs)==1, (st,arm,m,k,rs)
                tally = collections.Counter()
                recs = list(H.records(rs[0], tally=tally))
                print(st, arm, m, k, len(recs), dict(tally), sorted(set(r.trial for r in recs)), len(set(r.scene_id for r in recs)), sorted(set(r.scene_id for r in recs))[:3], sum(int(bool(r.score.get('success'))) for r in recs), list(recs[0].score.keys())[:8])
