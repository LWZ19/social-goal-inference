import sys, json, collections
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
SIX = ('no_usable_constraint','constraint_unusable','constraints_cut','exclusion_broken','people_cut','far_from_target')
def sig(r):
    res=r.result; fl=res.get('flags') or {}; sol=res.get('solver') or {}; cs=res.get('constraints') or []
    out={}
    out['six']=any(fl.get(f) for f in SIX)
    st=[c.get('status') for c in cs if isinstance(c,dict)]
    out['status_not_used']=any(s!='used' for s in st)
    out['broken>0']=any((c.get('broken_m') or 0)>0 for c in cs if isinstance(c,dict))
    out['own_space']=bool(sol.get('own_space_broken_m'))
    k=sol.get('keys') or {}
    out['excluded>0']=(k.get('excluded_m') or 0)>0 or (k.get('excluded_total_m') or 0)>0
    out['not_ahead_reason']=sol.get('not_ahead_reason') is not None
    out['six|broken']=out['six'] or out['broken>0']
    out['six|own']=out['six'] or out['own_space']
    out['six|excl']=out['six'] or out['excluded>0']
    out['any_nonfrontal']=any(v for f,v in fl.items() if f!='standard_goal_frontal')
    out['frontal_false_std']=res.get('source')=='standard' and not fl.get('standard_goal_frontal')
    out['six|frontal_false']=out['six'] or out['frontal_false_std']
    out['status_counts']=tuple(sorted(collections.Counter(st).items()))
    return out
for model,stage in [('dsv41','s20'),('glm53','s19'),('qwen38_low','s19')]:
  for arms in [('real','staffed','plain'),('ph',),('dev_seed0','dev_seed1','dev_seed2')]:
    for method in ['con']:
        recs=[r for a in arms for run in H.runs(stage=stage,arm=a,model=model,method=method) for r in H.records(run)]
        if not recs: continue
        c=collections.Counter()
        for r in recs:
            s=sig(r); ok=bool(r.score['success'])
            for k,v in s.items():
                if k=='status_counts': continue
                if v: c[(k,ok)]+=1
        print(model,arms,method,'n',len(recs),'succ',sum(bool(r.score['success']) for r in recs))
        print('   '+'; '.join(f'{k} {c[(k,True)]}/{c[(k,False)]}' for k in sorted({k for k,_ in c})))
# status values seen
st=collections.Counter()
for run in H.runs(stage='s20',model='dsv41',method='con'):
    for r in H.records(run):
        for c in r.result.get('constraints') or []:
            st[(c.get('status'),c.get('reason'))]+=1
print(st.most_common(20))
