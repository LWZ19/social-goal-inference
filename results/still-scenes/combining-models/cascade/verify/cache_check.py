import sys, json, statistics as st
from collections import Counter
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
sys.path.insert(0, 'results/still-scenes/combining-models/cascade/verify')
from verify import load, is_gs, ok
OUT=[]
def p(*a):
    s=' '.join(str(x) for x in a); print(s); OUT.append(s)
G19=load('s19','glm53'); Q19=load('s19','qwen38_low'); G23=load('s23','glm53'); Q21=load('s21','qwen38_low')
S23=load('s23','sonnet55_low'); M23=load('s23','gemini38_low')
def stage_lat(D, stage, field='latency_s', only_gs=True):
    v=[]
    for k,r in D.items():
        if only_gs and not is_gs(r): continue
        v.append(sum((c.get(field) or 0) for c in r.result.get('calls') or [] if c['stage']==stage))
    return round(st.mean(v),2)
for nm,D in [('G19',G19),('Q19',Q19),('G23',G23),('Q21',Q21),('S23',S23),('M23',M23)]:
    p(nm,'per-stage mean call latency_s over own goal-step:',{s:stage_lat(D,s) for s in ['plan','context','people','constraints']},
      '| timings_s keys mean:', {kk: round(st.mean((r.timings_s or {}).get(kk,0) for r in D.values() if is_gs(r)),2) for kk in ['plan','locate','context','goal','geometry']})
    allm=[r.latency_s for r in D.values() if r.latency_s is not None]
    p('   measures.latency_s mean all records', round(st.mean(allm),1), 'n', len(allm))
def ans(r, stage):
    for c in r.result.get('calls') or []:
        if c['stage']==stage:
            return c.get('raw_text')
    return None
# what do call dicts hold
r=next(iter(G23.values())); p('call keys:', sorted(r.result['calls'][0].keys()))
# do s23 GLM plan / people match s19 GLM for the same key?
for stage in ['plan','context','people','constraints']:
    same=sum(1 for k in G23 if k in G19 and ans(G23[k],stage) is not None and ans(G23[k],stage)==ans(G19[k],stage))
    n=sum(1 for k in G23 if ans(G23[k],stage) is not None)
    p(f'G23 vs G19 identical {stage} answer: {same}/{n}')
for stage in ['plan','context','people','constraints']:
    same=sum(1 for k in Q21 if k in Q19 and ans(Q21[k],stage) is not None and ans(Q21[k],stage)==ans(Q19[k],stage))
    n=sum(1 for k in Q21 if ans(Q21[k],stage) is not None)
    p(f'Q21 vs Q19 identical {stage} answer: {same}/{n}')
# people lists equal (parsed result)
p('G23 vs G19 identical result.people:', sum(1 for k in G23 if G23[k].people is not None and G23[k].people==G19[k].people), '/', sum(1 for k in G23 if G23[k].people is not None))
p('G23 vs G19 identical result.target:', sum(1 for k in G23 if G23[k].target==G19[k].target), '/', len(G23))
p('G23 vs G19 identical pose_world:', sum(1 for k in G23 if G23[k].pose_world==G19[k].pose_world), '/', len(G23))
p('G23 vs G19 goal-step sets equal:', {k for k in G23 if is_gs(G23[k])}=={k for k in G19 if is_gs(G19[k])})
# target varies across trials? count scene/episodes where targets differ across trials, G23
from collections import defaultdict
tg=defaultdict(set)
for k,r in G23.items(): tg[(k[0],k[1])].add(json.dumps(r.target,sort_keys=True))
p('G23 episodes with >1 distinct target across trials:', sum(1 for v in tg.values() if len(v)>1), 'of', len(tg))
with open('results/still-scenes/combining-models/cascade/verify/cache_check_out.txt','w') as f: f.write('\n'.join(OUT)+'\n')
