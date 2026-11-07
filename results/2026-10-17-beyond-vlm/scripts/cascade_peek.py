import json,glob,math,sys,os,random
os.chdir('runs')
def load(arm,m):
    d=sorted(glob.glob(f'*_s19_{arm}_{m}_con'))[-1]
    return {(r['scene_id'],r['episode_id'],r['trial']):r for r in map(json.loads,open(d+'/results.jsonl'))}
G={};Q={}
for a in ['real','staffed','plain']:
    G.update(load(a,'glm53'));Q.update(load(a,'qwen38_low'))
def prov(r,stages=None):
    return sum((c.get('latency_s') or 0) for c in r['result']['calls'] if stages is None or c['stage'] in stages)
def goalstep(r): return r['result'].get('solver') is not None
keys=sorted(G)
ok=lambda r: bool(r['score']['success'])
gs=[k for k in keys if goalstep(G[k])]
print('GLM goal-step records',len(gs),'GLM succ all',sum(ok(G[k]) for k in keys),'Qwen succ all',sum(ok(Q[k]) for k in keys))
from collections import Counter
c=Counter((ok(G[k]),ok(Q[k]),goalstep(G[k])) for k in keys); print('(glm,qwen,goalstep):',c)
# among goal-step GLM failures: target correct?
f=[k for k in gs if not ok(G[k])]
print('GLM goal-step fails',len(f),'target_correct',sum(bool(G[k]['score']['target_correct']) for k in f),
      'qwen succ on them',sum(ok(Q[k]) for k in f),
      'qwen succ & glm target correct',sum(ok(Q[k]) and G[k]['score']['target_correct'] for k in f))
print('same matched target glm vs qwen (goal-step):',sum(G[k]['score']['matched_target_id']==Q[k]['score']['matched_target_id'] for k in gs))
# latencies
import statistics as st
print('GLM prov goal-step mean',st.mean(prov(G[k]) for k in gs),'all',st.mean(prov(G[k]) for k in keys))
print('Qwen prov goal-step mean',st.mean(prov(Q[k]) for k in gs if goalstep(Q[k])),'all',st.mean(prov(Q[k]) for k in keys))
print('Qwen people+constraints mean (its goal-step)',st.mean(prov(Q[k],{'people','constraints'}) for k in keys if goalstep(Q[k])))
# signals
def sig(k):
    r=G[k]['result']; cs=r.get('constraints') or []
    kinds=[ ((x.get('constraint') or {}).get('kind') if isinstance(x,dict) else None) for x in cs]
    ca=any(kk and 'after' in kk for kk in kinds)
    cand=r.get('candidates') or []
    gap=None
    if len(cand)>=2:
        a,b=cand[0]['keys'],cand[1]['keys']
        gap=abs((b.get('preference_m') or 0)-(a.get('preference_m') or 0))+abs((b.get('excluded_m') or 0)-(a.get('excluded_m') or 0))
    exc=(cand[0]['keys'].get('excluded_m') or 0) if cand else 0
    npeople=len(r.get('people') or [])
    # cross-trial disagreement
    p=r.get('pose_world')
    others=[G[(k[0],k[1],t)]['result'].get('pose_world') for t in range(3) if t!=k[2] and (k[0],k[1],t) in G]
    dis=max([math.dist(p[:2],o[:2]) for o in others if o and p] or [0])
    return dict(ca=ca,gap=gap if gap is not None else 0,exc=exc,np=npeople,dis=dis,kinds=kinds)
S={k:sig(k) for k in gs}
print('constraint kinds seen',Counter(kk for k in gs for kk in S[k]['kinds']).most_common())
def auroc(score,lab):
    pos=[s for s,l in zip(score,lab) if l];neg=[s for s,l in zip(score,lab) if not l]
    if not pos or not neg: return None
    return sum((p>n)+0.5*(p==n) for p in pos for n in neg)/len(pos)/len(neg)
lab=[not ok(G[k]) for k in gs]; gain=[(not ok(G[k])) and ok(Q[k]) for k in gs]
for name in ['ca','gap','exc','np','dis']:
    sc=[float(S[k][name]) for k in gs]
    if name=='gap': sc=[-x for x in sc]
    print(name,'AUROC fail',round(auroc(sc,lab),3),'AUROC gain',round(auroc(sc,gain),3))
# people_found share (GT) vs fail
print('dis by fail: mean fail',st.mean(S[k]['dis'] for k in gs if not ok(G[k])),'succ',st.mean(S[k]['dis'] for k in gs if ok(G[k])))
# cascade sweep using dis then ca as score; escalated -> Qwen score; latency add Qwen people+constraints
def curve(score):
    order=sorted(gs,key=lambda k:-score[k]); n=len(gs); out=[]
    base=sum(ok(G[k]) for k in keys)
    for frac in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0]:
        esc=set(order[:round(frac*n)])
        succ=sum(ok(Q[k]) if k in esc else ok(G[k]) for k in keys)
        lat=st.mean(prov(G[k])+(prov(Q[k],{'people','constraints'}) if k in esc else 0) for k in gs)
        out.append((frac,succ,round(lat,1)))
    return out
comb={k:S[k]['dis']+ (5 if S[k]['ca'] else 0) for k in gs}
print('dis curve',curve({k:S[k]['dis'] for k in gs}))
print('ca+dis curve',curve(comb))
print('oracle curve',curve({k:(1 if (not ok(G[k]) and ok(Q[k])) else 0)+random.random()*0.01 for k in gs}))
random.seed(0); print('random curve',curve({k:random.random() for k in gs}))
# one extra sample only: disagreement with next trial
def dis1(k):
    p=G[k]['result'].get('pose_world'); o=G.get((k[0],k[1],(k[2]+1)%3),{}).get('result',{}).get('pose_world')
    return math.dist(p[:2],o[:2]) if p and o else 0
sc=[dis1(k) for k in gs]; print('dis1 AUROC fail',round(auroc(sc,lab),3),'gain',round(auroc(sc,gain),3))
# conservative plan-reuse: escalated success only if qwen ok and glm target correct
def curve2(score):
    order=sorted(gs,key=lambda k:-score[k]); n=len(gs); out=[]
    for frac in [0,0.2,0.3,0.4,0.5,0.6,1.0]:
        esc=set(order[:round(frac*n)])
        succ=sum((ok(Q[k]) and bool(G[k]['score']['target_correct'])) if k in esc else ok(G[k]) for k in keys)
        lat=st.mean(prov(G[k])+(prov(Q[k],{'people','constraints'}) if k in esc else 0) for k in gs)
        out.append((frac,succ,round(lat,1)))
    return out
print('plan-reuse dis curve',curve2({k:S[k]['dis'] for k in gs}))
print('plan-reuse ca+dis',curve2(comb))
print('plan-reuse dis1',curve2({k:dis1(k) for k in gs}))
print('ca rate',sum(S[k]['ca'] for k in gs),'fails with ca',sum(S[k]['ca'] for k in gs if not ok(G[k])))
print('by situation glm fail/qwen fix:',Counter((G[k]['labels']['situation'],ok(G[k]),ok(Q[k])) for k in gs))
