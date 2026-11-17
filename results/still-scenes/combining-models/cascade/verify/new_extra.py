import sys, io, contextlib
from collections import Counter
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
sys.path.insert(0, 'results/still-scenes/combining-models/cascade/verify')
import verify as V
OUT=[]
def p(*a):
    s=' '.join(str(x) for x in a); print(s); OUT.append(s)
G23=V.load('s23','glm53'); Q21=V.load('s21','qwen38_low'); S23=V.load('s23','sonnet55_low'); M23=V.load('s23','gemini38_low')
# situation breakdown NEW-3
gs=[k for k in sorted(G23) if V.is_gs(G23[k])]
p('NEW-3 both-fail on G goal-step by situation:', Counter(G23[k].situation for k in gs if not V.ok(G23[k]) and not V.ok(M23[k])))
p('NEW-3 G ok / Gemini fail on G goal-step by situation:', Counter(G23[k].situation for k in gs if V.ok(G23[k]) and not V.ok(M23[k])))
p('NEW-3 Gemini target_correct on those 20 both-fail:', sum(bool(M23[k].score.get('target_correct')) for k in gs if not V.ok(G23[k]) and not V.ok(M23[k])))
p('NEW-3 Gemini target_correct on those 13:', sum(bool(M23[k].score.get('target_correct')) for k in gs if V.ok(G23[k]) and not V.ok(M23[k])))
p('NEW-3 Gemini goal-step on those 13:', sum(V.is_gs(M23[k]) for k in gs if V.ok(G23[k]) and not V.ok(M23[k])))
# s23 GLM con plan cached from where? compare to s23 GLM cot/fsc
def plan_txt(r):
    for c in r.result.get('calls') or []:
        if c['stage']=='plan': return c.get('raw_text')
for meth in ['cot','fsc']:
    other={}
    for arm in V.ARMS3:
        for r in H.records(H.runs(stage='s23',arm=arm,model='glm53',method=meth)[0]): other[(r.scene_id,r.episode_id,r.trial)]=r
    p(f's23 GLM con plan raw_text == s23 GLM {meth} plan raw_text:', sum(1 for k in G23 if k in other and plan_txt(G23[k])==plan_txt(other[k])), '/', len(G23))
for meth in ['cot','fsc']:
    other={}
    for arm in V.ARMS3:
        for r in H.records(H.runs(stage='s21',arm=arm,model='qwen38_low',method=meth)[0]): other[(r.scene_id,r.episode_id,r.trial)]=r
    p(f's21 Qwen con plan raw_text == s21 Qwen {meth} plan raw_text:', sum(1 for k in Q21 if k in other and plan_txt(Q21[k])==plan_txt(other[k])), '/', len(Q21))
# replay for NEW pairings
for nm,Q in [('NEW-1 Q21',Q21),('NEW-2 S23',S23),('NEW-3 M23',M23)]:
    V.OUT.clear()
    with contextlib.redirect_stdout(io.StringIO()):
        V.analyse(G23,Q,nm,do_replay=True)
    for line in V.OUT:
        if line.startswith('replay') or line.startswith('plan-reuse'): p(nm,'|',line)
open('results/still-scenes/combining-models/cascade/verify/new_extra_out.txt','w').write('\n'.join(OUT)+'\n')
