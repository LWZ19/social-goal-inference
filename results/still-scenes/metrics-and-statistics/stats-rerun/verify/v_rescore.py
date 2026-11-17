"""Rescore old-wording E and C records with the current scorer and redo the layout sign-flip test."""
import sys, collections, itertools, time
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
import numpy as np
OUT=[]
def p(*a):
    s=' '.join(str(x) for x in a); print(s, flush=True); OUT.append(s)
SIGNS = np.array(list(itertools.product([1, -1], repeat=14)), dtype=np.int64)
OLD = {'qwen': (('s16b','qwen38_low','fsc'),('s19','qwen38_low','con')),
       'glm': (('s16b','glm53','fsc'),('s19','glm53','con')),
       'ds': (('s20','dsv41','fsc'),('s20','dsv41','con'))}
def load(stage, model, method):
    out = {}; mism = []
    for arm in ('real','staffed','plain'):
        ref, = H.runs(stage=stage, arm=arm, model=model, method=method)
        for r in H.records(ref):
            rec_s = int(bool(r.score['success']))
            now = H.score_pose(r, r.pose_world) if r.pose_world is not None else H.rescore(r)
            now_s = int(bool(now.success))
            out[(r.scene_id, r.episode_id, r.trial)] = (r.scene_id[len(arm)+1:], rec_s, now_s)
            if rec_s != now_s: mism.append((r.scene_id, r.episode_id, r.trial, rec_s, now_s))
    return out, mism
t0=time.time()
for m,(e,c) in OLD.items():
    E, me = load(*e); C, mc = load(*c)
    p(f'[{m}] mismatches recorded vs now: E {len(me)} {me[:6]}  C {len(mc)} {mc[:6]}  ({time.time()-t0:.0f}s)')
    for which, idx in (('recorded',1),('rescored now',2)):
        lays = sorted(set(v[0] for v in C.values()))
        d = [sum(C[k][idx]-E[k][idx] for k in C if C[k][0]==l) for l in lays]
        T=sum(d); S=SIGNS@np.array(d)
        p(f'   {which}: C={sum(v[idx] for v in C.values())} E={sum(v[idx] for v in E.values())} d={d} two-sided p={np.mean(np.abs(S)>=abs(T)):.6f}')
open('results/still-scenes/metrics-and-statistics/stats-rerun/verify/v_rescore.txt','w').write('\n'.join(OUT)+'\n')
