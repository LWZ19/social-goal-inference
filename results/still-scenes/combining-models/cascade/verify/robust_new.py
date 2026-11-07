import sys, math, statistics as st
sys.path.insert(0, 'results/2026-10-17-beyond-vlm/rerun/cascade/verify')
import verify as V
OUT=[]
def p(*a):
    s=' '.join(str(x) for x in a); print(s); OUT.append(s)
G=V.load('s23','glm53')
for qn,Q in [('s21 qwen',V.load('s21','qwen38_low')),('s23 sonnet',V.load('s23','sonnet55_low')),('s23 gemini',V.load('s23','gemini38_low'))]:
    keys=sorted(G); gs=[k for k in keys if V.is_gs(G[k])]; n=len(gs)
    lab=[not V.ok(G[k]) for k in gs]; gain=[(not V.ok(G[k])) and V.ok(Q[k]) for k in gs]
    def d(k, off):
        a=G[k].pose_world; b=G.get((k[0],k[1],(k[2]+off)%3)); b=b.pose_world if b else None
        return math.dist(a[:2],b[:2]) if a and b else 0.0
    def at(score, frac):
        order=sorted(gs,key=lambda k:-score[k]); esc=set(order[:round(frac*n)])
        return sum(V.ok(Q[k]) if k in esc else V.ok(G[k]) for k in keys)
    for off in (1,2):
        sc={k:d(k,off) for k in gs}
        p(f'{qn} dis1 partner t+{off}: AUROC fail {V.auroc([sc[k] for k in gs],lab):.3f} gain {V.auroc([sc[k] for k in gs],gain):.3f}; @0.5 {at(sc,0.5)}')
open('results/2026-10-17-beyond-vlm/rerun/cascade/verify/robust_new_out.txt','w').write('\n'.join(OUT)+'\n')
