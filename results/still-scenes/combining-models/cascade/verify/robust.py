import sys, math, statistics as st
sys.path.insert(0, 'results/still-scenes/combining-models/cascade/verify')
import verify as V
OUT=[]
def p(*a):
    s=' '.join(str(x) for x in a); print(s); OUT.append(s)
G=V.load('s19','glm53'); Q=V.load('s19','qwen38_low')
keys=sorted(G); gs=[k for k in keys if V.is_gs(G[k])]; n=len(gs); m=round(0.5*n)
lab=[not V.ok(G[k]) for k in gs]; gain=[(not V.ok(G[k])) and V.ok(Q[k]) for k in gs]
def d(k, off):
    a=G[k].pose_world; b=G.get((k[0],k[1],(k[2]+off)%3)); b=b.pose_world if b else None
    return math.dist(a[:2],b[:2]) if a and b else 0.0
def at(score, frac):
    order=sorted(gs,key=lambda k:-score[k]); esc=set(order[:round(frac*n)])
    succ=sum(V.ok(Q[k]) if k in esc else V.ok(G[k]) for k in keys)
    lat_prov=st.mean(V.prov(G[k])+(V.prov(Q[k],{'people','constraints'}) if k in esc else 0) for k in gs)
    lat_rec=st.mean(G[k].latency_s+((Q[k].timings_s or {}).get('goal',0) if k in esc else 0) for k in gs)
    return succ, round(lat_prov,1), round(lat_rec,1)
for off,name in [(1,'dis1 partner t+1 (judge)'),(2,'dis1 partner t+2 (alternative)')]:
    sc={k:d(k,off) for k in gs}
    p(f'{name}: AUROC fail {V.auroc([sc[k] for k in gs],lab):.3f} gain {V.auroc([sc[k] for k in gs],gain):.3f}; '
      + '; '.join(f'@{f}: succ/prov-lat/record-lat {at(sc,f)}' for f in [0.3,0.4,0.5,0.6]))
dis={k:max(d(k,1),d(k,2)) for k in gs}
p('dis (3 samples): '+'; '.join(f'@{f}: {at(dis,f)}' for f in [0,0.3,0.4,0.5,0.6]))
# ties at cutoff for dis
order=sorted(gs,key=lambda k:-dis[k]); cut=dis[order[m-1]]
p('dis value at cutoff', round(cut,4), 'records with that value', sum(1 for k in gs if dis[k]==cut), 'inside top-70', sum(1 for k in order[:m] if dis[k]==cut))
open('results/still-scenes/combining-models/cascade/verify/robust_out.txt','w').write('\n'.join(OUT)+'\n')
