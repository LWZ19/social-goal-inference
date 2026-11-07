"""Post hoc, in-sample diagnostic (not declared, not part of any verdict): the best setting of each variant over the
whole grid on all 549 primary records (no folds), and V3's concentration spread. Reads s_table.npz and rows_*.jsonl."""
import json, math, sys
import numpy as np
HERE = 'results/still-scenes/learned-models/soft-density'
PRIMARY = ['old_qwen', 'old_glm', 'old_deepseek', 'watch_qwen', 'watch_glm', 'watch_deepseek']
st = np.load(f'{HERE}/s_table.npz')
rows = {}
for c in PRIMARY:
    for l in open(f'{HERE}/rows_{c}.jsonl'):
        x = json.loads(l); rows[x['key']] = x
keys = st['keys'].tolist()
prim = np.array([k in rows for k in keys])
top1 = np.array([rows[k]['top1'] if k in rows else False for k in keys])
L = lambda S: np.log(1e-3 + S)
S1, S2, S3 = st['S1'][prim], st['S2'][prim], st['S3'][prim]
G, T, SG, K = st['gammas'], st['Ts'], st['sigmas'], st['kappas']
print(f'n primary {prim.sum()}; V0 (point mass) mean log {np.mean(L(top1[prim].astype(float))):.4f}, mean S {top1[prim].mean():.4f}')
for name, S in (('V1', S1), ('V2', S2)):
    m = L(S).mean(axis=0)
    gi, ti, ki = np.unravel_index(np.argmax(m), m.shape)
    print(f'{name} in-sample best: gamma {G[gi]} T {T[ti]:.4f} kappa {K[ki]}  mean log {m[gi,ti,ki]:.4f}  mean S {S[:,gi,ti,ki].mean():.4f}')
    for g in range(len(G)):
        a = np.unravel_index(np.argmax(m[g]), m[g].shape)
        print(f'   gamma {G[g]:2d}: best mean log {m[g].max():.4f} at T {T[a[0]]:.4f} kappa {K[a[1]]}; smallest T, kappa inf: mean S {S[:,g,0,-1].mean():.4f}')
m3 = L(S3).mean(axis=0)
si, ki = np.unravel_index(np.argmax(m3), m3.shape)
print(f'V3 in-sample best: sigma {SG[si]:.4f} kappa {K[ki]}  mean log {m3[si,ki]:.4f}  mean S {S3[:,si,ki].mean():.4f}')
print('V3 mean log by sigma (best kappa): ' + ', '.join(f'{SG[i]:.3f}:{m3[i].max():.3f}' for i in range(0, len(SG), 3)))
c3 = np.array([x['V3']['conc'] for x in rows.values()])
c1 = np.array([x['V1']['conc'] for x in rows.values()])
print(f'held-out c_V3: min {c3.min():.6f}, share == 1.0 exactly {np.mean(c3 == 1.0):.3f}, share > 0.999 {np.mean(c3 > 0.999):.3f}')
print(f'held-out c_V1: quartiles {np.percentile(c1, [0, 25, 50, 75, 100]).round(3).tolist()}')
