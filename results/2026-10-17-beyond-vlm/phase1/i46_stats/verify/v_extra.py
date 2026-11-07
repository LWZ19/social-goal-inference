import pickle, os, collections
from math import comb
OUT = os.path.dirname(os.path.abspath(__file__))
D = pickle.load(open(os.path.join(OUT, 'data.pkl'), 'rb'))
LAY = pickle.load(open(os.path.join(OUT, 'layouts.pkl'), 'rb'))
lines = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); lines.append(s)
def pw(n, k, N=14):
    p = k / N
    return sum(comb(n, m) * p**m * (1 - p)**(n - m) for m in range(6, n + 1))
P('exact power, all-positive cells, P(Bin(n,k/14)>=6):')
for k in (6, 7, 8, 9, 11):
    P(' k =', k, [round(pw(n, k), 4) for n in (8, 14, 20, 30)])
P('per-situation exact: lines 4 of 5 nonzero, n=8:', round(pw(8, 4, 5), 4), ' n=6:', round(pw(6, 4, 5), 4))
P('groups 2 of 3 nonzero n=6/8/14:', [round(pw(n, 2, 3), 4) for n in (6, 8, 14)])
P('\nE-R at nobody-near layouts (eval_00..05), new and old wording, success equality:')
for w, mods in (('old', ['Qwen', 'GLM', 'DeepSeek']), ('new', ['Qwen', 'GLM', 'DeepSeek', 'Haiku', 'Sonnet', 'Gemini'])):
    for m in mods:
        R = D[(w, m, 'R')]; E = D[(w, m, 'E')]
        nn = [k for k in R if int(k[1].split('_eval_')[1][:2]) <= 5]
        diff = sum(R[k] != E[k] for k in nn)
        P(f' {w} {m:8s} nobody-near records {len(nn)}, R != E in {diff}')
P('\nnobody-near layouts in E-R sums (new):')
for m in ['Qwen', 'GLM', 'DeepSeek', 'Haiku', 'Sonnet', 'Gemini']:
    d = LAY[('new', m, 'E-R')]
    P(' ', m, {l[:7]: int(v) for l, v in sorted(d.items())})
open(os.path.join(OUT, 'v_extra_output.txt'), 'w').write('\n'.join(lines) + '\n')
