"""Q2: search for a pairing that gives 4 wins / 9 losses, qwen3.6 vs 'Qwen low' (and other Qwen 3.8 runs)."""
import sys, collections, itertools
from math import comb
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H

def sign_p(w, l):
    n = w + l; k = min(w, l)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)

FIELDS = ('success', 'in_region', 'target_correct', 'heading_ok', 'reachable')
def load(ref):
    d = {}
    for rec in H.records(ref):
        d[(rec.scene_id, rec.episode_id, rec.trial)] = ({f: bool(rec.score.get(f)) for f in FIELDS}, rec.situation)
    return d

A = {m: load(H.runs(stage='s8', model='qwen36_35', method=m)[0]) for m in ('none', 'convlm', 'conorc')}
B = {}
for st, arm, model in [('s10', 'ph', 'qwen38_low'), ('s10', 'ph', 'qwen38'), ('s10', 'ph', 'qwen38_medium'),
                       ('s16b', 'ph', 'qwen38_low'), ('s19', 'ph', 'qwen38_low'), ('s8', 'ph', 'qwen38')]:
    for r in H.runs(stage=st, arm=arm, model=model):
        B[r.name] = load(r)

def per_ep(d, f, agg):
    g = collections.defaultdict(list)
    for (s, e, t), (sc, sit) in d.items():
        g[(s, e)].append(sc[f])
    if agg == 'maj': return {k: sum(v) * 2 > len(v) for k, v in g.items()}
    if agg == 'any': return {k: any(v) for k, v in g.items()}
    if agg == 'all': return {k: all(v) for k, v in g.items()}
    if agg == 'count': return {k: sum(v) for k, v in g.items()}

def sit_of(d):
    return {(s, e): sit for (s, e, t), (sc, sit) in d.items()}

hits, lines = [], []
def report(tag, pairs, sits=None):
    # pairs: list of (key, a, b) with a,b numbers (bool or count)
    w = sum(1 for k, a, b in pairs if a > b); l = sum(1 for k, a, b in pairs if b > a)
    line = f"{tag:95s} n={len(pairs):3d} A={sum(a for _,a,_ in pairs):3d} B={sum(b for _,_,b in pairs):3d} W={w:2d} L={l:2d} p={sign_p(w,l):.3f}"
    lines.append(line)
    if (w, l) in ((4, 9), (9, 4)): hits.append(line)
    if sits is not None:
        for s in sorted(set(sits.values())):
            sub = [(k, a, b) for k, a, b in pairs if sits.get(k[:2]) == s]
            w2 = sum(1 for k, a, b in sub if a > b); l2 = sum(1 for k, a, b in sub if b > a)
            l3 = f"{tag+' | '+s:95s} n={len(sub):3d} W={w2:2d} L={l2:2d}"
            lines.append(l3)
            if (w2, l2) in ((4, 9), (9, 4)): hits.append(l3)

for am, a in A.items():
    asit = sit_of(a)
    for bn, b in B.items():
        btrials = sorted(set(t for _, _, t in b))
        for f in FIELDS:
            tag0 = f"qwen36 {am} vs {bn} [{f}]"
            # record keyed, when b has 3 trials
            if len(btrials) > 1:
                pairs = [(k, a[k][0][f], b[k][0][f]) for k in sorted(set(a) & set(b))]
                report(tag0 + " rec(s,e,t)", pairs, asit)
            else:
                bt = btrials[0]
                for t in (0, 1, 2):
                    pairs = [((s, e, t), a[(s, e, t)][0][f], b[(s, e, bt)][0][f]) for (s, e, tt) in sorted(b) if (s, e, t) in a]
                    report(tag0 + f" qwen36 trial{t} vs single", pairs, asit)
                pairs = [((s, e, t), a[(s, e, t)][0][f], b[(s, e, bt)][0][f]) for (s, e, t) in sorted(a) if (s, e, bt) in b]
                report(tag0 + " 84 vs broadcast single", pairs, asit)
            for agg in ('maj', 'any', 'all', 'count'):
                pa, pb = per_ep(a, f, agg), per_ep(b, f, agg if len(btrials) > 1 or agg != 'count' else 'count')
                if len(btrials) == 1 and agg == 'count':
                    continue
                pairs = [(k, int(pa[k]), int(pb[k])) for k in sorted(set(pa) & set(pb))]
                report(tag0 + f" ep-{agg}", pairs, asit)
            if len(btrials) == 1:
                pa = per_ep(a, f, 'maj'); pb = per_ep(b, f, 'any')
                pairs = [(k, int(pa[k]), int(pb[k])) for k in sorted(set(pa) & set(pb))]
                report(tag0 + " ep-maj vs single", pairs, asit)

out = 'results/2026-10-17-beyond-vlm/rerun/i44/verify/v2_q2.txt'
with open(out, 'w') as fh:
    fh.write('HITS (4/9 or 9/4):\n' + '\n'.join(hits) + '\n\nALL:\n' + '\n'.join(lines) + '\n')
print('HITS:'); print('\n'.join(hits))
print('\nsuccess rows, not by situation:')
print('\n'.join(l for l in lines if '[success]' in l and ' | ' not in l))
