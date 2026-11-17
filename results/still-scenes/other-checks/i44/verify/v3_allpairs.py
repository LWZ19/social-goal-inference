"""Which pairs of placeholder runs (s8, s10, s16b ph, s19 ph) give 4 wins / 9 losses on recorded success?"""
import sys, collections, itertools
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
refs = H.runs(stage='s8') + H.runs(stage='s10') + H.runs(stage='s16b', arm='ph') + H.runs(stage='s19', arm='ph')
D = {r.name: {(x.scene_id, x.episode_id, x.trial): bool(x.score.get('success')) for x in H.records(r)} for r in refs}
def eps(d):
    g = collections.defaultdict(list)
    for (s, e, t), v in d.items(): g[(s, e)].append(v)
    return g
out = []
for a, b in itertools.permutations(D, 2):
    da, db = D[a], D[b]
    ta = len(set(k[2] for k in da)); tb = len(set(k[2] for k in db))
    res = []
    if ta == tb:
        ks = set(da) & set(db)
        res.append(('rec', sum(da[k] and not db[k] for k in ks), sum(db[k] and not da[k] for k in ks)))
    ga, gb = eps(da), eps(db)
    ks = set(ga) & set(gb)
    for agg, f in [('maj', lambda v: sum(v) * 2 > len(v)), ('any', any), ('all', all), ('t0', lambda v: v[0])]:
        res.append((agg, sum(f(ga[k]) and not f(gb[k]) for k in ks), sum(f(gb[k]) and not f(ga[k]) for k in ks)))
    if ta == tb:
        res.append(('count', sum(sum(ga[k]) > sum(gb[k]) for k in ks), sum(sum(gb[k]) > sum(ga[k]) for k in ks)))
    for kind, w, l in res:
        if (w, l) == (4, 9):
            out.append(f"{a} vs {b} [{kind}] W={w} L={l}")
txt = '\n'.join(out)
print(txt)
open('results/still-scenes/other-checks/i44/verify/v3_allpairs.txt', 'w').write(txt + '\n')
