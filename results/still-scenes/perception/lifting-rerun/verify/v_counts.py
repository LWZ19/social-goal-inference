import sys, collections
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
from sgi.evaluate import match_target
OUT = 'results/2026-10-17-beyond-vlm/rerun/lifting/verify/v_counts.txt'
lines = []
def p(*a):
    s = ' '.join(str(x) for x in a); print(s); lines.append(s)
sel = [('s19','qwen38_low'),('s19','glm53'),('s20','dsv41')]
tot = collections.Counter()
for arms_name, arms in (('realistic', ('real','staffed','plain')), ('ph', ('ph',)), ('dev', ('dev_seed0','dev_seed1','dev_seed2'))):
    for st, m in sel:
        c = collections.Counter()
        for arm in arms:
            for r in H.runs(stage=st, arm=arm, model=m, method='con'):
                for rec in H.records(r):
                    c['n'] += 1
                    if rec.score['target_correct']:
                        continue
                    c['wrong'] += 1
                    t = rec.target
                    st_ = rec.result.get('status')
                    c['status:' + str(st_)] += 1
                    if t is None:
                        c['no_point'] += 1
                        continue
                    c['with_point'] += 1
                    sc, gt, th, _ = H._against(rec)
                    mt = match_target(t['world'], sc.truth, gt, th)
                    if mt is not None:
                        c['point_actually_matches'] += 1
                    kinds = sorted({sc.truth.targets[tid].kind for tid in gt.target_ids})
                    c['kind:' + '/'.join(kinds)] += 1
        p(arms_name, st, m, dict(c))
p('done')
open(OUT, 'w').write('\n'.join(lines) + '\n')
