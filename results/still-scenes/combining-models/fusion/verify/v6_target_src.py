"""Where do the s23 con plan/context cache hits come from? Compare target point with s19/s20 con and s23 cot/fsc same model."""
import sys, collections, json, io
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
OUT = io.StringIO()
def p(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.write(s + '\n')
def idx(stage, arm, model, method):
    rs = H.runs(stage=stage, arm=arm, model=model, method=method)
    if len(rs) != 1: return None
    return {(r.scene_id, r.episode_id, r.trial): r for r in H.records(rs[0])}
first = True
for model, oldst in [('glm53', 's19'), ('dsv41', 's20')]:
    c = collections.Counter()
    for arm in ['real', 'staffed', 'plain']:
        new = idx('s23', arm, model, 'con')
        others = {'old_con': idx(oldst, arm, model, 'con')}
        for meth in ['cot', 'fsc', 'fsc3']:
            o = idx('s23', arm, model, meth)
            if o is not None: others['s23_' + meth] = o
        for k, r in new.items():
            if first:
                p('target keys', sorted((r.target or {}).keys())); p('target', json.dumps(r.target)[:300]); first = False
            for name, o in others.items():
                q = o.get(k)
                if q is None: continue
                c[name + ' n'] += 1
                c[name + ' same point'] += (r.target or {}).get('point') == (q.target or {}).get('point')
                c[name + ' same plan-out-tokens'] += (r.timings.get('plan') or {}).get('output_tokens') == (q.timings.get('plan') or {}).get('output_tokens')
    p(model, dict(sorted(c.items())))
open('results/still-scenes/combining-models/fusion/verify/v6_target_src_out.txt', 'w').write(OUT.getvalue())
