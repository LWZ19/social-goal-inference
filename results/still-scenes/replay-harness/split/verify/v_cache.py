"""Cache structure: s22 trials 3-5 freshness; s23 C goal-stage caching; old-wording not-ok keys."""
import sys, collections, io
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
OUT = 'results/2026-10-17-beyond-vlm/rerun/i01_split/verify/v_cache_out.txt'
buf = io.StringIO()
def p(*a):
    s = ' '.join(str(x) for x in a); print(s); buf.write(s + '\n')
REAL = ('real', 'staffed', 'plain')
def load(stage, model, method):
    out = []
    for arm in REAL:
        (r,) = H.runs(stage=stage, arm=arm, model=model, method=method)
        out += list(H.records(r))
    return out
def cc(recs):
    c = collections.Counter()
    for r in recs:
        for stg, t in r.timings.items():
            c[(stg, 'calls')] += t['calls']; c[(stg, 'cached')] += t['cached']
    return dict(c)
key = lambda r: (r.arm, r.scene_id, r.episode_id, r.trial)
tw = lambda r: None if not r.target else tuple(round(c, 4) for c in r.target.get('world') or ())
for meth in ('cot', 'fsc'):
    s22 = load('s22', 'qwen38_low', meth)
    for tset in ((0, 1, 2), (3, 4, 5)):
        p(f's22 {meth} trials {tset}:', cc([r for r in s22 if r.trial in tset]))
    # s22 cot vs fsc trials 3-5 share target?
s22c = {key(r): r for r in load('s22', 'qwen38_low', 'cot') if r.trial >= 3}
s22f = {key(r): r for r in load('s22', 'qwen38_low', 'fsc') if r.trial >= 3}
p('s22 t3-5 cot vs fsc same target world:', sum(tw(s22c[k]) == tw(s22f[k]) for k in s22c), len(s22c))
s16 = {(k[0], k[1], k[2], k[3] - 3): r for k, r in s22c.items()}
old = {key(r): r for r in load('s16b', 'qwen38_low', 'cot')}
p('s22 t3-5 cot target == s16b same key (trial-3):', sum(tw(old[k]) == tw(r) for k, r in s16.items() if k in old), len(s16))
for model, oldst in (('glm53', 's19'), ('dsv41', 's20')):
    newc = load('s23', model, 'con')
    p(f's23 {model} con cache:', cc(newc))
    o = {key(r): r for r in load(oldst, model, 'con')}
    samep = sum(1 for r in newc if r.people is not None and key(r) in o and o[key(r)].people == r.people)
    withp = sum(1 for r in newc if r.people is not None)
    p(f'   s23 {model} C people identical to {oldst} same key: {samep}/{withp}')
for model in ('qwen38_low',):
    p('s21 qwen con cache:', cc(load('s21', model, 'con')))
for model in ('haiku55', 'sonnet55_low', 'gemini38_low'):
    p(f's23 {model} con cache:', cc(load('s23', model, 'con')))
# old GLM not-ok keys
for meth, st in (('cot', 's16b'), ('fsc', 's16b'), ('con', 's19')):
    p('GLM old notok', meth, sorted(key(r) for r in load(st, 'glm53', meth) if r.result.get('status') != 'ok'))
open(OUT, 'w').write(buf.getvalue())
