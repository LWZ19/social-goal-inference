import sys, collections, statistics as st, io
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
OUT = io.StringIO()
def p(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.write(s + '\n')
def key(r): return (r.scene_id, r.episode_id, r.trial)
def rtgs(r): return r.solver_step and bool(r.score.get('target_correct'))
R = {n: list(H.records(n)) for n in ['s10_qwen38_none_con','s10_qwen38_low_con','s8_convlm_qwen38','s19_ph_qwen38_low_con']}
A = {key(r): r for r in R['s10_qwen38_none_con']}; B = {key(r): r for r in R['s10_qwen38_low_con']}
m1 = {k for k in A if k in B and rtgs(A[k]) and rtgs(B[k])}
C = {key(r): r for r in R['s8_convlm_qwen38']}; D = {key(r): r for r in R['s19_ph_qwen38_low_con']}
m2 = {k for k in C if k in D and rtgs(C[k]) and rtgs(D[k])}
m2t0 = {k for k in m2 if k[2]==0}
p('s10 matched', len(m1), 's8/s19 t0 matched', len(m2t0), 'overlap', len(m1 & m2t0))
p('none-side records counted twice in the 52 pool:', len(m1 & m2t0))
# dedup variants
def cnt(ks, X): return sum(bool(X[k].score['success']) for k in ks)
p('variant a (drop s10 pair): n', len(m2), 'none', cnt(m2, C), 'low', cnt(m2, D))
m2t12 = m2 - m2t0
p('variant b (drop s8/s19 t0): n', len(m1)+len(m2t12), 'none', cnt(m1, A)+cnt(m2t12, C), 'low', cnt(m1, B)+cnt(m2t12, D))
# variant c: union of none draws once: s10 matched records not in m2t0 added to m2
extra = m1 - m2t0
p('variant c (s8/s19 all + s10 matched keys not already in t0):', len(m2)+len(extra), 'none', cnt(m2, C)+cnt(extra, A), 'low', cnt(m2, D)+cnt(extra, B))
# GLM s10 reasoning per run
p('\nGLM (x-test-1) s10 reasoning_tokens per run')
for ref in H.runs(stage='s10'):
    if 'glm' not in ref.model: continue
    n=pos=rt=out=0
    for r in H.records(ref):
        for c in r.result.get('calls') or []:
            u = c.get('usage') or {}
            n += 1; pos += (u.get('reasoning_tokens') or 0) > 0; rt += u.get('reasoning_tokens') or 0; out += u.get('output_tokens') or 0
    p(ref.name, 'calls', n, 'pos', pos, 'rt/out %.3f' % (rt/out if out else 0))
# glm low all methods aggregate
n=pos=rt=out=0
for ref in H.runs(stage='s10', model='glm53_low'):
    for r in H.records(ref):
        for c in r.result.get('calls') or []:
            u = c.get('usage') or {}
            n += 1; pos += (u.get('reasoning_tokens') or 0) > 0; rt += u.get('reasoning_tokens') or 0; out += u.get('output_tokens') or 0
p('s10 glm53_low all methods: calls', n, 'pos', pos, 'rt/out %.3f' % (rt/out))
# key presence of reasoning_tokens in qwen usage
pres = collections.Counter()
for ref in H.runs():
    for r in H.records(ref):
        for c in r.result.get('calls') or []:
            if 'qwen3.8' in (c.get('model') or ''):
                pres['reasoning_tokens' in (c.get('usage') or {})] += 1
p('qwen3.8 calls with reasoning_tokens key present:', dict(pres))
# exact means I22-why
rs = [r for n in ['s19_real_qwen38_low_con','s19_staffed_qwen38_low_con','s19_plain_qwen38_low_con'] for r in H.records(n)]
for s in ('people','constraints'):
    cs = [c for r in rs for c in r.result.get('calls') or [] if c['stage']==s]
    p(s, 'n', len(cs), 'mean tok %.2f' % st.mean((c.get('usage') or {}).get('output_tokens') or 0 for c in cs), 'mean lat %.3f' % st.mean(c.get('latency_s') or 0 for c in cs))
# fresh latency s10 low
p('\ns10 low fresh calls by stage:', collections.Counter(c['stage'] for r in R['s10_qwen38_low_con'] for c in r.result.get('calls') or [] if not c.get('cached')))
# search for any listed qwen con run combination giving 74/84 at ph
for ref in H.runs():
    if 'qwen' in ref.model and ref.method in ('con','convlm'):
        rr = list(H.records(ref)); s = sum(bool(r.score['success']) for r in rr)
        lat = [r.latency_s for r in rr if r.latency_s is not None]
        p(ref.name, '%d/%d' % (s, len(rr)), 'lat med %.1f mean %.1f' % (st.median(lat), st.mean(lat)))
open('results/still-scenes/thinking-levels/verify/v2_extra.txt','w').write(OUT.getvalue())
