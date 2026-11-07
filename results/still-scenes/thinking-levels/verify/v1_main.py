"""Independent verifier for cluster 'thinking' (I22). Harness only."""
import sys, collections, statistics as st, io, json
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
OUT = io.StringIO()
def p(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.write(s + '\n')

def recs(name):
    return list(H.records(name))
def key(r): return (r.scene_id, r.episode_id, r.trial)
def rtgs(r): return r.solver_step and bool(r.score.get('target_correct'))

# ---- effort per run
p('== reasoning_effort seen in calls, trials, n records')
names = ['s8_convlm_qwen38','s8_none_qwen38','s8_conorc_qwen38','s10_qwen38_none_con','s10_qwen38_low_con','s10_qwen38_medium_con',
         's19_ph_qwen38_low_con','s19_real_qwen38_low_con','s19_staffed_qwen38_low_con','s19_plain_qwen38_low_con',
         's21_real_qwen38_low_con','s21_staffed_qwen38_low_con','s21_plain_qwen38_low_con']
R = {n: recs(n) for n in names}
for n in names:
    eff = collections.Counter((c['stage'], c.get('reasoning_effort')) for r in R[n] for c in r.result.get('calls') or [])
    trials = collections.Counter(r.trial for r in R[n])
    succ = sum(bool(r.score.get('success')) for r in R[n])
    p(n, 'n', len(R[n]), 'succ', succ, 'trials', dict(trials), 'effort', dict(eff))

# ---- NEW-T1a: reasoning_tokens field
p('\n== reasoning_tokens field across listed runs: by (stage, model) non-null count, >0 count, sum rt / sum out')
agg = collections.defaultdict(lambda: [0,0,0,0,0])
for ref in H.runs():
    for r in H.records(ref):
        for c in r.result.get('calls') or []:
            u = c.get('usage') or {}
            a = agg[(ref.stage, c.get('model'))]
            a[0] += 1
            rt = u.get('reasoning_tokens')
            a[1] += rt is not None
            a[2] += (rt or 0) > 0
            a[3] += rt or 0
            a[4] += u.get('output_tokens') or 0
for k in sorted(agg, key=str):
    a = agg[k]
    p(k, 'calls', a[0], 'nonnull', a[1], 'pos', a[2], 'rt/out %.3f' % (a[3]/a[4] if a[4] else 0))
qwen_calls = sum(a[0] for k,a in agg.items() if k[1] and 'qwen3.8' in k[1])
qwen_nonnull = sum(a[1] for k,a in agg.items() if k[1] and 'qwen3.8' in k[1])
p('qwen3.8 calls total', qwen_calls, 'nonnull reasoning_tokens', qwen_nonnull)
think_tags = sum(1 for n in names for r in R[n] for c in r.result.get('calls') or [] if '<think' in (c.get('raw_text') or ''))
p('raw_text with <think in qwen runs examined:', think_tags)

# ---- T1: hidden share
p('\n== T1 chars/token calibration at effort none (all calls, unique by (stage, raw_text, out))')
def uniq_calls(runnames, stages=('plan','context','people','constraints')):
    seen = set(); out = []
    for n in runnames:
        for r in R[n]:
            for c in r.result.get('calls') or []:
                if c['stage'] not in stages: continue
                u = c.get('usage') or {}
                if not u.get('output_tokens') or not c.get('raw_text'): continue
                k = (c['stage'], c['raw_text'], u.get('output_tokens'))
                if k in seen: continue
                seen.add(k); out.append(c)
    return out
cal = {}
for label, src in [('s8_convlm', ['s8_convlm_qwen38']), ('s10_none', ['s10_qwen38_none_con']), ('s8+s10none', ['s8_convlm_qwen38','s10_qwen38_none_con'])]:
    cs = uniq_calls(src)
    by = collections.defaultdict(lambda: [0,0,0])
    for c in cs:
        b = by[c['stage']]; b[0] += len(c['raw_text']); b[1] += c['usage']['output_tokens']; b[2] += 1
    cal[label] = {s: b[0]/b[1] for s,b in by.items()}
    p(label, {s: ('%.2f c/t n=%d' % (b[0]/b[1], b[2])) for s,b in by.items()})
p('\n== T1 hidden share = 1 - sum(len/cpt)/sum(out), unique calls, calibration s8_convlm (and s10_none)')
groups = [('s10_low', ['s10_qwen38_low_con']), ('s10_medium', ['s10_qwen38_medium_con']), ('s19_ph', ['s19_ph_qwen38_low_con']),
          ('s19_realistic', ['s19_real_qwen38_low_con','s19_staffed_qwen38_low_con','s19_plain_qwen38_low_con']),
          ('s21_realistic', ['s21_real_qwen38_low_con','s21_staffed_qwen38_low_con','s21_plain_qwen38_low_con'])]
for label, src in groups:
    cs = uniq_calls(src)
    row = []
    for s in ('plan','context','people','constraints'):
        sc = [c for c in cs if c['stage']==s]
        if not sc: continue
        out = sum(c['usage']['output_tokens'] for c in sc)
        vis8 = sum(len(c['raw_text'])/cal['s8_convlm'][s] for c in sc)
        vis10 = sum(len(c['raw_text'])/cal['s10_none'][s] for c in sc)
        per = sorted(1 - len(c['raw_text'])/cal['s8_convlm'][s]/c['usage']['output_tokens'] for c in sc)
        q = lambda f: per[int(f*(len(per)-1))]
        row.append('%s n=%d share %.1f%% (s10cal %.1f%%) IQR %.0f-%.0f%% p10-p90 %.0f-%.0f%%' % (s, len(sc), 100*(1-vis8/out), 100*(1-vis10/out), 100*q(.25), 100*q(.75), 100*q(.1), 100*q(.9)))
    p(label, ' | '.join(row))
# chars/4
p('\n== T1 alt: chars/4 estimator')
for label, src in groups:
    cs = uniq_calls(src)
    row = []
    for s in ('people','constraints'):
        sc = [c for c in cs if c['stage']==s]
        if not sc: continue
        out = sum(c['usage']['output_tokens'] for c in sc); vis = sum(len(c['raw_text'])/4 for c in sc)
        row.append('%s %.1f%%' % (s, 100*(1-vis/out)))
    p(label, ' '.join(row))

# ---- matched token ratio s10 none vs low / medium
p('\n== s10 matched none vs low/medium output tokens per stage (NEW-gate-tokens; T1 cross-check)')
def calls_by(rs):
    d = {}
    for r in rs:
        for c in r.result.get('calls') or []:
            d.setdefault((r.scene_id, r.episode_id, c['stage']), []).append(c)
    return d
none_c = calls_by(R['s10_qwen38_none_con'])
for other in ['s10_qwen38_low_con','s10_qwen38_medium_con']:
    oc = calls_by(R[other])
    for s in ('plan','context','people','constraints'):
        ks = [k for k in none_c if k[2]==s and k in oc]
        a = sum((c.get('usage') or {}).get('output_tokens') or 0 for k in ks for c in none_c[k])
        b = sum((c.get('usage') or {}).get('output_tokens') or 0 for k in ks for c in oc[k])
        p(other, s, 'pairs', len(ks), 'none', a, 'other', b, 'ratio %.1f%%' % (100*a/b if b else 0), 'implied hidden %.1f%%' % (100*(1-a/b) if b else 0))
# people/constraints call is one per (scene, trial)? count
p('people calls per record s10 none:', collections.Counter(len([c for c in r.result.get('calls') or [] if c['stage']=='people']) for r in R['s10_qwen38_none_con']))

# ---- T2
p('\n== T2 s10 none vs low right-target goal steps')
def rtgs_table(a, b, label):
    A = {key(r): r for r in R[a]}; B = {key(r): r for r in R[b]}
    ua = [r for r in R[a] if rtgs(r)]; ub = [r for r in R[b] if rtgs(r)]
    m = [k for k in A if k in B and rtgs(A[k]) and rtgs(B[k])]
    sa = sum(bool(A[k].score['success']) for k in m); sb = sum(bool(B[k].score['success']) for k in m)
    p(label, 'matched', len(m), a, sa, b, sb, '| unmatched', a, '%d/%d' % (sum(bool(r.score['success']) for r in ua), len(ua)), b, '%d/%d' % (sum(bool(r.score['success']) for r in ub), len(ub)),
      '| keys in both', len(set(A)&set(B)))
    return m, A, B
m, A, B = rtgs_table('s10_qwen38_none_con', 's10_qwen38_low_con', 'T2')
rtgs_table('s10_qwen38_none_con', 's10_qwen38_medium_con', 'T2-med')
p('records success none %d/%d low %d/%d' % (sum(bool(r.score['success']) for r in R['s10_qwen38_none_con']), len(A), sum(bool(r.score['success']) for r in R['s10_qwen38_low_con']), len(B)))
p('records lost (low ok, none fail) / gained:')
for k in sorted(A):
    a, b = A[k], B.get(k)
    if b is None: continue
    if bool(a.score['success']) != bool(b.score['success']):
        p('  ', k, a.situation, 'none ok' if a.score['success'] else 'none FAIL', 'low ok' if b.score['success'] else 'low FAIL',
          '| none: src', a.result.get('source'), 'tc', a.score.get('target_correct'), 'ctx_missed', a.measures.get('context_missed'), 'plan_exact', a.measures.get('plan_exact'),
          'dist', a.score.get('distance_to_region_m'), 'solver_step', a.solver_step, '| low: src', b.result.get('source'), 'tc', b.score.get('target_correct'))

# ---- T3 pooling search
p('\n== T3 pooled right-target goal steps: candidate pools')
pairs = [('s10_qwen38_none_con','s10_qwen38_low_con'), ('s8_convlm_qwen38','s19_ph_qwen38_low_con'), ('s8_convlm_qwen38','s10_qwen38_low_con'),
         ('s10_qwen38_none_con','s19_ph_qwen38_low_con'), ('s10_qwen38_none_con','s10_qwen38_medium_con')]
res = {}
for a, b in pairs:
    mm, AA, BB = rtgs_table(a, b, f'{a} vs {b}')
    res[(a,b)] = (mm, AA, BB)
    if a == 's8_convlm_qwen38' and b == 's19_ph_qwen38_low_con':
        for t in (0,1,2):
            mt = [k for k in mm if k[2]==t]
            p('   trial', t, 'matched', len(mt), 'none', sum(bool(AA[k].score['success']) for k in mt), 'low', sum(bool(BB[k].score['success']) for k in mt))
        for k in mm:
            if not (AA[k].score['success'] and BB[k].score['success']):
                p('   fail', k, AA[k].situation, 'none', AA[k].score['success'], 'low', BB[k].score['success'])
mm1 = res[('s10_qwen38_none_con','s10_qwen38_low_con')]; mm2 = res[('s8_convlm_qwen38','s19_ph_qwen38_low_con')]
tot = len(mm1[0]) + len(mm2[0])
sn = sum(bool(mm1[1][k].score['success']) for k in mm1[0]) + sum(bool(mm2[1][k].score['success']) for k in mm2[0])
sl = sum(bool(mm1[2][k].score['success']) for k in mm1[0]) + sum(bool(mm2[2][k].score['success']) for k in mm2[0])
p('POOL s10 matched + s8/s19ph matched:', tot, 'none', sn, 'low', sl)
mt12 = [k for k in mm2[0] if k[2] in (1,2)]
p('POOL dedup (s10 matched + s8/s19ph trials 1,2):', len(mm1[0]) + len(mt12), 'none', sum(bool(mm1[1][k].score['success']) for k in mm1[0]) + sum(bool(mm2[1][k].score['success']) for k in mt12),
  'low', sum(bool(mm1[2][k].score['success']) for k in mm1[0]) + sum(bool(mm2[2][k].score['success']) for k in mt12))
for k in mm1[0]:
    if not (mm1[1][k].score['success'] and mm1[2][k].score['success']): p('   s10 fail', k)

# re-solve the pool
p('\n== T3 re-solved with current solver')
moved = 0; rn = 0; rl = 0
for (mmx, AA, BB) in (mm1, mm2):
    for k in mmx:
        ra = H.resolve(AA[k]); rb = H.resolve(BB[k])
        moved += (not ra.reproduced) + (not rb.reproduced)
        rn += bool(ra.score.success); rl += bool(rb.score.success)
p('re-solved pool none', rn, 'low', rl, 'not reproduced places', moved)

# ---- NEW-T3a duplicate check: s10 none vs s8 convlm trial 0
p('\n== s10_qwen38_none_con vs s8_convlm_qwen38 trial 0: identical call texts?')
S8 = {key(r): r for r in R['s8_convlm_qwen38']}
same = 0; tot = 0
for r in R['s10_qwen38_none_con']:
    k = (r.scene_id, r.episode_id, 0)
    o = S8.get(k)
    tot += 1
    if o is None: continue
    ta = [(c['stage'], c.get('raw_text'), (c.get('usage') or {}).get('output_tokens')) for c in r.result.get('calls') or []]
    tb = [(c['stage'], c.get('raw_text'), (c.get('usage') or {}).get('output_tokens')) for c in o.result.get('calls') or []]
    same += ta == tb and r.score['success'] == o.score['success']
p('s10 none records', tot, 'identical to s8 convlm trial 0 (calls+success)', same, '; s10 trial values', set(r.trial for r in R['s10_qwen38_none_con']))
p('cached flags s10 none:', collections.Counter(c.get('cached') for r in R['s10_qwen38_none_con'] for c in r.result.get('calls') or []))
p('cached flags s10 low:', collections.Counter(c.get('cached') for r in R['s10_qwen38_low_con'] for c in r.result.get('calls') or []))
p('cached flags s19 ph:', collections.Counter(c.get('cached') for r in R['s19_ph_qwen38_low_con'] for c in r.result.get('calls') or []))
p('cached flags s8 convlm:', collections.Counter(c.get('cached') for r in R['s8_convlm_qwen38'] for c in r.result.get('calls') or []))
# is s10 low = s19_ph trial0 overlap?
S19 = {key(r): r for r in R['s19_ph_qwen38_low_con']}
same = 0
for r in R['s10_qwen38_low_con']:
    o = S19.get((r.scene_id, r.episode_id, 0))
    if o is None: continue
    ta = [(c['stage'], c.get('raw_text')) for c in r.result.get('calls') or [] if c['stage'] in ('people','constraints')]
    tb = [(c['stage'], c.get('raw_text')) for c in o.result.get('calls') or [] if c['stage'] in ('people','constraints')]
    same += ta == tb and bool(ta)
p('s10 low records whose people+constraints texts equal s19_ph trial 0:', same)

# ---- T4
p('\n== T4 Qwen C success and latency per record')
for n in ['s8_convlm_qwen38','s10_qwen38_none_con','s10_qwen38_low_con','s10_qwen38_medium_con','s19_ph_qwen38_low_con']:
    lat = [r.latency_s for r in R[n] if r.latency_s is not None]
    prov = [sum(c.get('latency_s') or 0 for c in r.result.get('calls') or []) for r in R[n]]
    provt = [sum(c.get('latency_total_s') or 0 for c in r.result.get('calls') or []) for r in R[n]]
    succ = sum(bool(r.score['success']) for r in R[n])
    p(n, 'succ %d/%d' % (succ, len(R[n])), 'measures.latency_s median %.1f mean %.1f' % (st.median(lat), st.mean(lat)), '| sum call latency_s median %.1f mean %.1f' % (st.median(prov), st.mean(prov)),
      '| sum latency_total_s median %.1f mean %.1f' % (st.median(provt), st.mean(provt)))
p('any listed qwen low con run on ph:', [r.folder for r in H.runs() if r.arm=='ph' and 'qwen38' in r.model and r.method=='con'])
p('s12 runs listed:', H.runs(stage='s12'))

# ---- I22-why
p('\n== I22-why s19 realistic qwen low con per-call means')
rs = [r for n in ['s19_real_qwen38_low_con','s19_staffed_qwen38_low_con','s19_plain_qwen38_low_con'] for r in R[n]]
p('records', len(rs), 'succ', sum(bool(r.score['success']) for r in rs))
for s in ('plan','context','people','constraints'):
    cs = [c for r in rs for c in r.result.get('calls') or [] if c['stage']==s]
    L = [c.get('latency_s') or 0 for c in cs]; T = [(c.get('usage') or {}).get('output_tokens') or 0 for c in cs]
    u = {}
    for c in cs: u[(c.get('raw_text'), (c.get('usage') or {}).get('output_tokens'))] = c
    uL = [c.get('latency_s') or 0 for c in u.values()]; uT = [(c.get('usage') or {}).get('output_tokens') or 0 for c in u.values()]
    p(s, 'calls', len(cs), 'mean lat %.1f med %.1f mean tok %.0f' % (st.mean(L), st.median(L), st.mean(T)), '| unique', len(u), 'mean lat %.1f tok %.0f' % (st.mean(uL), st.mean(uT)))
open('results/2026-10-17-beyond-vlm/rerun/thinking/verify/v1_main.txt','w').write(OUT.getvalue())
