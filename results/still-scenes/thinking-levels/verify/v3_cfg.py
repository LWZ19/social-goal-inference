import sys, io
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
OUT = io.StringIO()
def p(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.write(s + '\n')
for n in ['s8_convlm_qwen38','s10_qwen38_none_con','s10_qwen38_low_con','s19_ph_qwen38_low_con']:
    raw, cfg = H.recorded_config(H.run_ref(n))
    flat = {k: v for k, v in raw.items() if not isinstance(v, (dict, list))}
    p(n, {k: v for k, v in flat.items() if any(w in k.lower() for w in ('commit','git','version','sha','date'))})
    p('   top keys', list(raw.keys())[:40])
# unmatched dedup
def rtgs(r): return r.solver_step and bool(r.score.get('target_correct'))
s8 = list(H.records('s8_convlm_qwen38'))
p('s8 t0 rtgs', sum(rtgs(r) for r in s8 if r.trial==0), 'succ', sum(rtgs(r) and r.score['success'] for r in s8 if r.trial==0))
p('unmatched none dedup (s8 only):', sum(rtgs(r) and r.score['success'] for r in s8), '/', sum(rtgs(r) for r in s8))
open('results/still-scenes/thinking-levels/verify/v3_cfg.txt','w').write(OUT.getvalue())
