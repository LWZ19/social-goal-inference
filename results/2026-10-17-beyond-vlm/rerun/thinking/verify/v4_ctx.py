import sys, io, json, collections
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
OUT = io.StringIO()
def p(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.write(s + '\n')
for n in ['s8_convlm_qwen38','s10_qwen38_low_con','s19_ph_qwen38_low_con']:
    ks = collections.Counter(); ins = []
    for r in H.records(n):
        for c in r.result.get('calls') or []:
            if c['stage'] != 'context': continue
            try: ks[tuple(sorted(json.loads(c['raw_text']).keys()))] += 1
            except Exception: ks['unparsed'] += 1
            ins.append((c.get('usage') or {}).get('input_tokens'))
    p(n, 'context answer keys', dict(ks), 'context input tokens set', sorted(set(ins))[:6])
open('results/2026-10-17-beyond-vlm/rerun/thinking/verify/v4_ctx.txt','w').write(OUT.getvalue())
