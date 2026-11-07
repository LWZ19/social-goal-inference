import sys, io, collections
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
OUT = io.StringIO()
def p(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.write(s + '\n')
# matched input token difference per stage, same scene/episode
def inp(n):
    d = {}
    for r in H.records(n):
        for c in r.result.get('calls') or []:
            d[(r.scene_id, r.episode_id, r.trial, c['stage'])] = (c.get('usage') or {}).get('input_tokens')
    return d
I = {n: inp(n) for n in ['s8_convlm_qwen38','s10_qwen38_none_con','s10_qwen38_low_con','s10_qwen38_medium_con','s19_ph_qwen38_low_con']}
for a, b in [('s10_qwen38_none_con','s10_qwen38_low_con'), ('s10_qwen38_low_con','s10_qwen38_medium_con'), ('s10_qwen38_low_con','s19_ph_qwen38_low_con'), ('s8_convlm_qwen38','s19_ph_qwen38_low_con')]:
    for s in ('plan','context','people','constraints'):
        diffs = collections.Counter(I[b][k] - I[a][k] for k in I[a] if k[3]==s and k in I[b] and I[a][k] is not None and I[b][k] is not None)
        p(a, '->', b, s, 'input token diff (b-a):', dict(diffs.most_common(4)))
open('results/2026-10-17-beyond-vlm/rerun/thinking/verify/v5_inp.txt','w').write(OUT.getvalue())
