"""Q1 detail: matched fresh calls (same method, scene, episode, trial, stage) and the convlm goal step per record."""
import sys, statistics as st
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
def fresh(ref):
    d = {}
    for x in H.records(ref):
        for c in x.result.get('calls') or []:
            if c['stage'] in ('people', 'constraints') and not c.get('cached'):
                d[(x.scene_id, x.episode_id, x.trial, c['stage'])] = (c['latency_s'], (c.get('usage') or {}).get('output_tokens') or 0)
    return d
for meth in ('convlm', 'conorc'):
    a = fresh(H.runs(stage='s8', model='qwen36_35', method=meth)[0]); b = fresh(H.runs(stage='s8', model='qwen38', method=meth)[0])
    for stg in ('people', 'constraints'):
        ks = [k for k in a if k in b and k[3] == stg]
        if not ks: continue
        more = sum(a[k][1] > b[k][1] for k in ks)
        ta = st.median([a[k][1] / a[k][0] for k in ks]); tb = st.median([b[k][1] / b[k][0] for k in ks])
        print(f"{meth} {stg}: pairs {len(ks)} tok/s {ta:.1f} vs {tb:.1f} x{ta/tb:.2f}; qwen36 more tokens in {more}/{len(ks)}")
for m in ('qwen36_35', 'qwen38'):
    lat = []
    for x in H.records(H.runs(stage='s8', model=m, method='convlm')[0]):
        cs = [c for c in x.result.get('calls') or [] if c['stage'] in ('people', 'constraints')]
        if len(cs) == 2 and not any(c.get('cached') for c in cs): lat.append(sum(c['latency_s'] for c in cs))
    print(m, 'convlm goal step, both calls fresh: n', len(lat), 'median', round(st.median(lat), 2))
