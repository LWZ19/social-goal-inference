"""N1: C goal-step call (people + constraints) latency and output tokens: qwen3.6 off, Qwen 3.8 off (s8), Qwen 3.8 low (s10 + s19 ph).
Three readings: (a) every call, (b) dedup key (stage, scene, episode, trial, raw_text, latency) as the first agent states,
(c) truly distinct calls: key (stage, raw_text, latency_s, output_tokens), (d) fresh calls only."""
import sys, collections, statistics as st
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
groups = {
    'qwen36 off (s8 convlm+conorc)': [H.runs(stage='s8', model='qwen36_35', method=m)[0] for m in ('convlm', 'conorc')],
    'qwen38 off (s8 convlm+conorc)': [H.runs(stage='s8', model='qwen38', method=m)[0] for m in ('convlm', 'conorc')],
    'qwen38 low (s10 con + s19 ph con)': H.runs(stage='s10', model='qwen38_low', method='con') + H.runs(stage='s19', arm='ph', model='qwen38_low', method='con'),
    'qwen38 low (s10 con only)': H.runs(stage='s10', model='qwen38_low', method='con'),
}
med = st.median
res = {}
for g, refs in groups.items():
    calls = []
    for r in refs:
        for rec in H.records(r):
            for c in rec.result.get('calls') or []:
                if c['stage'] not in ('people', 'constraints'): continue
                u = c.get('usage') or {}
                calls.append(dict(stage=c['stage'], scene=rec.scene_id, ep=rec.episode_id, trial=rec.trial, raw=c.get('raw_text'),
                                  lat=c.get('latency_s') or 0.0, ot=u.get('output_tokens') or 0, rt=u.get('reasoning_tokens') or 0,
                                  cached=bool(c.get('cached')), run=r.name))
    readings = {}
    readings['a all'] = calls
    seen, b = set(), []
    for c in calls:
        k = (c['stage'], c['scene'], c['ep'], c['trial'], c['raw'], c['lat'])
        if k not in seen: seen.add(k); b.append(c)
    readings['b agent key'] = b
    seen, d = set(), []
    for c in calls:
        k = (c['stage'], c['raw'], round(c['lat'], 9), c['ot'])
        if k not in seen: seen.add(k); d.append(c)
    readings['c distinct'] = d
    readings['d fresh'] = [c for c in calls if not c['cached']]
    res[g] = readings
    print(f"== {g}: runs {[r.name for r in refs]}")
    for name, cs in readings.items():
        v = [c for c in cs if c['lat'] and c['ot']]
        if not v: print(f"  {name:12s} n=0"); continue
        print(f"  {name:12s} n={len(cs):4d} median lat {med([c['lat'] for c in cs]):6.2f} s  median out tok {med([c['ot'] for c in cs]):7.1f}  median tok/s {med([c['ot']/c['lat'] for c in v]):6.1f}  median reasoning tok {med([c['rt'] for c in cs]):.0f}")
print()
for name in ('a all', 'b agent key', 'c distinct', 'd fresh'):
    try:
        q36 = med([c['lat'] for c in res['qwen36 off (s8 convlm+conorc)'][name]])
        q38 = med([c['lat'] for c in res['qwen38 off (s8 convlm+conorc)'][name]])
        lo = med([c['lat'] for c in res['qwen38 low (s10 con + s19 ph con)'][name]])
        lo10 = med([c['lat'] for c in res['qwen38 low (s10 con only)'][name]])
        t38 = med([c['ot'] for c in res['qwen38 off (s8 convlm+conorc)'][name]]); tlo = med([c['ot'] for c in res['qwen38 low (s10 con + s19 ph con)'][name]])
        print(f"{name:12s} 36off/38off lat x{q36/q38:.2f}; 38off/38low lat x{q38/lo:.2f} (s10 only x{q38/lo10:.2f}); 38off/38low out tok x{t38/tlo:.2f}")
    except Exception as e:
        print(name, 'n/a', e)
