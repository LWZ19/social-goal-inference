"""Q1 (tok/s, tokens) and Q3 (C success counts) for s8 qwen36_35 vs qwen38. Harness only."""
import sys, collections, statistics as st
from math import comb
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H

def sign_p(w, l):
    n = w + l; k = min(w, l)
    if n == 0: return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)

med = lambda v: st.median(v) if v else float('nan')
print("== Q1: per-call output tokens / latency_s, s8 runs")
pool = {m: {'all': [], 'fresh': [], 'fresh_tok': [], 'fresh_lat': [], 'all_tok': []} for m in ('qwen36_35', 'qwen38')}
bystage = collections.defaultdict(list)
for meth in ('none', 'convlm', 'conorc'):
    for m in ('qwen36_35', 'qwen38'):
        r = H.runs(stage='s8', model=m, method=meth)[0]
        tps_all, tps_fr, stages = [], [], collections.Counter()
        for rec in H.records(r):
            for c in rec.result.get('calls') or []:
                u = c.get('usage') or {}
                lat = c.get('latency_s'); ot = u.get('output_tokens')
                stages[(c['stage'], bool(c.get('cached')))] += 1
                if lat and ot:
                    tps_all.append(ot / lat); pool[m]['all'].append(ot / lat); pool[m]['all_tok'].append(ot)
                    if not c.get('cached'):
                        tps_fr.append(ot / lat); pool[m]['fresh'].append(ot / lat)
                        pool[m]['fresh_tok'].append(ot); pool[m]['fresh_lat'].append(lat)
                        bystage[(m, c['stage'])].append((ot / lat, ot))
        print(f"{r.name:28s} all-calls tok/s med {med(tps_all):6.1f} (n={len(tps_all)})  fresh med {med(tps_fr):6.1f} (n={len(tps_fr)})  calls(stage,cached)={dict(stages)}")
for kind in ('all', 'fresh'):
    a, b = med(pool['qwen36_35'][kind]), med(pool['qwen38'][kind])
    print(f"pooled {kind:5s}: qwen36 {a:.1f} (n={len(pool['qwen36_35'][kind])}) vs qwen38 {b:.1f} (n={len(pool['qwen38'][kind])}) ratio x{a/b:.2f}")
for kind in ('fresh_tok', 'all_tok', 'fresh_lat'):
    a, b = med(pool['qwen36_35'][kind]), med(pool['qwen38'][kind])
    print(f"pooled median {kind}: qwen36 {a:.2f} vs qwen38 {b:.2f} ratio x{a/b:.2f}")
for stg in ('people', 'constraints'):
    a = bystage[('qwen36_35', stg)]; b = bystage[('qwen38', stg)]
    print(f"fresh {stg}: tok/s {med([x for x,_ in a]):.1f} vs {med([x for x,_ in b]):.1f} (x{med([x for x,_ in a])/med([x for x,_ in b]):.2f}); out tok {med([y for _,y in a]):.0f} vs {med([y for _,y in b]):.0f}; n {len(a)}/{len(b)}")

print("\n== Q3: s8 C runs, recorded success and rescored, qwen36_35 vs qwen38")
tot = collections.Counter()
for meth in ('convlm', 'conorc', 'none'):
    S = {}
    for m in ('qwen36_35', 'qwen38'):
        r = H.runs(stage='s8', model=m, method=meth)[0]
        d = {}
        for rec in H.records(r):
            d[(rec.scene_id, rec.episode_id, rec.trial)] = (bool(rec.score.get('success')), bool(H.rescore(rec).success))
        S[m] = d
    keys = sorted(set(S['qwen36_35']) & set(S['qwen38']))
    a = sum(S['qwen36_35'][k][0] for k in keys); b = sum(S['qwen38'][k][0] for k in keys)
    ar = sum(S['qwen36_35'][k][1] for k in keys); br = sum(S['qwen38'][k][1] for k in keys)
    w = sum(S['qwen36_35'][k][0] and not S['qwen38'][k][0] for k in keys)
    l = sum(S['qwen38'][k][0] and not S['qwen36_35'][k][0] for k in keys)
    print(f"{meth:7s} n={len(keys)} recorded {a} vs {b}; rescored {ar} vs {br}; wins {w} losses {l} p={sign_p(w,l):.3f}")
    if meth != 'none':
        tot['a'] += a; tot['b'] += b; tot['w'] += w; tot['l'] += l; tot['n'] += len(keys)
print(f"convlm+conorc n={tot['n']} {tot['a']} vs {tot['b']}; wins {tot['w']} losses {tot['l']} p={sign_p(tot['w'],tot['l']):.3f}")
