"""I03 Stage A: fit-data failure analysis for the one general fix (FROZEN.md section 3). Fit scenes only.
For the chosen setting and for the recall-1 settings, where the misses' depth went and what the false bodies are."""
import collections, json, math, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import fitdata as FD, people_scorer as PS, blobs as B
from evaluate import diagnose

chosen = json.loads((HERE / 'chosen.json').read_text())['setting']
alts = {'chosen': chosen,
        'recall1': dict(z_lo=0.10, z_hi=2.20, top_lo=1.50, split_r=0.15, foot_lo=0.15, foot_hi=1.00),
        'z_hi2.20_split0.35': dict(chosen, z_hi=2.20),
        'z_hi2.05_split0.35': dict(chosen, z_hi=2.05),
        'split0.25': dict(chosen, split_r=0.25),
        'split0.15': dict(chosen, split_r=0.15)}
which = sys.argv[1] if len(sys.argv) > 1 else 'chosen'
s = alts[which]
print('setting', which, B.setting_key(s))
diag = collections.Counter(); fb = collections.Counter(); fbrows = []
for domain, sset, sid, loader in FD.fit_refs():
    sc, th = FD.load(loader)
    kb, allb = B.detect(sc.observation, s, keep_cells=True)
    r = PS.score_people([(b['x'], b['y'], None) for b in kb], sc, th, exempt_xy=PS.exempt_points(sc.truth))
    kept_ids = {b['id'] for b in kb}
    mids = {kb[n - 1]['id'] for n in r['matches']}
    vis = {p.id: p for p in PS.truth_info(sc, th)['visible']}
    for m in r['misses']:
        p = vis[m['id']]
        d, info = diagnose((p.x, p.y), allb, kept_ids, mids, s)
        diag[(domain, d)] += 1
        if domain == 'F-real' or which == 'chosen':
            print(f"  MISS {domain} {sid} {m['id']} role={m['role']} lower_hidden={m['lower_hidden']} -> {d} {info}")
    for n in r['false_numbers']:
        b = kb[n - 1]
        near = min((math.dist((b['x'], b['y']), (p.x, p.y)) for p in vis.values()), default=None)
        fb[domain] += 1
        fbrows.append((domain, sid, round(b['top'], 2), round(b['footprint'], 2), b['cells'], None if near is None else round(near, 2)))
print(dict(diag)); print(dict(fb))
if '--fb' in sys.argv:
    for row in fbrows[:80]:
        print('  FALSE', row)
    tops = collections.Counter(round(r[2], 1) for r in fbrows)
    print('false tops', sorted(tops.items()))
    feet = collections.Counter(round(r[3], 1) for r in fbrows)
    print('false footprints', sorted(feet.items()))
