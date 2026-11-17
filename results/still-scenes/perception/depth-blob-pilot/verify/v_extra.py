"""Extra checks: misses at the variants that reproduce 39/42 and 37/42, roles, facing by axis input and situation."""
import json, math, sys
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v_match import match, axerr, SC
OUT = Path(__file__).resolve().parent
L = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); L.append(s)

say('=== roles and in-image counts per set ===')
for S in ('plain', 'real', 'staffed', 'ph', 'seed0', 'seed1', 'seed2'):
    ppl = [p for s in SC if s['set'] == S for p in s['people']]
    roles = Counter(p['role'] for p in ppl)
    out = [p for p in ppl if not p['in_img']]
    say(f"{S:8s} people={len(ppl)} roles={dict(roles)} in_img={sum(p['in_img'] for p in ppl)} visible={sum(p['visible'] for p in ppl)} "
        f"out_of_image={len(out)} roles_out={dict(Counter(p['role'] for p in out))} ids_out={dict(Counter(p['id'] for p in out))}")

for variant, foot, R in (('ws_0.25', 'pca', 0.6), ('ws_0.35', 'pca', 0.6), ('vor_0.35', 'bbox', 0.6), ('ws_0.25', 'pca', 0.4)):
    say(f'=== misses at {variant} {foot} R={R} ===')
    n = h = 0
    for s in SC:
        allb = s['blobs'][variant]
        kept = [b for b in allb if b.get(f'kept_{foot}')]
        mp = match(s['people'], kept, R)
        for i, p in enumerate(s['people']):
            if p['in_img'] and i not in mp:
                n += 1; h += p['low_hidden'] >= 6
                near = sorted(allb, key=lambda b: math.hypot(b['pos'][0] - p['x'], b['pos'][1] - p['y']))[:2]
                desc = '; '.join(f"d={math.hypot(b['pos'][0]-p['x'], b['pos'][1]-p['y']):.2f} top={b['top']} foot={b['foot_pca']} kept={b.get('kept_pca')}" for b in near)
                say(f"  {s['scene_id']:34s} {p['id']} role={p['role']} lower_hidden={p['low_hidden']}/8 seen={p['low_seen']}/8 | nearest: {desc}")
    say(f'  total misses={n}, lower body hidden={h}, not hidden={n-h}')

say('=== facing by axis input (ws_0.35 pca R=0.4 and ws_0.25 pca R=0.4), per set and situation ===')
for variant in ('ws_0.25', 'ws_0.35'):
    acc = defaultdict(lambda: defaultdict(list))
    for s in SC:
        kept = [b for b in s['blobs'][variant] if b.get('kept_pca')]
        mp = match(s['people'], kept, 0.4)
        sit = 'line' if 'waiting_line' in s['scene_id'] else 'group' if 'group' in s['scene_id'] else 'audience' if 'audience' in s['scene_id'] else 'other'
        S = s['set']
        grp = 'realistic' if S in ('plain', 'real', 'staffed') else 'ph+3seeds'
        for i, j in mp.items():
            b = kept[j]; p = s['people'][i]
            for key in ('ax_pts', 'ax_cells', 'ax_band'):
                if b.get(key) is not None:
                    hit = axerr(b[key], p['yaw']) <= math.pi / 4
                    acc[(S, key)]['all'].append(hit); acc[(grp, key)][sit].append(hit)
                    if S in ('ph', 'seed0'): acc[('ph+seed0', key)]['all'].append(hit)
    for key in ('ax_pts', 'ax_cells', 'ax_band'):
        row = []
        for S in ('plain', 'real', 'staffed', 'ph', 'ph+seed0'):
            v = acc[(S, key)]['all']; row.append(f"{S}={np.mean(v):.3f}({sum(v)}/{len(v)})")
        say(f'  {variant} {key:8s} ' + '  '.join(row))
        for grp in ('realistic', 'ph+3seeds'):
            say(f'     {grp}: ' + '  '.join(f"{k}={sum(v)}/{len(v)}" for k, v in sorted(acc[(grp, key)].items())))
(OUT / 'v_extra.txt').write_text('\n'.join(L) + '\n')
