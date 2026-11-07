import json, sys, math
from pathlib import Path
sys.path.insert(0, 'results/2026-11-01-push/moving')
import summarise_fresh as S
R = Path('results/2026-11-01-push/moving/runs/sonnet55_low')
arms = ['CT', 'CT2', 'CTd', 'CTe']
scen = sorted(p.name for p in R.iterdir() if p.is_dir())
for sc in scen:
    for seed in (10, 11, 12):
        row = []
        for arm in arms:
            f = R / sc / f'seed{seed}' / arm / 'episode.json'
            if not f.exists():
                row.append(f'{arm}:-'); continue
            e = json.loads(f.read_text())
            m = S.measures(e)
            a = e['answers'][0]
            r0 = e['resolves'][0] if e['resolves'] else {}
            pose = a.get('pose_world')
            row.append(f"{arm}: sh {m['share_strict']:.2f} end {int(m['end_strict'])} cut {m['cut']:3d} first {m['first_strict'] if m['first_strict'] is None else round(m['first_strict'])} lat {a['latency_s']:.1f} cached {''.join('c' if c else 'n' for c in a.get('cached', []))} ppl0 {r0.get('people')} turn0 {int(bool(r0.get('in_turn')))} pose {None if pose is None else (round(pose[0],1), round(pose[1],1))} drop {e['summary'].get('dropped')}")
        print(f'{sc} s{seed}')
        for x in row: print('   ', x)
