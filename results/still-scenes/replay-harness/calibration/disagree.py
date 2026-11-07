"""List the goal steps where P disagrees with conorc, with perception counts and constraint kinds of both arms."""
import json, sys, collections
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
HERE = 'results/2026-10-17-beyond-vlm/phase1/i01_calibration'
rows = [json.loads(l) for l in open(f'{HERE}/rows2.jsonl')]
orc = {}
for m in sorted({r['model'] for r in rows}):
    for rec in H.records(H.runs(stage='s8', model=m, method='conorc')[0]):
        orc[(m, rec.scene_id, rec.episode_id, rec.trial)] = rec
out = []
cls = collections.Counter()
for r in rows:
    if not r['solver_step'] or r['P'] == r['orc_now']:
        continue
    o = orc[(r['model'], r['scene'], r['episode'], r['trial'])]
    kind = ('W not recovered' if r['orc_now'] and not r['C_replay_ok'] else
            'L not reproduced' if r['C_replay_ok'] and not r['orc_now'] else 'P broke a success')
    cls[(r['model'], kind)] += 1
    ok = [u['constraint']['kind'] for u in o.constraints or []]
    sc = o.score
    out.append(f"{r['model']:10s} {r['scene']:32s} {r['episode']} t{r['trial']} {r['situation']:19s} {kind:17s} "
               f"convlm {r['C_replay_ok']!s:5s} P {r['P']!s:5s} conorc {r['orc_now']!s:5s} | det {r['n_det']} vis {r['n_vis']} "
               f"miss {r['n_missed']} spur {r['n_spurious']} | convlm kinds {sorted(r['kinds'])} | conorc kinds {sorted(ok)} "
               f"| conorc in_region {sc.get('in_region')} reach {sc.get('reachable')} dist {sc.get('distance_to_region_m')}")
out.append('')
out.append('counts: ' + ', '.join(f'{m} {k}: {v}' for (m, k), v in sorted(cls.items())))
open(f'{HERE}/disagree.txt', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
