"""P1: package matching rule on further subsets (GLM s19 / s23, DeepSeek s20 / s23)."""
import sys
sys.path.insert(0, 'results/2026-10-17-beyond-vlm/rerun/perception')
import perception as P
for stage, model in [('s19', 'glm53'), ('s20', 'dsv41'), ('s23', 'glm53'), ('s23', 'dsv41')]:
    rows = [a for a in (P.analyse(r) for r in P.recs_of(stage, model, P.REAL)) if a is not None]
    lines = [a for a in rows if a['rec'].measures.get('members_recall') is not None]
    for name, sel in [('all line goal steps', lines),
                      ('failed line goal steps (any cause)', [a for a in lines if not a['rec'].score.get('success')]),
                      ('failed, right target', [a for a in lines if not a['rec'].score.get('success') and a['rec'].score.get('target_correct')]),
                      ('successful line goal steps', [a for a in lines if a['rec'].score.get('success')])]:
        missed = sum(len(a['missed_ca']) for a in sel)
        nf = sum(sum(m not in a['found_ids'] for m in a['missed_ca']) for a in sel)
        print(f'{stage} {model} {name}: records {len(sel)}, missed members {missed}, never found {nf}'
              + (f' = {nf/missed:.3f}' if missed else ''))

print('\n--- never-found missed members: missed outright, or a spare (unmatched) detection within 1.0 m (mislocated)?')
import math
for stage, model in [('s19', 'glm53'), ('s20', 'dsv41'), ('s23', 'glm53'), ('s23', 'dsv41')]:
    rows = [a for a in (P.analyse(r) for r in P.recs_of(stage, model, P.REAL)) if a is not None]
    lines = [a for a in rows if a['rec'].measures.get('members_recall') is not None]
    for name, sel in [('all line goal steps', lines), ('P3 fails', [a for a in lines if P.is_p3_fail(a)])]:
        nf = spare = 0
        gaps = []
        for a in sel:
            sc, th, vis = P.scene_info(a['rec'])
            pos = {p.id: (p.x, p.y) for p in vis}
            spares = [d for d in a['det'] if d.number not in a['matched']]
            for m in a['missed_ca']:
                if m in a['found_ids']:
                    continue
                nf += 1
                g = min((math.dist(d.xy, pos[m]) for d in spares), default=float('inf'))
                spare += g <= 1.0
                if g < float('inf'):
                    gaps.append(g)
        gaps.sort()
        print(f'{stage} {model} {name}: never-found missed members {nf}; with a spare detection within 1.0 m {spare}; '
              f'without one {nf - spare}' + (f'; nearest spare median {gaps[len(gaps)//2]:.2f} m' if gaps else ''))
