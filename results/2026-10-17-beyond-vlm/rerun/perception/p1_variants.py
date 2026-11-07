"""P1 definition search: which reading of 'missed members never found' gives the report's ~81% (GLM s19 con, realistic).
Also: is the phantom the last listed person the solver queues behind (s23 / s19 GLM lines)."""
import math, sys, collections
sys.path.insert(0, 'results/2026-10-17-beyond-vlm/rerun/perception')
import perception as P
from sgi.types import DetectedPerson

def rows_of(stage, model):
    rows = [a for a in (P.analyse(r) for r in P.recs_of(stage, model, P.REAL)) if a is not None]
    for a in rows: a['p3fail'] = P.is_p3_fail(a)
    return rows

for stage, model in [('s19', 'glm53'), ('s20', 'dsv41'), ('s23', 'glm53'), ('s23', 'dsv41')]:
    rows = rows_of(stage, model)
    print(f'\n=== {stage} {model}')
    lines = [a for a in rows if a['rec'].measures.get('members_recall') is not None]
    fails = [a for a in lines if a['p3fail']]
    nonsucc = [a for a in lines if not a['rec'].score.get('success')]
    # V1: radius sweep, one-to-one greedy vs any detection within r
    for sel_name, sel in [('all lines', lines), ('P3 fails', fails), ('all failed line goal steps', nonsucc)]:
        res = []
        for r in (0.25, 0.3, 0.5, 0.75, 1.0):
            missed = nf = 0
            for a in sel:
                sc, th, vis = P.scene_info(a['rec'])
                pos = {p.id: (p.x, p.y) for p in vis}
                for m in a['missed_ca']:
                    missed += 1
                    nf += not any(math.dist(d.xy, pos[m]) <= r for d in a['det'])
            res.append(f'{r}m:{nf}/{missed}={nf/missed:.3f}' if missed else f'{r}m:-')
        print(f'  V1 any-detection-within-r, {sel_name}: ' + '  '.join(res))
    # V2: record-level shares: sum (1-found_share)*n_vis / sum (1-recall)*n_members
    for sel_name, sel in [('all lines', lines), ('P3 fails', fails)]:
        unfound_vis = sum((a['n_vis'] - len(a['found_ids'])) for a in sel)
        unfound_mem = sum(sum(m not in a['found_ids'] for m in a['members']) for a in sel)
        missed = sum(len(a['missed_ca']) for a in sel)
        nmem = sum(len(a['members']) for a in sel)
        print(f'  V2 {sel_name}: members {nmem}, missed {missed}, unfound members (any, named or not) {unfound_mem} '
              f'-> unfound/missed {unfound_mem/missed:.3f}; unfound visible people (members or not) {unfound_vis} -> {unfound_vis/missed:.3f}')
    # V3: per-record mean share
    shares = [sum(m not in a['found_ids'] for m in a['missed_ca']) / len(a['missed_ca']) for a in lines if a['missed_ca']]
    print(f'  V3 mean per-record share over line records with a miss: {sum(shares)/len(shares):.3f} (n {len(shares)})')
    # V4: lines + groups pooled (groups: named in any people list)
    gm = sum(len(a['missed_any']) for a in rows if a['rec'].situation == 'conversation_group' and a['has_cons'] and a['members'])
    gn = sum(sum(m not in a['found_ids'] for m in a['missed_any']) for a in rows if a['rec'].situation == 'conversation_group' and a['has_cons'] and a['members'])
    lm = sum(len(a['missed_ca']) for a in lines); ln = sum(sum(m not in a['found_ids'] for m in a['missed_ca']) for a in lines)
    print(f'  V4 lines+groups pooled: {ln+gn}/{lm+gm} = {(ln+gn)/(lm+gm):.3f}')
    # V5: missed by any constraint (not only come_after) at lines
    m5 = sum(len(a['missed_any']) for a in lines); n5 = sum(sum(m not in a['found_ids'] for m in a['missed_any']) for a in lines)
    print(f'  V5 lines, named = any constraint people list: {n5}/{m5} = {n5/m5:.3f}' if m5 else '  V5 -')
    # V6: lines but excluding records where nothing was stated under come_after
    sel = [a for a in lines if a['come_after_stated']]
    m6 = sum(len(a['missed_ca']) for a in sel); n6 = sum(sum(m not in a['found_ids'] for m in a['missed_ca']) for a in sel)
    print(f'  V6 lines with a come_after stated: {n6}/{m6} = {n6/m6:.3f} (records {len(sel)})')
    sel = [a for a in fails if a['come_after_stated']]
    m7 = sum(len(a['missed_ca']) for a in sel); n7 = sum(sum(m not in a['found_ids'] for m in a['missed_ca']) for a in sel)
    print(f'  V7 P3 fails with a come_after stated: {n7}/{m7} = {n7/m7:.3f} (records {len(sel)})')
    # V8: distinct (arm, scene, trial) people answers x episode? lines distinct by (arm, scene, episode, trial) is all
    # Phantom-as-last check
    queued = phantom_listed = 0; dists = []
    for a in lines:
        if not a['rec'].solver_step: continue
        far = {d.number for d, dm in zip(a['det'], a['det_min_all']) if dm > 1.0}
        listed = [d for d in a['det'] if d.number in a['ca_nums']]
        if not (far & a['ca_nums']) or not listed: continue
        phantom_listed += 1
        sc, th, vis = P.scene_info(a['rec'])
        tgt = a['rec'].target['world'] if a['rec'].target else None
        if tgt is None: continue
        last = max(listed, key=lambda d: math.dist(d.xy, tgt[:2]))
        if last.number in far:
            queued += 1
            pw = a['rec'].pose_world
            dists.append(math.dist(pw[:2], last.xy))
    dists.sort()
    print(f'  phantom: line goal steps with a phantom listed under come_after {phantom_listed}; the phantom is the listed '
          f'person farthest from the target (the one the solver queues behind) in {queued}; place-to-phantom distance median '
          f'{dists[len(dists)//2] if dists else float("nan"):.2f} m (min {dists[0] if dists else float("nan"):.2f}, max {dists[-1] if dists else float("nan"):.2f})')
