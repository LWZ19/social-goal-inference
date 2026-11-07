"""Scan of plausible P1 definitions to find which reproduce ~0.81 (GLM s19 con)."""
import sys, math, itertools
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
from sgi.gt.derive import is_visible
sys.path.insert(0, 'results/still-scenes/perception/perception-rerun/verify')
from v_main import my_match
OUT = __file__.replace('.py', '_out.txt')
L = []
def P(*a):
    s = ' '.join(map(str, a)); print(s); L.append(s)
rows = []
for r in H.runs(stage='s19', model='glm53', method='con'):
    if r.arm.startswith('dev'): continue
    th = H.recorded_config(r)[1].thresholds
    for rec in H.records(r):
        if not rec.solver_step: continue
        sc = H.scene(r, rec.scene_id)
        vis = [p for p in sc.truth.people if is_visible(p, sc.observation, th)]
        m = my_match(rec.people, vis, th.person_radius_m)
        ep = sc.truth.episodes[rec.episode_id]
        chosen = rec.score.get('matched_target_id') or ep.target_ids[0]
        g = sc.truth.group_for(rec.episode_id, chosen)
        if g is None: continue
        allmem = [p for p in sc.truth.people if p.group == g]
        for vis_only in (True, False):
            members = [p for p in allmem if (p in vis) or not vis_only]
            for naming in ('come_after', 'shared', 'any'):
                if rec.situation == 'waiting_line' and naming != 'come_after' and naming != 'any': continue
                if naming == 'come_after': kinds = {'come_after'}
                elif naming == 'shared': kinds = {'keep_out_of_shared_space'}
                else: kinds = None
                listed = {n for c in rec.constraints if (kinds is None or c['constraint']['kind'] in kinds) for n in c['constraint']['people']}
                named = {m.get(n) for n in listed}
                missed = [p for p in members if p.id not in named]
                for found_rule in ('match', 'r0.5', 'r0.25'):
                    if found_rule == 'match': nf = [p for p in missed if p.id not in set(m.values())]
                    else:
                        rad = float(found_rule[1:])
                        nf = [p for p in missed if not any(math.dist(d['xy'], (p.x, p.y)) <= rad for d in rec.people)]
                    s = rec.score
                    rows.append(dict(arm=r.arm, sit=rec.situation, vis_only=vis_only, naming=naming, found=found_rule,
                                     miss=len(missed), nf=len(nf), success=s.get('success'),
                                     fail=bool(s.get('target_correct') and not s.get('success') and (not s.get('in_region') or not s.get('reachable')))))
hits = []
for arms_name, arms in (('realistic', {'real', 'staffed', 'plain'}), ('+ph', {'real', 'staffed', 'plain', 'ph'}), ('ph', {'ph'})):
    for sits in ('lines', 'groups', 'both'):
        for vis_only in (True, False):
            for gnaming in ('shared', 'any'):
                for lnaming in ('come_after', 'any'):
                    for found in ('match', 'r0.5', 'r0.25'):
                        for subset in ('all', 'failed', 'P3fail'):
                            def ok(x):
                                if x['arm'] not in arms or x['vis_only'] != vis_only or x['found'] != found: return False
                                if x['sit'] == 'waiting_line':
                                    if sits == 'groups' or x['naming'] != lnaming: return False
                                else:
                                    if sits == 'lines' or x['naming'] != gnaming: return False
                                if subset == 'failed' and x['success']: return False
                                if subset == 'P3fail' and not x['fail']: return False
                                return True
                            sub = [x for x in rows if ok(x)]
                            a = sum(x['nf'] for x in sub); b = sum(x['miss'] for x in sub)
                            if b:
                                hits.append((abs(a / b - 0.81), f'{arms_name:9s} {sits:6s} vis_only={vis_only!s:5s} groupname={gnaming:6s} linename={lnaming:10s} found={found:5s} {subset:6s}: {a}/{b} = {a/b:.3f}'))
hits = sorted(set(hits))
P('closest to 0.81 (deduplicated):')
for h in hits[:30]: P('  ', h[1])
open(OUT, 'w').write('\n'.join(L) + '\n')
