"""F2/N2 steelman test ('the variance sits in the constraints'): in each mixed episode (mixedC subset), for every
pair (S succeeding, F failing) of C goal steps with the right target, solve F again with S's people (F's constraints)
and with S's constraints (F's people), via H.resolve. Counts are per pair and per episode (any pair fixes)."""
import sys, collections, io, itertools, time
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
OUT = io.StringIO()
def p(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.write(s + '\n')
def load(stage, model):
    eps = collections.defaultdict(list)
    for arm in ['real', 'staffed', 'plain']:
        for rec in H.records(H.runs(stage=stage, arm=arm, model=model, method='con')[0]):
            eps[(arm, rec.scene_id, rec.episode_id)].append(rec)
    return eps
for st, m in [('s19', 'glm53'), ('s20', 'dsv41'), ('s23', 'glm53'), ('s23', 'dsv41')]:
    t0 = time.time(); eps = load(st, m)
    c = collections.Counter(); ep = collections.Counter()
    for k, v in eps.items():
        cs = [x for x in v if x.result.get('source') == 'constraints' and x.score['target_correct']]
        if len(cs) < 2: continue
        s = [bool(x.score['success']) for x in cs]
        if not (any(s) and not all(s)): continue
        ep['mixed'] += 1
        fix_p = fix_c = fix_both = False; samecount_ep = True
        for S, F in itertools.product([x for x in cs if x.score['success']], [x for x in cs if not x.score['success']]):
            base = H.resolve(F)
            c['pairs'] += 1
            c['base reproduced'] += base.reproduced
            c['base success (should be 0)'] += bool(base.score.success)
            same = len(S.people) == len(F.people)
            samecount_ep &= same
            c['pairs same people count'] += same
            rp = H.resolve(F, people=S.people)
            rc = H.resolve(F, constraints=[y['constraint'] for y in S.constraints])
            rb = H.resolve(F, people=S.people, constraints=[y['constraint'] for y in S.constraints])
            c['F + S people -> success'] += bool(rp.score.success)
            c['F + S constraints -> success'] += bool(rc.score.success)
            c['F + S both -> success'] += bool(rb.score.success)
            if same:
                c['[same count] F + S people -> success'] += bool(rp.score.success)
                c['[same count] F + S constraints -> success'] += bool(rc.score.success)
            fix_p |= bool(rp.score.success); fix_c |= bool(rc.score.success); fix_both |= bool(rb.score.success)
        ep['people swap fixes (any pair)'] += fix_p
        ep['constraints swap fixes (any pair)'] += fix_c
        ep['only people swap fixes'] += fix_p and not fix_c
        ep['only constraints swap fixes'] += fix_c and not fix_p
        ep['both single swaps fix'] += fix_p and fix_c
        ep['neither single swap fixes'] += not fix_p and not fix_c
        ep['joint swap fixes'] += fix_both
    p(f'{st} {m} ({time.time()-t0:.0f}s) episodes:', dict(ep))
    p(f'   pairs:', dict(c))
open('results/2026-10-17-beyond-vlm/rerun/fusion/verify/v10_swap_out.txt', 'w').write(OUT.getvalue())
