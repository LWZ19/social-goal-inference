"""Independent verification of M1 (I48 flags) and M2 (I47 standard-goal fallback), through research_harness only."""
import sys, collections
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H

REAL = ('real', 'staffed', 'plain')
SIX = ('no_usable_constraint', 'constraint_unusable', 'constraints_cut', 'exclusion_broken', 'people_cut',
       'far_from_target')
OTHER_WARN = ('marked_point_off_floor', 'standard_goal_blocked', 'heading_fallback', 'heading_degenerate')

out = []
def p(*a):
    s = ' '.join(str(x) for x in a)
    print(s); out.append(s)


def load(stage, model, arms):
    recs = []
    for arm in arms:
        rs = H.runs(stage=stage, model=model, method='con', arm=arm)
        assert len(rs) == 1, (stage, model, arm, rs)
        t = collections.Counter()
        recs += list(H.records(rs[0], tally=t))
        p(f'  {rs[0].folder}: {dict(t)}')
    return recs


def flagcount(recs, names, label):
    rec_s = rec_f = fire_s = fire_f = 0
    per = collections.Counter()
    for r in recs:
        fl = r.result.get('flags') or {}
        on = [n for n in names if fl.get(n)]
        ok = bool(r.score.get('success'))
        for n in on:
            per[(n, ok)] += 1
        if on:
            if ok: rec_s += 1
            else: rec_f += 1
        if ok: fire_s += len(on)
        else: fire_f += len(on)
    ns = sum(bool(r.score.get('success')) for r in recs)
    p(f'  {label}: n={len(recs)} succ={ns} fail={len(recs)-ns} | flagged records succ={rec_s} fail={rec_f} '
      f'| firings succ={fire_s} fail={fire_f} | per flag (name, success): {dict(sorted(per.items()))}')


p('== M1: s20 dsv41 con ==')
p('realistic (real, staffed, plain):')
ds_real = load('s20', 'dsv41', REAL)
flagcount(ds_real, SIX, 'six constraint flags')
flagcount(ds_real, SIX + OTHER_WARN, 'six + 4 other warn flags (not frontal)')
flagcount([r for r in ds_real if r.solver_step], SIX, 'six, solver steps only')
p('all seven s20 dsv41 con runs:')
ds_all = ds_real + load('s20', 'dsv41', ('ph', 'dev_seed0', 'dev_seed1', 'dev_seed2'))
flagcount(ds_all, SIX, 'six constraint flags, 7 runs')
flagcount(ds_all, SIX + OTHER_WARN, 'six + 4 other, 7 runs')
for arm in ('ph', 'dev_seed0', 'dev_seed1', 'dev_seed2'):
    flagcount([r for r in ds_all if r.arm == arm], SIX, f'six, {arm}')

p('\n== M2: standard-goal misses under C, realistic ==')
models = {'qwen38_low': ('s19',), 'glm53': ('s19',), 'dsv41': ('s20',)}
for model, (stage,) in models.items():
    p(f'{model}:')
    recs = ds_real if model == 'dsv41' else load(stage, model, REAL)
    std = [r for r in recs if r.result.get('source') == 'standard']
    miss = [r for r in std if not r.score.get('success') and r.score.get('target_correct')]
    miss_out = [r for r in miss if not r.score.get('in_region')]
    noone = [r for r in miss_out if r.situation == 'no_one']
    fb = lambda rs: sum(not (r.result.get('flags') or {}).get('standard_goal_frontal') for r in rs)
    p(f'  n={len(recs)} standard-source={len(std)} std fail & target_correct={len(miss)} of which outside region='
      f'{len(miss_out)} (situations {dict(collections.Counter(r.situation for r in miss_out))})')
    p(f'  no_one std misses outside region: {len(noone)}, fallback (frontal False) among them: {fb(noone)}; '
      f'among all std misses outside region: {fb(miss_out)}; among all std fail&tc: {fb(miss)}')
    p(f'  other properties of the no_one misses: reachable={sum(bool(r.score.get("reachable")) for r in noone)} '
      f'blocked={sum(bool((r.result.get("flags") or {}).get("standard_goal_blocked")) for r in noone)} '
      f'heading_ok={sum(bool(r.score.get("heading_ok")) for r in noone)}')
    # all standard fails (target incorrect too), for context
    p(f'  all standard-source fails: {sum(not r.score.get("success") for r in std)}; '
      f'target incorrect: {sum(not r.score.get("success") and not r.score.get("target_correct") for r in std)}')
    # N2 split: no_one, standard, target_correct, by frontal and success
    base = [r for r in std if r.situation == 'no_one' and r.score.get('target_correct')]
    c = collections.Counter((bool((r.result.get('flags') or {}).get('standard_goal_frontal')), bool(r.score.get('success'))) for r in base)
    p(f'  N2: no_one std target_correct (frontal, success): {dict(sorted(c.items()))}')
    if model == 'dsv41':
        e2 = [r for r in recs if '_eval_02_' in r.scene_id]
        p(f'  N3: eval_02 scenes {sorted(set(r.scene_id for r in e2))}: n={len(e2)} succ={sum(bool(r.score.get("success")) for r in e2)}; '
          f'std misses there {sum(1 for r in noone if "_eval_02_" in r.scene_id)} of {len(noone)}; '
          f'scenes of the others: {collections.Counter(r.scene_id for r in noone if "_eval_02_" not in r.scene_id)}')
        nonstd = [r for r in e2 if not r.score.get('success') and r.result.get('source') != 'standard']
        p(f'  N3: eval_02 non-standard fails: {len(nonstd)} sources {collections.Counter(r.result.get("source") for r in nonstd)} '
          f'in_region {collections.Counter(bool(r.score.get("in_region")) for r in nonstd)}')
    if model != 'dsv41':
        p('  N1 flags:')
        flagcount(recs, SIX, f'six flags {model}')
        flagcount([r for r in recs if r.solver_step], SIX, f'six flags {model} solver steps')

open('results/2026-10-17-beyond-vlm/rerun/flags_misc/verify/v_m1_m2.txt', 'w').write('\n'.join(out) + '\n')
