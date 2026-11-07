"""I02 re-computation: fusing trials and models (F1-F4 of the report, plus the new-wording rows).

Reads runs only through research_harness (step-0 harness). No model calls, nothing tuned.
Ports the logic of the judge scripts RD/scripts/bounds.py, mixedC.py and cross.py.

Definitions
  episode      = (arm, scene_id, episode_id) over the realistic arms real, staffed, plain (3 x 28 = 84);
                 each episode has trials 0, 1, 2 of one run per arm.
  success      = the recorded score['success'] (as the judge scripts used); a rescore column checks it now.
  any-of-3     = at least 1 of 3 trials succeeds; all-3 = all 3 succeed; majority-of-3 = at least 2 of 3.
  per-trial mean = total successes / 3 (successes per trial, averaged over the 3 trials).
  F2 subset    = C goal steps with the right target: trials with result.source == 'constraints' and
                 score.target_correct; an episode counts when it has >= 2 such trials and their success is
                 mixed (some succeed, some fail) -- mixedC.py.
                 people count = len(result.people).
                 kinds (judge-literal) = sorted multiset of str(entry.get('kind')) over result.constraints:
                 every entry is a wrapper {constraint, status, reason, broken_m}, so this is a tuple of 'None'
                 of length n, i.e. it compares the NUMBER of constraints only.
                 kinds (true) = sorted multiset of entry['constraint']['kind'].
                 kind set (true) = set of entry['constraint']['kind'].
  F3           = 6 samples per episode (3 GLM + 3 DeepSeek trials); count episodes with >= 3 successes.
"""
import collections
import itertools
import math
import sys
import time

sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H  # noqa: E402

ARMS = ('real', 'staffed', 'plain')
OUT = []
RESCORE = '--no-rescore' not in sys.argv


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    OUT.append(s)


_RS = {}


def ok(rec, rescored=False):
    if not rescored:
        return bool(rec.score['success'])
    key = (rec.run, rec.scene_id, rec.episode_id, rec.trial)
    if key not in _RS:
        _RS[key] = bool(H.rescore(rec).success)
    return _RS[key]


def load(stage, model, method='con'):
    """episode -> [rec trial0, trial1, trial2] over the realistic arms."""
    eps = {}
    for arm in ARMS:
        refs = H.runs(stage=stage, arm=arm, model=model, method=method)
        assert len(refs) == 1, (stage, arm, model, method, refs)
        for rec in H.records(refs[0]):
            eps.setdefault((arm, rec.scene_id, rec.episode_id), []).append(rec)
    for k, v in eps.items():
        v.sort(key=lambda r: r.trial)
        assert [r.trial for r in v] == [0, 1, 2], (stage, model, k, [r.trial for r in v])
    assert len(eps) == 84, (stage, model, len(eps))
    return eps


def counts(eps, rescored=False):
    s = {k: [ok(r, rescored) for r in v] for k, v in eps.items()}
    tot = sum(map(sum, s.values()))
    per_trial = [sum(v[t] for v in s.values()) for t in range(3)]
    return dict(n=len(s), total=tot, mean=tot / 3, per_trial=per_trial,
                any=sum(any(v) for v in s.values()), all=sum(all(v) for v in s.values()),
                maj=sum(sum(v) >= 2 for v in s.values()))


def plen(p):
    if p is None:
        return None
    return len(p) if isinstance(p, list) else len(p.get('people', p))


def kinds_judge(c):
    if c is None:
        return None
    L = c if isinstance(c, list) else c.get('constraints', [])
    return tuple(sorted(str(y.get('kind')) for y in L if isinstance(y, dict)))


def _kind(y):
    inner = y.get('constraint') if isinstance(y.get('constraint'), dict) else y
    return str(inner.get('kind'))


def kinds_true(c):
    if c is None:
        return None
    return tuple(sorted(_kind(y) for y in c if isinstance(y, dict)))


def kindset_true(c):
    if c is None:
        return None
    return frozenset(_kind(y) for y in c if isinstance(y, dict))


def nconstr(c):
    return None if c is None else len(c)


def mixed(eps, rescored=False):
    """mixedC.py: C goal steps with the right target, >=2 of them, mixed success."""
    res = collections.Counter()
    sits = collections.Counter()
    for k, v in eps.items():
        cs = [x for x in v if x.result.get('source') == 'constraints' and x.score['target_correct']]
        if len(cs) < 2:
            continue
        s = [ok(x, rescored) for x in cs]
        if any(s) and not all(s):
            res['mixed'] += 1
            sits[v[0].situation] += 1
            res['people_count_differs'] += len(set(plen(x.people) for x in cs)) > 1
            res['kinds_judge_literal_differs'] += len(set(kinds_judge(x.constraints) for x in cs)) > 1
            res['n_constraints_differs'] += len(set(nconstr(x.constraints) for x in cs)) > 1
            res['kinds_true_multiset_differs'] += len(set(kinds_true(x.constraints) for x in cs)) > 1
            res['kind_set_true_differs'] += len(set(kindset_true(x.constraints) for x in cs)) > 1
            res['people_same_but_kinds_true_differ'] += (len(set(plen(x.people) for x in cs)) == 1
                                                          and len(set(kinds_true(x.constraints) for x in cs)) > 1)
        elif not any(s):
            res['all_C_fail'] += 1
        else:
            res['all_C_succeed'] += 1
    return res, sits


def mixed_any(eps, rescored=False):
    """Looser reading: every episode whose 3 trials have mixed success; people/kinds compared over its C steps."""
    res = collections.Counter()
    for k, v in eps.items():
        s = [ok(x, rescored) for x in v]
        if any(s) and not all(s):
            res['mixed_all_trials'] += 1
            cs = [x for x in v if x.result.get('source') == 'constraints']
            if len(cs) >= 2:
                res['with_>=2_C_steps'] += 1
                res['people_count_differs'] += len(set(plen(x.people) for x in cs)) > 1
                res['kinds_true_multiset_differs'] += len(set(kinds_true(x.constraints) for x in cs)) > 1
    return res


def cross(groups, rescored=False):
    """groups: list of episode dicts; pooled samples per episode."""
    keys = groups[0].keys()
    S = {k: [ok(r, rescored) for g in groups for r in g[k]] for k in keys}
    n = len(next(iter(S.values())))
    out = dict(n_eps=len(S), samples=n, any=sum(any(v) for v in S.values()),
               all=sum(all(v) for v in S.values()), mean_per_sample=sum(map(sum, S.values())) / n)
    for t in range(1, n + 1):
        out[f'>={t}'] = sum(sum(v) >= t for v in S.values())
    return out


def fmt_counts(name, c):
    return (f"{name:<28} n={c['n']} total={c['total']}/252 per-trial={c['per_trial']} mean={c['mean']:.2f} "
            f"any={c['any']} maj={c['maj']} all={c['all']}")


def block_F(label, eps_g, eps_d):
    say(f'\n=== {label} ===')
    for name, e in [('GLM', eps_g), ('DeepSeek', eps_d)]:
        say(fmt_counts(name + ' recorded', counts(e)))
        if RESCORE:
            say(fmt_counts(name + ' rescored now', counts(e, True)))
    say('-- F2 (mixedC): C goal steps with right target, >=2 such trials, mixed success')
    for name, e in [('GLM', eps_g), ('DeepSeek', eps_d)]:
        r, sits = mixed(e)
        say(f'{name:<9} recorded ', dict(r), 'situations', dict(sits))
        if RESCORE:
            r2, _ = mixed(e, True)
            say(f'{name:<9} rescored ', dict(r2))
        say(f'{name:<9} looser (mixed over all 3 trials) ', dict(mixed_any(e)))
    say('-- F3 (cross.py): GLM + DeepSeek pooled, 6 samples per episode')
    c = cross([eps_g, eps_d])
    say('recorded', {k: (round(v, 2) if isinstance(v, float) else v) for k, v in c.items()})
    if RESCORE:
        c2 = cross([eps_g, eps_d], True)
        say('rescored', {k: (round(v, 2) if isinstance(v, float) else v) for k, v in c2.items()})


def cache_check(eps_new, eps_old, name):
    same_people = same_target = same_source = same_constraints = both_people = near_target = 0
    n = 0
    for k, v in eps_new.items():
        for a, b in zip(v, eps_old[k]):
            n += 1
            same_target += (a.target or {}).get('point') == (b.target or {}).get('point')
            if a.target and b.target and a.target.get('world') and b.target.get('world'):
                near_target += math.dist(a.target['world'][:2], b.target['world'][:2]) <= 0.05
            same_source += a.result.get('source') == b.result.get('source')
            if a.people is not None and b.people is not None:
                both_people += 1
                same_people += a.people == b.people
            same_constraints += kinds_true(a.constraints) == kinds_true(b.constraints)
    say(f'{name}: records {n}; same marked target point {same_target}; target world within 0.05 m {near_target}; '
        f'same source {same_source}; both have people '
        f'{both_people}, identical people lists {same_people}; same true constraint-kind multiset {same_constraints}')


def main():
    t0 = time.time()
    say('I02 fusion re-computation (research_harness only); rescore check:', RESCORE)
    g19, d20 = load('s19', 'glm53'), load('s20', 'dsv41')
    g23, d23 = load('s23', 'glm53'), load('s23', 'dsv41')

    block_F('Old wording: s19 glm53 con / s20 dsv41 con (realistic arms, 84 episodes)', g19, d20)
    block_F('New wording: s23 glm53 con / s23 dsv41 con (realistic arms, 84 episodes)', g23, d23)

    say('\n=== Cache check: s23 vs s19/s20, same (arm, scene, episode, trial) ===')
    cache_check(g23, g19, 'GLM s23 vs s19')
    cache_check(d23, d20, 'DeepSeek s23 vs s20')
    for name, e in [('GLM s23', g23), ('DeepSeek s23', d23)]:
        c = collections.Counter()
        for v in e.values():
            for r in v:
                for st, tm in r.timings.items():
                    c[st + ' calls'] += tm['calls']
                    c[st + ' cached'] += tm['cached']
        say(name, 'call cache counts', dict(sorted(c.items())))

    say('\n=== 6-model table, C (con), new wording: s21 qwen38_low + s23 others; realistic arms, 84 episodes ===')
    models = [('qwen38_low', 's21'), ('glm53', 's23'), ('dsv41', 's23'), ('haiku55', 's23'),
              ('sonnet55_low', 's23'), ('gemini38_low', 's23')]
    E = {m: (g23 if m == 'glm53' else d23 if m == 'dsv41' else load(st, m)) for m, st in models}
    for rs in ([False, True] if RESCORE else [False]):
        say(f'-- {"rescored now" if rs else "recorded score"}')
        for m, st in models:
            say(fmt_counts(f'{m} ({st})', counts(E[m], rs)))
        allc = cross([E[m] for m, _ in models], rs)
        say('pooled 18 samples:', {k: allc[k] for k in ['any', '>=6', '>=9', '>=10', '>=12', 'all']},
            f"mean/sample={allc['mean_per_sample']:.2f}")
        keys = list(E['glm53'].keys())
        majs = {k: sum(sum(ok(r, rs) for r in E[m][k]) >= 2 for m, _ in models) for k in keys}
        say('vote of per-model majorities (6 voters): >=3 of 6', sum(v >= 3 for v in majs.values()),
            '| >=4 of 6 (strict majority)', sum(v >= 4 for v in majs.values()),
            '| any model majority', sum(v >= 1 for v in majs.values()))
        say('pairs (6 samples): any / >=3 / >=4 / all')
        for (a, _), (b, _) in itertools.combinations(models, 2):
            c = cross([E[a], E[b]], rs)
            say(f'  {a:>12} + {b:<12} any={c["any"]} >=3={c[">=3"]} >=4={c[">=4"]} all={c["all"]}')

    say('\n=== Context: old-wording 3-model table, C, realistic arms (s19 qwen38_low, s19 glm53, s20 dsv41) ===')
    q19 = load('s19', 'qwen38_low')
    for m, e in [('qwen38_low s19', q19), ('glm53 s19', g19), ('dsv41 s20', d20)]:
        say(fmt_counts(m, counts(e)))
    c = cross([q19, g19, d20])
    say('pooled 9 samples:', {k: c[k] for k in ['any', '>=3', '>=4', '>=5', 'all']})

    say(f'\nelapsed {time.time() - t0:.0f} s')
    with open('results/2026-10-17-beyond-vlm/'
              'rerun/fusion/fusion_out.txt', 'w') as f:
        f.write('\n'.join(OUT) + '\n')


if __name__ == '__main__':
    main()
