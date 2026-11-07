"""I39 model-escalation cascade, re-computed through research_harness only.

Port of analysis-scripts/cascade_peek.py (the judge's script), with its definitions kept:
  - records of the realistic sets (real, staffed, plain), method con, paired by (scene_id, episode_id, trial);
  - goal step of the cheap model = result.solver is not None (judge's goalstep());
  - ok = recorded score.success; escalation candidates = the cheap model's goal-step records only;
  - escalated record takes the strong model's recorded success (curve) or, for the plan-reuse version,
    strong-model success AND cheap-model target_correct (curve2, judge's proxy);
  - latency per goal-step record = provider seconds (sum of per-call latency_s) of the cheap model plus, when
    escalated, the strong model's people + constraints provider seconds (judge's accounting);
  - signals: dis = max xy distance of this trial's pose to the other two trials' poses (3 samples), dis1 = distance
    to the next trial (k+1 mod 3; 2 samples), ca = any constraint kind containing 'after', gap/exc/np as judge;
  - random curve = random.seed(0) then one draw per goal-step key in sorted order (judge's replica), plus the
    analytic expectation of random escalation.
Additions (stated as such in the output): dis1 run through curve(); a sequential-full-strong latency column; a
plan-reuse estimate that re-runs the current solver (H.resolve) on the cheap record with its own target and the
strong model's people and constraints; brief-style latency (mean record measures.latency_s over goal-step records).
"""
import collections
import math
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H  # noqa: E402

OUT = Path(__file__).resolve().parent
ARMS = ('real', 'staffed', 'plain')
FRACS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
FRACS2 = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]
LOG = []


def out(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def load(stage, model):
    d = {}
    for arm in ARMS:
        refs = H.runs(stage=stage, arm=arm, model=model, method='con')
        assert len(refs) == 1, (stage, arm, model, refs)
        for r in H.records(refs[0]):
            k = (r.scene_id, r.episode_id, r.trial)
            assert k not in d
            d[k] = r
    return d


def prov(r, stages=None):
    return sum(v['latency_s'] for s, v in r.timings.items() if stages is None or s in stages)


def goalstep(r):
    return r.solver is not None


def ok(r):
    return bool(r.score['success'])


def tc(r):
    return bool(r.score.get('target_correct'))


def auroc(score, lab):
    pos = [s for s, l in zip(score, lab) if l]
    neg = [s for s, l in zip(score, lab) if not l]
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / len(pos) / len(neg)


def r3(x):
    return None if x is None else round(x, 3)


def analyse(tag, G, Q, resolve_reuse=True):
    out('=' * 100)
    out(f'PAIR {tag}')
    keys = sorted(G)
    missing = set(G) ^ set(Q)
    out(f'records: cheap {len(G)}, strong {len(Q)}, unpaired {len(missing)}')
    keys = [k for k in keys if k in Q]
    N = len(keys)
    gs = [k for k in keys if goalstep(G[k])]
    n = len(gs)
    base = sum(ok(G[k]) for k in keys)
    qall = sum(ok(Q[k]) for k in keys)
    out(f'N paired={N}; cheap goal-step records={n}; cheap succ all={base}/{N} ({100*base/N:.1f}%); '
        f'strong succ all={qall}/{N} ({100*qall/N:.1f}%)')
    c = collections.Counter((ok(G[k]), ok(Q[k]), goalstep(G[k])) for k in keys)
    out('(cheap ok, strong ok, cheap goalstep) counts:', dict(sorted(c.items())))
    f = [k for k in gs if not ok(G[k])]
    gain_ks = [k for k in gs if not ok(G[k]) and ok(Q[k])]
    loss_ks = [k for k in gs if ok(G[k]) and not ok(Q[k])]
    out(f'K1-type: cheap goal-step fails={len(f)}; strong succ on them={sum(ok(Q[k]) for k in f)}; '
        f'cheap target_correct on them={sum(tc(G[k]) for k in f)}; strong ok & cheap target correct='
        f'{sum(ok(Q[k]) and tc(G[k]) for k in f)}')
    out(f'   strong-only gains among goal-step={len(gain_ks)}; losses (cheap ok, strong fail)={len(loss_ks)}; '
        f'cheap non-goal-step fails={sum(not ok(G[k]) for k in keys if not goalstep(G[k]))} (not escalatable)')
    out(f'   same matched target cheap vs strong (cheap goal-step): '
        f'{sum(G[k].score.get("matched_target_id") == Q[k].score.get("matched_target_id") for k in gs)}/{n}')
    out('   by situation (cheap goal-step): (situation, cheap ok, strong ok) ->',
        dict(sorted(collections.Counter((G[k].situation, ok(G[k]), ok(Q[k])) for k in gs).items())))

    # latency
    def lat_lines(D, name):
        own_gs = [k for k in keys if goalstep(D[k])]
        ss = [k for k in keys if D[k].solver_step]
        rec_lat = [D[k].latency_s for k in own_gs if D[k].latency_s is not None]
        out(f'   {name}: own goal-step n={len(own_gs)} (solver_step n={len(ss)}); '
            f'record latency_s mean (median) over own goal-step = {st.mean(rec_lat):.1f} ({st.median(rec_lat):.1f}); '
            f'provider-only mean over own goal-step = {st.mean(prov(D[k]) for k in own_gs):.1f}; '
            f'record latency_s all = {st.mean(D[k].latency_s for k in keys):.1f}; '
            f'people+constraints provider over own goal-step = '
            f'{st.mean(prov(D[k], {"people", "constraints"}) for k in own_gs):.1f}; '
            f'cached calls share = {sum(v["cached"] for k in keys for v in D[k].timings.values())}/'
            f'{sum(v["calls"] for k in keys for v in D[k].timings.values())}')
    out('latency (K5-type; brief definition = mean record measures.latency_s over goal-step episodes):')
    lat_lines(G, 'cheap ')
    lat_lines(Q, 'strong')
    out(f'   judge: cheap provider mean over cheap goal-step = {st.mean(prov(G[k]) for k in gs):.1f}; '
        f'strong full provider over cheap goal-step keys = {st.mean(prov(Q[k]) for k in gs):.1f}')

    # signals
    def sig(k):
        r = G[k].result
        cs = r.get('constraints') or []
        kinds = [((x.get('constraint') or {}).get('kind') if isinstance(x, dict) else None) for x in cs]
        ca = any(kk and 'after' in kk for kk in kinds)
        cand = r.get('candidates') or []
        gap = None
        if len(cand) >= 2:
            a, b = cand[0]['keys'], cand[1]['keys']
            gap = abs((b.get('preference_m') or 0) - (a.get('preference_m') or 0)) + \
                abs((b.get('excluded_m') or 0) - (a.get('excluded_m') or 0))
        exc = (cand[0]['keys'].get('excluded_m') or 0) if cand else 0
        npeople = len(r.get('people') or [])
        p = r.get('pose_world')
        others = [G[(k[0], k[1], t)].pose_world for t in range(3) if t != k[2] and (k[0], k[1], t) in G]
        dis = max([math.dist(p[:2], o[:2]) for o in others if o and p] or [0])
        o1 = G.get((k[0], k[1], (k[2] + 1) % 3))
        o1 = o1.pose_world if o1 is not None else None
        dis1 = math.dist(p[:2], o1[:2]) if p and o1 else 0
        return dict(ca=ca, gap=gap if gap is not None else 0, exc=exc, np=npeople, dis=dis, dis1=dis1, kinds=kinds)

    S = {k: sig(k) for k in gs}
    out('constraint kinds seen:', collections.Counter(kk for k in gs for kk in S[k]['kinds']).most_common())
    lab = [not ok(G[k]) for k in gs]
    gain = [(not ok(G[k])) and ok(Q[k]) for k in gs]
    out(f'AUROC over cheap goal-step records (n={n}; fails={sum(lab)}; gains={sum(gain)}):')
    au = {}
    for name in ['ca', 'gap', 'exc', 'np', 'dis', 'dis1']:
        sc = [float(S[k][name]) for k in gs]
        if name == 'gap':
            sc = [-x for x in sc]
        au[name] = (auroc(sc, lab), auroc(sc, gain))
        out(f'   {name:5s} AUROC fail={r3(au[name][0])}  AUROC gain={r3(au[name][1])}')
    if any(not x for x in lab) and any(lab):
        out(f'   dis mean: fail={st.mean(S[k]["dis"] for k in gs if not ok(G[k])):.3f} '
            f'succ={st.mean(S[k]["dis"] for k in gs if ok(G[k])):.3f}')

    def curve(score, fracs=FRACS, reuse=None):
        order = sorted(gs, key=lambda k: -score[k])
        res = []
        for frac in fracs:
            esc = set(order[:round(frac * n)])
            if reuse is None:
                succ = sum(ok(Q[k]) if k in esc else ok(G[k]) for k in keys)
            else:
                succ = sum(reuse(k) if k in esc else ok(G[k]) for k in keys)
            lat = st.mean(prov(G[k]) + (prov(Q[k], {'people', 'constraints'}) if k in esc else 0) for k in gs)
            lat_full = st.mean(prov(G[k]) + (prov(Q[k]) if k in esc else 0) for k in gs)
            res.append((frac, len(esc), succ, round(lat, 1), round(lat_full, 1)))
        return res

    def show(name, cv):
        out(f'   {name}: ' + '  '.join(f'{fr:.1f}[{m}]:{s} ({la}s/{lf}s)' for fr, m, s, la, lf in cv))

    dis = {k: S[k]['dis'] for k in gs}
    dis1 = {k: S[k]['dis1'] for k in gs}
    comb = {k: S[k]['dis'] + (5 if S[k]['ca'] else 0) for k in gs}
    rng = random.Random(12345)
    orc = {k: (1 if (not ok(G[k]) and ok(Q[k])) else 0) + rng.random() * 0.01 for k in gs}
    random.seed(0)
    rnd = {k: random.random() for k in gs}
    out(f'curves (Qwen-own-plan version): frac[escalated]:successes/{N} (judge latency s = cheap provider + strong '
        f'people+constraints when escalated / sequential full strong provider when escalated), mean over cheap '
        f'goal-step records')
    C = {}
    for name, sc in [('dis', dis), ('ca+dis', comb), ('dis1', dis1), ('oracle', orc), ('random seed0', rnd)]:
        C[name] = curve(sc)
        show(name, C[name])
    delta = sum(ok(Q[k]) - ok(G[k]) for k in gs)
    out('   random expectation: ' + '  '.join(f'{fr:.1f}:{base + round(fr * n) / n * delta:.1f}' for fr in FRACS))
    # random spread over seeds at 0.5
    vals = []
    for s in range(2000):
        rr = random.Random(s)
        order = rr.sample(gs, n)
        esc = set(order[:round(0.5 * n)])
        vals.append(sum(ok(Q[k]) if k in esc else ok(G[k]) for k in keys))
    vals.sort()
    out(f'   random @0.5 over 2000 seeds: mean {st.mean(vals):.1f}, 5-95% [{vals[100]}, {vals[1899]}]')

    out(f'plan-reuse curves (judge proxy: escalated ok = strong ok AND cheap target_correct):')
    reuse_proxy = lambda k: ok(Q[k]) and tc(G[k])  # noqa: E731
    C2 = {}
    for name, sc in [('dis', dis), ('ca+dis', comb), ('dis1', dis1), ('oracle', orc)]:
        C2[name] = curve(sc, FRACS2, reuse_proxy)
        show(name, C2[name])
    best = max((s, name, fr) for name in ['dis', 'ca+dis', 'dis1'] for fr, m, s, la, lf in C2[name])
    out(f'   plan-reuse proxy, best over non-oracle gates: {best[0]} ({best[1]} @ {best[2]}); '
        f'ceiling (escalate all) = {C2["dis"][-1][2]}; oracle best = {max(x[2] for x in C2["oracle"])}')

    res_reuse = None
    if resolve_reuse:
        # solver replay: cheap record, its own target, strong model's people and constraints
        rs = {}
        nfall = 0
        nrep = 0
        for k in gs:
            q = Q[k]
            if q.people is not None and q.constraints is not None:
                constraints = [x['constraint'] if isinstance(x, dict) and 'constraint' in x else x
                               for x in q.constraints]
                try:
                    rv = H.resolve(G[k], people=q.people, constraints=constraints)
                    rs[k] = bool(rv.score.success)
                    nrep += 1
                except Exception as e:  # noqa: BLE001
                    out('   resolve error', k, repr(e)[:200])
                    rs[k] = reuse_proxy(k)
                    nfall += 1
            else:
                rs[k] = reuse_proxy(k)
                nfall += 1
        out(f'plan-reuse curves (solver replay: H.resolve on the cheap record with its own target and plan, strong '
            f'people+constraints; {nrep} replayed, {nfall} with no strong constraints fall back to the proxy):')
        out(f'   replay ok on cheap goal-step = {sum(rs.values())}/{n}; on cheap fails = '
            f'{sum(rs[k] for k in f)}/{len(f)}; proxy on cheap fails = {sum(reuse_proxy(k) for k in f)}/{len(f)}')
        C3 = {}
        for name, sc in [('dis', dis), ('ca+dis', comb), ('dis1', dis1)]:
            C3[name] = curve(sc, FRACS2, lambda k: rs[k])
            show(name, C3[name])
        orc3 = {k: (1 if (not ok(G[k]) and rs[k]) else 0) + rng.random() * 0.01 for k in gs}
        C3['oracle'] = curve(orc3, FRACS2, lambda k: rs[k])
        show('oracle', C3['oracle'])
        best3 = max((s, name, fr) for name in ['dis', 'ca+dis', 'dis1'] for fr, m, s, la, lf in C3[name])
        out(f'   solver-replay plan reuse, best over non-oracle gates: {best3[0]} ({best3[1]} @ {best3[2]}); '
            f'ceiling = {C3["dis"][-1][2]}; oracle best = {max(x[2] for x in C3["oracle"])}')
        res_reuse = dict(best=best3, ceiling=C3['dis'][-1][2], oracle=max(x[2] for x in C3['oracle']))

    summ = dict(N=N, n=n, base=base, qall=qall, fails=len(f), qfix=sum(ok(Q[k]) for k in f), au=au, C=C, C2=C2,
                best_reuse=best, res_reuse=res_reuse, rand_mean=st.mean(vals))
    return summ


def main():
    pairs = [
        ('s19 glm53 -> s19 qwen38_low (judge pair)', ('s19', 'glm53'), ('s19', 'qwen38_low')),
        ('NEW s23 glm53 -> s21 qwen38_low', ('s23', 'glm53'), ('s21', 'qwen38_low')),
        ('NEW s23 glm53 -> s23 sonnet55_low', ('s23', 'glm53'), ('s23', 'sonnet55_low')),
        ('NEW s23 glm53 -> s23 gemini38_low', ('s23', 'glm53'), ('s23', 'gemini38_low')),
    ]
    cache = {}
    summary = {}
    for tag, a, b in pairs:
        for x in (a, b):
            if x not in cache:
                cache[x] = load(*x)
        summary[tag] = analyse(tag, cache[a], cache[b])
    out('=' * 100)
    out('SUMMARY (successes of N; escalation fractions are of the cheap model\'s goal-step records)')
    for tag, s in summary.items():
        C = s['C']
        at = lambda name, fr: next(x[2] for x in C[name] if x[0] == fr)  # noqa: E731
        out(f'{tag}: N={s["N"]} cheap={s["base"]} strong={s["qall"]} goal-step n={s["n"]} | fails {s["fails"]}, '
            f'strong fixes {s["qfix"]} | oracle@0.3={at("oracle", 0.3)} | dis@0.5={at("dis", 0.5)} '
            f'ca+dis@0.5={at("ca+dis", 0.5)} dis1@0.5={at("dis1", 0.5)} random-seed0@0.5={at("random seed0", 0.5)} '
            f'random-mean@0.5={s["rand_mean"]:.1f} | AUROC fail/gain dis={r3(s["au"]["dis"][0])}/{r3(s["au"]["dis"][1])} '
            f'dis1={r3(s["au"]["dis1"][0])}/{r3(s["au"]["dis1"][1])} | plan-reuse proxy best={s["best_reuse"][0]} '
            f'| plan-reuse replay best={None if s["res_reuse"] is None else s["res_reuse"]["best"][0]}')
    (OUT / 'cascade_out.txt').write_text('\n'.join(LOG) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
