"""Independent verification of I39 (model-escalation cascade). Reads runs only through research_harness.
Ports cascade_peek.py's definitions; adds tie-robustness and random-expectation checks."""
import sys, math, random, statistics as st
from collections import Counter
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H

OUT = []
def p(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.append(s)

ARMS3 = ['real', 'staffed', 'plain']

def load(stage, model):
    d = {}
    for arm in ARMS3:
        rs = H.runs(stage=stage, arm=arm, model=model, method='con')
        assert len(rs) == 1, (stage, arm, model, rs)
        for r in H.records(rs[0]):
            k = (r.scene_id, r.episode_id, r.trial)
            assert k not in d
            d[k] = r
    return d

def ok(r): return bool(r.score.get('success'))
def prov(r, stages=None):
    return sum((c.get('latency_s') or 0) for c in (r.result.get('calls') or []) if stages is None or c['stage'] in stages)
def prov_total(r, stages=None):
    return sum((c.get('latency_total_s') or 0) for c in (r.result.get('calls') or []) if stages is None or c['stage'] in stages)
def fresh(r, stages=None):
    return sum((c.get('latency_s') or 0) for c in (r.result.get('calls') or [])
               if (stages is None or c['stage'] in stages) and not c.get('cached'))
def is_gs(r): return r.result.get('solver') is not None

def auroc(score, lab):
    pos = [s for s, l in zip(score, lab) if l]; neg = [s for s, l in zip(score, lab) if not l]
    if not pos or not neg: return float('nan')
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / len(pos) / len(neg)

def analyse(G, Q, label, do_replay=False):
    p('=' * 100); p(label)
    keys = sorted(set(G) & set(Q))
    p('G records', len(G), 'Q records', len(Q), 'paired', len(keys), 'unpaired G', len(set(G) - set(Q)), 'unpaired Q', len(set(Q) - set(G)))
    gs = [k for k in keys if is_gs(G[k])]
    gs_prop = [k for k in keys if G[k].solver_step]
    p('G goal-step (solver not None)', len(gs), '| via Record.solver_step', len(gs_prop), '| same set', set(gs) == set(gs_prop))
    baseG = sum(ok(G[k]) for k in keys); baseQ = sum(ok(Q[k]) for k in keys)
    p(f'G success {baseG}/{len(keys)} = {baseG/len(keys):.3f}; Q success {baseQ}/{len(keys)} = {baseQ/len(keys):.3f}')
    p('(G ok, Q ok, G goalstep) counts:', sorted(Counter((ok(G[k]), ok(Q[k]), is_gs(G[k])) for k in keys).items()))
    fails = [k for k in gs if not ok(G[k])]
    qfix = [k for k in fails if ok(Q[k])]
    tc = [k for k in fails if G[k].score.get('target_correct')]
    p(f'K1-type: G goal-step fails {len(fails)}; Q succeeds on {len(qfix)}; G target_correct among fails {len(tc)}; Q ok & G tc {sum(1 for k in qfix if G[k].score.get("target_correct"))}')
    p('G non-goal-step fails', sum(1 for k in keys if not is_gs(G[k]) and not ok(G[k])),
      '| G ok & Q fail on goal-step', sum(1 for k in gs if ok(G[k]) and not ok(Q[k])))
    # signals
    def pose(k):
        r = G.get(k); return r.pose_world if r is not None else None
    dis, dis1, ca = {}, {}, {}
    for k in gs:
        pw = pose(k)
        oth = [pose((k[0], k[1], t)) for t in range(3) if t != k[2]]
        ds = [math.dist(pw[:2], o[:2]) for o in oth if o and pw]
        dis[k] = max(ds) if ds else 0.0
        o1 = pose((k[0], k[1], (k[2] + 1) % 3))
        dis1[k] = math.dist(pw[:2], o1[:2]) if (pw and o1) else 0.0
        kinds = [((x.get('constraint') or {}).get('kind') if isinstance(x, dict) else None) for x in (G[k].constraints or [])]
        ca[k] = any(kk and 'after' in kk for kk in kinds)
    lab = [not ok(G[k]) for k in gs]; gain = [(not ok(G[k])) and ok(Q[k]) for k in gs]
    for nm, sc in [('dis', dis), ('dis1', dis1), ('ca', {k: float(ca[k]) for k in gs})]:
        s = [sc[k] for k in gs]
        p(f'AUROC {nm}: fail {auroc(s, lab):.3f} gain {auroc(s, gain):.3f}')
    p('dis zero count', sum(1 for k in gs if dis[k] == 0), 'dis1 zero count', sum(1 for k in gs if dis1[k] == 0))
    n = len(gs)
    def curve(score, fracs, succ_fn, order_keys=None):
        order = sorted(gs if order_keys is None else order_keys, key=lambda k: -score[k])
        res = []
        for f in fracs:
            esc = set(order[:round(f * n)])
            succ = sum(succ_fn(k) if k in esc else ok(G[k]) for k in keys)
            lat = st.mean(prov(G[k]) + (prov(Q[k], {'people', 'constraints'}) if k in esc else 0) for k in gs)
            res.append((f, round(f * n), succ, round(lat, 1)))
        return res
    FR = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    sQ = lambda k: ok(Q[k])
    sPR = lambda k: ok(Q[k]) and bool(G[k].score.get('target_correct'))
    orc = {k: 1.0 if (not ok(G[k]) and ok(Q[k])) else 0.0 for k in gs}
    p('oracle curve (frac, n_esc, succ, judge-lat):', curve(orc, FR, sQ))
    p('dis curve:', curve(dis, FR, sQ))
    p('dis1 curve:', curve(dis1, FR, sQ))
    comb = {k: dis[k] + (5 if ca[k] else 0) for k in gs}
    p('ca+dis curve:', curve(comb, FR, sQ))
    random.seed(0)
    rsc = {k: random.random() for k in gs}
    p('random seed0 curve:', curve(rsc, FR, sQ))
    # expectation of random at 50%
    vals = []
    m = round(0.5 * n)
    for sd in range(2000):
        rr = random.Random(sd); sel = set(rr.sample(gs, m))
        vals.append(sum(ok(Q[k]) if k in sel else ok(G[k]) for k in keys))
    vals.sort()
    p(f'random @50% over 2000 seeds: mean {st.mean(vals):.1f}, 5% {vals[100]}, 95% {vals[1899]}; analytic = {baseG + len(qfix)*m/n - sum(1 for k in gs if ok(G[k]) and not ok(Q[k]))*m/n:.1f}')
    # tie robustness for dis/dis1 at 50%: shuffle order within ties
    for nm, sc in [('dis', dis), ('dis1', dis1)]:
        vs = []
        for sd in range(500):
            rr = random.Random(1000 + sd); sh = gs[:]; rr.shuffle(sh)
            order = sorted(sh, key=lambda k: -sc[k]); sel = set(order[:m])
            vs.append(sum(ok(Q[k]) if k in sel else ok(G[k]) for k in keys))
        p(f'{nm} @50% with random tie-breaking: min {min(vs)} mean {st.mean(vs):.1f} max {max(vs)}')
    PR = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]
    p('plan-reuse dis:', curve(dis, PR, sPR))
    p('plan-reuse ca+dis:', curve(comb, PR, sPR))
    p('plan-reuse dis1:', curve(dis1, PR, sPR))
    p('plan-reuse oracle:', curve({k: 1.0 if (not ok(G[k]) and sPR(k)) else 0.0 for k in gs}, PR, sPR))
    # latencies
    gG = [k for k in keys if is_gs(G[k])]; gQ = [k for k in keys if is_gs(Q[k])]
    for nm, D, g in [('G', G, gG), ('Q', Q, gQ)]:
        lat = [D[k].latency_s for k in g if D[k].latency_s is not None]
        p(f'{nm} own goal-step n={len(g)}: measures.latency_s mean {st.mean(lat):.1f} median {st.median(lat):.1f}; '
          f'prov latency_s sum mean {st.mean(prov(D[k]) for k in g):.1f}; latency_total_s sum mean {st.mean(prov_total(D[k]) for k in g):.1f}; '
          f'fresh-only latency_s sum mean {st.mean(fresh(D[k]) for k in g):.1f}')
        calls = [c for k in keys for c in (D[k].result.get('calls') or [])]
        p(f'   {nm} calls {len(calls)}, cached {sum(1 for c in calls if c.get("cached"))}; by stage cached:',
          sorted(Counter((c['stage'], bool(c.get('cached'))) for c in calls).items()))
    # replay
    if do_replay:
        rep = {}; fb = 0; err = Counter()
        for k in gs:
            g, q = G[k], Q[k]
            if not (q.people is not None and q.constraints is not None and g.target is not None):
                rep[k] = None; fb += 1; continue
            try:
                rv = H.resolve(g, target=g.target, people=q.people, constraints=[x['constraint'] for x in q.constraints])
                rep[k] = bool(rv.score.success)
            except Exception as e:
                rep[k] = None; fb += 1; err[type(e).__name__] += 1
        p('replay: replayed', sum(1 for v in rep.values() if v is not None), 'fallback', fb, dict(err))
        sRP = lambda k: rep[k] if rep[k] is not None else sPR(k)
        p('replay ceiling (all escalated):', sum(sRP(k) if k in set(gs) else ok(G[k]) for k in keys),
          '| fixes among G fails', sum(1 for k in fails if sRP(k)))
        p('replay dis:', curve(dis, PR, sRP)); p('replay dis1:', curve(dis1, PR, sRP))
    return dict(keys=keys, gs=gs)

if __name__ == '__main__':
    replay = '--replay' in sys.argv
    G19 = load('s19', 'glm53'); Q19 = load('s19', 'qwen38_low')
    analyse(G19, Q19, 'REPORTED: s19 glm53 con -> s19 qwen38_low con (real+staffed+plain)', do_replay=replay)
    G23 = load('s23', 'glm53')
    analyse(G23, load('s21', 'qwen38_low'), 'NEW-1: s23 glm53 con -> s21 qwen38_low con')
    analyse(G23, load('s23', 'sonnet55_low'), 'NEW-2: s23 glm53 con -> s23 sonnet55_low con')
    analyse(G23, load('s23', 'gemini38_low'), 'NEW-3: s23 glm53 con -> s23 gemini38_low con')
    with open('results/still-scenes/combining-models/cascade/verify/verify_out%s.txt' % ('_replay' if replay else ''), 'w') as f:
        f.write('\n'.join(OUT) + '\n')
