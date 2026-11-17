"""P1 alternatives: all s19 GLM arms (incl. ph, dev seeds), per-record means, loose 'found' (any detection within 0.5 m)."""
import sys, math, statistics
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
from sgi.gt.derive import is_visible
sys.path.insert(0, 'results/still-scenes/perception/perception-rerun/verify')
from v_main import my_match
OUT = __file__.replace('.py', '_out.txt')
L = []
def P(*a):
    s = ' '.join(map(str, a)); print(s); L.append(s)

for stage, model in (('s19', 'glm53'), ('s20', 'dsv41')):
    by_arm = {}
    for r in H.runs(stage=stage, model=model, method='con'):
        th = H.recorded_config(r)[1].thresholds
        acc = by_arm.setdefault(r.arm, dict(miss=0, nf=0, nf_loose=0, fr=[], miss_f=0, nf_f=0, nfl_f=0, n=0))
        for rec in H.records(r):
            if not rec.solver_step or rec.situation != 'waiting_line':
                continue
            sc = H.scene(r, rec.scene_id)
            vis = [p for p in sc.truth.people if is_visible(p, sc.observation, th)]
            m = my_match(rec.people, vis, th.person_radius_m)
            ep = sc.truth.episodes[rec.episode_id]
            chosen = rec.score.get('matched_target_id') or ep.target_ids[0]
            g = sc.truth.group_for(rec.episode_id, chosen)
            members = [p for p in vis if g is not None and p.group == g]
            listed = {n for c in rec.constraints if c['constraint']['kind'] == 'come_after' for n in c['constraint']['people']}
            named = {m.get(n) for n in listed}
            missed = [p for p in members if p.id not in named]
            nf = [p for p in missed if p.id not in set(m.values())]
            nfl = [p for p in missed if not any(math.dist(d['xy'], (p.x, p.y)) <= 2 * th.person_radius_m for d in rec.people)]
            acc['n'] += 1
            acc['miss'] += len(missed); acc['nf'] += len(nf); acc['nf_loose'] += len(nfl)
            if missed: acc['fr'].append(len(nf) / len(missed))
            s = rec.score
            if s.get('target_correct') and not s.get('success') and (not s.get('in_region') or not s.get('reachable')):
                acc['miss_f'] += len(missed); acc['nf_f'] += len(nf); acc['nfl_f'] += len(nfl)
    P(f'== {stage} {model} con, line solver steps, by arm')
    tot = dict(miss=0, nf=0, nf_loose=0, fr=[], miss_f=0, nf_f=0, nfl_f=0, n=0)
    for arm, a in by_arm.items():
        P(f'  {arm:10s} n {a["n"]:3d}: nf/missed {a["nf"]}/{a["miss"]}  loose {a["nf_loose"]}/{a["miss"]}  failures {a["nf_f"]}/{a["miss_f"]} loose {a["nfl_f"]}/{a["miss_f"]}')
        for k in tot: tot[k] += a[k]
    def rr(a, b): return f'{a}/{b} = {a/b:.3f}' if b else 'n/a'
    P(f'  ALL arms: {rr(tot["nf"], tot["miss"])}; loose {rr(tot["nf_loose"], tot["miss"])}; failures {rr(tot["nf_f"], tot["miss_f"])}, loose {rr(tot["nfl_f"], tot["miss_f"])}; per-record mean of share {statistics.mean(tot["fr"]):.3f} (n {len(tot["fr"])})')
    rl = {k: v for k, v in by_arm.items() if k in ('real', 'staffed', 'plain')}
    a = sum(v['nf'] for v in rl.values()); b = sum(v['miss'] for v in rl.values())
    fr = [x for v in rl.values() for x in v['fr']]
    P(f'  realistic: {rr(a, b)}; per-record mean {statistics.mean(fr):.3f} (n {len(fr)}); loose {rr(sum(v["nf_loose"] for v in rl.values()), b)}')
    nr = {k: v for k, v in by_arm.items() if k not in ('real', 'staffed', 'plain')}
    a = sum(v['nf'] for v in nr.values()); b = sum(v['miss'] for v in nr.values())
    P(f'  placeholder+dev: {rr(a, b)}')
open(OUT, 'w').write('\n'.join(L) + '\n')
