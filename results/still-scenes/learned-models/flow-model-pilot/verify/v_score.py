"""Verifier step 2b (project venv): score my own sidecar samples and recompute every metric and C5's criteria.

  val   validation units (my vseed190-199): every one of the 64 raw samples scored by the PACKAGE scorer
        (sgi.evaluate.score, oracle target, robot freespace cached per scene exactly as research_harness._reach_once
        does); top pose = densest of the 64 raw positions (sigma 0.15 m); coverage = share of my extended region
        lattice cells with a raw sample within 0.2 m (+1e-9); B0/B1 baselines by sgi.solver.solve + package score.
  dev   development records: H.score_poses on top pose, hit@5 places, the 64 snapped samples and the raw top;
        P(success), Lift_room, hit@k; compared per record with the compute agent's rows; I04 V1/V3 comparison;
        C5 criteria; secondary bootstrap and breakdowns.
Writes verify/out/*.jsonl and verify/out/summary_<part>.json.
"""
from __future__ import annotations

import collections
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, 'results/2026-10-17-beyond-vlm/phase1/i01_repairs')
import v_common as V  # noqa: E402
import research_harness as H  # noqa: E402
import repairs  # noqa: E402

P = Path(V.PILOT)
VER = Path(V.VER)
OUT = VER / 'out'
MODELS = ('main', 'loso_waiting_line', 'loso_conversation_group', 'loso_audience')
SITS = ('no_one', 'waiting_line', 'conversation_group', 'audience')
CELLS = ('old_qwen', 'old_glm', 'old_deepseek', 'watch_qwen', 'watch_glm', 'watch_deepseek')


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


class CachedFS:
    def __init__(self, real, truth, robot, grid, joined):
        self.real, self.truth, self.robot, self.grid, self.joined = real, truth, robot, grid, joined

    def __getattr__(self, n):
        return getattr(self.real, n)

    def robot_grid(self, truth, at, th):
        if truth is self.truth and at is self.robot:
            return self.grid
        return self.real.robot_grid(truth, at, th)

    def reachable(self, on, s, g):
        if on is self.grid and tuple(s) == (self.robot.x, self.robot.y):
            c = on.index(*g)
            return c is not None and bool(self.joined[c])
        return self.real.reachable(on, s, g)


# ------------------------------------------------------------------------------------------------------- val
def part_val():
    import sgi.evaluate as EV
    from sgi.config import load_config
    from sgi.freespace import connected, robot_grid
    from sgi.geometry import make_poses, project
    from sgi.gt.derive import is_visible, oracle_target, prepare
    from sgi.scene import load_scene
    from sgi.solver import solve
    from sgi.stages.goal import standard_goal
    from sgi.types import Constraint, DetectedPerson, GoalResult
    cfg = load_config()
    th = cfg.thresholds
    samp = {}
    for m in MODELS:
        for ss in ('proto', 'alt'):
            z = np.load(VER / 'samples' / f'val_{m}_{ss}.npz')
            samp[(m, ss)] = ({str(k): i for i, k in enumerate(z['keys'])}, z['x'])
    rows = collections.defaultdict(list)
    base = []
    t0 = time.monotonic()
    real_fs = EV.freespace
    for k in range(190, 200):
        info = json.loads((VER / 'data' / f'vseed{k}.json').read_text())
        z = np.load(VER / 'data' / f'vseed{k}.npz')
        by_scene = collections.defaultdict(list)
        for ui, u in enumerate(info['units']):
            by_scene[(u['split'], u['scene'])].append((ui, u))
        for (split, sid), units in by_scene.items():
            folder = Path(V.SEEDS) / f'seed{k}' / split / sid
            H.guard_scene(folder)
            inputs, truth = load_scene(folder)
            truths = prepare(inputs, truth, cfg)
            V.check_dev_scene(sid, [et.gt.situation for et in truths.values()])
            obs = inputs.observation
            robot = obs.robot
            grid = robot_grid(truth, robot, th)
            joined = connected(grid, (robot.x, robot.y))
            # oracle people (OraclePeople's rule, written out) with the true ids
            people, num = [], {}
            for person in truth.people:
                if not is_visible(person, obs, th):
                    continue
                mk = project(obs.camera, (person.x, person.y, th.person_mark_height_m))
                if mk is None:
                    continue
                num[person.id] = len(people) + 1
                people.append(DetectedPerson(number=len(people) + 1, xy=(person.x, person.y), yaw=person.yaw,
                                             pixel=tuple(mk[:2]), label=None))
            EV.freespace = CachedFS(real_fs, truth, robot, grid, joined)
            try:
                for ui, u in units:
                    key = u['key']
                    gt = truths[u['episode']].gt
                    tgt = truth.targets[u['target']]
                    tg = oracle_target(obs, tgt, th)
                    assert tg is not None, key                    # verified: no lift failure in 190-199
                    ox, oy, yaw = u['frame'][:3]

                    def sc(pose):
                        pw, pr = make_poses(robot, pose[:2], pose[2])
                        return EV.score(GoalResult(status='ok', source='constraints', pose_world=pw, pose_robot=pr,
                                                   target=tg), obs, truth, gt, th)
                    # baselines
                    std = standard_goal(obs, tg, th)
                    grp = truth.group_for(u['episode'], u['target'])
                    mem = [] if grp is None else [num[p.id] for p in truth.group(grp) if p.id in num]
                    need = num.get(tgt.person_id) if tgt.person_id else None
                    ref = [Constraint.model_validate(c) for c in repairs.reference_constraints(gt.situation, mem, need)]
                    b = dict(key=key, sit=gt.situation)
                    for nm, cons in (('B0', []), ('B1', ref)):
                        try:
                            b[nm] = bool(sc(solve(obs, tg, people, cons, std, th).pose_world).success)
                        except ValueError as e:
                            b[nm] = False
                            b[nm + '_err'] = str(e)
                    base.append(b)
                    ext = z['ext_xy'][z['ext_off'][ui]:z['ext_off'][ui + 1]]
                    for (m, ss), (idx, X) in samp.items():
                        if key not in idx:
                            continue
                        x = X[idx[key]]
                        px, py = 3.2 * x[:, 0], 3.2 * x[:, 1]
                        c, s = math.cos(yaw), math.sin(yaw)
                        wx, wy = ox + c * px - s * py, oy + s * px + c * py
                        hd = np.arctan2(x[:, 3], x[:, 2]) + yaw
                        succ = np.array([bool(sc((wx[j], wy[j], hd[j])).success) for j in range(64)])
                        rp = np.stack([px, py], 1)
                        d2 = ((rp[:, None] - rp[None]) ** 2).sum(-1)
                        dn = np.exp(-d2 / (2 * 0.15 ** 2)).sum(1)
                        top = int(sorted(range(64), key=lambda j: (-dn[j], j))[0])
                        if len(ext):
                            dd = np.sqrt(((ext[:, None, :] - rp[None, :, :]) ** 2).sum(-1)).min(1)
                            cov = float(np.mean(dd <= 0.2 + 1e-9))
                        else:
                            cov = None
                        rows[(m, ss)].append(dict(key=key, sit=gt.situation, row=u['row'], scene=sid, seed=k,
                                                  labelled=u['n_cells'] > 0, top=bool(succ[top]),
                                                  share=float(succ.mean()), cov=cov))
            finally:
                EV.freespace = real_fs
        print(f'{time.monotonic() - t0:7.1f} s  seed{k} scored', flush=True)
    for (m, ss), rr in rows.items():
        with open(OUT / f'val_{m}_{ss}.jsonl', 'w') as f:
            f.write('\n'.join(json.dumps(r) for r in rr) + '\n')
    with open(OUT / 'val_baselines.jsonl', 'w') as f:
        f.write('\n'.join(json.dumps(r) for r in base) + '\n')
    # summary + comparison with the compute agent
    S = {}
    bmap = {b['key']: b for b in base}
    their_b = {r['key']: r for r in jl(P / 'rows_val_baselines.jsonl')}
    S['baselines_vs_theirs'] = dict(B0_agree=sum(bmap[k]['B0'] == their_b[k]['B0'] for k in bmap),
                                    B1_agree=sum(bmap[k]['B1'] == their_b[k]['B1'] for k in bmap), n=len(bmap))
    for (m, ss), rr in sorted(rows.items()):
        theirs = {r['key']: r for r in jl(P / f'rows_val_{m}.jsonl')}
        g = {}
        for s in SITS + ('POOLED',):
            q = [r for r in rr if s == 'POOLED' or r['sit'] == s]
            if not q:
                continue
            cv = [r['cov'] for r in q if r['cov'] is not None]
            g[s] = dict(n=len(q), n_scenes=len({(r['seed'], r['scene']) for r in q}), top=sum(r['top'] for r in q),
                        share=float(np.mean([r['share'] for r in q])), cov=float(np.mean(cv)) if cv else None,
                        B0=sum(bmap[r['key']]['B0'] for r in q), B1=sum(bmap[r['key']]['B1'] for r in q),
                        unlabelled=sum(not r['labelled'] for r in q),
                        top_on_unlabelled=sum(r['top'] for r in q if not r['labelled']))
        cmp = dict(n=len(rr), top_agree=sum(r['top'] == theirs[r['key']]['top_success'] for r in rr),
                   share_maxdiff=max(abs(r['share'] - theirs[r['key']]['share_success']) for r in rr),
                   cov_maxdiff=max(abs((r['cov'] or 0) - (theirs[r['key']]['coverage'] or 0)) for r in rr))
        S[f'{m}/{ss}'] = dict(groups=g, vs_theirs=cmp)
    (OUT / 'summary_val.json').write_text(json.dumps(S, indent=1))
    print(json.dumps(S, indent=1))


# ------------------------------------------------------------------------------------------------------- dev
def part_dev():
    from v_build_dev import record_list
    recs = {f"{r.ref.folder}/{r.scene_id}/{r.episode_id}/{r.trial}": (c, r) for c, r in record_list()}
    meta = {}
    for p in sorted((VER / 'data').glob('vdev_*.json')):
        for m in json.loads(p.read_text()):
            meta[m['key']] = m
    assert len(meta) == 549 and set(meta) == set(recs)
    i04 = {}
    for f in sorted((Path(V.ROOT) / 'phase1d' / 'i04').glob('rows_old_*.jsonl')) + \
            sorted((Path(V.ROOT) / 'phase1d' / 'i04').glob('rows_watch_*.jsonl')):
        for r in jl(f):
            i04[r['key']] = r
    assert set(meta) <= set(i04)
    rho_chk = max(abs(meta[k]['rho_b'] - i04[k]['rho_room']) for k in meta)
    nF_chk = sum(meta[k]['nF_b'] != i04[k]['n_cells'] for k in meta)
    t0 = time.monotonic()
    allrows = {}
    files = [('b', 'main'), ('a', 'main'), ('b', 'loso_waiting_line'), ('b', 'loso_conversation_group'),
             ('b', 'loso_audience')]
    for setting, model in files:
        for ss in ('proto', 'alt'):
            z = np.load(VER / 'samples' / f'dev_{setting}_{model}_{ss}.npz')
            rows = []
            for i, key in enumerate(z['keys']):
                key = str(key)
                cell, rec = recs[key]
                m = meta[key]
                top = tuple(map(float, z['top'][i]))
                npl = int(z['n_places'][i])
                places = [tuple(map(float, z['places'][i][j])) for j in range(npl)]
                snapped = [(float(z['sp'][i][j, 0]), float(z['sp'][i][j, 1]), float(z['head'][i][j])) for j in range(64)]
                raw_top = tuple(map(float, z['raw_top'][i]))
                uniq = list(dict.fromkeys([top] + places + snapped + [raw_top]))
                ok = {p: bool(s.success) for p, s in zip(uniq, H.score_poses(rec, uniq))}
                valid = z['valid'][i].astype(bool)
                fb = bool(z['fallback'][i])
                s64 = np.array([ok[p] for p in snapped])
                psucc = float(ok[top]) if fb else float(s64[valid].mean())
                hit = [int(any(ok[p] for p in places[:k + 1])) for k in range(5)]
                rows.append(dict(key=key, cell=cell, model=m['model'], sit=m['sit'], scene=m['scene'],
                                 layout=i04[key]['layout'], top1=ok[top], psucc=psucc, psucc64=float(s64.mean()),
                                 valid_share=float(valid.mean()), fallback=fb, hit=hit, raw_top1=ok[raw_top],
                                 rho=m[f'rho_{setting}'], C=m['C_rec'], P=m['P_top1']))
            allrows[(setting, model, ss)] = rows
            with open(OUT / f'dev_{setting}_{model}_{ss}.jsonl', 'w') as f:
                f.write('\n'.join(json.dumps(r) for r in rows) + '\n')
            print(f'{time.monotonic() - t0:7.1f} s  {setting} {model} {ss}: {len(rows)}', flush=True)
    S = dict(rho_b_vs_i04_maxdiff=rho_chk, nF_b_vs_i04_mismatch=nF_chk,
             C_rec_vs_rescore_mismatch=sum(m['C_rec'] != m['C_now'] for m in meta.values()))
    # compare with compute agent rows
    theirs = {}
    for st in ('b', 'a'):
        for c in CELLS:
            for r in jl(P / f'rows_{st}_{c}.jsonl'):
                theirs[(st, 'main', r['key'])] = r
    for c in CELLS:
        for mname in ('loso_waiting_line', 'loso_conversation_group', 'loso_audience'):
            fp = P / f'rows_b_{c}_{mname}.jsonl'
            if fp.exists():
                for r in jl(fp):
                    theirs[('b', mname, r['key'])] = r
    for (st, model, ss), rows in allrows.items():
        cm = collections.Counter()
        for r in rows:
            t = theirs[(st, model, r['key'])]
            cm['n'] += 1
            cm['top1'] += r['top1'] == t['top1']
            cm['fallback'] += r['fallback'] == t['fallback']
            cm['hit5'] += r['hit'][4] == t['hit'][4]
            cm['hit_all'] += r['hit'] == t['hit']
            cm['raw_top1'] += r['raw_top1'] == t['raw_top1']
            cm['psucc_eq'] += abs(r['psucc'] - t['psucc']) < 1e-12
            cm['rho_eq'] += abs(r['rho'] - t['rho_room']) < 1e-12
            cm['C_eq'] += r['C'] == t['C_top1']
            cm['P_eq'] += r['P'] == t['P_top1']
        S[f'vs_theirs/{st}/{model}/{ss}'] = dict(cm)

    def agg(rr):
        n = len(rr)
        t1 = sum(r['top1'] for r in rr)
        ps = sum(r['psucc'] for r in rr)
        rho = sum(r['rho'] for r in rr)
        return dict(n=n, top1=t1, top1_pct=100 * t1 / n, psucc_mean=ps / n, lift_room=ps / rho, lift_i14=t1 / rho,
                    hit=[sum(r['hit'][k] for r in rr) for k in range(5)], fallback=sum(r['fallback'] for r in rr),
                    raw_top1=sum(r['raw_top1'] for r in rr), C=sum(r['C'] for r in rr), P=sum(r['P'] for r in rr),
                    C_pct=100 * sum(r['C'] for r in rr) / n, P_pct=100 * sum(r['P'] for r in rr) / n,
                    valid_share=float(np.mean([r['valid_share'] for r in rr])),
                    psucc64=float(np.mean([r['psucc64'] for r in rr])))

    def i04agg(rr, v):
        Ssum = sum(i04[r['key']][v]['S'] for r in rr)
        rho = sum(i04[r['key']]['rho_room'] for r in rr)
        return dict(meanS=Ssum / len(rr), lift=Ssum / rho, hit5=sum(i04[r['key']][v]['hit_pkg'][4] for r in rr),
                    lex_hit5=sum(i04[r['key']]['lex_hit'][4] for r in rr))
    groups = [(c, lambda r, c=c: r['cell'] == c) for c in CELLS]
    groups += [(f'MODEL {m}', lambda r, m=m: r['cell'].split('_', 1)[1] == m) for m in ('qwen', 'glm', 'deepseek')]
    groups += [('POOLED', lambda r: True)]
    for ss in ('proto', 'alt'):
        G = {}
        for st in ('b', 'a'):
            rr_all = allrows[(st, 'main', ss)]
            for g, f in groups:
                G[f'{st}:{g}'] = agg([r for r in rr_all if f(r)])
            for s in SITS:
                q = [r for r in rr_all if r['sit'] == s]
                G[f'{st}:sit {s}'] = agg(q)
        rb = allrows[('b', 'main', ss)]
        G['i04:POOLED'] = dict(V1=i04agg(rb, 'V1'), V3=i04agg(rb, 'V3'))
        for m in ('qwen', 'glm', 'deepseek'):
            q = [r for r in rb if r['cell'].split('_', 1)[1] == m]
            G[f'i04:MODEL {m}'] = dict(V1=i04agg(q, 'V1'), V3=i04agg(q, 'V3'))
        # criteria
        per = {m: dict(flow_b=G[f'b:MODEL {m}']['top1_pct'], C=G[f'b:MODEL {m}']['C_pct'],
                       flow_a=G[f'a:MODEL {m}']['top1_pct'], P=G[f'a:MODEL {m}']['P_pct'], n=G[f'b:MODEL {m}']['n'])
               for m in ('qwen', 'glm', 'deepseek')}
        fl, v1, v3 = G['b:POOLED'], G['i04:POOLED']['V1'], G['i04:POOLED']['V3']
        crit = dict(per_model=per,
                    S1_count_one_sided=sum(p['flow_b'] >= p['C'] - 3 for p in per.values()),
                    S1_count_two_sided=sum(abs(p['flow_b'] - p['C']) <= 3 for p in per.values()),
                    S2=bool(fl['lift_room'] > v1['lift'] and fl['hit'][4] > v1['hit5']),
                    K1=all(p['flow_a'] < p['P'] - 10 for p in per.values()),
                    K2=all(p['flow_b'] < p['C'] - 10 for p in per.values()),
                    K3=bool(fl['lift_room'] <= v3['lift'] and fl['psucc_mean'] <= v3['meanS']),
                    pooled=dict(flow_lift=fl['lift_room'], V1_lift=v1['lift'], V3_lift=v3['lift'],
                                flow_hit5=fl['hit'][4], V1_hit5=v1['hit5'], V3_hit5=v3['hit5'],
                                flow_psucc=fl['psucc_mean'], V3_meanS=v3['meanS'], V1_meanS=v1['meanS']))
        crit['S1'] = crit['S1_count_one_sided'] >= 2
        G['criteria'] = crit
        # secondary: layout bootstrap of flow(b) - C (I04 / FROZEN 6 secondary recipe)
        real_l = sorted({r['layout'] for r in rb if r['cell'].startswith('old_')})
        watch_l = sorted({r['layout'] for r in rb if r['cell'].startswith('watch_')})
        rng = np.random.default_rng(20261018)
        dr = rng.integers(0, len(real_l), size=(2000, len(real_l)))
        dw = rng.integers(0, len(watch_l), size=(2000, len(watch_l)))
        boot = {}
        for g, f in groups[6:]:
            q = [r for r in rb if f(r)]
            sr, nr = np.zeros(len(real_l)), np.zeros(len(real_l))
            sw, nw = np.zeros(len(watch_l)), np.zeros(len(watch_l))
            for r in q:
                d = int(r['top1']) - int(r['C'])
                if r['cell'].startswith('old_'):
                    sr[real_l.index(r['layout'])] += d
                    nr[real_l.index(r['layout'])] += 1
                else:
                    sw[watch_l.index(r['layout'])] += d
                    nw[watch_l.index(r['layout'])] += 1
            bb = 100 * (sr[dr].sum(1) + sw[dw].sum(1)) / (nr[dr].sum(1) + nw[dw].sum(1))
            boot[g] = dict(est=100 * (sr.sum() + sw.sum()) / (nr.sum() + nw.sum()),
                           lo=float(np.percentile(bb, 2.5)), hi=float(np.percentile(bb, 97.5)))
        G['bootstrap'] = boot
        # LOSO on development records (secondary)
        lo = {}
        for mname in ('loso_waiting_line', 'loso_conversation_group', 'loso_audience'):
            q = allrows[('b', mname, ss)]
            keys = {r['key'] for r in q}
            mm = [r for r in rb if r['key'] in keys]
            lo[mname] = dict(n=len(q), loso_top1=sum(r['top1'] for r in q), main_top1=sum(r['top1'] for r in mm),
                             C=sum(r['C'] for r in mm), fallbacks=sum(r['fallback'] for r in q))
        G['loso_dev'] = lo
        S[ss] = G
    (OUT / 'summary_dev.json').write_text(json.dumps(S, indent=1))
    print(json.dumps({k: v for k, v in S.items() if not isinstance(v, dict) or k.startswith('vs_')}, indent=1))
    for ss in ('proto', 'alt'):
        print(ss, json.dumps(S[ss]['criteria'], indent=1))


if __name__ == '__main__':
    {'val': part_val, 'dev': part_dev}[sys.argv[1]]()
