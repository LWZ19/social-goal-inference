import json, collections, random, statistics
OUT = 'results/2026-10-17-beyond-vlm/phase1/i01_calibration/verify/'
rows = [json.loads(l) for l in open(OUT + 'v1_rows.jsonl')]
MODELS = ['glm53', 'ornith15', 'qwen36_35', 'qwen38', 'qwen3vl']
lines = []
P = lambda *a: lines.append(' '.join(str(x) for x in a))

# config reading
cv = collections.Counter(tuple(r['cfg_v']) for r in rows)
co = collections.Counter(tuple(r['cfg_o']) for r in rows)
cn = collections.Counter(tuple(r['cfg_n']) for r in rows)
P('CONFIG convlm', dict(cv)); P('CONFIG conorc', dict(co)); P('CONFIG none', dict(cn))
P('same plan/locate/context raw text vo', sum(r['same_plan_ctx_vo'] for r in rows), 'vn', sum(r['same_plan_ctx_vn'] for r in rows),
  'all cached', sum(r['plan_ctx_all_cached'] for r in rows), 'same target', sum(r['same_target_vo'] for r in rows),
  'same plan', sum(r['same_plan_vo'] for r in rows), 'n calls dist', dict(collections.Counter(r['n_plan_ctx_calls'] for r in rows)))
P('steps v', sum(r['step_v'] for r in rows), 'steps o', sum(r['step_o'] for r in rows), 'both', sum(r['step_v'] and r['step_o'] for r in rows),
  'xor', sum(r['step_v'] != r['step_o'] for r in rows))
nonstep = [r for r in rows if not r['step_v']]
P('non-step identical verdict (now)', sum(r['now_v'] == r['now_o'] for r in nonstep), 'of', len(nonstep),
  'successes among non-step', sum(r['now_v'] for r in nonstep), sum(r['now_o'] for r in nonstep))
P('cons cached o', collections.Counter(str(r['cons_cached_o']) for r in rows), 'cons cached v', collections.Counter(str(r['cons_cached_v']) for r in rows))
P('people cached v', collections.Counter(str(r['people_cached_v']) for r in rows))

def verdict(r, arm):
    # arm: 'v' convlm, 'o' conorc, or repair name; resolved now where a step, rescored now otherwise
    if arm == 'o':
        return r['res_o'] if r['step_o'] else r['now_o']
    if arm == 'v':
        return r['res_v'] if r['step_v'] else r['now_v']
    return r['res_' + arm] if r['step_v'] else r['now_v']

P('\nreproduction: res_v==rec_v on steps', sum(r['res_v'] == r['rec_v'] for r in rows if r['step_v']),
  'moved_v>1e-6', [(r['model'], r['scene'], r['episode'], r['trial'], round(r['moved_v'], 4)) for r in rows if r['step_v'] and r['moved_v'] > 1e-6],
  'moved_o>1e-6', [(r['model'], r['scene'], r['episode'], r['trial'], round(r['moved_o'], 4)) for r in rows if r['step_o'] and r['moved_o'] > 1e-6])
P('rescore==recorded v', sum(r['now_v'] == r['rec_v'] for r in rows), 'o', sum(r['now_o'] == r['rec_o'] for r in rows))
P('res_o==rec_o on steps', sum(r['res_o'] == r['rec_o'] for r in rows if r['step_o']))
P('found share recompute eq', sum(abs((r['n_match'] / r['n_vis'] if r['n_vis'] else -1) - (r['rec_found_share'] if r['rec_found_share'] is not None else -1)) < 1e-9 and (r['n_det'] - r['n_match']) == r['rec_unmatched'] for r in rows if r['step_v']))
P('P people == conorc people', sum(r['P_people_eq_o'] for r in rows if r['step_v']))

P('\nTABLE per model (verdicts now; recorded totals in brackets)')
hdr = ['model', 'conv', 'conorc', 'real', 'P', 'repl', 'share', 'agreeG', 'agreeAll', 'baseG', 'baseAll', 'amb', 'pairs', 'R1', 'R2', 'R3', 'R4', 'Ps', 'P1m']
P(' | '.join(hdr))
tot = collections.Counter()
def stats(sub):
    s = {}
    s['n'] = len(sub)
    s['v'] = sum(verdict(r, 'v') for r in sub); s['o'] = sum(verdict(r, 'o') for r in sub)
    s['rv'] = sum(r['rec_v'] for r in sub); s['ro'] = sum(r['rec_o'] for r in sub)
    for k in ['P', 'R1', 'R2', 'R3', 'R4', 'Ps', 'P1m']:
        s[k] = sum(verdict(r, k) for r in sub)
    g = [r for r in sub if r['step_v']]
    s['g'] = len(g)
    s['agG'] = sum(verdict(r, 'P') == verdict(r, 'o') for r in g)
    s['agA'] = sum(verdict(r, 'P') == verdict(r, 'o') for r in sub)
    s['bG'] = sum(verdict(r, 'v') == verdict(r, 'o') for r in g)
    s['bA'] = sum(verdict(r, 'v') == verdict(r, 'o') for r in sub)
    s['amb'] = sum(r['amb'] for r in g); s['amb_all'] = sum(r['amb_all'] for r in g); s['pairs'] = sum(r['n_match'] for r in g)
    s['amb_recs'] = sum(r['amb'] > 0 for r in g)
    s['det'] = sum(r['n_det'] for r in g); s['vis'] = sum(r['n_vis'] for r in g)
    s['miss'] = s['vis'] - s['pairs']; s['spur'] = s['det'] - s['pairs']
    errs = [e for r in g for e in r['err_xy']]
    s['med_err'] = statistics.median(errs) if errs else None
    # flips on goal steps
    s['real_win'] = sum(verdict(r, 'o') and not verdict(r, 'v') for r in g)
    s['real_loss'] = sum(not verdict(r, 'o') and verdict(r, 'v') for r in g)
    s['P_win'] = sum(verdict(r, 'P') and not verdict(r, 'v') for r in g)
    s['P_loss'] = sum(not verdict(r, 'P') and verdict(r, 'v') for r in g)
    s['P_win_realwin'] = sum(verdict(r, 'P') and not verdict(r, 'v') and verdict(r, 'o') for r in g)
    s['P_loss_where_o_succ'] = sum(not verdict(r, 'P') and verdict(r, 'v') and verdict(r, 'o') for r in g)
    s['realloss_reproduced'] = sum(not verdict(r, 'o') and verdict(r, 'v') and not verdict(r, 'P') for r in g)
    return s

for model in MODELS + ['ALL']:
    sub = [r for r in rows if model == 'ALL' or r['model'] == model]
    s = stats(sub)
    real = s['o'] - s['v']; repl = s['P'] - s['v']
    share = repl / real if real else float('nan')
    P(f"{model} | {s['v']} [{s['rv']}] | {s['o']} [{s['ro']}] | {real:+d} [{s['ro']-s['rv']:+d}] | {s['P']} | {repl:+d} | {share:.2f} | "
      f"{s['agG']}/{s['g']}={s['agG']/s['g']:.3f} | {s['agA']}/{s['n']}={s['agA']/s['n']:.3f} | {s['bG']/s['g']:.3f} | {s['bA']/s['n']:.3f} | "
      f"{s['amb']}/{s['pairs']}={s['amb']/s['pairs']:.3f} (all-true {s['amb_all']}, recs {s['amb_recs']}) | "
      + ' | '.join(f"{k} {s[k]-s['v']:+d}" for k in ['R1', 'R2', 'R3', 'R4', 'Ps', 'P1m']))
    P(f"    perception det/vis/match/miss/spur {s['det']}/{s['vis']}/{s['pairs']}/{s['miss']}/{s['spur']} med_err {s['med_err']:.3f}; "
      f"flips realwin {s['real_win']} realloss {s['real_loss']} Pwin {s['P_win']} (realwins {s['P_win_realwin']}) Ploss {s['P_loss']} (o succ {s['P_loss_where_o_succ']}) realloss_reproduced {s['realloss_reproduced']}")

# by situation
P('\nBy situation (pooled goal steps)')
for sit in sorted(set(r['situation'] for r in rows)):
    g = [r for r in rows if r['situation'] == sit and r['step_v']]
    allsit = [r for r in rows if r['situation'] == sit]
    if not g:
        P(sit, 'no goal steps; records', len(allsit)); continue
    s = stats(allsit)
    P(sit, f"goal {s['g']} agreeP {s['agG']}/{s['g']}={s['agG']/s['g']:.3f} base {s['bG']}/{s['g']}={s['bG']/s['g']:.3f} realwin {s['real_win']} realloss {s['real_loss']}")

# disagreement sources
g = [r for r in rows if r['step_v']]
dis = [r for r in g if verdict(r, 'P') != verdict(r, 'o')]
perfect = sum(r['n_det'] == r['n_match'] == r['n_vis'] for r in dis)
P('\nP disagreements on goal steps', len(dis), 'with perfect matching', perfect)
for r in dis:
    P('  ', r['model'], r['scene'], r['episode'], r['trial'], 'v', verdict(r, 'v'), 'o', verdict(r, 'o'), 'P', verdict(r, 'P'),
      'det/vis/match', r['n_det'], r['n_vis'], r['n_match'])

# noise ceiling: conorc self-agreement across trials of one (scene, episode), both goal steps, right target, same matched id
P('\nNoise ceiling (conorc self-agreement across trial pairs)')
by = collections.defaultdict(list)
for r in rows:
    if r['step_o'] and r['matched_target_o'] is not None:
        by[(r['model'], r['scene'], r['episode'])].append(r)
tot_p = tot_a = 0
for model in MODELS:
    pa = ag = 0
    for k, lst in by.items():
        if k[0] != model: continue
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                a, b = lst[i], lst[j]
                if a['matched_target_o'] != b['matched_target_o']: continue
                pa += 1; ag += verdict(a, 'o') == verdict(b, 'o')
    P(model, f'{ag}/{pa}={ag/pa:.3f}'); tot_p += pa; tot_a += ag
P('ALL', f'{tot_a}/{tot_p}={tot_a/tot_p:.3f}')
# note: matched_target_o is recorded score field 'matched_target_id' (target correct implies id among targets)

# bootstrap pooled share and agreement, clusters (model, scene, trial)
random.seed(1)
cl = collections.defaultdict(list)
for r in rows:
    cl[(r['model'], r['scene'], r['trial'])].append(r)
def boot(model_filter, B=2000):
    keys_by_model = collections.defaultdict(list)
    for k in cl:
        if model_filter is None or k[0] == model_filter:
            keys_by_model[k[0]].append(k)
    shares, reals, agrs = [], [], []
    for _ in range(B):
        sub = []
        for m, ks in keys_by_model.items():
            for _k in range(len(ks)):
                sub.extend(cl[random.choice(ks)])
        v = sum(verdict(r, 'v') for r in sub); o = sum(verdict(r, 'o') for r in sub); p = sum(verdict(r, 'P') for r in sub)
        g = [r for r in sub if r['step_v']]
        reals.append(o - v)
        agrs.append(sum(verdict(r, 'P') == verdict(r, 'o') for r in g) / len(g))
        if o - v > 0:
            shares.append((p - v) / (o - v))
    q = lambda xs, a: sorted(xs)[int(a * (len(xs) - 1))]
    return (q(reals, .025), q(reals, .975)), (q(shares, .025), q(shares, .975)), (q(agrs, .025), q(agrs, .975)), sum(x <= 0 for x in reals)
for mf in [None] + MODELS:
    rr, sh, ag, nonpos = boot(mf)
    P('bootstrap', mf or 'ALL', 'real', rr, 'share', tuple(round(x, 2) for x in sh), 'agreeG', tuple(round(x, 3) for x in ag), 'real<=0 in', nonpos)

# people answer shared across episodes of (scene, trial)?
pt = collections.defaultdict(set)
for r in rows:
    pt[(r['model'], r['scene'], r['trial'])].add(json.dumps(r['people_text_v']))
P('\n(scene,trial) clusters', len(pt), 'with one distinct people answer', sum(len(s) == 1 for s in pt.values()))
open(OUT + 'v2_tables.txt', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
