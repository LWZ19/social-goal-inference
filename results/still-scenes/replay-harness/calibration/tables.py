"""Tables of I01 Step 2 (the s8 calibration) from rows2.jsonl (calibrate2.py) and rows.jsonl (calibrate.py, the
cross-check). Also the conorc self-agreement ceiling across trials (read through the harness). Writes tables.txt."""
from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H  # noqa: E402

HERE = Path(__file__).resolve().parent
MODELS = ['glm53', 'ornith15', 'qwen36_35', 'qwen38', 'qwen3vl']
ARMS = ['R1', 'R2', 'R3', 'R4', 'P', 'R4s', 'Ps', 'P1m', 'R3_1m', 'R0', 'R5', 'P5']
SITS = ['no_one', 'waiting_line', 'conversation_group']

rows = [json.loads(line) for line in (HERE / 'rows2.jsonl').open()]
mine = {(r['model'], r['scene'], r['episode'], r['trial']): r for r in map(json.loads, (HERE / 'rows.jsonl').open())}
out = []
p = out.append


def k(r):
    return (r['model'], r['scene'], r['episode'], r['trial'])


def stats(rs, arm):
    """real gain (orc_now - C_now), replayed gain (arm - C_now), agreement with orc_now (all, goal steps)."""
    c = sum(r['C_replay_ok'] if r['solver_step'] else r['C_rec'] for r in rs)
    o = sum(r['orc_now'] for r in rs)
    a = sum(r[arm] for r in rs)
    st = [r for r in rs if r['solver_step']]
    agree_all = sum(r[arm] == r['orc_now'] for r in rs)
    agree_st = sum(r[arm] == r['orc_now'] for r in st)
    return dict(n=len(rs), n_st=len(st), C=c, orc=o, rep=a, real=o - c, gain=a - c,
                agree_all=agree_all, agree_st=agree_st)


def share(g, real):
    return float('nan') if real == 0 else g / real


# ------------------------------------------------------------------------------------------------ checks
p('I01 Step 2: the s8 calibration (placeholder panel, 84 records per model and arm; 3 trials x 28 episodes).')
p('convlm = s8_convlm_<model> (people: vlm, constraints: vlm); conorc = s8_conorc_<model> (people: oracle,')
p('constraints: vlm); both goal: constraints. Read from config.yaml (H.recorded_config) and every record\'s config.')
p('')
p('CHECKS')
p(f"  keys paired: {len(rows)} convlm records, each with its conorc record on (scene, episode, trial)")
p(f"  same target (recorded GroundedTarget identical): {sum(r['same_target'] for r in rows)} of {len(rows)}"
  " (plan, locate and context raw answers identical in 420 of 420: check_cache.txt)")
p(f"  goal step in both arms: {sum(r['solver_step'] and r['orc_solver_step'] for r in rows)}; in convlm only "
  f"{sum(r['solver_step'] and not r['orc_solver_step'] for r in rows)}; in conorc only "
  f"{sum(r['orc_solver_step'] and not r['solver_step'] for r in rows)}")
nong = [r for r in rows if not r['solver_step']]
p(f"  non-goal-step records with the same verdict in both arms: {sum(r['C_rec'] == r['orc_rec'] for r in nong)} of {len(nong)}")
p(f"  rescore = recorded: convlm {sum(r['C_rescore'] == r['C_rec'] for r in rows)} of {len(rows)}, "
  f"conorc {sum(r['orc_rescore'] == r['orc_rec'] for r in rows)} of {len(rows)}")
st_all = [r for r in rows if r['solver_step']]
p(f"  convlm goal steps reproduced by the solver now: {sum(r['reproduced'] for r in st_all)} of {len(st_all)}; "
  f"moved: " + ', '.join(f"{r['model']} {r['scene']} {r['episode']} t{r['trial']} {r['moved_m']:.3f} m "
                          f"(rec {r['C_rec']}, now {r['C_replay_ok']})" for r in st_all if not r['reproduced']))
om = [r for r in rows if r['orc_solver_step'] and (r['orc_moved_m'] or 0) > 1e-6]
p(f"  conorc goal steps moved now: {len(om)}: " + ', '.join(
    f"{r['model']} {r['scene']} {r['episode']} t{r['trial']} {r['orc_moved_m']:.3f} m (rec {r['orc_rec']}, now {r['orc_now']})"
    for r in om))
p(f"  P's people set = conorc's people set (true positions of visible people with a projectable mark): "
  f"{sum(r['P_people_equal_orc'] for r in st_all)} of {len(st_all)} goal steps")
p(f"  recorded measures vs recomputed matching: found share {sum(abs((r['n_matched'] / r['n_vis'] if r['n_vis'] else -1) - (mine[k(r)]['found_share_recorded'] if mine[k(r)]['found_share_recorded'] is not None else -1)) < 1e-9 for r in st_all)} of {len(st_all)}, "
  f"unmatched {sum(r['n_spurious'] == mine[k(r)]['unmatched_recorded'] for r in st_all)} of {len(st_all)}")
cross = collections.Counter()
for r in st_all:
    m = mine[k(r)]
    for a, b in (('R1', 'R1'), ('R2', 'R2'), ('R3', 'R3'), ('R4s', 'R4'), ('Ps', 'P')):
        cross[a] += r[a] == m[b]
p("  independent implementation (calibrate.py, written before i01_repairs existed) vs shared code: "
  + ', '.join(f"{a} {cross[a]}/{len(st_all)}" for a in ('R1', 'R2', 'R3', 'R4s', 'Ps')))
p(f"  convlm C now vs recorded over all records: {sum((r['C_replay_ok'] if r['solver_step'] else r['C_rec']) for r in rows)} vs "
  f"{sum(r['C_rec'] for r in rows)}; conorc now vs recorded {sum(r['orc_now'] for r in rows)} vs {sum(r['orc_rec'] for r in rows)}")
p('')

# ------------------------------------------------------------------------------------------------ table 1
p('TABLE 1. Real and replayed gain per model (solver of the code now for every arm; recorded in brackets).')
p('  real gain = conorc - convlm successes (n 84). replayed gain = repair - convlm successes. share = replayed / real.')
p('  agreement = records where the repaired convlm verdict equals the conorc verdict; "all" over 84 records (the')
p('  non-goal-step records are identical by construction), "goal" over convlm goal steps only.')
p(f"{'model':10s}{'n':>4s}{'goal':>5s}{'convlm':>7s}{'conorc':>7s}{'real':>6s}   {'arm':5s}{'rep':>5s}{'gain':>6s}{'share':>7s}"
  f"{'agree all':>14s}{'agree goal':>14s}")
for model in MODELS + ['pooled']:
    rs = [r for r in rows if model == 'pooled' or r['model'] == model]
    rec_c, rec_o = sum(r['C_rec'] for r in rs), sum(r['orc_rec'] for r in rs)
    for i, arm in enumerate(['none'] + ARMS[:5] + ['Ps', 'P1m']):
        if arm == 'none':
            d = stats([dict(r, none=(r['C_replay_ok'] if r['solver_step'] else r['C_rec'])) for r in rs], 'none')
        else:
            d = stats(rs, arm)
        head = (f"{model:10s}{d['n']:4d}{d['n_st']:5d}{d['C']:4d}({rec_c:2d}){d['orc']:4d}({rec_o:2d}){d['real']:+4d}"
                f"({rec_o - rec_c:+d})" if i == 0 else ' ' * 47)
        p(f"{head:47s}{arm:5s}{d['rep']:5d}{d['gain']:+6d}{share(d['gain'], d['real']):7.2f}"
          f"{d['agree_all']:5d}/{d['n']:<3d}{d['agree_all'] / d['n']:5.1%} {d['agree_st']:4d}/{d['n_st']:<3d}{d['agree_st'] / d['n_st']:5.1%}")
    p('')
p('  arm none = convlm as recorded, solved again (the do-nothing baseline for agreement).')
p('  R4 and P strip dropped numbers from the constraints (i01_repairs primary); R4s/Ps leave constraints as written;')
p('  P1m = P with matching widened to 1.0 m.')
p('')

# ------------------------------------------------------------------------------------------------ table 2
p('TABLE 2. Paired flips on goal steps: real wins W (conorc success, convlm fail), real losses L (the reverse);')
p('P wins (P success, convlm fail) and P losses; overlap = records where P moves the verdict the same way as conorc.')
p(f"{'model':10s}{'goal':>5s}{'W':>4s}{'L':>4s}{'W-L':>5s}  {'Pw':>4s}{'Pl':>4s}{'Pw-Pl':>6s}  {'Pw in W':>8s}{'Pl in L':>8s}"
  f"{'W kept fail by P':>18s}{'P agrees W':>12s}")
for model in MODELS + ['pooled']:
    st = [r for r in rows if r['solver_step'] and (model == 'pooled' or r['model'] == model)]
    W = [r for r in st if r['orc_now'] and not r['C_replay_ok']]
    L = [r for r in st if not r['orc_now'] and r['C_replay_ok']]
    Pw = [r for r in st if r['P'] and not r['C_replay_ok']]
    Pl = [r for r in st if not r['P'] and r['C_replay_ok']]
    p(f"{model:10s}{len(st):5d}{len(W):4d}{len(L):4d}{len(W) - len(L):+5d}  {len(Pw):4d}{len(Pl):4d}{len(Pw) - len(Pl):+6d}  "
      f"{sum(r['orc_now'] for r in Pw):8d}{sum(not r['orc_now'] for r in Pl):8d}{sum(not r['P'] for r in W):18d}"
      f"{sum(r['P'] for r in W):6d}/{len(W)}")
p('')

# ------------------------------------------------------------------------------------------------ table 3
p('TABLE 3. Agreement of P with conorc on goal steps, by situation (pooled over the five models), and the')
p('do-nothing baseline (convlm as is).')
for s in SITS:
    st = [r for r in rows if r['solver_step'] and r['situation'] == s]
    if not st:
        p(f"  {s}: no goal steps")
        continue
    W = sum(r['orc_now'] and not r['C_replay_ok'] for r in st)
    L = sum(not r['orc_now'] and r['C_replay_ok'] for r in st)
    p(f"  {s:20s} n {len(st):3d}  convlm {sum(r['C_replay_ok'] for r in st):3d} conorc {sum(r['orc_now'] for r in st):3d} "
      f"P {sum(r['P'] for r in st):3d}  W {W} L {L}  agree P {sum(r['P'] == r['orc_now'] for r in st)} "
      f"({sum(r['P'] == r['orc_now'] for r in st) / len(st):.1%}), none {sum(r['C_replay_ok'] == r['orc_now'] for r in st)} "
      f"({sum(r['C_replay_ok'] == r['orc_now'] for r in st) / len(st):.1%})")
p('')

# ------------------------------------------------------------------------------------------------ table 4
p('TABLE 4. Ambiguous matches (one-to-one matching within 0.5 m, against the visible true people).')
p('  A matched pair is ambiguous when its detection is within 0.5 m of two or more visible true people, or its true')
p('  person is within 0.5 m of two or more detections. "all people" counts every true person (visible or not).')
p(f"{'model':10s}{'goal':>5s}{'det':>5s}{'vis':>5s}{'match':>6s}{'miss':>5s}{'spur':>5s}{'amb':>5s}{'share':>7s}"
  f"{'amb(all)':>9s}{'share':>7s}{'rec w/ amb':>11s}{'det>=2 true':>12s}{'true>=2 det':>12s}{'med err m':>10s}")
for model in MODELS + ['pooled']:
    st = [r for r in rows if r['solver_step'] and (model == 'pooled' or r['model'] == model)]
    pairs = [q for r in st for q in r['pairs']]
    amb = sum(q['ambiguous'] for q in pairs)
    amb_all = sum(q['ambiguous_all'] for q in pairs)
    recs = sum(any(q['ambiguous'] for q in r['pairs']) for r in st)
    gaps = sorted(q['gap'] for q in pairs)
    med = gaps[len(gaps) // 2] if gaps else float('nan')
    p(f"{model:10s}{len(st):5d}{sum(r['n_det'] for r in st):5d}{sum(r['n_vis'] for r in st):5d}{len(pairs):6d}"
      f"{sum(r['n_missed'] for r in st):5d}{sum(r['n_spurious'] for r in st):5d}{amb:5d}{amb / max(len(pairs), 1):7.1%}"
      f"{amb_all:9d}{amb_all / max(len(pairs), 1):7.1%}{recs:6d}/{len(st):<4d}"
      f"{sum(q['det_near_true'] >= 2 for q in pairs):12d}{sum(q['true_near_det'] >= 2 for q in pairs):12d}{med:10.3f}")
p('  Records with any ambiguous match, of goal steps with at least one match: ' + ', '.join(
    f"{m} {sum(any(q['ambiguous'] for q in r['pairs']) for r in rows if r['solver_step'] and r['model'] == m)}/"
    f"{sum(bool(r['pairs']) for r in rows if r['solver_step'] and r['model'] == m)}" for m in MODELS))
p('')

# ------------------------------------------------------------------------------------------------ table 5: ceiling
p('TABLE 5. Noise ceiling: conorc agreeing with itself across trials. conorc has the same (true) people in every')
p('trial and a fresh constraints draw per trial; pairs of trials of one (scene, episode) that are both goal steps')
p('with the same recorded target, verdict = solved again now (rows2 orc_now). Also convlm vs convlm, same pairing')
p('(different people draws and constraints draws). 5a: identical recorded GroundedTarget (rare: the target point is')
p('redrawn per trial). 5b: looser, both trials have the right target with the same matched target id (the target')
p('point may differ by a few pixels, so the solver input differs slightly too).')
by = {k(r): r for r in rows}


def table5_pairs(orc_target):
    for model in MODELS + ['pooled']:
        c = collections.Counter()
        for mm in (MODELS if model == 'pooled' else [model]):
            for sc, ep in sorted({(r['scene'], r['episode']) for r in rows if r['model'] == mm}):
                for a, b in ((0, 1), (0, 2), (1, 2)):
                    ka, kb = (mm, sc, ep, a), (mm, sc, ep, b)
                    if ka not in orc_target or kb not in orc_target:
                        continue
                    ta, tb = orc_target[ka], orc_target[kb]
                    if not (ta[1] and tb[1] and ta[0] == tb[0]):
                        continue
                    ra, rb = by[ka], by[kb]
                    c['pairs'] += 1
                    c['orc'] += ra['orc_now'] == rb['orc_now']
                    c['lvm'] += ra['C_replay_ok'] == rb['C_replay_ok']
                    c['P'] += ra['P'] == rb['P']
                    c['P_vs_orc_cross'] += ra['P'] == rb['orc_now']
        n = max(c['pairs'], 1)
        p(f"    {model:10s} pairs {c['pairs']:3d}: conorc vs conorc {c['orc']} ({c['orc'] / n:.1%}); convlm vs convlm "
          f"{c['lvm']} ({c['lvm'] / n:.1%}); P vs P {c['P']} ({c['P'] / n:.1%}); P(trial a) vs conorc(trial b) "
          f"{c['P_vs_orc_cross']} ({c['P_vs_orc_cross'] / n:.1%})")


for label, mode in (('5a', 'exact'), ('5b', 'id')):
    p(f'  {label}')
    orc_target = {}
    for model in MODELS:
        for rec in H.records(H.runs(stage='s8', model=model, method='conorc')[0]):
            tkey = (json.dumps(rec.target, sort_keys=True) if mode == 'exact' else
                    (rec.score.get('matched_target_id') if rec.score.get('target_correct') else None))
            ok = rec.solver_step and tkey is not None
            orc_target[(model, rec.scene_id, rec.episode_id, rec.trial)] = (tkey, ok)
    table5_pairs(orc_target)
p('')

# ------------------------------------------------------------------------------------------------ bootstrap
p('TABLE 6. Bootstrap (2,000 resamples of (scene, trial) clusters within each model; a people answer is shared by')
p('the two episodes of a scene and trial). 95% percentile intervals.')
rng = random.Random(6244)
clusters = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    clusters[r['model']][(r['scene'], r['trial'])].append(r)


def summary(rs):
    d = stats(rs, 'P')
    return d['real'], d['gain'], d['agree_st'] / max(d['n_st'], 1), d['agree_all'] / d['n']


for model in MODELS + ['pooled']:
    reps = []
    for _ in range(2000):
        rs = []
        for mm in (MODELS if model == 'pooled' else [model]):
            keys = list(clusters[mm])
            for _ in keys:
                rs.extend(clusters[mm][rng.choice(keys)])
        reps.append(summary(rs))
    shares = sorted(g / r for r, g, _, _ in reps if r > 0)
    nonpos = sum(r <= 0 for r, _, _, _ in reps)
    ag = sorted(a for _, _, a, _ in reps)
    aa = sorted(a for _, _, _, a in reps)
    rg = sorted(r for r, _, _, _ in reps)

    def ci(v):
        return f"[{v[int(0.025 * len(v))]:.2f}, {v[int(0.975 * len(v)) - 1]:.2f}]" if v else '[n/a]'
    p(f"  {model:10s} real gain {ci(rg)}; share recovered {ci(shares)} (resamples with real gain <= 0: {nonpos}); "
      f"agreement goal {ci(ag)}, all {ci(aa)}")
p('')

# ------------------------------------------------------------------------------------------------ criteria
p('CRITERIA (pre-registered in REPORT I01; repair = P; agreement on goal steps and on all records; ambiguity per')
p('matched pair, visible true people).')
for model in MODELS + ['pooled']:
    rs = [r for r in rows if model == 'pooled' or r['model'] == model]
    d = stats(rs, 'P')
    st = [r for r in rs if r['solver_step']]
    pairs = [q for r in st for q in r['pairs']]
    amb = sum(q['ambiguous'] for q in pairs) / max(len(pairs), 1)
    sh = share(d['gain'], d['real'])
    ag_st, ag_all = d['agree_st'] / d['n_st'], d['agree_all'] / d['n']
    succ_st = sh >= 0.5 and ag_st >= 0.8 and amb <= 0.10
    succ_all = sh >= 0.5 and ag_all >= 0.8 and amb <= 0.10
    kill_st = sh < 1 / 3 or ag_st < 0.65 or amb > 0.20
    kill_all = sh < 1 / 3 or ag_all < 0.65 or amb > 0.20
    p(f"  {model:10s} real {d['real']:+d}, replayed {d['gain']:+d}, share {sh:.2f}, agreement goal {ag_st:.1%} / all {ag_all:.1%}, "
      f"ambiguous {amb:.1%} -> success: goal {succ_st}, all {succ_all}; kill: goal {kill_st}, all {kill_all}")

(HERE / 'tables.txt').write_text('\n'.join(out) + '\n')
print('\n'.join(out))
