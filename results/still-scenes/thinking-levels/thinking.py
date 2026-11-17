"""I22 (thinking set per stage): re-computation of the REPORT's judge-computed numbers, read only through
research_harness. No model call, no network, nothing tuned. Prints every table and saves it to thinking_tables.txt
beside this file.

Claims checked:
  T1  share of Qwen's people/constraints output tokens that is hidden thinking (report: about 65-87%)
  T2  s10 right-target goal steps, none vs low (report: 11 of 11 and 11 of 11); where the all-none loss comes from
  T3  pooled right-target goal steps, none vs low (report: 49 of 52 vs 51 of 52)
  T4  Qwen C 65 (none) vs 74 (low) of 84, at 8.4 vs 26.2 s per episode (brief 3.1)
  NEW per-call stage means behind the I22 "Why" line (people 16.0 s / 1,199 tokens; constraints 31.7 s / 2,351)
"""
import collections
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H  # noqa: E402

OUT = Path(__file__).resolve().parent / 'thinking_tables.txt'
_lines = []


def p(*args):
    line = ' '.join(str(a) for a in args)
    print(line)
    _lines.append(line)


def recs(name):
    return list(H.records(H.run_ref(name)))


def keyed(name):
    return {(r.scene_id, r.episode_id, r.trial): r for r in recs(name)}


def ok(r):
    return bool(r.score.get('success'))


def right_target_goal_step(r):
    """target_correct and a goal step of the constraint method's solver (Record.solver_step)."""
    return r.solver_step and bool(r.score.get('target_correct'))


def q(v, f):
    v = sorted(v)
    return v[min(len(v) - 1, int(f * len(v)))] if v else float('nan')


STAGES = ('plan', 'context', 'people', 'constraints')

# ---------------------------------------------------------------------------------------------------------------
p('=' * 110)
p('S0. Which usage field holds reasoning tokens? (every listed run, per stage of the list and model)')
p('=' * 110)
agg = collections.defaultdict(collections.Counter)
think_tags = 0
for ref in H.runs():
    for r in H.records(ref):
        for c in r.result.get('calls') or []:
            u = c.get('usage') or {}
            a = agg[(ref.stage, ref.model)]
            a['calls'] += 1
            a['has_field'] += 'reasoning_tokens' in u
            a['non_null'] += u.get('reasoning_tokens') is not None
            a['positive'] += (u.get('reasoning_tokens') or 0) > 0
            a['rt_sum'] += u.get('reasoning_tokens') or 0
            a['out_sum'] += u.get('output_tokens') or 0
            if '<think>' in (c.get('raw_text') or ''):
                think_tags += 1
p(f"{'stage':6s} {'model':14s} {'calls':>6s} {'field':>6s} {'nonnull':>7s} {'pos':>6s} {'rt/out':>7s}")
for (stg, m), a in sorted(agg.items()):
    p(f"{stg:6s} {m:14s} {a['calls']:6d} {a['has_field']:6d} {a['non_null']:7d} {a['positive']:6d} "
      f"{(a['rt_sum'] / a['out_sum'] if a['out_sum'] else 0):7.2f}")
p("calls whose raw_text holds a '<think>' tag:", think_tags)
p('Reading: usage.reasoning_tokens exists on every call but is null on every Qwen call (all Qwen runs); the gateway')
p('returns Qwen thinking as message.reasoning_content and folds its tokens into usage.output_tokens (ledger, 9 Oct).')
p('So the hidden share must be estimated as 1 - visible/output, with visible tokens estimated from raw_text.')

# ---------------------------------------------------------------------------------------------------------------
p('')
p('=' * 110)
p('T1. Hidden-thinking share of Qwen output tokens, by stage')
p('=' * 110)


def calls_of(names, dedupe=True):
    """(stage, output_tokens, chars, cached) per call; dedupe drops cache copies (same stage, text and tokens)."""
    seen, out = set(), []
    for n in names:
        for r in recs(n):
            for c in r.result.get('calls') or []:
                u = c.get('usage') or {}
                txt = c.get('raw_text') or ''
                key = (c['stage'], txt, u.get('output_tokens'))
                if dedupe and key in seen:
                    continue
                seen.add(key)
                out.append((c['stage'], u.get('output_tokens') or 0, len(txt), bool(c.get('cached'))))
    return out


CAL = ['s8_convlm_qwen38']          # effort none, 3 trials; its trial 0 is byte-identical to s10_qwen38_none_con
cal = calls_of(CAL)
cpt = {}
p('Calibration (effort none, assumed no hidden reasoning): chars per output token, unique calls of', CAL)
for s in STAGES:
    v = [(o, ch) for stg, o, ch, _ in cal if stg == s]
    cpt[s] = sum(ch for _, ch in v) / sum(o for o, _ in v)
    p(f'  {s:12s} n={len(v):3d} mean out {st.mean(o for o, _ in v):6.0f} mean chars {st.mean(ch for _, ch in v):6.0f} '
      f'chars/token {cpt[s]:.2f}')

SETS = {
    's10_qwen38_low_con (placeholder, 1 trial, a3c3fd0)': ['s10_qwen38_low_con'],
    's10_qwen38_medium_con (placeholder, 1 trial)': ['s10_qwen38_medium_con'],
    's19_ph_qwen38_low_con (placeholder, 3 trials)': ['s19_ph_qwen38_low_con'],
    's19_{real,staffed,plain}_qwen38_low_con (realistic, brief 3.6)': [f's19_{a}_qwen38_low_con' for a in ('real', 'staffed', 'plain')],
    's21_{real,staffed,plain}_qwen38_low_con (realistic, current wording)': [f's21_{a}_qwen38_low_con' for a in ('real', 'staffed', 'plain')],
}
T1_ROWS = {}
for label, names in SETS.items():
    p('')
    p(label)
    p(f"  {'stage':12s} {'n':>4s} {'mean out':>8s} {'chars':>6s} | {'E1 calib':>8s} {'p25':>5s} {'med':>5s} {'p75':>5s} | {'E2 c/4':>6s}")
    for s in STAGES:
        v = [(o, ch) for stg, o, ch, _ in calls_of(names) if stg == s and o > 0]
        if not v:
            continue
        so, sc = sum(o for o, _ in v), sum(ch for _, ch in v)
        e1 = 1 - (sc / cpt[s]) / so
        per = [max(0.0, 1 - (ch / cpt[s]) / o) for o, ch in v]
        e2 = 1 - (sc / 4) / so
        T1_ROWS[(label, s)] = (len(v), e1, e2, q(per, .25), q(per, .5), q(per, .75))
        p(f'  {s:12s} {len(v):4d} {so / len(v):8.0f} {sc / len(v):6.0f} | {e1:8.1%} {q(per, .25):5.0%} {q(per, .5):5.0%} '
          f'{q(per, .75):5.0%} | {e2:6.1%}')

p('')
p('E3 (s10 only): matched calls none vs low on the same scene/episode (people: per scene; constraints: per episode)')
p('    hidden share = 1 - sum(out at none)/sum(out at low), assuming the visible answer is as long at both levels')
for lo in ('s10_qwen38_low_con', 's10_qwen38_medium_con'):
    A, B = keyed('s10_qwen38_none_con'), keyed(lo)
    for s in STAGES:
        pair = []
        for k in A.keys() & B.keys():
            ca = [c for c in A[k].result['calls'] if c['stage'] == s]
            cb = [c for c in B[k].result['calls'] if c['stage'] == s]
            if ca and cb:
                pair.append((ca[0]['usage']['output_tokens'], cb[0]['usage']['output_tokens'],
                             len(ca[0].get('raw_text') or ''), len(cb[0].get('raw_text') or '')))
        if pair:
            so_n, so_l = sum(x[0] for x in pair), sum(x[1] for x in pair)
            p(f'  none vs {lo[11:]:13s} {s:12s} pairs {len(pair):2d} out none {so_n / len(pair):6.0f} low {so_l / len(pair):6.0f} '
              f'-> none/low {so_n / so_l:5.1%}, hidden {1 - so_n / so_l:5.1%}; chars none/low {sum(x[2] for x in pair) / sum(x[3] for x in pair):5.2f}')

# ---------------------------------------------------------------------------------------------------------------
p('')
p('=' * 110)
p('T2. s10, one trial of 28: right-target goal steps, none vs low (and medium)')
p('=' * 110)
NONE = keyed('s10_qwen38_none_con')
for lo in ('s10_qwen38_low_con', 's10_qwen38_medium_con'):
    LOW = keyed(lo)
    ka = {k for k, r in NONE.items() if right_target_goal_step(r)}
    kb = {k for k, r in LOW.items() if right_target_goal_step(r)}
    both = ka & kb
    p(f'none vs {lo}:')
    p(f'  unmatched: none {sum(ok(NONE[k]) for k in ka)} of {len(ka)}, {lo[11:]} {sum(ok(LOW[k]) for k in kb)} of {len(kb)}')
    p(f'  matched (right-target goal step in both runs): none {sum(ok(NONE[k]) for k in both)} of {len(both)}, '
      f'{lo[11:]} {sum(ok(LOW[k]) for k in both)} of {len(both)}')
    p(f'  records overall: none {sum(map(ok, NONE.values()))} of {len(NONE)}, {lo[11:]} {sum(map(ok, LOW.values()))} of {len(LOW)}')


def cls(r):
    if ok(r):
        return 'ok'
    if not r.score.get('target_correct'):
        return 'target wrong (plan/locate)'
    if r.result.get('source') != 'constraints':
        return (f"right target, {r.result.get('source')} path"
                + (" (context missed people)" if r.measures.get('context_missed') else " (no people needed; stop missed)"))
    return 'right target, C goal step failed'


LOW = keyed('s10_qwen38_low_con')
p('')
p('Failure class of every record (first that applies), none and low:')
for nm, D in (('none', NONE), ('low', LOW)):
    p(f'  {nm:5s}', dict(collections.Counter(cls(r) for r in D.values())))
p('Paired (none class -> low class), records whose outcome differs:')
pc = collections.Counter((cls(NONE[k]), cls(LOW[k])) for k in NONE if ok(NONE[k]) != ok(LOW[k]))
for (a, b), n in sorted(pc.items(), key=lambda x: -x[1]):
    p(f'  {n:2d}  none: {a:40s} low: {b}')
lost = [k for k in NONE if ok(LOW[k]) and not ok(NONE[k])]
gained = [k for k in NONE if ok(NONE[k]) and not ok(LOW[k])]
p(f'  lost at none {len(lost)}, gained at none {len(gained)}; net {len(lost) - len(gained)}')
p('  lost at none, detail (scene, episode, situation, none class, context_missed, plan_exact):')
for k in sorted(lost):
    r = NONE[k]
    p(f'    {k[0]:32s} {k[1]} {r.situation:20s} {cls(r):42s} ctx_missed={r.measures.get("context_missed")} '
      f'plan_exact={r.measures.get("plan_exact")}')
    sc = r.score
    p(f'      score: in_region={sc.get("in_region")} dist={sc.get("distance_to_region_m")} heading_ok={sc.get("heading_ok")} '
      f'reachable={sc.get("reachable")} gap={sc.get("min_body_gap_m")} status={r.result.get("status")} '
      f'low source={LOW[k].result.get("source")}')
p('  the one low failure:', [(k, LOW[k].situation, cls(LOW[k])) for k in LOW if not ok(LOW[k])])

# ---------------------------------------------------------------------------------------------------------------
p('')
p('=' * 110)
p('T3. Pooled right-target goal steps, none vs low (gate: none within 2 of low)')
p('=' * 110)


def matched(a, b, resolve=False):
    A, B = keyed(a), keyed(b)
    both = sorted({k for k, r in A.items() if right_target_goal_step(r)} & {k for k, r in B.items() if right_target_goal_step(r)})
    def s(D, k):
        return bool(H.resolve(D[k]).score.success) if resolve else ok(D[k])
    return both, {k: s(A, k) for k in both}, {k: s(B, k) for k in both}


PAIRS = [('s10_qwen38_none_con', 's10_qwen38_low_con'), ('s8_convlm_qwen38', 's19_ph_qwen38_low_con')]
pool = {}
for resolve in (False, True):
    tot_n = tot_a = tot_b = 0
    tag = 're-solved by the current solver' if resolve else 'as recorded'
    p(f'{tag}:')
    for a, b in PAIRS:
        both, sa, sb = matched(a, b, resolve)
        pool[(a, b, resolve)] = (both, sa, sb)
        tot_n += len(both); tot_a += sum(sa.values()); tot_b += sum(sb.values())
        by_t = collections.Counter(k[2] for k in both)
        p(f'  {a:22s} vs {b:24s} matched {len(both):2d} (by trial {dict(by_t)}): none {sum(sa.values())}, low {sum(sb.values())}')
    p(f'  POOLED: none {tot_a} of {tot_n}, low {tot_b} of {tot_n}  (difference {tot_b - tot_a})')

p('')
p('Variants (as recorded):')
both, sa, sb = pool[PAIRS[1] + (False,)]
for k in both:
    pass
for label, keep in (('s8 vs s19_ph alone', lambda k: True),
                    ('s8 vs s19_ph trials 1-2 only (drops the s8 trial 0 that duplicates s10 none)', lambda k: k[2] != 0)):
    kk = [k for k in both if keep(k)]
    p(f'  {label}: none {sum(sa[k] for k in kk)} of {len(kk)}, low {sum(sb[k] for k in kk)} of {len(kk)}')
b10, s10a, s10b = pool[PAIRS[0] + (False,)]
kk = [k for k in both if k[2] != 0]
p(f'  de-duplicated pool, s10 matched + s8/s19_ph trials 1-2: none {sum(s10a.values()) + sum(sa[k] for k in kk)} of '
  f'{len(b10) + len(kk)}, low {sum(s10b.values()) + sum(sb[k] for k in kk)} of {len(b10) + len(kk)}')
A10, A8 = keyed('s10_qwen38_none_con'), keyed('s8_convlm_qwen38')
same = sum(1 for (sc, ep, t), r in A10.items()
           if [c.get('raw_text') for c in r.result['calls']] == [c.get('raw_text') for c in A8[(sc, ep, 0)].result['calls']])
p(f'  identity: s10_qwen38_none_con records whose call texts equal s8_convlm_qwen38 trial 0: {same} of {len(A10)}')
p('  fails among matched (as recorded):')
for (a, b) in PAIRS:
    both, sa, sb = pool[(a, b, False)]
    p(f'    {a}: {[ (k, keyed(a)[k].situation) for k in both if not sa[k]]}')
    p(f'    {b}: {[ (k, keyed(b)[k].situation) for k in both if not sb[k]]}')
unmatched_none = sum(ok(r) for n in ('s10_qwen38_none_con', 's8_convlm_qwen38') for r in recs(n) if right_target_goal_step(r))
unmatched_none_n = sum(1 for n in ('s10_qwen38_none_con', 's8_convlm_qwen38') for r in recs(n) if right_target_goal_step(r))
unmatched_low = sum(ok(r) for n in ('s10_qwen38_low_con', 's19_ph_qwen38_low_con') for r in recs(n) if right_target_goal_step(r))
unmatched_low_n = sum(1 for n in ('s10_qwen38_low_con', 's19_ph_qwen38_low_con') for r in recs(n) if right_target_goal_step(r))
p(f'  unmatched pool (each run its own right-target goal steps): none {unmatched_none} of {unmatched_none_n}, '
  f'low {unmatched_low} of {unmatched_low_n}')

# ---------------------------------------------------------------------------------------------------------------
p('')
p('=' * 110)
p('T4. Qwen C success and seconds per episode, none vs low, placeholder (brief 3.1: 65 vs 74 of 84, 8.4 vs 26.2 s)')
p('=' * 110)
p('Runs of stage s12 (the 3-trial low/medium runs of the ledger) on the list:', H.runs(stage='s12'))
p(f"{'run':26s} {'n':>3s} {'succ':>4s} {'trials':>14s} | {'rec lat mean':>12s} {'median':>6s} | {'prov mean':>9s} | {'fresh prov':>10s} {'cached calls':>12s}")
for n in ('s8_convlm_qwen38', 's10_qwen38_none_con', 's10_qwen38_low_con', 's10_qwen38_medium_con', 's19_ph_qwen38_low_con',
          's10_qwen38_none_fsc', 's10_qwen38_low_fsc'):
    R = recs(n)
    lat = [r.latency_s for r in R]
    prov = [sum(t['latency_s'] for t in r.timings.values()) for r in R]
    fresh = [sum(t['fresh_latency_s'] for t in r.timings.values()) for r in R]
    cached = sum(t['cached'] for r in R for t in r.timings.values())
    calls = sum(t['calls'] for r in R for t in r.timings.values())
    p(f"{n:26s} {len(R):3d} {sum(map(ok, R)):4d} {str(dict(collections.Counter(r.trial for r in R))):>14s} | "
      f"{st.mean(lat):12.1f} {st.median(lat):6.1f} | {st.mean(prov):9.1f} | {st.mean(fresh):10.1f} {cached:5d}/{calls:<6d}")

# ---------------------------------------------------------------------------------------------------------------
p('')
p('=' * 110)
p('NEW. Per call by stage, Qwen C on the realistic sets (s19 real/staffed/plain qwen38_low_con; brief 3.6), all calls')
p('     (s19 calls are cache hits: latencies are those of the calls that first produced them)')
p('=' * 110)
names = [f's19_{a}_qwen38_low_con' for a in ('real', 'staffed', 'plain')]
by = collections.defaultdict(list)
succ = n_all = 0
for nm in names:
    for r in recs(nm):
        succ += ok(r); n_all += 1
        for c in r.result['calls']:
            by[c['stage']].append((c.get('latency_s') or 0, (c.get('usage') or {}).get('output_tokens') or 0))
p(f'  records {n_all}, success {succ}')
for s in STAGES:
    v = by[s]
    p(f'  {s:12s} calls {len(v):4d} latency mean {st.mean(a for a, _ in v):5.1f} (median {st.median(a for a, _ in v):5.1f}) '
      f'mean output tokens {st.mean(b for _, b in v):6.0f}')

OUT.write_text('\n'.join(_lines) + '\n', encoding='utf-8')
print(f'\nsaved {OUT}')
