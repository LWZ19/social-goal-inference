"""I21 latency replay and latency facts (claims L1-L8 and the stage-23 per-stage medians), read only through
research_harness. No model call, no network, nothing written outside this folder.

Ports of the judge scripts (RD/scripts/cp.py, cp2.py, lat.py), with the run lists taken from the harness:
  cp.py  per record with a people AND a constraints call; per-stage sums of call latency_s;
         serial = plan+locate+context+people+constraints; concurrent = max(plan+locate+context, people)+constraints;
         per-scene-shared: the first record of each (scene_id, trial) in file order pays people, later ones
         pay pre+constraints in both columns. `seen` is reset per run folder.
  cp2.py per record with a constraints call (people 0 if absent); no locate;
         serial=p+x+pe+k, A1=max(p+x,pe)+k, A2=max(p,pe)+max(x,k), A2_fastpeople=p+max(x,k).
  lat.py per stage, median of call latency_s (here split by cached flag).
Output: printed and saved to latency_out.txt beside this file.
"""
import collections
import io
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H  # noqa: E402

OUT = Path(__file__).resolve().parent / 'latency_out.txt'
buf = io.StringIO()


def p(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    buf.write(s + '\n')


REAL = ('real', 'staffed', 'plain')
WL = H.whitelist()
_REC = {}


def recs(ref):
    if ref.folder not in _REC:
        _REC[ref.folder] = list(H.records(ref, wl=WL))
    return _REC[ref.folder]


def mean(v):
    return st.mean(v) if v else float('nan')


def med(v):
    return st.median(v) if v else float('nan')


def g(L, s, field='latency_s'):
    return L[s][field] if s in L else 0.0


# ---------------------------------------------------------------- cp.py port
def replay(ref, field='latency_s'):
    rows, seen = [], set()
    for r in recs(ref):
        L = r.timings
        if 'constraints' not in L or 'people' not in L:
            continue
        pre = g(L, 'plan', field) + g(L, 'locate', field) + g(L, 'context', field)
        pe, k = g(L, 'people', field), g(L, 'constraints', field)
        s, c = pre + pe + k, max(pre, pe) + k
        key = (r.scene_id, r.trial)
        if key in seen:
            ss = cs = pre + k
        else:
            seen.add(key)
            ss, cs = s, c
        rows.append(dict(run=ref.name, scene=r.scene_id, ep=r.episode_id, trial=r.trial, pre=pre, people=pe,
                         constraints=k, serial=s, cp=c, ser_share=ss, cp_share=cs, rec_latency=r.latency_s,
                         people_cached=L['people']['cached']))
    return rows


def fmt_replay(label, rows):
    if not rows:
        return f'{label:44s} n=0'
    S, C = mean([x['serial'] for x in rows]), mean([x['cp'] for x in rows])
    SS, CS = mean([x['ser_share'] for x in rows]), mean([x['cp_share'] for x in rows])
    return (f'{label:44s} n={len(rows):4d} serial {S:5.1f} cp {C:5.1f} ({100 * (1 - C / S):4.1f}%) | '
            f'per-scene-shared serial {SS:5.1f} cp {CS:5.1f} ({100 * (1 - CS / SS):4.1f}%)')


def replay_group(title, refs, field='latency_s'):
    p(f'\n## {title}  [field={field}]')
    allrows, per_run = [], []
    for ref in refs:
        rows = replay(ref, field)
        p(fmt_replay(ref.name, rows))
        allrows += rows
        if rows:
            per_run.append((mean([x['serial'] for x in rows]), mean([x['cp'] for x in rows]),
                            mean([x['ser_share'] for x in rows]), mean([x['cp_share'] for x in rows])))
    p(fmt_replay('POOLED (record-weighted)', allrows))
    if per_run:
        m = [mean([t[i] for t in per_run]) for i in range(4)]
        p(f'{"MEAN OF PER-RUN MEANS":44s}        serial {m[0]:5.1f} cp {m[1]:5.1f} ({100 * (1 - m[1] / m[0]):4.1f}%) | '
          f'per-scene-shared serial {m[2]:5.1f} cp {m[3]:5.1f} ({100 * (1 - m[3] / m[2]):4.1f}%)')
    if allrows:
        p(f'{"POOLED medians":44s}        serial {med([x["serial"] for x in allrows]):5.1f} '
          f'cp {med([x["cp"] for x in allrows]):5.1f}; mean pre {mean([x["pre"] for x in allrows]):.1f} '
          f'people {mean([x["people"] for x in allrows]):.1f} constraints {mean([x["constraints"] for x in allrows]):.1f}; '
          f'people slower than pre in {sum(x["people"] > x["pre"] for x in allrows)}/{len(allrows)}; '
          f'distinct (scene,trial) per run summed = {sum(len({(x["scene"], x["trial"]) for x in allrows if x["run"] == ref.name}) for ref in refs)}')
    return allrows


p('# I21 latency replay and latency facts (rerun via research_harness)')

# ================================================================ L1, L2, L3
q19 = [r for r in H.runs(stage='s19', model='qwen38_low', method='con', wl=WL) if r.arm in REAL]
g19 = [r for r in H.runs(stage='s19', model='glm53', method='con', wl=WL) if r.arm in REAL]
L1rows = replay_group('L1/L3 Qwen s19 realistic C (real, staffed, plain)', q19)
L2rows = replay_group('L2 GLM s19 realistic C (real, staffed, plain)', g19)
replay_group('L1 variant: Qwen s19 realistic C with latency_total_s', q19, 'latency_total_s')
replay_group('Other subsets: Qwen s19 ph + dev seeds',
             [r for r in H.runs(stage='s19', model='qwen38_low', method='con', wl=WL) if r.arm not in REAL])
replay_group('Other subsets: GLM s19 ph + dev seeds',
             [r for r in H.runs(stage='s19', model='glm53', method='con', wl=WL) if r.arm not in REAL])
replay_group('Other subsets: Qwen s19 all 7 arms', H.runs(stage='s19', model='qwen38_low', method='con', wl=WL))
replay_group('Other subsets: GLM s19 all 7 arms', H.runs(stage='s19', model='glm53', method='con', wl=WL))
q21rows = replay_group('Qwen s21 realistic C (new wording; people cached from s19, constraints fresh)',
                       [r for r in H.runs(stage='s21', method='con', wl=WL) if r.arm in REAL])
replay_group('DeepSeek s20 realistic C (stage-20 timings not to be quoted: rate-limited)',
             [r for r in H.runs(stage='s20', method='con', wl=WL) if r.arm in REAL])
s23rows = {}
for m in ('glm53', 'dsv41', 'haiku55', 'sonnet55_low', 'gemini38_low'):
    s23rows[m] = replay_group(f'NEW s23 {m} realistic C',
                              [r for r in H.runs(stage='s23', model=m, method='con', wl=WL) if r.arm in REAL])

# ================================================================ L4 (cp2.py port)
p('\n## L4 cp2.py port: per record with a constraints call; p,x,pe,k = plan, context, people, constraints latency_s')
p('   columns: mean/median/p90 (p90 = sorted[int(0.9 n)]); A1-A2 = mean saving of A2 over A1')


def cp2(ref):
    A = collections.defaultdict(list)
    for r in recs(ref):
        L = r.timings
        if 'constraints' not in L:
            continue
        pp, x, pe, k = g(L, 'plan'), g(L, 'context'), g(L, 'people'), g(L, 'constraints')
        A['serial'].append(pp + x + pe + k)
        A['A1'].append(max(pp + x, pe) + k)
        A['A2'].append(max(pp, pe) + max(x, k))
        A['A2_fastpeople'].append(pp + max(x, k))
        A['k'].append(k)
    return A


def q90(v):
    return sorted(v)[int(0.9 * len(v))]


savings = []
pooledA = collections.defaultdict(lambda: collections.defaultdict(list))
for ref in [r for r in H.runs(wl=WL) if r.method == 'con']:
    A = cp2(ref)
    if not A:
        continue
    grp = f'{ref.stage} {ref.model} ' + ('realistic' if ref.arm in REAL else 'ph/dev')
    for kk, v in A.items():
        pooledA[grp][kk] += v
    sv = mean(A['A1']) - mean(A['A2'])
    savings.append((sv, ref.name, len(A['A1'])))
    p(f'{ref.name:42s} n={len(A["A1"]):3d} ' + ' '.join(f'{n}:{mean(v):.1f}/{med(v):.1f}/p90 {q90(v):.1f}'
                                                      for n, v in A.items()) + f'  A1-A2={sv:.2f}')
p('pooled by group:')
for grp, A in pooledA.items():
    p(f'  {grp:32s} n={len(A["A1"]):4d} serial {mean(A["serial"]):.1f} A1 {mean(A["A1"]):.1f} A2 {mean(A["A2"]):.1f} '
      f'A2_fastpeople {mean(A["A2_fastpeople"]):.1f}  A1-A2 mean {mean(A["A1"]) - mean(A["A2"]):.2f}  '
      f'A1-A2 median {med(A["A1"]) - med(A["A2"]):.2f}')
savings.sort(reverse=True)
p('largest per-run mean savings A1-A2 (s):', ', '.join(f'{n} {s:.2f} (n {k})' for s, n, k in savings[:5]))
s19sv = [t for t in savings if t[1].startswith('s19')]
p('largest among s19 runs:', ', '.join(f'{n} {s:.2f}' for s, n, k in s19sv[:3]))

# ================================================================ L5: C vs E, same accounting
p('\n## L5 E (s16b qwen38_low fsc, realistic) and R (s16b cot), same accounting: per record with a goal call,')
p('   serial = plan+locate+context+goal latency_s; C = s19 Qwen realistic as in L1')


def e_rows(refs):
    rows = []
    for ref in refs:
        for r in recs(ref):
            L = r.timings
            if 'goal' not in L:
                continue
            rows.append(dict(run=ref.name, arm=ref.arm, scene=r.scene_id, ep=r.episode_id, trial=r.trial,
                             serial=g(L, 'plan') + g(L, 'locate') + g(L, 'context') + g(L, 'goal'),
                             goal=g(L, 'goal'), rec_latency=r.latency_s))
    return rows


E16 = e_rows([r for r in H.runs(stage='s16b', model='qwen38_low', method='fsc', wl=WL) if r.arm in REAL])
R16 = e_rows([r for r in H.runs(stage='s16b', model='qwen38_low', method='cot', wl=WL) if r.arm in REAL])
Cs, Cc = mean([x['serial'] for x in L1rows]), mean([x['cp'] for x in L1rows])
Es, Rs = mean([x['serial'] for x in E16]), mean([x['serial'] for x in R16])
p(f'E s16b fsc goal-step records n={len(E16)} serial mean {Es:.1f} (median {med([x["serial"] for x in E16]):.1f}); '
  f'E goal call mean {mean([x["goal"] for x in E16]):.1f} median {med([x["goal"] for x in E16]):.1f}')
p(f'R s16b cot goal-step records n={len(R16)} serial mean {Rs:.1f}')
p(f'C s19 n={len(L1rows)}: serial {Cs:.1f}, concurrent {Cc:.1f}; saving {100 * (1 - Cc / Cs):.1f}% per goal-step episode')
p(f'ratio C/E: serial {Cs / Es:.2f}x, concurrent {Cc / Es:.2f}x; C/R concurrent {Cc / Rs:.2f}x')
arm_of = {ref.name: ref.arm for ref in q19}
ckey = {(arm_of[x['run']], x['scene'], x['ep'], x['trial']): x for x in L1rows}
ekey = {(x['arm'], x['scene'], x['ep'], x['trial']): x for x in E16}
both = sorted(set(ckey) & set(ekey))
p(f'matched episodes (arm, scene, episode, trial) goal step in both C and E: n={len(both)}; '
  f'C serial {mean([ckey[k]["serial"] for k in both]):.1f} cp {mean([ckey[k]["cp"] for k in both]):.1f} '
  f'E {mean([ekey[k]["serial"] for k in both]):.1f}; ratio cp/E {mean([ckey[k]["cp"] for k in both]) / mean([ekey[k]["serial"] for k in both]):.2f}x')
# saving over every C record (standard-goal ones gain nothing)
allC = []
for ref in q19:
    for r in recs(ref):
        L = r.timings
        pre = g(L, 'plan') + g(L, 'locate') + g(L, 'context')
        pe, k = g(L, 'people'), g(L, 'constraints')
        allC.append((pre + pe + k, max(pre, pe) + k))
p(f'over all {len(allC)} C records (incl. standard-goal): serial {mean([a for a, b in allC]):.1f} '
  f'cp {mean([b for a, b in allC]):.1f} ({100 * (1 - mean([b for a, b in allC]) / mean([a for a, b in allC])):.1f}%)')
Creal = [x for x in L1rows if x['run'].startswith('s19_real_')]
Ereal = [x for x in E16 if x['arm'] == 'real']
creal_rec = [r.latency_s for ref in q19 if ref.arm == 'real' for r in recs(ref) if 'constraints' in r.timings]
p(f'real set alone: C n={len(Creal)} serial {mean([x["serial"] for x in Creal]):.1f} cp {mean([x["cp"] for x in Creal]):.1f} '
  f'(record latency_s mean {mean(creal_rec):.1f}); E n={len(Ereal)} serial {mean([x["serial"] for x in Ereal]):.1f}; '
  f'ratio cp/E {mean([x["cp"] for x in Creal]) / mean([x["serial"] for x in Ereal]):.2f}x, serial/E '
  f'{mean([x["serial"] for x in Creal]) / mean([x["serial"] for x in Ereal]):.2f}x')
# Same comparison at stage 21 (new wording), E = s21 fsc
E21 = e_rows([r for r in H.runs(stage='s21', method='fsc', wl=WL) if r.arm in REAL])
p(f'stage 21: C s21 con n={len(q21rows)} serial {mean([x["serial"] for x in q21rows]):.1f} cp {mean([x["cp"] for x in q21rows]):.1f}; '
  f'E s21 fsc n={len(E21)} serial {mean([x["serial"] for x in E21]):.1f}; ratio cp/E '
  f'{mean([x["cp"] for x in q21rows]) / mean([x["serial"] for x in E21]):.2f}x')

# ================================================================ L6: what measures.latency_s is; does 61.6 reproduce
p('\n## L6 measures.latency_s versus sum(result.timings_s), sum(call latency_total_s), sum(call latency_s), all listed runs')
n_all = n_eq_ts = n_none = 0
res_tot, res_lat = [], []
for ref in H.runs(wl=WL):
    for r in recs(ref):
        n_all += 1
        if r.latency_s is None:
            n_none += 1
            continue
        if abs(r.latency_s - sum(r.timings_s.values())) < 1e-6:
            n_eq_ts += 1
        res_tot.append(r.latency_s - sum(t['latency_total_s'] for t in r.timings.values()))
        res_lat.append(r.latency_s - sum(t['latency_s'] for t in r.timings.values()))
p(f'records {n_all}; latency_s missing {n_none}; equal to sum(timings_s) within 1e-6: {n_eq_ts}')
p(f'latency_s - sum(call latency_total_s): median {med(res_tot):.3f} s, min {min(res_tot):.3f}, max {max(res_tot):.3f}')
p(f'latency_s - sum(call latency_s):       median {med(res_lat):.3f} s, min {min(res_lat):.3f}, max {max(res_lat):.3f}')


def brief_row(label, refs):
    R = [r for ref in refs for r in recs(ref)]
    if not R:
        return
    rec = [r.latency_s for r in R]
    prov = [sum(t['latency_s'] for t in r.timings.values()) for r in R]
    gs = [r for r in R if 'goal' in r.timings or 'constraints' in r.timings]
    sg = [r for r in R if r.result.get('source') == 'standard']
    gsl = [r.latency_s for r in gs]
    gsp = [sum(t['latency_s'] for t in r.timings.values()) for r in gs]
    calls = sum(t['calls'] for r in R for t in r.timings.values())
    p(f'{label:34s} n={len(R):3d} rec {mean(rec):5.1f} ({med(rec):5.1f}) | prov {mean(prov):5.1f} ({med(prov):5.1f}) | '
      f'goal-step n={len(gs):3d} rec {mean(gsl):5.1f} ({med(gsl):5.1f}) prov {mean(gsp):5.1f} ({med(gsp):5.1f}) | '
      f'standard n={len(sg):3d} rec {mean([r.latency_s for r in sg]):5.1f} ({med([r.latency_s for r in sg]):5.1f}) | '
      f'calls/ep {calls / len(R):.2f}')


p('\nBrief section 3.6 table, recomputed (mean (median)); goal-step = record with a goal or constraints call;')
p('brief: Qwen R 20.8(21.6) 20.6(21.4) 28.1(27.5) 11.2(10.9) 2.57 | Qwen E 20.7(22.9) 20.6(22.7) 28.1(26.9) 11.1(10.8) 2.57 |')
p('       Qwen C 39.8(44.9) 39.5(44.5) 61.6(59.3) 11.2(10.8) 3.13 | GLM R 6.3(5.9) 5.4(5.7) 8.7(7.1) | GLM E 5.8(4.7) 4.7(4.5) 7.6(5.7) |')
p('       GLM C 8.0(7.8) 6.9(7.4) 11.6(10.1) 3.5(3.3) 3.10')
for lab, st_, mo, me in [('Qwen R s16b cot', 's16b', 'qwen38_low', 'cot'), ('Qwen E s16b fsc', 's16b', 'qwen38_low', 'fsc'),
                         ('Qwen C s19 con', 's19', 'qwen38_low', 'con'), ('GLM R s16b cot', 's16b', 'glm53', 'cot'),
                         ('GLM E s16b fsc', 's16b', 'glm53', 'fsc'), ('GLM C s19 con', 's19', 'glm53', 'con'),
                         ('Qwen R s21 cot', 's21', 'qwen38_low', 'cot'), ('Qwen E s21 fsc', 's21', 'qwen38_low', 'fsc'),
                         ('Qwen C s21 con', 's21', 'qwen38_low', 'con'),
                         ('Qwen R s22 cot (6 trials)', 's22', 'qwen38_low', 'cot'),
                         ('Qwen E s22 fsc (6 trials)', 's22', 'qwen38_low', 'fsc')]:
    brief_row(lab, [r for r in H.runs(stage=st_, model=mo, method=me, wl=WL) if r.arm in REAL])
p('per-stage provider latency_s mean (median) [mean output tokens], C realistic, s19 (brief: Qwen plan 6.4(5.8)[443], '
  'context 6.0(5.7)[417], people 16.0(17.0)[1199], constraints 31.7(29.4)[2351]; E goal 14.5(13.4))')
for lab, refs in [('Qwen s19 con', q19), ('GLM s19 con', g19),
                  ('Qwen s16b fsc', [r for r in H.runs(stage='s16b', model='qwen38_low', method='fsc', wl=WL) if r.arm in REAL])]:
    by = collections.defaultdict(list)
    for ref in refs:
        for r in recs(ref):
            for c in r.result.get('calls') or []:
                by[c['stage']].append((c.get('latency_s') or 0, (c.get('usage') or {}).get('output_tokens') or 0))
    p('  ' + lab + ': ' + '; '.join(f'{s} {mean([a for a, b in v]):.1f}({med([a for a, b in v]):.1f})[{mean([b for a, b in v]):.0f}] n{len(v)}'
                                   for s, v in by.items()))

# ================================================================ L7: cached calls
p('\n## L7 cached calls: what latency they carry, and whether measures.latency_s counts them')
for st_ in ('s19', 's21'):
    cnt = collections.Counter()
    for ref in H.runs(stage=st_, wl=WL):
        for r in recs(ref):
            for s, t in r.timings.items():
                cnt[(s, 'calls')] += t['calls']
                cnt[(s, 'cached')] += t['cached']
                cnt[(s, 'cached_with_latency>0')] += sum(1 for c in r.result.get('calls') or []
                                                        if c['stage'] == s and c.get('cached') and (c.get('latency_s') or 0) > 0)
    p(f'{st_}: ' + '; '.join(f'{s} {cnt[(s, "cached")]}/{cnt[(s, "calls")]} cached, {cnt[(s, "cached_with_latency>0")]} cached with latency_s>0'
                             for s in ('plan', 'locate', 'context', 'goal', 'people', 'constraints') if cnt[(s, 'calls')]))


def calls_by_key(refs, stage, cached=None):
    out = {}
    for ref in refs:
        for r in recs(ref):
            for c in r.result.get('calls') or []:
                if c['stage'] != stage or (cached is not None and bool(c.get('cached')) != cached):
                    continue
                out[(ref.arm, r.scene_id, r.episode_id, r.trial)] = c
    return out


def compare(label, a, b):
    keys = sorted(set(a) & set(b))
    same_text = [k for k in keys if a[k].get('raw_text') == b[k].get('raw_text')]
    eq_lat = sum(1 for k in same_text if a[k].get('latency_s') == b[k].get('latency_s'))
    eq_tot = sum(1 for k in same_text if a[k].get('latency_total_s') == b[k].get('latency_total_s'))
    eq_tok = sum(1 for k in same_text if (a[k].get('usage') or {}) == (b[k].get('usage') or {}))
    p(f'  {label}: matched keys {len(keys)}, same raw_text {len(same_text)}, of those identical latency_s {eq_lat}, '
      f'identical latency_total_s {eq_tot}, identical usage {eq_tok}')


s21 = lambda me: [r for r in H.runs(stage='s21', method=me, wl=WL) if r.arm in REAL]  # noqa: E731
s23 = lambda mo, me: [r for r in H.runs(stage='s23', model=mo, method=me, wl=WL) if r.arm in REAL]  # noqa: E731
p('cached call versus the fresh call that first produced it (key = arm, scene, episode, trial):')
compare('s21 fsc plan (cached) vs s21 cot plan (fresh)', calls_by_key(s21('fsc'), 'plan', True), calls_by_key(s21('cot'), 'plan', False))
compare('s21 con plan (cached) vs s21 cot plan (fresh)', calls_by_key(s21('con'), 'plan', True), calls_by_key(s21('cot'), 'plan', False))
compare('s21 con context (cached) vs s21 cot context (fresh)', calls_by_key(s21('con'), 'context', True), calls_by_key(s21('cot'), 'context', False))
for mo in ('glm53', 'dsv41', 'haiku55', 'sonnet55_low', 'gemini38_low'):
    compare(f's23 {mo} con plan (cached) vs s23 cot plan (fresh)', calls_by_key(s23(mo, 'con'), 'plan', True),
            calls_by_key(s23(mo, 'cot'), 'plan', False))
p('cached call versus an earlier cached copy (both cached; first producer not in the list):')
compare('s21 con people vs s19 Qwen con people', calls_by_key(s21('con'), 'people'), calls_by_key(q19, 'people'))
compare('s23 glm53 con people vs s19 GLM con people', calls_by_key(s23('glm53', 'con'), 'people'), calls_by_key(g19, 'people'))
compare('s23 dsv41 con people vs s20 DeepSeek con people', calls_by_key(s23('dsv41', 'con'), 'people'),
        calls_by_key([r for r in H.runs(stage='s20', method='con', wl=WL) if r.arm in REAL], 'people'))
compare('s19 Qwen con plan vs s16b Qwen cot plan', calls_by_key(q19, 'plan'),
        calls_by_key([r for r in H.runs(stage='s16b', model='qwen38_low', method='cot', wl=WL) if r.arm in REAL], 'plan'))
compare('s19 Qwen con context vs s16b Qwen cot context', calls_by_key(q19, 'context'),
        calls_by_key([r for r in H.runs(stage='s16b', model='qwen38_low', method='cot', wl=WL) if r.arm in REAL], 'context'))
# within-run people sharing: second record of a (scene, trial) carries the first one's people latency
for lab, refs in [('s19 Qwen', q19), ('s23 haiku55', s23('haiku55', 'con'))]:
    same = tot = 0
    for ref in refs:
        first = {}
        for r in recs(ref):
            c = [c for c in r.result.get('calls') or [] if c['stage'] == 'people']
            if not c:
                continue
            key = (r.scene_id, r.trial)
            if key in first:
                tot += 1
                same += c[0].get('latency_s') == first[key].get('latency_s') and bool(c[0].get('cached'))
            else:
                first[key] = c[0]
    p(f'  {lab}: later records of a (scene, trial) whose people call is cached with the first record\'s latency_s: {same}/{tot}')

p('measures.latency_s includes cached calls? residual = latency_s - sum(latency_total_s of ALL calls); '
  'and latency_s - sum(latency_total_s of FRESH calls only)')
for lab, refs in [('s19 Qwen realistic con (all cached)', q19), ('s19 GLM realistic con (all cached)', g19),
                  ('s21 con (plan, context, people cached)', s21('con')), ('s21 fsc (plan, context cached)', s21('fsc')),
                  ('s21 cot (all fresh)', s21('cot'))]:
    R = [r for ref in refs for r in recs(ref)]
    ra = [r.latency_s - sum(t['latency_total_s'] for t in r.timings.values()) for r in R]
    rf = [r.latency_s - sum(t['fresh_latency_total_s'] for t in r.timings.values()) for r in R]
    cached_share = [sum(t['latency_total_s'] - t['fresh_latency_total_s'] for t in r.timings.values()) for r in R]
    p(f'  {lab:40s} n={len(R)} mean latency_s {mean([r.latency_s for r in R]):.1f}; residual vs all calls median {med(ra):.3f} '
      f'[min {min(ra):.3f}, max {max(ra):.3f}]; vs fresh-only median {med(rf):.2f}; mean cached latency_total_s per record {mean(cached_share):.1f}')
gs = [r for ref in s21('con') for r in recs(ref) if 'constraints' in r.timings]
gd = [r.timings_s.get('goal', 0) - g(r.timings, 'people', 'latency_total_s') - g(r.timings, 'constraints', 'latency_total_s') for r in gs]
p(f'  s21 con goal-step records n={len(gs)}: timings_s.goal - (people + constraints latency_total_s) median {med(gd):.3f} '
  f'[min {min(gd):.3f}, max {max(gd):.3f}]; people cached in {sum(r.timings["people"]["cached"] for r in gs)}/{len(gs)}')

# ================================================================ L8 and NEW: fresh-call medians per stage
p('\n## L8 stage 21 fresh-call medians (cached false), latency_s; goal split by method (cot = R, fsc = E)')


def fresh_table(label, refs):
    by = collections.defaultdict(list)
    tot = collections.defaultdict(list)
    cachedn = collections.Counter()
    folders = collections.defaultdict(collections.Counter)
    for ref in refs:
        for r in recs(ref):
            for c in r.result.get('calls') or []:
                s = c['stage']
                if s == 'goal':
                    s = 'goal R' if ref.method == 'cot' else 'goal E'
                if c.get('cached'):
                    cachedn[s] += 1
                    continue
                by[s].append(c.get('latency_s') or 0.0)
                tot[s].append(c.get('latency_total_s') or 0.0)
                folders[s][ref.method] += 1
    order = ['plan', 'locate', 'context', 'goal R', 'goal E', 'people', 'constraints']
    p(f'{label}')
    for s in order:
        if s in by or cachedn[s]:
            p(f'    {s:12s} fresh n={len(by[s]):4d} median latency_s {med(by[s]):5.1f} (mean {mean(by[s]):5.1f}; '
              f'median latency_total_s {med(tot[s]):5.1f}) cached excluded {cachedn[s]:4d}; fresh by folder {dict(folders[s])}')
    return by


s21by = fresh_table('s21 Qwen 3.8 low, realistic, 9 folders', [r for r in H.runs(stage='s21', wl=WL) if r.arm in REAL])
p('\n## NEW stage 23 fresh-call medians per model (same definition); load during s23: 6 gateway jobs under 85/min,'
  ' 4 Anthropic at 40/min, 3 Gemini at 60/min')
for m in ('glm53', 'dsv41', 'haiku55', 'sonnet55_low', 'gemini38_low'):
    fresh_table(f's23 {m}, realistic, 9 folders', [r for r in H.runs(stage='s23', model=m, wl=WL) if r.arm in REAL])

OUT.write_text(buf.getvalue(), encoding='utf-8')
print(f'\nsaved {OUT}')
