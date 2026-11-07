"""I44 re-run: a faster mixture-of-experts model (qwen3.6:35b against Qwen 3.8).

Reads runs only through research_harness. No model call, no network. Writes i44_tables.txt beside this file.

Q1  tokens per second: output_tokens / latency_s per call, s8 runs *_qwen36_35 against *_qwen38 (both thinking
    off), fresh calls only (primary), then the judge's lat.py reading (every call, cached copies included, per run),
    and per stage the output tokens and the latency. Also against Qwen 3.8 low (s10, s16b ph, s19 ph) for context.
Q2  paired wins and losses of qwen3.6:35b against several Qwen 3.8 runs, keyed (scene_id, episode_id, trial) or,
    against a one-trial run, (scene_id, episode_id) per qwen3.6 trial; recorded success.
Q3  qwen3.6:35b against Qwen 3.8 with C on s8 (convlm, conorc), recorded and rescored now.
"""
import collections
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H  # noqa: E402

OUT = Path(__file__).resolve().parent / 'i44_tables.txt'
_lines = []


def out(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    _lines.append(s)


def med(v):
    return st.median(v) if v else float('nan')


def sign_p(w, l):
    """Two-sided exact sign test on the discordant pairs."""
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def run1(stage, name):
    rs = [r for r in H.runs(stage=stage) if r.name == name]
    assert len(rs) == 1, (stage, name, rs)
    return rs[0]


REC = {}


def recs(run):
    if run.folder not in REC:
        REC[run.folder] = list(H.records(run))
    return REC[run.folder]


def calls_of(run):
    for x in recs(run):
        for c in x.result.get('calls') or []:
            u = c.get('usage') or {}
            yield x, c, u


# ---------------------------------------------------------------- Q1
out('=' * 100)
out('Q1. Output tokens per second, s8 qwen3.6:35b (qwen36_35) against Qwen 3.8 (qwen38), both reasoning_effort none')
out('tok/s = usage.output_tokens / latency_s per call (latency_s = last back-end attempt); calls with 0 tokens or 0 s dropped')
out('=' * 100)

S8 = {m: [r for r in H.runs(stage='s8', model=m)] for m in ('qwen36_35', 'qwen38')}
for m, rs in S8.items():
    out(f'{m}: runs', ', '.join(r.folder for r in rs))


def collect(rs, fresh_only=True, dedup=False):
    """{stage: [(latency_s, output_tokens, reasoning_tokens, tok/s)]}"""
    by = collections.defaultdict(list)
    seen = set()
    for r in rs:
        for x, c, u in calls_of(r):
            if fresh_only and c.get('cached'):
                continue
            if dedup:
                key = (c['stage'], x.scene_id, x.episode_id, x.trial, c.get('raw_text'), c.get('latency_s'))
                if key in seen:
                    continue
                seen.add(key)
            lat = c.get('latency_s') or 0.0
            tok = u.get('output_tokens') or 0
            rt = u.get('reasoning_tokens') or 0
            by[c['stage']].append((lat, tok, rt, tok / lat if lat and tok else None))
    return by


def table(label, data):
    out(f'\n-- {label}')
    out(f'{"model":<12}{"stage":<13}{"n calls":>8}{"med lat s":>11}{"med out tok":>12}{"med tok/s":>11}{"mean tok/s":>11}{"sum tok/sum s":>14}')
    res = {}
    for m, by in data.items():
        allrows = [row for v in by.values() for row in v]
        for stage, rows in list(by.items()) + [('ALL', allrows)]:
            tps = [r[3] for r in rows if r[3] is not None]
            stok = sum(r[1] for r in rows if r[3] is not None)
            slat = sum(r[0] for r in rows if r[3] is not None)
            res[(m, stage)] = dict(n=len(rows), lat=med([r[0] for r in rows]), tok=med([r[1] for r in rows]),
                                   tps=med(tps), mtps=(st.mean(tps) if tps else float('nan')),
                                   agg=(stok / slat if slat else float('nan')), ntps=len(tps))
            d = res[(m, stage)]
            out(f'{m:<12}{stage:<13}{d["n"]:>8}{d["lat"]:>11.2f}{d["tok"]:>12.0f}{d["tps"]:>11.1f}{d["mtps"]:>11.1f}{d["agg"]:>14.1f}')
    ms = list(data)
    if len(ms) == 2:
        a, b = ms
        out(f'ratio {a}/{b}:')
        for stage in sorted({s for (mm, s) in res if mm == a} & {s for (mm, s) in res if mm == b}):
            A, B = res[(a, stage)], res[(b, stage)]
            out(f'   {stage:<12} med tok/s x{A["tps"] / B["tps"]:.2f}   sum-ratio tok/s x{A["agg"] / B["agg"]:.2f}   '
                f'med out tok x{A["tok"] / B["tok"]:.2f}   med latency x{A["lat"] / B["lat"]:.2f}')
    return res


R_fresh = table('s8, fresh calls only (primary definition), pooled over none/convlm/conorc',
                {m: collect(rs, fresh_only=True) for m, rs in S8.items()})
R_dedup = table('s8, every call once (cached copies included but each distinct call counted once)',
                {m: collect(rs, fresh_only=False, dedup=True) for m, rs in S8.items()})
R_all = table('s8, every call as recorded (cached copies repeated across runs), pooled',
              {m: collect(rs, fresh_only=False) for m, rs in S8.items()})

# lat.py per run (all calls incl. cached, median tok/s over all calls of the run; per stage medians)
out('\n-- the judge lat.py reading, per run (every call, cached included); success = recorded score.success')
for m, rs in S8.items():
    for r in rs:
        rr = recs(r)
        succ = sum(1 for x in rr if x.score.get('success'))
        by = collections.defaultdict(list)
        tps = []
        for x, c, u in calls_of(r):
            by[c['stage']].append((c['latency_s'], u.get('output_tokens') or 0))
            if c['latency_s'] and u.get('output_tokens'):
                tps.append(u['output_tokens'] / c['latency_s'])
        s = ' '.join(f'{k}:{med([a for a, b in v]):.1f}s/{med([b for a, b in v]):.0f}t' for k, v in by.items())
        out(f'{r.name:<26} n={len(rr)} succ {succ} tok/s med {med(tps):.0f}  {s}')

# matched calls: the same (scene, episode, trial, stage) in the same s8 method, both fresh
out('\n-- matched calls: same s8 method, same (scene, episode, trial, stage), fresh in both models')
for meth in ('convlm', 'conorc'):
    a = run1('s8', f's8_{meth}_qwen36_35')
    b = run1('s8', f's8_{meth}_qwen38')
    ca = {(x.scene_id, x.episode_id, x.trial, c['stage']): (c, u) for x, c, u in calls_of(a) if not c.get('cached')}
    cb = {(x.scene_id, x.episode_id, x.trial, c['stage']): (c, u) for x, c, u in calls_of(b) if not c.get('cached')}
    for stage in ('people', 'constraints'):
        keys = [k for k in ca if k in cb and k[3] == stage]
        if not keys:
            continue
        tr = [(ca[k][1].get('output_tokens') or 0) / max(1, (cb[k][1].get('output_tokens') or 0)) for k in keys]
        lr = [(ca[k][0]['latency_s'] or 0) / (cb[k][0]['latency_s'] or 1) for k in keys]
        sa = [(ca[k][1].get('output_tokens') or 0) / ca[k][0]['latency_s'] for k in keys]
        sb = [(cb[k][1].get('output_tokens') or 0) / cb[k][0]['latency_s'] for k in keys]
        more = sum(1 for k in keys if (ca[k][1].get('output_tokens') or 0) > (cb[k][1].get('output_tokens') or 0))
        faster = sum(1 for k in keys if ca[k][0]['latency_s'] < cb[k][0]['latency_s'])
        out(f'{meth:<7}{stage:<12} pairs {len(keys):>3}  med tok ratio x{med(tr):.2f} (qwen36 more in {more}/{len(keys)})  '
            f'med latency ratio x{med(lr):.2f} (qwen36 faster in {faster}/{len(keys)})  '
            f'med tok/s {med(sa):.1f} vs {med(sb):.1f} (x{med(sa) / med(sb):.2f})')

# per record: the goal step of convlm (people + constraints), fresh calls only, where both calls are fresh
out('\n-- s8 convlm goal step (people + constraints calls) per record where both calls are fresh')
for m in ('qwen36_35', 'qwen38'):
    r = run1('s8', f's8_convlm_{m}')
    lat, tok = [], []
    for x in recs(r):
        cs = [c for c in x.result.get('calls') or [] if c['stage'] in ('people', 'constraints')]
        if len(cs) == 2 and not any(c.get('cached') for c in cs):
            lat.append(sum(c['latency_s'] for c in cs))
            tok.append(sum((c.get('usage') or {}).get('output_tokens') or 0 for c in cs))
    out(f'{m:<12} records {len(lat):>3}  med latency {med(lat):.2f} s  med output tokens {med(tok):.0f}  '
        f'mean latency {st.mean(lat):.2f} s  mean output tokens {st.mean(tok):.0f}')

# context: Qwen 3.8 low (thinking low) on the placeholder, fresh calls only
out('\n-- context: Qwen 3.8 low on the placeholder (reasoning tokens are billed as output tokens), fresh calls only')
LOW = [run1('s10', 's10_qwen38_low_cot'), run1('s10', 's10_qwen38_low_fsc'), run1('s10', 's10_qwen38_low_con'),
       run1('s16b', 's16b_ph_qwen38_low_cot'), run1('s16b', 's16b_ph_qwen38_low_fsc')]
NONE10 = [run1('s10', 's10_qwen38_none_cot'), run1('s10', 's10_qwen38_none_fsc'), run1('s10', 's10_qwen38_none_con')]
for label, rs in (('s10 qwen38 none', NONE10), ('s10+s16b ph qwen38 low', LOW)):
    nc = sum(1 for r in rs for _ in calls_of(r))
    nf = sum(1 for r in rs for _, c, _ in calls_of(r) if not c.get('cached'))
    out(f'{label}: calls {nc}, fresh {nf}')
table('Qwen 3.8 none (s10) against low (s10 + s16b ph), fresh calls only',
      {'q38_none': collect(NONE10), 'q38_low': collect(LOW)})
table('qwen3.6:35b none (s8, fresh) against Qwen 3.8 low (s10 + s16b ph, fresh)',
      {'qwen36_35': collect(S8['qwen36_35']), 'q38_low': collect(LOW)})

# the C goal-step calls of Qwen 3.8 low: s10 low con (fresh and cached) and s19 ph low con (all cache hits; their
# latency_s is that of the call that first produced them). Each distinct call once.
LOWC = [run1('s10', 's10_qwen38_low_con'), run1('s19', 's19_ph_qwen38_low_con')]
NONEC = [run1('s8', 's8_convlm_qwen38'), run1('s8', 's8_conorc_qwen38')]
Q36C = [run1('s8', 's8_convlm_qwen36_35'), run1('s8', 's8_conorc_qwen36_35')]


def only(by, stages=('people', 'constraints')):
    return {k: v for k, v in by.items() if k in stages}


table('C goal-step calls, each distinct call once (cached copies counted once): qwen3.6 off, Qwen 3.8 off, Qwen 3.8 low',
      {'qwen36_off': only(collect(Q36C, fresh_only=False, dedup=True)),
       'q38_off': only(collect(NONEC, fresh_only=False, dedup=True)),
       'q38_low': only(collect(LOWC, fresh_only=False, dedup=True))})
table('the same, Qwen 3.8 off against Qwen 3.8 low (thinking alone)',
      {'q38_off': only(collect(NONEC, fresh_only=False, dedup=True)),
       'q38_low': only(collect(LOWC, fresh_only=False, dedup=True))})

# ---------------------------------------------------------------- Q2
out('\n' + '=' * 100)
out('Q2. Paired wins/losses, qwen3.6:35b (A) against Qwen 3.8 runs (B), recorded score.success')
out('win = A succeeds and B fails; loss = A fails and B succeeds; two-sided exact sign test on discordant pairs')
out('=' * 100)

# scene identity across scene folders (the s19 ph runs record keep/placeholder_first14)
listing = {}
for sref in H.distinct_scenes():
    for lst in sref.listed:
        listing[tuple(lst) if isinstance(lst, (list, tuple)) else lst] = sref.folder


def succ_map(run, by_trial=True):
    m = {}
    for x in recs(run):
        k = (x.scene_id, x.episode_id, x.trial) if by_trial else (x.scene_id, x.episode_id)
        m[k] = bool(x.score.get('success'))
    return m


def paired(a, b, label):
    A, B = succ_map(a), succ_map(b)
    keys = sorted(set(A) & set(B))
    w = sum(1 for k in keys if A[k] and not B[k])
    l = sum(1 for k in keys if B[k] and not A[k])
    out(f'{label:<64} pairs {len(keys):>3}  A {sum(A[k] for k in keys):>3}  B {sum(B[k] for k in keys):>3}  '
        f'wins {w:>2} losses {l:>2}  p {sign_p(w, l):.3f}')
    return w, l, len(keys)


def paired_one_trial(a, b, label):
    """b has one trial (0): pair each trial of a with b by (scene, episode)."""
    B = succ_map(b, by_trial=False)
    tot_w = tot_l = 0
    for t in sorted({x.trial for x in recs(a)}):
        A = {(x.scene_id, x.episode_id): bool(x.score.get('success')) for x in recs(a) if x.trial == t}
        keys = sorted(set(A) & set(B))
        w = sum(1 for k in keys if A[k] and not B[k])
        l = sum(1 for k in keys if B[k] and not A[k])
        tot_w += w
        tot_l += l
        out(f'{label + f" [A trial {t}]":<64} pairs {len(keys):>3}  A {sum(A[k] for k in keys):>3}  '
            f'B {sum(B[k] for k in keys):>3}  wins {w:>2} losses {l:>2}  p {sign_p(w, l):.3f}')
    out(f'{label + " [A trials 0-2 summed]":<64} wins {tot_w} losses {tot_l}')


q36 = {meth: run1('s8', f's8_{meth}_qwen36_35') for meth in ('none', 'convlm', 'conorc')}
q38 = {meth: run1('s8', f's8_{meth}_qwen38') for meth in ('none', 'convlm', 'conorc')}
s19 = run1('s19', 's19_ph_qwen38_low_con')
s16c = run1('s16b', 's16b_ph_qwen38_low_cot')
s16f = run1('s16b', 's16b_ph_qwen38_low_fsc')
s10 = {k: run1('s10', f's10_qwen38_{k}') for k in ('low_con', 'low_cot', 'low_fsc', 'none_con', 'medium_con')}

# scene folders used
for r in (q36['convlm'], s19, s16c, s10['low_con']):
    out(f'scenes of {r.name}: {r.scenes_dir}')
same = all(listing.get((q36['convlm'].scenes_dir, sid)) == listing.get((s19.scenes_dir, sid))
           for sid in {x.scene_id for x in recs(s19)})
out(f's19 ph scenes are the s8 scenes by content (distinct_scenes): {same}')

out('\n-- same model family, thinking off on both sides (s8 against s8)')
for meth in ('none', 'convlm', 'conorc'):
    paired(q36[meth], q38[meth], f's8 {meth} qwen36_35 vs s8 {meth} qwen38')
out('\n-- against Qwen 3.8 low, three trials (s19 ph con, s16b ph cot/fsc)')
for meth in ('convlm', 'conorc'):
    paired(q36[meth], s19, f's8 {meth} qwen36_35 vs s19 ph qwen38_low con')
    paired(q36[meth], s16f, f's8 {meth} qwen36_35 vs s16b ph qwen38_low fsc (E)')
    paired(q36[meth], s16c, f's8 {meth} qwen36_35 vs s16b ph qwen38_low cot (R)')
out('\n-- against Qwen 3.8 low/none/medium, one trial (s10, 28 records), by (scene, episode) per qwen3.6 trial')
for meth in ('convlm', 'conorc'):
    for k in ('low_con', 'none_con', 'medium_con'):
        paired_one_trial(q36[meth], s10[k], f's8 {meth} qwen36_35 vs s10 qwen38_{k}')
out('\n-- trial 0 only, keyed (scene, episode, trial)')
for meth in ('convlm', 'conorc'):
    for k in ('low_con',):
        paired(q36[meth], s10[k], f's8 {meth} qwen36_35 vs s10 qwen38_{k} (trial 0 keys)')

# ---------------------------------------------------------------- Q3
out('\n' + '=' * 100)
out('Q3. C with qwen3.6:35b against Qwen 3.8 (thinking off) on the s8 placeholder episodes (84 records a run)')
out('=' * 100)
tot = collections.Counter()
for meth in ('convlm', 'conorc'):
    for m, r in (('qwen36_35', q36[meth]), ('qwen38', q38[meth])):
        rr = recs(r)
        rec_s = sum(1 for x in rr if x.score.get('success'))
        now = sum(1 for x in rr if H.rescore(x).success)
        tot[(m, 'rec')] += rec_s
        tot[(m, 'now')] += now
        out(f'{r.name:<26} n {len(rr)}  recorded success {rec_s}  rescored now {now}')
out(f'convlm + conorc, of 168: qwen36_35 recorded {tot[("qwen36_35", "rec")]} (now {tot[("qwen36_35", "now")]}), '
    f'qwen38 recorded {tot[("qwen38", "rec")]} (now {tot[("qwen38", "now")]})')
w1, l1, n1 = paired(q36['convlm'], q38['convlm'], 'convlm qwen36_35 vs qwen38 (84)')
w2, l2, n2 = paired(q36['conorc'], q38['conorc'], 'conorc qwen36_35 vs qwen38 (84)')
out(f'{"convlm + conorc pooled (168)":<64} wins {w1 + w2} losses {l1 + l2}  p {sign_p(w1 + w2, l1 + l2):.3f}')
out('Note: report.tex:1263 says 136 vs 127 of 168, p = 0.20. 136 - 69 = 67 and 127 - 65 = 62; the harness list holds')
out('no second 84-record C run of each model besides conorc (71 vs 67), so that 168 is not a pair of listed runs.')

OUT.write_text('\n'.join(_lines) + '\n')
print(f'\nwritten {OUT}')
