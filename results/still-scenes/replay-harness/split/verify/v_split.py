"""Independent verification of i01_split claims (recorded score fields only)."""
import sys, collections, io
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
OUT = 'results/2026-10-17-beyond-vlm/rerun/i01_split/verify/v_split_out.txt'
buf = io.StringIO()
def p(*a):
    s = ' '.join(str(x) for x in a)
    print(s); buf.write(s + '\n')

REAL = ('real', 'staffed', 'plain')
SITS = ('no_one', 'waiting_line', 'conversation_group')
M = {'R': 'cot', 'E': 'fsc', 'C': 'con'}

def load(stage, model, method, arms=REAL, trials=None):
    out = []
    for arm in arms:
        rs = H.runs(stage=stage, arm=arm, model=model, method=method)
        assert len(rs) == 1, (stage, arm, model, method, rs)
        for r in H.records(rs[0]):
            if trials is not None and r.trial not in trials:
                continue
            out.append(r)
    return out

def key(r):
    return (r.arm, r.scene_id, r.episode_id, r.trial)

def cls(r):
    s = r.score
    if r.result.get('status') != 'ok' or not s.get('target_correct'):
        return 'wrong'
    if s.get('success'):
        return 'succ'
    if not s.get('in_region'):
        return 'out'
    if not s.get('reachable'):
        return 'unreach'
    if not s.get('heading_ok'):
        return 'heading'
    return 'other'

def summarise(label, recs):
    n = len(recs)
    succ = sum(bool(r.score.get('success')) for r in recs)
    found = sum(bool(r.score.get('target_correct')) and r.result.get('status') == 'ok' for r in recs)
    found_raw = sum(bool(r.score.get('target_correct')) for r in recs)
    notok = sum(r.result.get('status') != 'ok' for r in recs)
    sg = sum(bool(r.score.get('success')) and bool(r.score.get('target_correct')) for r in recs)
    row = [f'{label:28s} n={n:3d} succ={succ:3d} found={found:3d}(raw {found_raw}) notok={notok} givenT={sg}/{found}=' + (f'{100*sg/found:.1f}%' if found else 'na')]
    for sit in SITS:
        rr = [r for r in recs if r.situation == sit]
        f = sum(bool(r.score.get('target_correct')) for r in rr)
        s = sum(bool(r.score.get('success')) for r in rr)
        row.append(f'{sit[:5]}: n={len(rr)} found={f} succ={s} givenT={s}/{f}' + (f'={100*s/f:.1f}%' if f else ''))
    c = collections.Counter(cls(r) for r in recs)
    outunr = sum(cls(r) == 'out' and not r.score.get('reachable') for r in recs)
    row.append(f'fails={n-succ}: wrong={c["wrong"]} out={c["out"]} unreach={c["unreach"]} heading={c["heading"]} other={c["other"]} (out&unr={outunr})')
    p(' | '.join(row))
    return dict(n=n, succ=succ, found=found, c=c)

NEW = [('qwen38_low', 's21'), ('glm53', 's23'), ('dsv41', 's23'), ('haiku55', 's23'), ('sonnet55_low', 's23'), ('gemini38_low', 's23')]
data = {}
p('=== NEW WORDING ===')
for model, st in NEW:
    for m, meth in M.items():
        data[(model, m)] = load(st, model, meth)
        summarise(f'{model} {m} ({st})', data[(model, m)])

p('\n=== c0 totals R/E/C ===')
for model, st in NEW:
    p(model, '/'.join(str(sum(bool(r.score['success']) for r in data[(model, m)])) for m in 'REC'), [len(data[(model, m)]) for m in 'REC'])

p('\n=== c1 lines given target, R ===')
for model, st in NEW:
    rr = [r for r in data[(model, 'R')] if r.situation == 'waiting_line']
    f = [r for r in rr if r.score['target_correct']]
    s = sum(bool(r.score['success']) for r in f)
    p(f'{model}: {s}/{len(f)} = {100*s/len(f):.1f}%  (found {len(f)}/{len(rr)})')

p('\n=== c2 C people scenes ===')
for model, st in NEW:
    rr = [r for r in data[(model, 'C')] if r.situation in ('waiting_line', 'conversation_group')]
    s = sum(bool(r.score['success']) for r in rr)
    c = collections.Counter((r.situation, cls(r)) for r in rr if not r.score['success'])
    p(f'{model}: {s}/{len(rr)}  fails {dict(c)}')

p('\n=== c3 C gap to Sonnet by situation and failure class ===')
base = {sit: sum(bool(r.score['success']) for r in data[('sonnet55_low', 'C')] if r.situation == sit) for sit in SITS}
for model, st in NEW:
    rr = data[(model, 'C')]
    tot = sum(bool(r.score['success']) for r in rr)
    parts = {}
    for sit in SITS:
        x = [r for r in rr if r.situation == sit]
        parts[sit] = (sum(bool(r.score['success']) for r in x) - base[sit],
                      dict(collections.Counter(cls(r) for r in x if not r.score['success'])))
    p(f'{model}: total {tot} gap {tot-242}:', parts)
# no_one failures of Qwen C detail
p('Qwen C no_one failures:')
for r in data[('qwen38_low', 'C')]:
    if r.situation == 'no_one' and not r.score['success']:
        p('  ', r.arm, r.scene_id, r.episode_id, r.trial, cls(r), r.result.get('source'), round(r.score.get('distance_to_region_m') or 0, 3), r.score.get('reachable'))
p('Gemini C no_one failures:')
for r in data[('gemini38_low', 'C')]:
    if r.situation == 'no_one' and not r.score['success']:
        p('  ', r.arm, r.scene_id, r.episode_id, r.trial, cls(r), r.result.get('source'), round(r.score.get('distance_to_region_m') or 0, 3))
# no_one wrong targets per model on C
p('wrong targets at no_one, C:', {model: sum(cls(r) == 'wrong' for r in data[(model, 'C')] if r.situation == 'no_one') for model, _ in NEW})
p('wrong targets at no_one, R:', {model: sum(cls(r) == 'wrong' for r in data[(model, 'R')] if r.situation == 'no_one') for model, _ in NEW})

p('\n=== n1 target shared across methods ===')
def tw(r):
    t = r.result.get('target')
    if not t or not t.get('world'):
        return None
    return tuple(round(c, 4) for c in t['world'])
def shared(groups, label):
    maps = [{key(r): r for r in g} for g in groups]
    ks = set(maps[0])
    for mm in maps[1:]:
        ks &= set(mm)
    same_w = sum(len({tw(mm[k]) for mm in maps}) == 1 for k in ks)
    same_tc = sum(len({bool(mm[k].score['target_correct']) for mm in maps}) == 1 for k in ks)
    p(f'{label}: keys={len(ks)} same world={same_w} same target_correct={same_tc}')
for model, st in NEW:
    shared([data[(model, m)] for m in 'REC'], f'{model} new')
    # cache of plan/context in E, C
    for m in 'EC':
        cc = collections.Counter()
        for r in data[(model, m)]:
            for stg in ('plan', 'locate', 'context'):
                t = r.timings.get(stg)
                if t:
                    cc[(stg, 'calls')] += t['calls']; cc[(stg, 'cached')] += t['cached']
        p(f'   {m} cache:', dict(cc))
    cc = collections.Counter()
    for r in data[(model, 'R')]:
        for stg, t in r.timings.items():
            cc[(stg, 'calls')] += t['calls']; cc[(stg, 'cached')] += t['cached']
    p('   R cache:', dict(cc))

p('\n=== n2 found by situation (R records) ===')
for model, st in NEW:
    rr = data[(model, 'R')]
    f = [sum(bool(r.score['target_correct']) for r in rr if r.situation == sit) for sit in SITS]
    notok = sum(r.result.get('status') != 'ok' for m in 'REC' for r in data[(model, m)])
    p(model, f, sum(f), f'{100*sum(f)/len(rr):.1f}%', 'notok all methods', notok)

p('\n=== n11 E-R ===')
d = {model: sum(bool(r.score['success']) for r in data[(model, 'E')]) - sum(bool(r.score['success']) for r in data[(model, 'R')]) for model, _ in NEW}
p(d, 's23 sum', sum(v for k, v in d.items() if k != 'qwen38_low'), 'all', sum(d.values()))

p('\n=== n9 wrong targets by scene (R) ===')
for model, st in NEW:
    c = collections.Counter(r.scene_id.split('_', 1)[1] for r in data[(model, 'R')] if not r.score['target_correct'])
    p(model, sum(c.values()), dict(c.most_common()))
    # also with arm prefix kept
    c2 = collections.Counter(r.scene_id for r in data[(model, 'R')] if not r.score['target_correct'])
    p('   ', dict(c2.most_common()))

p('\n=== n10 right target, in region, unreachable ===')
for model, st in NEW:
    for m in 'REC':
        rr = [r for r in data[(model, m)] if cls(r) == 'unreach']
        gap0 = sum((r.score.get('min_body_gap_m') is not None and r.score['min_body_gap_m'] <= 0.0) for r in rr)
        gap2 = sum((r.score.get('min_body_gap_m') is not None and r.score['min_body_gap_m'] <= 0.002) for r in rr)
        no1 = sum(r.situation == 'no_one' for r in rr)
        p(f'{model} {m}: unreach={len(rr)} gap<=0: {gap0} gap<=0.002: {gap2} no_one: {no1}')

# ===== old wording =====
p('\n=== n5 OLD WORDING ===')
old = {}
for model, src in (('qwen38_low', {'R': ('s16b', 'cot'), 'E': ('s16b', 'fsc'), 'C': ('s19', 'con')}),
                   ('glm53', {'R': ('s16b', 'cot'), 'E': ('s16b', 'fsc'), 'C': ('s19', 'con')}),
                   ('dsv41', {'R': ('s20', 'cot'), 'E': ('s20', 'fsc'), 'C': ('s20', 'con')})):
    for m, (st, meth) in src.items():
        old[(model, m)] = load(st, model, meth)
        summarise(f'OLD {model} {m} ({st})', old[(model, m)])
    tot = '/'.join(str(sum(bool(r.score['success']) for r in old[(model, m)])) for m in 'REC')
    lines = []
    for m in 'REC':
        rr = [r for r in old[(model, m)] if r.situation == 'waiting_line' and r.score['target_correct']]
        s = sum(bool(r.score['success']) for r in rr)
        lines.append(f'{s}/{len(rr)}={100*s/len(rr):.1f}%')
    ppl = sum(bool(r.score['success']) for r in old[(model, 'C')] if r.situation != 'no_one')
    foundR = sum(bool(r.score['target_correct']) for r in old[(model, 'R')])
    notok = sum(r.result.get('status') != 'ok' for m in 'REC' for r in old[(model, m)])
    notok_sit = collections.Counter((m, r.situation) for m in 'REC' for r in old[(model, m)] if r.result.get('status') != 'ok')
    p(f'OLD {model}: totals {tot}; found R {foundR}; lines givenT {lines}; C people {ppl}/144; notok {notok} {dict(notok_sit)}')
    shared([old[(model, m)] for m in 'REC'], f'OLD {model}')
    shared([old[(model, m)] for m in 'RE'], f'OLD {model} R,E only')

# ===== Qwen 9-draw pool =====
p('\n=== n6 Qwen 9-draw pool ===')
s22 = {m: load('s22', 'qwen38_low', M[m]) for m in 'RE'}
p('s22 trials:', sorted(collections.Counter(r.trial for r in s22['R']).items()))
pool = {}
for m in 'RE':
    s22f = [r for r in s22[m] if r.trial in (3, 4, 5)]
    pool[m] = old[('qwen38_low', m)] + s22f + data[('qwen38_low', m)]
    summarise(f'POOL9 qwen {m}', pool[m])
    p('   s22 trials 3-5 alone succ:', sum(bool(r.score['success']) for r in s22f), len(s22f))
    rr = [r for r in pool[m] if r.situation == 'waiting_line' and r.score['target_correct']]
    s = sum(bool(r.score['success']) for r in rr)
    p(f'   lines givenT {s}/{len(rr)}={100*s/len(rr):.1f}%')
p('E-R pool9:', sum(bool(r.score['success']) for r in pool['E']) - sum(bool(r.score['success']) for r in pool['R']))
pool['C'] = old[('qwen38_low', 'C')] + data[('qwen38_low', 'C')]
summarise('POOL6 qwen C', pool['C'])
rr = [r for r in pool['C'] if r.situation == 'waiting_line' and r.score['target_correct']]
p('   C lines givenT', sum(bool(r.score['success']) for r in rr), len(rr), 'people', sum(bool(r.score['success']) for r in pool['C'] if r.situation != 'no_one'))

p('\n=== n7 s22 duplicates ===')
for m in 'RE':
    ref = {key(r): r for r in old[('qwen38_low', m)]}
    dup = [r for r in s22[m] if r.trial in (0, 1, 2)]
    same = sum(key(r) in ref and ref[key(r)].score == r.score and ref[key(r)].pose_world == r.pose_world for r in dup)
    same_t = sum(key(r) in ref and tw(ref[key(r)]) == tw(r) for r in dup)
    fresh = [r for r in s22[m] if r.trial in (3, 4, 5)]
    # fresh vs s16b same keys mod 3
    p(f'{m}: dup records {len(dup)} same score+pose {same} same target {same_t}')

open(OUT, 'w').write(buf.getvalue())
