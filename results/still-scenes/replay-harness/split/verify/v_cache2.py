"""Are Qwen s21 C people answers copies of s19's? Are Haiku/Sonnet/Gemini people cache hits shared between the two
episodes of one scene and trial?"""
import sys, collections, io, json
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
OUT = 'results/still-scenes/replay-harness/split/verify/v_cache2_out.txt'
buf = io.StringIO()
def p(*a):
    s = ' '.join(str(x) for x in a); print(s); buf.write(s + '\n')
REAL = ('real', 'staffed', 'plain')
def load(stage, model, method):
    out = []
    for arm in REAL:
        (r,) = H.runs(stage=stage, arm=arm, model=model, method=method)
        out += list(H.records(r))
    return out
key = lambda r: (r.arm, r.scene_id, r.episode_id, r.trial)
pj = lambda x: json.dumps(x, sort_keys=True)
new = load('s21', 'qwen38_low', 'con'); old = {key(r): r for r in load('s19', 'qwen38_low', 'con')}
withp = [r for r in new if r.people is not None]
p('Qwen s21 C people identical to s19 same key:', sum(key(r) in old and pj(old[key(r)].people) == pj(r.people) for r in withp), '/', len(withp))
# any s19 key (any trial/episode) with identical people?
oldset = collections.defaultdict(set)
for k, r in old.items():
    if r.people is not None:
        oldset[(k[0], k[1])].add(pj(r.people))
p('Qwen s21 C people identical to some s19 answer of same arm+scene:', sum(pj(r.people) in oldset[(r.arm, r.scene_id)] for r in withp), '/', len(withp))
s19p = [r for r in old.values() if r.people is not None]
p('s19 qwen C people cache:', sum(r.timings.get('people', {}).get('cached', 0) for r in s19p), '/', len(s19p))
# compare success at people scenes s19 vs s21
for st, recs in (('s19', list(old.values())), ('s21', new)):
    p(st, 'qwen C people-scene success', sum(bool(r.score['success']) for r in recs if r.situation != 'no_one'), 'of', sum(r.situation != 'no_one' for r in recs))
# within s21: do the two episodes of one scene+trial share the people answer?
for model, st in (('qwen38_low', 's21'), ('haiku55', 's23'), ('sonnet55_low', 's23'), ('gemini38_low', 's23')):
    recs = [r for r in load(st, model, 'con') if r.people is not None]
    g = collections.defaultdict(list)
    for r in recs:
        g[(r.arm, r.scene_id, r.trial)].append(r)
    shared = sum(1 for v in g.values() if len(v) > 1 and len({pj(x.people) for x in v}) == 1)
    multi = sum(1 for v in g.values() if len(v) > 1)
    cached_second = sum(r.timings.get('people', {}).get('cached', 0) for r in recs)
    p(f'{model} {st}: scene+trial groups with >1 people record {multi}, of which people answer identical {shared}; people cached {cached_second}/{len(recs)}')
open(OUT, 'w').write(buf.getvalue())
