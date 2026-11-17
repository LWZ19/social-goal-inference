"""N1 on the new-wording sets (s21 Qwen, s23 others): E failures at groups with the right target, judge's definitions,
from snap_records.jsonl (written by snap.py). Saves n1_s23.txt."""
import json, collections
HERE = 'results/still-scenes/depth-fix/first-check'
rows = [json.loads(l) for l in open(f'{HERE}/snap_records.jsonl')]
out = []
def p(s): print(s); out.append(s)
p("E (fsc) failures with the right target at groups; near = distance to region <= 0.3 m (0 inside); on a body (judge) = "
  "min_body_gap_m < 0.3; overlap = min_body_gap_m < 0; fixed = success after the depth-only snap (snapD) / with C's people (snapP)")
for method in ('fsc', 'cot'):
    tot = collections.Counter()
    for model in ['qwen38_low', 'glm53', 'dsv41', 'haiku55', 'sonnet55_low', 'gemini38_low']:
        c = collections.Counter()
        for r in rows:
            if r['model'] != model or r['method'] != method or r['sit'] != 'conversation_group':
                continue
            if r['success'] or not r['target_ok'] or r['now']['dist'] is None:
                continue
            s = r['now']; near = s['dist'] <= 0.3; body = s['gap'] is not None and s['gap'] < 0.3
            c['n'] += 1; c['near'] += near; c['near&body'] += near and body
            c['near&overlap'] += near and s['gap'] is not None and s['gap'] < 0
            if near and body:
                c['nb_fixD'] += r['snapD']['success']; c['nb_fixP'] += r['snapP']['success']
        tot.update(c)
        p(f"{method} {model:13s} n {c['n']:3d} near {c['near']:3d} near&on-body {c['near&body']:3d} near&overlap {c['near&overlap']:3d}; of near&on-body fixed snapD {c['nb_fixD']}, snapP {c['nb_fixP']}")
    p(f"{method} {'pooled':13s} n {tot['n']:3d} near {tot['near']:3d} near&on-body {tot['near&body']:3d} near&overlap {tot['near&overlap']:3d}; of near&on-body fixed snapD {tot['nb_fixD']}, snapP {tot['nb_fixP']}")
open(f'{HERE}/n1_s23.txt', 'w').write('\n'.join(out) + '\n')
