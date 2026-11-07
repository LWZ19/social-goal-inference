"""Per-record comparison of my vrows_<set>.jsonl with the compute agent's rows_<set>.jsonl (their outputs, read as
data; none of their code is imported)."""
import collections
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
THEIRS = OUT.parent
CLS = {'choice error': 'choice', 'occluder capture': 'occluder', 'background leak': 'background',
       'floor slip': 'floor slip', 'near-depth miss': 'near-depth'}


def key(r):
    return (r['run'], r['scene'], r['episode'], r['trial'])


def main(name):
    mine = {key(r): r for r in map(json.loads, open(OUT / f'vrows_{name}.jsonl'))}
    theirs = {key(r): r for r in map(json.loads, open(THEIRS / f'rows_{name}.jsonl'))}
    print(f'{name}: mine {len(mine)} theirs {len(theirs)}; only mine {len(set(mine) - set(theirs))}, only theirs '
          f'{len(set(theirs) - set(mine))}')
    diff = collections.Counter()
    examples = collections.defaultdict(list)
    for k in sorted(set(mine) & set(theirs)):
        a, b = mine[k], theirs[k]
        if a['has_point'] != b['has_point']:
            diff['has_point'] += 1
            continue
        if not a['has_point']:
            continue
        checks = [('right', a['right'], b['right_before']),
                  ('s_G1', a['s_G1'], b['s_cls']), ('s_G2', a['s_G2'], b.get('s_cls_G2')),
                  ('T1', a['T1_G1'], b['T1']), ('T2', a['T2_G1'], b['T2']), ('T1_G2', a['T1_G2'], b.get('T1_G2')),
                  ('nlab', a['nlab_G1'], b.get('win_labels'))]
        if not a['right']:
            checks.append(('lclass', a['lclass'], CLS.get(b.get('lift_class'), b.get('lift_class'))))
        for f in ('oracle', 'V1', 'V2', 'V3', 'V4', 'front'):
            fa, fb = a['fx'][f], b['fx'][f]
            checks.append((f + '.moved', fa['moved'], fb['moved']))
            checks.append((f + '.right_after', fa['right_after'], fb['right_after']))
            checks.append((f + '.succ_after', fa['succ_after'], fb.get('succ_after')))
            if fa['moved'] and fb['moved'] and 'world' in fb:
                d = max(abs(x - y) for x, y in zip(fa['world'], fb['world']))
                checks.append((f + '.world_close', d < 1e-9, True))
        for name_, x, y in checks:
            if x != y:
                diff[name_] += 1
                if len(examples[name_]) < 4:
                    examples[name_].append((k, x, y))
    print('  differences:', dict(diff) if diff else 'none')
    for n, ex in examples.items():
        for e in ex:
            print('   ', n, e)


if __name__ == '__main__':
    for n in sys.argv[1:]:
        main(n)
