"""Record-by-record comparison of the verifier's vrows_*.jsonl with the first agent's rows_*.jsonl, plus clustering
counts (distinct people answers and layouts behind the flips). Prints to stdout."""
import collections
import json
import os

D = 'results/2026-10-17-beyond-vlm/phase1/i01_repairs'
ORDER = ['old_qwen38_low', 'old_glm53', 'old_dsv41', 'new_qwen38_low', 'new_glm53', 'new_dsv41', 'new_haiku55',
         'new_sonnet55_low', 'new_gemini38_low']
ARMS = [('replayC', 'C_replay_ok'), ('R0', 'R0'), ('R1', 'R1'), ('R2', 'R2'), ('R3', 'R3'), ('R4', 'R4'),
        ('R4s', 'R4s'), ('P', 'P'), ('Ps', 'Ps'), ('R5', 'R5'), ('R6', 'R6'), ('R6x', 'R6x'), ('P_1m', 'P1m'),
        ('R3_1m', 'R3_1m')]


def key(r):
    return (r['arm'], r['scene'], r['episode'], r['trial'])


for k in ORDER:
    vp, fp = f'{D}/verify/vrows_{k}.jsonl', f'{D}/rows_{k}.jsonl'
    if not (os.path.exists(vp) and os.path.exists(fp)):
        continue
    v = {key(r): r for r in map(json.loads, open(vp))}
    f = {key(r): r for r in map(json.loads, open(fp))}
    assert set(v) == set(f), k
    dis = collections.Counter()
    ex = collections.defaultdict(list)
    for kk, r in v.items():
        g = f[kk]
        if r['C'] != g['C_rec'] or r['solver_step'] != g['solver_step']:
            dis['C/solver_step'] += 1
        if r['solver_step']:
            lf_v = r['situation'] == 'waiting_line' and r['target_correct'] and not r['C'] and (not r['in_region'] or not r['reachable'])
            if lf_v != g['line_failure']:
                dis['line_failure'] += 1
        for a, b in ARMS:
            if a in ('replayC',) and not r['solver_step']:
                continue
            if a not in r:
                continue
            if bool(r[a]) != bool(g[b]):
                dis[a] += 1
                ex[a].append(kk)
    n_ss = sum(r['solver_step'] for r in v.values())
    print(f'{k}: records {len(v)}, solver steps {n_ss}; disagreements: {dict(dis) or 0}')
    for a, lst in ex.items():
        print('   ', a, lst[:5])

# clustering of flips on the pre-registered sets
for k in ('old_glm53', 'old_dsv41'):
    v = [json.loads(x) for x in open(f'{D}/verify/vrows_{k}.jsonl')]
    lf = [r for r in v if r['solver_step'] and r['situation'] == 'waiting_line' and r['target_correct'] and not r['C']
          and (not r['in_region'] or not r['reachable'])]
    for a in ('P', 'R4', 'R3', 'R5'):
        fl = [r for r in lf if r[a]]
        answers = {(r['arm'], r['scene'], r['trial']) for r in fl}
        layouts = {r['scene'].split('_eval_')[1][:2] for r in fl}
        print(f'{k} {a}: flips {len(fl)} of {len(lf)}; distinct people answers (arm, scene, trial) {len(answers)}; '
              f'layouts {sorted(layouts)}')
    answers_all = {(r['arm'], r['scene'], r['trial']) for r in lf}
    print(f'{k}: line failures {len(lf)} come from {len(answers_all)} distinct (arm, scene, trial) people answers; '
          f'layouts {sorted({r["scene"].split("_eval_")[1][:2] for r in lf})}')
