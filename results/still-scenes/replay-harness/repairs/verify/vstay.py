import json
for k in ('old_glm53', 'old_dsv41', 'new_glm53', 'new_dsv41'):
    v = [json.loads(x) for x in open(f'results/still-scenes/replay-harness/repairs/verify/vrows_{k}.jsonl')]
    lf = [r for r in v if r['solver_step'] and r['situation'] == 'waiting_line' and r['target_correct'] and not r['C'] and (not r['in_region'] or not r['reachable'])]
    stay = [r for r in lf if not r['P']]
    print(k, 'LF', len(lf), 'P stays', len(stay), 'R5 fixes', sum(r['R5'] for r in stay), 'P_1m fixes', sum(r['P_1m'] for r in stay),
          '| R6 flips on goal-step wrong targets', sum(r['R6'] for r in v if r['solver_step'] and not r['target_correct']), 'of', sum(1 for r in v if r['solver_step'] and not r['target_correct']))
