import json, collections
D='results/still-scenes/replay-harness/calibration/'
mine = {(r['model'], r['scene'], r['episode'], r['trial']): r for r in map(json.loads, open(D+'verify/v1_rows.jsonl'))}
theirs = {(r['model'], r['scene'], r['episode'], r['trial']): r for r in map(json.loads, open(D+'rows2.jsonl'))}
assert set(mine) == set(theirs)
c = collections.Counter()
lines=[]
for k, m in mine.items():
    t = theirs[k]
    c['orc_now'] += (m['res_o'] if m['step_o'] else m['now_o']) == t['orc_now']
    c['step'] += m['step_v'] == t['solver_step']
    if m['step_v']:
        c['n_step'] += 1
        for a in ['R1','R2','R3','R4','P','Ps','P1m']:
            c[a] += m['res_'+a] == t[a]
        c['C'] += m['res_v'] == t['C_replay_ok']
        c['amb'] += m['amb'] == sum(q['ambiguous'] for q in t['pairs'])
        c['P_eq_Ps_mine'] += m['res_P'] == m['res_Ps']
lines.append(str(dict(c)))
open(D+'verify/v4_cross.txt','w').write('\n'.join(lines)+'\n'); print('\n'.join(lines))
