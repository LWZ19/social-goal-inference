"""Are the C (s19/s20 con) and E (s16b/s20 fsc) wrong-target points the same points (shared, cached plan call)?"""
import sys, collections
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
lines = []
for (sc_, mc), (se, me) in [(('s19', 'qwen38_low'), ('s16b', 'qwen38_low')), (('s19', 'glm53'), ('s16b', 'glm53')), (('s20', 'dsv41'), ('s20', 'dsv41'))]:
    c = collections.Counter()
    for arm in ('real', 'staffed', 'plain'):
        C = {(r.scene_id, r.episode_id, r.trial): r for r in H.records(H.runs(stage=sc_, arm=arm, model=mc, method='con')[0])}
        E = {(r.scene_id, r.episode_id, r.trial): r for r in H.records(H.runs(stage=se, arm=arm, model=me, method='fsc')[0])}
        for k, rc in C.items():
            re_ = E.get(k)
            if re_ is None:
                c['no E twin'] += 1
                continue
            pc = None if rc.target is None else tuple(rc.target['pixel'])
            pe = None if re_.target is None else tuple(re_.target['pixel'])
            wrong = not rc.score['target_correct']
            c[('C wrong' if wrong else 'C right', 'same point' if pc == pe else 'different point')] += 1
            if wrong:
                plan_c = [x for x in rc.result.get('calls', []) if x['stage'] in ('plan', 'locate')]
                c[('C wrong', 'C plan/locate calls cached', tuple(bool(x.get('cached')) for x in plan_c))] += 1
    lines.append(f"{mc}: {dict(c)}")
txt = '\n'.join(lines)
print(txt)
open('results/2026-10-17-beyond-vlm/rerun/lifting/same_points_out.txt', 'w').write(txt + '\n')
