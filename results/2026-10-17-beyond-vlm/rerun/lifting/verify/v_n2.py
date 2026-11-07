import sys, collections
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
D = 'results/2026-10-17-beyond-vlm/rerun/lifting/verify/'
lines = []
def p(*a):
    s = ' '.join(str(x) for x in a); print(s); lines.append(s)
pairs = [('qwen38_low', 's19', 's16b', 'fsc'), ('glm53', 's19', 's16b', 'fsc'), ('dsv41', 's20', 's20', 'fsc')]
for m, cst, est, emeth in pairs:
    same = diff = unpaired = 0
    wrongC = wrongE = 0; wrong_both = 0
    cached = collections.Counter()
    for arm in ('real', 'staffed', 'plain'):
        C = {(r.scene_id, r.episode_id, r.trial): r for run in H.runs(stage=cst, arm=arm, model=m, method='con') for r in H.records(run)}
        E = {(r.scene_id, r.episode_id, r.trial): r for run in H.runs(stage=est, arm=arm, model=m, method=emeth) for r in H.records(run)}
        for k, c in C.items():
            e = E.get(k)
            if e is None:
                unpaired += 1; continue
            pc = None if c.target is None else tuple(c.target['pixel'])
            pe = None if e.target is None else tuple(e.target['pixel'])
            if pc == pe: same += 1
            else: diff += 1
            cw = not c.score['target_correct']; ew = not e.score['target_correct']
            wrongC += cw; wrongE += ew; wrong_both += cw and ew
            if cw:
                for call in c.result.get('calls') or []:
                    if call['stage'] in ('plan', 'locate'):
                        cached[(call['stage'], bool(call.get('cached')))] += 1
        p(m, arm, 'C runs', [x.name for x in H.runs(stage=cst, arm=arm, model=m, method='con')], 'E runs', [x.name for x in H.runs(stage=est, arm=arm, model=m, method=emeth)])
    p(m, 'paired', same + diff, 'same pixel', same, 'different', diff, 'unpaired', unpaired, '| C wrong', wrongC, 'E wrong', wrongE, 'both', wrong_both, '| C-wrong plan/locate calls cached', dict(cached))
open(D + 'v_n2.txt', 'w').write('\n'.join(lines) + '\n')
