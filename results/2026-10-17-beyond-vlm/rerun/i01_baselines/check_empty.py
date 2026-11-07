import sys
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
ref = H.runs(stage='s23', arm='real', model='haiku55', method='con')[0]
rec = next(r for r in H.records(ref) if r.situation == 'waiting_line' and r.solver_step)
a = H.resolve(rec, constraints=[])
b = H.resolve(rec)
print('recorded kinds', [u['constraint']['kind'] for u in rec.constraints])
print('empty: uses', a.uses, 'pose', a.pose_world, 'success', a.score.success, 'constraints in result', a.result.constraints)
print('recorded: reproduced', b.reproduced, 'success', b.score.success, 'rec', rec.score['success'])
