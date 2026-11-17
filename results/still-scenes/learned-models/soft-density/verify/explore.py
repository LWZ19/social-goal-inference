import sys, collections
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
CELLS = [('new_qwen','s21','qwen38_low'),('new_glm','s23','glm53'),('new_deepseek','s23','dsv41'),('new_haiku','s23','haiku55'),
         ('new_sonnet','s23','sonnet55_low'),('new_gemini','s23','gemini38_low'),
         ('old_qwen','s19','qwen38_low'),('old_glm','s19','glm53'),('old_deepseek','s20','dsv41')]
for lab, st, m in CELLS[6:7]:
    for arm in ('real','staffed','plain'):
        refs = H.runs(stage=st, arm=arm, model=m, method='con')
        print(lab, arm, [r.folder for r in refs])
        for rec in H.records(refs[0]):
            if rec.solver_step:
                print(rec.scene_id, rec.episode_id, rec.trial, rec.situation, list(rec.result.keys()))
                print(rec.solver, rec.result.get('flags'))
                print(rec.candidates[0])
                break
for arm in ('dev_seed0',):
    refs = H.runs(stage='s19', arm=arm, model='glm53', method='con')
    print([r.folder for r in refs])
    c = collections.Counter()
    for rec in H.records(refs[0]):
        c[(rec.scene_id, rec.situation, rec.solver_step)] += 1
    print(c)
