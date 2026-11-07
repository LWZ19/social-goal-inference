import sys, json, collections
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
SIX = ('no_usable_constraint','constraint_unusable','constraints_cut','exclusion_broken','people_cut','far_from_target')
for stage,model in [('s20','dsv41'),('s23','dsv41'),('s19','glm53'),('s19','qwen38_low')]:
    for run in H.runs(stage=stage, model=model, method='con'):
        t=collections.Counter(); recs=list(H.records(run, tally=t))
        s=sum(1 for r in recs if r.score['success'] and any(r.result['flags'].get(f) for f in SIX))
        f=sum(1 for r in recs if not r.score['success'] and any(r.result['flags'].get(f) for f in SIX))
        print(stage, run.arm, model, dict(t), 'succ', sum(bool(r.score['success']) for r in recs), 'SIX flagged succ/fail', s, f)
