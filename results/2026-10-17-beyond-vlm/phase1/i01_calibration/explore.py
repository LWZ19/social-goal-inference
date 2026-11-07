import sys, collections, json
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
refs = H.runs(stage='s8')
for r in refs:
    doc, cfg = H.recorded_config(r)
    run = cfg.run
    print(r.folder, r.model, r.method, 'goal=', run.goal, 'people=', run.people, 'constraints=', getattr(run,'constraints',None),
          'context=', run.context_check, 'grounding=', run.grounding, 'scenes=', r.scenes_dir.name)
    t = collections.Counter()
    recs = list(H.records(r, tally=t))
    cfgs = collections.Counter(json.dumps({k: v for k, v in rec.raw['config'].items() if k != 'stages'}, sort_keys=True) for rec in recs)
    print('   tally', dict(t), 'record configs', dict(cfgs))
    sits = collections.Counter(rec.situation for rec in recs)
    print('   situations', dict(sits), 'success', sum(bool(rec.score.get('success')) for rec in recs), 'solver steps', sum(rec.solver_step for rec in recs))
