"""Exploration (no scoring): list the fit runs and their record counts through the harness."""
import sys, collections
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
PLAN = [("s19", "qwen38_low"), ("s19", "glm53"), ("s20", "dsv41")]
ARMS = ("real", "staffed", "plain", "ph", "dev_seed0", "dev_seed1", "dev_seed2")
for stage, model in PLAN:
    for arm in ARMS:
        refs = H.runs(stage=stage, arm=arm, model=model, method="con")
        assert len(refs) == 1, (stage, arm, model, refs)
        ref = refs[0]
        tally = collections.Counter()
        recs = list(H.records(ref, tally=tally))
        ss = [r for r in recs if r.solver_step]
        th = H.recorded_config(ref)[1].thresholds
        tgt = collections.Counter()
        sits = collections.Counter(r.situation for r in recs)
        print(ref.folder, ref.scenes_dir.name, ref.scenes_dir.parent.name, len(recs), len(ss), dict(tally), dict(sits),
              th.person_radius_m, th.person_mark_height_m, th.visibility_min_samples, th.max_people)
