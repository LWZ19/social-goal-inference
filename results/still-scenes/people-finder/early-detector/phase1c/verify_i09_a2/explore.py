import sys, collections
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
PLAN = {
 "old": [("s19","qwen38_low",("real","staffed","plain")),("s19","glm53",("real","staffed","plain")),("s20","dsv41",("real","staffed","plain"))],
 "new": [("s21","qwen38_low",("real","staffed","plain"))]+[("s23",m,("real","staffed","plain")) for m in ("glm53","dsv41","haiku55","sonnet55_low","gemini38_low")],
 "ph": [("s19","qwen38_low",("ph","dev_seed0","dev_seed1","dev_seed2")),("s19","glm53",("ph","dev_seed0","dev_seed1","dev_seed2")),("s20","dsv41",("ph","dev_seed0","dev_seed1","dev_seed2"))],
}
for w, plan in PLAN.items():
    for st, m, arms in plan:
        for a in arms:
            f = H.runs(stage=st, arm=a, model=m, method="con")
            assert len(f)==1, (st,a,m,f)
            ref=f[0]
            tally=collections.Counter()
            recs=list(H.records(ref, tally=tally))
            ss=sum(r.solver_step for r in recs)
            print(w, st, m, a, ref.folder, str(ref.scenes_dir)[-60:], len(recs), ss, dict(tally), H.recorded_config(ref)[1].thresholds.person_radius_m, H.recorded_config(ref)[1].thresholds.person_mark_height_m)
