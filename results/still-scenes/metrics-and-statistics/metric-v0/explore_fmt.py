"""Format check only (no metric is computed): situations, candidate counts, region tolerances, forbidden kinds."""
import sys, collections
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
ncand = collections.Counter(); sits = collections.Counter(); tol = collections.Counter(); forb = collections.Counter(); nreg = collections.Counter()
heads = collections.Counter()
for st, model in (('s20', 'dsv41'), ('s19', 'glm53')):
    for r in H.runs(stage=st, model=model, method='con'):
        if r.arm == 'ph':
            continue
        for rec in H.records(r):
            if not rec.solver_step:
                continue
            ncand[len(rec.candidates or [])] += 1
            sits[(r.arm[:3], rec.situation)] += 1
            sc = H.scene(rec.ref, rec.scene_id)
            gt = sc.truths[rec.episode_id].gt
            nreg[(rec.situation, len(gt.acceptable_regions or []))] += 1
            for g in gt.acceptable_regions or []:
                tol[(rec.situation, g.tolerance_deg)] += 1
                heads[(rec.situation, g.heading.yaw is None, g.target_id is None)] += 1
            for g in gt.forbidden_regions or []:
                forb[(rec.situation, g.kind)] += 1
print('ncand', ncand); print('sits', sits); print('tol', tol); print('nreg', nreg); print('forb', forb); print('heads (sit, face_point?, target_id None?)', heads)
