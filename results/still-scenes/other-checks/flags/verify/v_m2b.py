"""M2 nuance: of the standard-goal misses with standard_goal_frontal False, how many had no surface normal (fit failed)
vs a normal but no clear in-view step before it, vs the robot already inside the standoff. Current code on the recorded
target; harness only."""
import sys, math, collections
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
from sgi.geometry import surface_normal_xy
out = []
def p(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)
for stage, model in (('s19', 'qwen38_low'), ('s19', 'glm53'), ('s20', 'dsv41')):
    c = collections.Counter()
    for arm in ('real', 'staffed', 'plain'):
        ref = H.runs(stage=stage, model=model, method='con', arm=arm)[0]
        th = H.recorded_config(ref)[1].thresholds
        for r in H.records(ref):
            if r.score.get('success') or not r.score.get('target_correct') or r.result.get('source') != 'standard' or r.situation != 'no_one':
                continue
            fr = bool(r.result['flags'].get('standard_goal_frontal'))
            sc = H.scene(ref, r.scene_id); obs = sc.observation
            w = r.result['target']['world']
            dist = math.hypot(obs.robot.x - w[0], obs.robot.y - w[1])
            if dist < th.standoff_m:
                kind = 'inside standoff'
            else:
                n = surface_normal_xy(obs.depth, obs.camera, tuple(w), th.surface_radius_m, th.surface_min_points,
                                      th.surface_max_tilt, th.surface_max_rms_m, th.surface_min_spread_m, th.surface_max_rms_ratio)
                kind = 'no normal (fit failed)' if n is None else 'normal found, no clear in-view step'
            c[(fr, kind if not fr else 'frontal')] += 1
    p(model, dict(c))
open(__file__.replace('.py', '.txt'), 'w').write('\n'.join(out) + '\n')
