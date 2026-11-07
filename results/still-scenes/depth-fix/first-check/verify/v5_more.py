"""(verifier) Side checks from v2 records, the gate pooled over situations as I34 words it, gap<0.3 'on a body'
after the snap, and the depth-only snap on the old E sets (s16b Qwen/GLM, s20 DeepSeek fsc) where N1 came from."""
import sys, json, collections, io, math
import numpy as np
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
from sgi.solver import floor as c_floor
from sgi.stages.goal import _poses
V = 'results/2026-10-17-beyond-vlm/rerun/snap/verify/'
rows = [json.loads(l) for l in open(V + 'v2_snap_records.jsonl')]
OUT = io.StringIO()
def p(*a):
    print(*a); print(*a, file=OUT)
def klass(d, inreg):
    if d is None: return 'none'
    if inreg: return 'inreg'
    return 'near' if d <= 0.3 else ('mid' if d <= 1 else 'far')
mk = [r for r in rows if not r['ok'] and r['tc'] and r['source'] == 'direct_marking' and 'okD' in r]
x = [r for r in mk if r['model'] == 'dsv41' and r['sit'] == 'waiting_line' and not r['okD'] and r['gapD'] is not None and r['gapD'] < 0]
p('dsv41 lines still failing after snapD with gapD<0:', len(x), '; of them still failing after snapP with gapP<0:', sum(1 for r in x if not r['okP'] and r['gapP'] is not None and r['gapP'] < 0))
p('dsv41 lines still failing after snapP with gapP<0 (all):', sum(1 for r in mk if r['model']=='dsv41' and r['sit']=='waiting_line' and not r['okP'] and r['gapP'] is not None and r['gapP']<0))
g = [r for r in mk if r['model'] == 'gemini38_low' and r['sit'] == 'conversation_group' and not r['okP']]
p('gemini groups still failing after snapP:', len(g), collections.Counter(klass(r['dP'], (r['dP'] == 0)) for r in g))
p('\n-- gate pooled over lines and groups (as I34 words it), near = d <= 0.3 before the snap')
for who in [None, 'glm53', 'dsv41']:
    for meth in ['cot', 'fsc']:
        y = [r for r in mk if r['sit'] in ('waiting_line', 'conversation_group') and r['method'] == meth and (who is None or r['model'] == who) and r['d'] <= 0.3]
        p(f'  {who or "pooled6"} {meth}: near {len(y)} fixedD {sum(r["okD"] for r in y)} ({sum(r["okD"] for r in y)/len(y):.0%}) fixedP {sum(r["okP"] for r in y)} ({sum(r["okP"] for r in y)/len(y):.0%})')
p('\n-- still failing after snapD, on a body under N1 definition (gap<0.3) vs strict (gap<0), lines')
for m in ['qwen38_low', 'glm53', 'dsv41', 'haiku55', 'sonnet55_low', 'gemini38_low']:
    y = [r for r in mk if r['model'] == m and r['sit'] == 'waiting_line' and not r['okD']]
    p(f'  {m}: {len(y)} gap<0 {sum(1 for r in y if r["gapD"] < 0)} gap<0.3 {sum(1 for r in y if r["gapD"] < 0.3)}')

p('\n-- depth-only snap on the old E sets (s16b/s20 fsc, realistic arms): E failures with the right target, marked')
FL = {}
for model, stage in [('qwen38_low', 's16b'), ('glm53', 's16b'), ('dsv41', 's20')]:
    c = collections.Counter()
    for arm in ['real', 'staffed', 'plain']:
        (ref,) = H.runs(stage=stage, arm=arm, model=model, method='fsc')
        th = H.recorded_config(ref)[1].thresholds
        for rec in H.records(ref):
            s = rec.score
            if rec.result.get('source') != 'direct_marking' or rec.result.get('status') != 'ok' or rec.pose_world is None: continue
            sc = H.scene(ref, rec.scene_id)
            key = (str(ref.scenes_dir), rec.scene_id)
            if key not in FL: FL[key] = c_floor(sc.observation, [], th)
            fl = FL[key]
            x_, y_, _ = rec.pose_world
            cell = fl.index(x_, y_)
            if cell is not None and fl.selectable[cell]:
                sx, sy = x_, y_
            else:
                cen = fl.centres()[fl.selectable]; d = np.hypot(cen[:, 0] - x_, cen[:, 1] - y_); i = int(np.argmin(d)); sx, sy = cen[i]
            pw, _, _ = _poses(sc.observation.robot, (float(sx), float(sy)), rec.target['world'][:2], th)
            ok2 = H.score_pose(rec, pw).success
            if s.get('success'):
                c['succ'] += 1; c['broken'] += not ok2; continue
            if not s.get('target_correct'): continue
            sit = rec.situation; d0 = s.get('distance_to_region_m')
            c[(sit, 'n')] += 1; c[(sit, 'fixed')] += ok2
            if d0 is not None and d0 <= 0.3:
                c[(sit, 'near')] += 1; c[(sit, 'near_fixed')] += ok2
                g_ = s.get('min_body_gap_m')
                if g_ is not None and g_ < 0.3:
                    c[(sit, 'near&body')] += 1; c[(sit, 'near&body_fixed')] += ok2
            if d0 is not None and d0 > 1: c[(sit, 'far')] += 1
    p(f'  {model} {stage}: ' + ', '.join(f'{k}: {v}' for k, v in sorted(c.items(), key=str)))
open(V + 'v5_more.txt', 'w').write(OUT.getvalue())
