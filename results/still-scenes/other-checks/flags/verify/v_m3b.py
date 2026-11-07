"""M3 variants and N4 check: candidates of the goal set, GLM+DeepSeek only, raw scene forbidden lists, and R/E."""
import sys, collections, math, re
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
from sgi.polygons import point_in_polygon
from sgi.geometry import to_world_xy

src = open(__file__.replace('v_m3b.py', 'v_m3.py')).read()
exec(src.split("p('== M3 primary")[0])          # helpers only: orient, seg_inter, seg_meets_poly, runs_for, p, out
out.clear()

def P(poly): return [(float(x), float(y)) for x, y in poly]

def rows_for(refs, use_candidates=False, raw=False):
    recs = []
    for ref in refs:
        recs += list(H.records(ref))
    recs.sort(key=lambda r: (str(r.ref.scenes_dir), r.scene_id))
    rows = []
    for r in recs:
        pr = r.result.get('pose_robot')
        if r.result.get('status') != 'ok' or pr is None:
            continue
        sc = H.scene(r.ref, r.scene_id)
        robot = sc.observation.robot
        a = (robot.x, robot.y)
        if raw:
            ep = sc.truth.episodes[r.episode_id]
            regions = ep.forbidden_regions or []
        else:
            regions = sc.truths[r.episode_id].gt.forbidden_regions or []
        ends = [to_world_xy(robot, pr[0], pr[1])]
        if use_candidates:
            ends = [tuple(c['pose_world'][:2]) for c in (r.candidates or [])]
        for b in ends:
            A = any(seg_meets_poly(a, b, reg.polygon) for reg in regions)
            endin = any(point_in_polygon(b, P(reg.polygon)) for reg in regions)
            rows.append(dict(model=r.model, method=r.method, success=bool(r.score.get('success')), A=A, endin=endin))
    return rows

def show(rows, label):
    n = len(rows); a = sum(x['A'] for x in rows); b = sum(x['A'] and not x['endin'] for x in rows); e = sum(x['endin'] for x in rows)
    p(f'  {label}: n={n} A={a} ({100*a/max(n,1):.2f}%) B={b} ({100*b/max(n,1):.2f}%) end-inside={e}')

C = runs_for([('s19', 'qwen38_low'), ('s19', 'glm53'), ('s20', 'dsv41')])
p('== M3 variants on s19/s20 C realistic ==')
rows = rows_for(C)
show([x for x in rows if x['model'] in ('glm53', 'dsv41')], 'GLM + DeepSeek only')
show(rows_for(C, use_candidates=True), 'every candidate of the goal set (result.candidates), solver steps')
try:
    show(rows_for(C, raw=True), 'raw scene forbidden lists (not derived)')
except Exception as exc:
    p(f'  raw scene lists: not available ({exc!r})')

p('\n== N4: R and E, realistic, 3 models ==')
for meth in ('cot', 'fsc'):
    refs = []
    for stage, model in (('s16b', 'qwen38_low'), ('s16b', 'glm53'), ('s20', 'dsv41')):
        refs += [x for x in H.runs(stage=stage, model=model, method=meth) if x.arm in ('real', 'staffed', 'plain')]
    rr = rows_for(refs)
    show(rr, f'{meth} pooled')
    for m in ('qwen38_low', 'glm53', 'dsv41'):
        show([x for x in rr if x['model'] == m], f'  {meth} {m}')
open(__file__.replace('.py', '.txt'), 'w').write('\n'.join(out) + '\n')
