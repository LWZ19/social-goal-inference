"""Independent verification of M3 (I16): share of straight-line approaches, robot start to final pose, that meet a
forbidden polygon of the episode's derived truth. Through research_harness only."""
import sys, collections, math
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
from sgi.polygons import point_in_polygon, distance_to_boundary
from sgi.geometry import to_world_xy

REAL = ('real', 'staffed', 'plain')
out = []
def p(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); out.append(s)


def orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def seg_inter(p1, p2, q1, q2, eps=1e-12):
    d1, d2, d3, d4 = orient(q1, q2, p1), orient(q1, q2, p2), orient(p1, p2, q1), orient(p1, p2, q2)
    if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)):
        return True
    def on(a, b, c):
        return min(a[0], b[0]) - 1e-9 <= c[0] <= max(a[0], b[0]) + 1e-9 and min(a[1], b[1]) - 1e-9 <= c[1] <= max(a[1], b[1]) + 1e-9
    if abs(d1) <= eps and on(q1, q2, p1): return True
    if abs(d2) <= eps and on(q1, q2, p2): return True
    if abs(d3) <= eps and on(p1, p2, q1): return True
    if abs(d4) <= eps and on(p1, p2, q2): return True
    return False


def seg_meets_poly(a, b, poly):
    v = [(float(x), float(y)) for x, y in poly]
    if point_in_polygon(a, v) or point_in_polygon(b, v):
        return True
    return any(seg_inter(a, b, v[i - 1], v[i]) for i in range(len(v)))


def seg_poly_dist(a, b, poly, n=200):
    """min distance from points on the segment to the polygon (0 inside), sampled finely."""
    v = [(float(x), float(y)) for x, y in poly]
    best = math.inf
    for k in range(n + 1):
        t = k / n
        q = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        d = 0.0 if point_in_polygon(q, v) else distance_to_boundary(q, v)
        best = min(best, d)
        if best == 0.0:
            break
    return best


def runs_for(spec):
    refs = []
    for stage, model in spec:
        for arm in REAL:
            rs = H.runs(stage=stage, model=model, method='con', arm=arm)
            assert len(rs) == 1, (stage, model, arm)
            refs.append(rs[0])
    return refs


def analyse(refs, label, robot_r=0.30):
    rows = []
    recs = []
    for ref in refs:
        t = collections.Counter()
        recs += list(H.records(ref, tally=t))
        p(f'  {ref.folder}: {dict(t)}')
    recs.sort(key=lambda r: (str(r.ref.scenes_dir), r.scene_id))
    mism = 0
    for r in recs:
        pr = r.result.get('pose_robot')
        if r.result.get('status') != 'ok' or pr is None:
            rows.append(dict(model=r.model, ok=False, haspose=False, success=bool(r.score.get('success'))))
            continue
        sc = H.scene(r.ref, r.scene_id)
        robot = sc.observation.robot
        a = (robot.x, robot.y)
        b = to_world_xy(robot, pr[0], pr[1])
        pw = r.pose_world
        if pw is not None and math.dist(b, pw[:2]) > 1e-6:
            mism += 1
        regions = sc.truths[r.episode_id].gt.forbidden_regions or []
        meets = [seg_meets_poly(a, b, reg.polygon) for reg in regions]
        end_in = [point_in_polygon(b, [(float(x), float(y)) for x, y in reg.polygon]) for reg in regions]
        start_in = any(point_in_polygon(a, [(float(x), float(y)) for x, y in reg.polygon]) for reg in regions)
        rec_forb = any((r.score.get('forbidden') or {}).values())
        near = any(seg_poly_dist(a, b, reg.polygon) < robot_r for reg in regions) if regions else False
        rows.append(dict(model=r.model, ok=True, haspose=True, success=bool(r.score.get('success')),
                         nreg=len(regions), A=any(meets), endin=any(end_in), startin=start_in, recforb=rec_forb,
                         kinds=tuple(sorted({reg.kind for reg, m in zip(regions, meets) if m})), near=near,
                         source=r.result.get('source')))
    p(f'  pose_world vs pose_robot->world mismatches: {mism}')
    posed = [x for x in rows if x['haspose']]
    p(f'  {label}: records={len(rows)} with pose={len(posed)} successes with pose={sum(x["success"] for x in posed)}')
    p(f'  start inside a polygon: {sum(x["startin"] for x in posed)}; end-inside vs recorded score.forbidden '
      f'disagreements: {sum(x["endin"] != x["recforb"] for x in posed)}')
    def share(sel, key, name):
        n = len(sel); k = sum(key(x) for x in sel)
        p(f'    {name}: {k}/{n} = {100*k/max(n,1):.2f}%')
    for scope, sel in (('all C goal poses', posed), ('C successes', [x for x in posed if x['success']]),
                       ('episodes with a forbidden region', [x for x in posed if x['nreg'] > 0]),
                       ('all records (no pose = no crossing)', rows)):
        p(f'   scope {scope}:')
        share(sel, lambda x: x.get('A', False), 'Def A (closed segment meets polygon, end inside counts)')
        share(sel, lambda x: x.get('A', False) and not x['endin'], 'Def B (pass-through only, end outside every polygon)')
        share(sel, lambda x: x.get('near', False) and not x['endin'], 'Def C (segment within robot radius of polygon, end outside)')
    for m in sorted({x['model'] for x in posed}):
        sel = [x for x in posed if x['model'] == m]
        p(f'   {m}: A {sum(x["A"] for x in sel)}/{len(sel)}  B {sum(x["A"] and not x["endin"] for x in sel)}/{len(sel)}  '
          f'A on successes {sum(x["A"] for x in sel if x["success"])}/{sum(x["success"] for x in sel)}')
    p(f'   kinds crossed (A): {collections.Counter(k for x in posed if x["A"] for k in x["kinds"])}')
    p(f'   source of A records: {collections.Counter(x["source"] for x in posed if x["A"])}')
    return rows


p('== M3 primary: s19 qwen38_low, glm53 con; s20 dsv41 con; realistic ==')
analyse(runs_for([('s19', 'qwen38_low'), ('s19', 'glm53'), ('s20', 'dsv41')]), 's19/s20 C realistic')
p('\n== M3 variant: s23 C realistic, five models (glm53/dsv41 people answers are cache copies) ==')
analyse(runs_for([('s23', m) for m in ('glm53', 'dsv41', 'haiku55', 'sonnet55_low', 'gemini38_low')]), 's23 C realistic')

open('results/2026-10-17-beyond-vlm/rerun/flags_misc/verify/v_m3.txt', 'w').write('\n'.join(out) + '\n')
