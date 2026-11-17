"""Re-computation of three judge-computed numbers of the beyond-VLM report (flags_misc):

  M1 (I48)  solver flags on DeepSeek C successes vs failures (s20 dsv41 con, realistic arms), result.flags.
  M2 (I47)  standard-goal misses under C and how many are the fallback (standard_goal_frontal False), s19/s20.
  M3 (I16)  share of straight-line approaches (robot start -> final pose) that cross a forbidden polygon of the
            derived truth, over C records of s19/s20 realistic (R/E given as extra rows).

Reads only through research_harness. No model or network call. Writes only beside this file.
"""
import collections
import math
import sys
from pathlib import Path

sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H  # noqa: E402

OUT = Path(__file__).resolve().parent
ARMS = ('real', 'staffed', 'plain')
C_RUNS = {'qwen': ('s19', 'qwen38_low'), 'glm': ('s19', 'glm53'), 'ds': ('s20', 'dsv41')}
RE_RUNS = {'qwen': ('s16b', 'qwen38_low'), 'glm': ('s16b', 'glm53'), 'ds': ('s20', 'dsv41')}
SIX = ('no_usable_constraint', 'constraint_unusable', 'constraints_cut', 'exclusion_broken', 'people_cut',
       'far_from_target')
OTHER = ('marked_point_off_floor', 'standard_goal_blocked', 'heading_fallback', 'heading_degenerate')

lines = []


def out(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    lines.append(s)


def one_run(stage, model, arm, method):
    rs = H.runs(stage=stage, arm=arm, model=model, method=method)
    assert len(rs) == 1, (stage, model, arm, method, rs)
    return rs[0]


# ------------------------------------------------------------------ geometry helpers (M3)
def _orient(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _seg_seg_intersect(p1, p2, q1, q2, eps=1e-12):
    d1 = _orient(*q1, *q2, *p1)
    d2 = _orient(*q1, *q2, *p2)
    d3 = _orient(*p1, *p2, *q1)
    d4 = _orient(*p1, *p2, *q2)
    if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)):
        return True

    def on(a, b, c):  # c on segment ab, given collinear
        return min(a[0], b[0]) - 1e-9 <= c[0] <= max(a[0], b[0]) + 1e-9 and min(a[1], b[1]) - 1e-9 <= c[1] <= max(a[1], b[1]) + 1e-9
    if abs(d1) <= eps and on(q1, q2, p1):
        return True
    if abs(d2) <= eps and on(q1, q2, p2):
        return True
    if abs(d3) <= eps and on(p1, p2, q1):
        return True
    if abs(d4) <= eps and on(p1, p2, q2):
        return True
    return False


def _pt_seg(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def seg_poly(a, b, poly):
    """(intersects, end_inside, distance) of segment a-b against a polygon (boundary counts as inside)."""
    from sgi.polygons import point_in_polygon
    v = [(float(p[0]), float(p[1])) for p in poly]
    a_in, b_in = point_in_polygon(a, v), point_in_polygon(b, v)
    hit = a_in or b_in
    dist = 0.0 if hit else math.inf
    for i in range(len(v)):
        q1, q2 = v[i - 1], v[i]
        if not hit and _seg_seg_intersect(a, b, q1, q2):
            hit = True
            dist = 0.0
        if dist > 0:
            dist = min(dist, _pt_seg(*a, *q1, *q2), _pt_seg(*b, *q1, *q2), _pt_seg(*q1, *a, *b), _pt_seg(*q2, *a, *b))
    return hit, b_in, (0.0 if hit else dist)


# ------------------------------------------------------------------ M1
def m1():
    out('=' * 100)
    out('M1 (I48): records with a flag set, by recorded success; C realistic (real+staffed+plain)')
    out('  SIX = the six flags of the constraint method (types.py:90-97):', ', '.join(SIX))
    out('  WARN = SIX + ', ', '.join(OTHER), '(every flag except standard_goal_frontal, which True means the fit worked)')
    for key in ('ds', 'qwen', 'glm'):
        stage, model = C_RUNS[key]
        recs = [r for arm in ARMS for r in H.records(one_run(stage, model, arm, 'con'))]
        out('-' * 100)
        out(f'{key} ({stage} {model} con): n={len(recs)}  successes={sum(bool(r.score["success"]) for r in recs)}')
        for scope_name, keep in (('all records', lambda r: True),
                                 ('solver steps (source=constraints)', lambda r: r.result.get('source') == 'constraints'),
                                 ('H.solver_step', lambda r: r.solver_step)):
            sub = [r for r in recs if keep(r)]
            ns = sum(bool(r.score['success']) for r in sub)
            nf = len(sub) - ns
            row = [f'  [{scope_name}] n={len(sub)} succ={ns} fail={nf}']
            for set_name, fl in (('SIX', SIX), ('WARN', SIX + OTHER)):
                s = sum(1 for r in sub if r.score['success'] and any((r.result.get('flags') or {}).get(f) for f in fl))
                f_ = sum(1 for r in sub if not r.score['success'] and any((r.result.get('flags') or {}).get(f) for f in fl))
                row.append(f'{set_name}: flagged succ {s} / fail {f_}')
            # every truthy flag including frontal
            s = sum(1 for r in sub if r.score['success'] and any((r.result.get('flags') or {}).values()))
            f_ = sum(1 for r in sub if not r.score['success'] and any((r.result.get('flags') or {}).values()))
            row.append(f'ANY(incl. frontal): succ {s} / fail {f_}')
            out(' | '.join(row))
            # per-flag
            per = collections.Counter()
            for r in sub:
                for f, v in (r.result.get('flags') or {}).items():
                    if v:
                        per[(f, bool(r.score['success']))] += 1
            names = sorted({f for f, _ in per})
            out('      per flag (succ/fail): ' + '; '.join(f'{f} {per[(f, True)]}/{per[(f, False)]}' for f in names))
            # flag rates
            fl_s = sum(1 for r in sub if r.score['success'] and any((r.result.get('flags') or {}).get(f) for f in SIX))
            fl_f = sum(1 for r in sub if not r.score['success'] and any((r.result.get('flags') or {}).get(f) for f in SIX))
            if ns and nf:
                out(f'      SIX rate: {fl_s}/{ns}={fl_s / ns:.1%} of successes vs {fl_f}/{nf}={fl_f / nf:.1%} of failures;'
                    f' precision for failure {fl_f}/{fl_s + fl_f}={(fl_f / (fl_s + fl_f) if fl_s + fl_f else float("nan")):.1%}')


# ------------------------------------------------------------------ M2 + M3 (scenes)
def classify(r):
    """The brief section 3.5 order: success, wrong target, no target, standard goal, solver unreachable,
    solver outside region, heading."""
    s, res = r.score, r.result
    if s['success']:
        return 'success'
    if res.get('target') is None or res.get('status') == 'no_target':
        return 'no target marked'
    if not s['target_correct']:
        return 'wrong target'
    if res.get('source') == 'standard':
        return 'standard: nobody near' if r.situation == 'no_one' else 'standard: people matter (context missed)'
    if not s['reachable']:
        return 'solver: not reachable'
    if not s['in_region']:
        return 'solver: outside region'
    return 'solver: heading only'


def m2():
    out('=' * 100)
    out('M2 (I47): C realistic failure classes (brief 3.5 order) and the standard-goal fallback among standard-goal misses')
    out('  standard-goal miss = recorded failure, target correct, result.source == "standard"; fallback = '
        'flags.standard_goal_frontal False')
    from sgi.stages.goal import standard_goal
    from sgi.types import GroundedTarget
    for key in ('qwen', 'glm', 'ds'):
        stage, model = C_RUNS[key]
        cls = collections.Counter()
        misses = []
        std_all = collections.Counter()
        for arm in ARMS:
            run = one_run(stage, model, arm, 'con')
            th = H.recorded_config(run)[1].thresholds
            for r in H.records(run):
                c = classify(r)
                cls[c] += 1
                if r.result.get('source') == 'standard' and r.score['target_correct']:
                    fr = bool((r.result.get('flags') or {}).get('standard_goal_frontal'))
                    std_all[(r.situation == 'no_one', bool(r.score['success']), fr)] += 1
                if c.startswith('standard'):
                    fl = r.result.get('flags') or {}
                    sc = H.scene(run, r.scene_id)
                    # the current code's standard goal on the recorded target: does the flag reproduce?
                    g = standard_goal(sc.observation, GroundedTarget.model_validate(r.result['target']), th)
                    moved = math.dist(g.pose_world[:2], r.pose_world[:2])
                    now = H.score_pose(r, r.pose_world)
                    misses.append(dict(arm=arm, scene=r.scene_id, ep=r.episode_id, trial=r.trial, sit=r.situation,
                                       frontal=bool(fl.get('standard_goal_frontal')),
                                       blocked=bool(fl.get('standard_goal_blocked')),
                                       in_region=r.score['in_region'], reachable=r.score['reachable'],
                                       heading_ok=r.score['heading_ok'], d=r.score.get('distance_to_region_m'),
                                       frontal_now=bool(g.flags.standard_goal_frontal), moved_now=moved,
                                       success_now=bool(now.success), view=(r.labels.get('factors') or {}).get('view_angle_deg')))
        out('-' * 100)
        out(f'{key} ({stage} {model} con) n={sum(cls.values())} classes: ' + '; '.join(f'{k} {v}' for k, v in sorted(cls.items())))
        nn = [m for m in misses if m['sit'] == 'no_one']
        pm = [m for m in misses if m['sit'] != 'no_one']
        for name, sub in (('nobody near', nn), ('people matter', pm), ('all standard misses', misses)):
            out(f'  {name}: n={len(sub)}  fallback (frontal False)={sum(not m["frontal"] for m in sub)}'
                f'  frontal True={sum(m["frontal"] for m in sub)}  blocked={sum(m["blocked"] for m in sub)}'
                f'  outside region={sum(not m["in_region"] for m in sub)}  in region but heading/reach fail='
                f'{sum(m["in_region"] for m in sub)}')
        out(f'  check with the current code: flag reproduces in {sum(m["frontal"] == m["frontal_now"] for m in misses)}/{len(misses)},'
            f' pose within 1 cm in {sum(m["moved_now"] < 0.01 for m in misses)}/{len(misses)},'
            f' recorded pose fails when scored now in {sum(not m["success_now"] for m in misses)}/{len(misses)}')
        out('  standard-goal records with the right target, (nobody near, success, frontal): ' +
            '; '.join(f'{k} {v}' for k, v in sorted(std_all.items())))
        by_scene = collections.Counter((m['scene'], m['frontal']) for m in misses)
        out('  misses by (scene, frontal): ' + '; '.join(f'{k[0]}/{"frontal" if k[1] else "fallback"} {v}' for k, v in sorted(by_scene.items())))
        for m in misses:
            out(f'    {m["arm"]:8s} {m["scene"]:28s} {m["ep"]:4s} t{m["trial"]} {m["sit"]:18s} frontal={m["frontal"]!s:5s}'
                f' blocked={m["blocked"]!s:5s} in_region={m["in_region"]!s:5s} reach={m["reachable"]!s:5s}'
                f' heading={m["heading_ok"]!s:5s} d={m["d"] if m["d"] is None else round(m["d"], 2)}'
                f' view={None if m["view"] is None else round(m["view"], 1)} frontal_now={m["frontal_now"]} moved_now={m["moved_now"]:.3f}')


def m3():
    out('=' * 100)
    out('M3 (I16): straight segment from the robot start (scene observation.robot x, y) to the recorded final pose '
        '(result.pose_world x, y),')
    out('  tested against every forbidden polygon of the episode (derived truth: H.scene(...).truths[ep].gt.forbidden_regions;'
        ' the scorer uses the same list).')
    out('  def A: the closed segment meets a polygon (ending inside counts).  def B: the segment meets a polygon while the goal is outside'
        ' every polygon (a pass-through; the start is never inside one).')
    out('  def C: the robot body swept along the segment (distance from segment to polygon < robot_radius_m) meets a polygon'
        ' while the goal is outside every polygon.')
    out('  denominators: all records with a pose; successes; records of episodes that have a forbidden region.')
    rows = {}
    kinds_hit = collections.Counter()
    examples = []
    for arm in ARMS:                                # arm outermost so one arm's scenes stay cached
        for key in ('qwen', 'glm', 'ds'):
            for method_name, (stage, model), methods in (('C', C_RUNS[key], ('con',)),
                                                         ('R', RE_RUNS[key], ('cot',)),
                                                         ('E', RE_RUNS[key], ('fsc',))):
                for method in methods:
                    run = one_run(stage, model, arm, method)
                    th = H.recorded_config(run)[1].thresholds
                    acc = rows.setdefault((key, method_name), collections.Counter())
                    for r in H.records(run):
                        if r.pose_world is None or r.result.get('status') != 'ok':
                            acc['no pose'] += 1
                            continue
                        sc = H.scene(run, r.scene_id)
                        robot = sc.observation.robot
                        a = (float(robot.x), float(robot.y))
                        b = (float(r.pose_world[0]), float(r.pose_world[1]))
                        regions = sc.truths[r.episode_id].gt.forbidden_regions or []
                        succ = bool(r.score['success'])
                        hitA = hitB = hitC = False
                        goal_in = False
                        for reg in regions:
                            hit, b_in, dist = seg_poly(a, b, reg.polygon)
                            goal_in |= b_in
                            if hit:
                                hitA = True
                                kinds_hit[(method_name, reg.kind, 'A')] += 1
                            if dist < th.robot_radius_m:
                                hitC = True
                        for reg in regions:      # def B: crosses an edge with the goal outside every polygon
                            if not goal_in and seg_poly(a, b, reg.polygon)[0]:
                                hitB = True
                                kinds_hit[(method_name, reg.kind, 'B')] += 1
                        hitC = hitC and not goal_in
                        for den, keep in (('all', True), ('succ', succ), ('has_forbidden', bool(regions)),
                                          ('succ_has_forbidden', succ and bool(regions))):
                            if keep:
                                acc[(den, 'n')] += 1
                                acc[(den, 'A')] += hitA
                                acc[(den, 'B')] += hitB
                                acc[(den, 'C')] += hitC
                        acc['goal inside a forbidden polygon'] += goal_in
                        acc['goal-inside disagrees with recorded score.forbidden'] += goal_in != any((r.score.get('forbidden') or {}).values())
                        from sgi.polygons import point_in_polygon
                        acc['start inside a forbidden polygon'] += any(point_in_polygon(a, reg.polygon) for reg in regions)
                        if method_name == 'C' and hitA:
                            examples.append((key, arm, r.scene_id, r.episode_id, r.trial, r.situation, succ, goal_in,
                                             hitB, [reg.kind for reg in regions]))
    out('-' * 100)

    def fmt(c, den, d):
        n = c[(den, 'n')]
        return f'{c[(den, d)]}/{n}={c[(den, d)] / n:.1%}' if n else '-'
    for method_name in ('C', 'R', 'E'):
        pooled = collections.Counter()
        for key in ('qwen', 'glm', 'ds'):
            c = rows[(key, method_name)]
            pooled.update(c)
            out(f'{method_name} {key:5s}: ' + ' | '.join(
                f'{den}: A {fmt(c, den, "A")} B {fmt(c, den, "B")} C {fmt(c, den, "C")}'
                for den in ('all', 'succ', 'has_forbidden', 'succ_has_forbidden'))
                + f' | no pose {c["no pose"]} | goal inside {c["goal inside a forbidden polygon"]} | start inside {c["start inside a forbidden polygon"]}')
        out(f'{method_name} pooled: ' + ' | '.join(
            f'{den}: A {fmt(pooled, den, "A")} B {fmt(pooled, den, "B")} C {fmt(pooled, den, "C")}'
            for den in ('all', 'succ', 'has_forbidden', 'succ_has_forbidden'))
            + f' | no pose {pooled["no pose"]} | goal inside {pooled["goal inside a forbidden polygon"]} | start inside {pooled["start inside a forbidden polygon"]}')
    out('  goal-inside check against the recorded score.forbidden, disagreements: ' + ', '.join(
        f'{k[0]} {k[1]} {rows[k]["goal-inside disagrees with recorded score.forbidden"]}' for k in sorted(rows)))
    out('  hits by (method, kind, def): ' + '; '.join(f'{k} {v}' for k, v in sorted(kinds_hit.items())))
    out('  C records whose segment meets a forbidden polygon (def A):')
    for e in examples:
        out('    ' + ' '.join(str(x) for x in e))


def m1b():
    out('=' * 100)
    out('M1 check: recorded success vs the package scoring the recorded pose now (H.score_poses), C realistic; and the'
        ' 25/14 reading: flag firings (each set flag of SIX counted once per record) over all seven s20 dsv41 con runs')
    for key in ('ds', 'glm', 'qwen'):
        stage, model = C_RUNS[key]
        diff = n = 0
        for arm in ARMS:
            for r in H.records(one_run(stage, model, arm, 'con')):
                if r.pose_world is None or r.result.get('status') != 'ok':
                    continue
                n += 1
                diff += bool(H.score_pose(r, r.pose_world).success) != bool(r.score['success'])
        out(f'  {key}: records with a pose {n}, success differs now {diff}')
    for scope, arms in (('realistic', ARMS), ('ph', ('ph',)), ('dev seeds', ('dev_seed0', 'dev_seed1', 'dev_seed2')),
                        ('all seven', ARMS + ('ph', 'dev_seed0', 'dev_seed1', 'dev_seed2'))):
        rec_s = rec_f = fire_s = fire_f = ns = nf = 0
        for arm in arms:
            for r in H.records(one_run('s20', 'dsv41', arm, 'con')):
                fl = r.result.get('flags') or {}
                k = sum(bool(fl.get(f)) for f in SIX)
                if r.score['success']:
                    ns += 1; rec_s += k > 0; fire_s += k
                else:
                    nf += 1; rec_f += k > 0; fire_f += k
        out(f'  ds {scope:10s}: succ {ns} fail {nf} | records flagged succ {rec_s} / fail {rec_f} | flag firings succ {fire_s} / fail {fire_f}')


def m2b():
    out('=' * 100)
    out('I47 side figure: DeepSeek C on the oblique eval_02 layout (realistic arms), by class')
    stage, model = C_RUNS['ds']
    cls = collections.Counter()
    for arm in ARMS:
        for r in H.records(one_run(stage, model, arm, 'con')):
            if '_eval_02_' in r.scene_id:
                cls[classify(r)] += 1
    out(f'  n={sum(cls.values())} ' + '; '.join(f'{k} {v}' for k, v in sorted(cls.items())))


if __name__ == '__main__':
    which = sys.argv[1:] or ['m1', 'm1b', 'm2', 'm2b', 'm3']
    for w in which:
        globals()[w]()
        (OUT / f'{w}_output.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        lines.clear()
