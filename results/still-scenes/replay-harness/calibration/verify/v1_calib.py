"""Independent verifier for I01 Step 2 (s8 calibration). Written without reading phase1/i01_calibration scripts.

Per model: pair s8_convlm and s8_conorc records on (scene, episode, trial); rebuild true people (visible rule),
match detections to them nearest first within 2 * person_radius_m (copy of sgi.run._matched_people), apply
R1 (true xy), R2 (true yaw), R3 (missed visible people added with new numbers, pixel as OraclePeople),
R4 (unmatched detections dropped, numbers stripped from constraints), P = R1..R4, Ps = P without stripping.
Resolve convlm goal steps through H.resolve; records with no goal step keep their rescored verdict.
Writes rows to v1_rows.jsonl.
"""
import sys, json, math, collections
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
from sgi.gt.derive import is_visible
from sgi.geometry import project

OUT = 'results/still-scenes/replay-harness/calibration/verify/'
MODELS = ['glm53', 'ornith15', 'qwen36_35', 'qwen38', 'qwen3vl']


def match(detected, visible, width):
    pairs = sorted((math.dist(d['xy'], (p.x, p.y)), d['number'], i) for d in detected for i, p in enumerate(visible))
    m, taken = {}, set()
    for gap, num, i in pairs:
        if gap <= width and num not in m and i not in taken:
            m[num] = i
            taken.add(i)
    return m


def repaired(det, visible, m, th, obs, r1, r2, r3, r4):
    people = []
    for d in det:
        d = dict(d)
        if d['number'] in m:
            t = visible[m[d['number']]]
            if r1:
                d['xy'] = [t.x, t.y]
            if r2:
                d['yaw'] = t.yaw
        elif r4:
            continue
        people.append(d)
    if r3:
        nxt = max([d['number'] for d in det], default=0) + 1
        hit = set(m.values())
        for i, t in enumerate(visible):
            if i in hit:
                continue
            mk = project(obs.camera, (t.x, t.y, th.person_mark_height_m))
            if mk is None:
                continue
            people.append({'number': nxt, 'xy': [t.x, t.y], 'yaw': t.yaw, 'pixel': list(mk[:2]), 'label': None})
            nxt += 1
    return people


def strip(cons, dropped):
    out = []
    for c in cons:
        c = dict(c)
        c['people'] = [n for n in c.get('people', []) if n not in dropped]
        out.append(c)
    return out


rows = []
for model in MODELS:
    rv = H.runs(stage='s8', method='convlm', model=model)
    ro = H.runs(stage='s8', method='conorc', model=model)
    rn = H.runs(stage='s8', method='none', model=model)
    assert len(rv) == len(ro) == len(rn) == 1
    rv, ro, rn = rv[0], ro[0], rn[0]
    cfgv, cfgo, cfgn = (H.recorded_config(x)[1] for x in (rv, ro, rn))
    th = cfgv.thresholds
    assert cfgo.thresholds == th
    width = 2 * th.person_radius_m
    V = {(r.scene_id, r.episode_id, r.trial): r for r in H.records(rv)}
    O = {(r.scene_id, r.episode_id, r.trial): r for r in H.records(ro)}
    N = {(r.scene_id, r.episode_id, r.trial): r for r in H.records(rn)}
    assert set(V) == set(O) == set(N), model
    for key in sorted(V):
        v, o, n = V[key], O[key], N[key]
        row = {'model': model, 'scene': key[0], 'episode': key[1], 'trial': key[2], 'situation': v.situation,
               'cfg_v': [cfgv.run.people, cfgv.run.constraints, cfgv.run.goal, v.raw['config']['people'], v.raw['config']['constraints'], v.raw['config']['goal']],
               'cfg_o': [cfgo.run.people, cfgo.run.constraints, cfgo.run.goal, o.raw['config']['people'], o.raw['config']['constraints'], o.raw['config']['goal']],
               'cfg_n': [cfgn.run.people, cfgn.run.constraints, cfgn.run.goal, n.raw['config']['people'], n.raw['config']['constraints'], n.raw['config']['goal']]}
        # cache sharing: raw text of plan/locate/context calls, target, plan
        def stage_text(r, stages=('plan', 'locate', 'context')):
            return [(c['stage'], c.get('raw_text'), c.get('cached')) for c in (r.result.get('calls') or []) if c['stage'] in stages]
        tv, to, tn = stage_text(v), stage_text(o), stage_text(n)
        row['same_plan_ctx_vo'] = [(a[0], a[1]) for a in tv] == [(a[0], a[1]) for a in to]
        row['same_plan_ctx_vn'] = [(a[0], a[1]) for a in tv] == [(a[0], a[1]) for a in tn]
        row['plan_ctx_all_cached'] = all(a[2] for a in tv + to)
        row['n_plan_ctx_calls'] = len(tv)
        row['same_target_vo'] = v.target == o.target
        row['same_plan_vo'] = v.raw.get('plan') == o.raw.get('plan')
        row['people_text_v'] = [c.get('raw_text') for c in (v.result.get('calls') or []) if c['stage'] == 'people']
        row['people_cached_v'] = [c.get('cached') for c in (v.result.get('calls') or []) if c['stage'] == 'people']
        row['cons_cached_o'] = [c.get('cached') for c in (o.result.get('calls') or []) if c['stage'] == 'constraints']
        row['cons_cached_v'] = [c.get('cached') for c in (v.result.get('calls') or []) if c['stage'] == 'constraints']
        row['step_v'], row['step_o'] = v.solver_step, o.solver_step
        row['rec_v'], row['rec_o'] = bool(v.score.get('success')), bool(o.score.get('success'))
        row['now_v'], row['now_o'] = bool(H.rescore(v).success), bool(H.rescore(o).success)
        row['target_correct_v'] = v.score.get('target_correct')
        row['matched_target_v'] = v.score.get('matched_target_id')
        row['matched_target_o'] = o.score.get('matched_target_id')
        row['target_xy_o'] = None if not o.target else o.target.get('world')
        if o.solver_step:
            ro_ = H.resolve(o)
            row['res_o'], row['moved_o'] = bool(ro_.score.success), ro_.moved_m
        if v.solver_step:
            rv_ = H.resolve(v)
            row['res_v'], row['moved_v'] = bool(rv_.score.success), rv_.moved_m
            sc = H.scene(rv, key[0])
            obs = sc.observation
            visible = [p for p in sc.truth.people if is_visible(p, obs, th)]
            det = v.people
            m = match(det, visible, width)
            m1 = match(det, visible, 2 * width)
            row['n_det'], row['n_vis'], row['n_match'] = len(det), len(visible), len(m)
            row['n_true_all'] = len(sc.truth.people)
            row['rec_found_share'] = v.measures.get('people_found_share')
            row['rec_unmatched'] = v.measures.get('people_unmatched')
            row['err_xy'] = [math.dist(d['xy'], (visible[m[d['number']]].x, visible[m[d['number']]].y)) for d in det if d['number'] in m]
            # ambiguity per matched pair
            amb = 0
            amb_all = 0
            for d in det:
                if d['number'] not in m:
                    continue
                t = visible[m[d['number']]]
                near_t = sum(math.dist(d['xy'], (p.x, p.y)) <= width for p in visible)
                near_t_all = sum(math.dist(d['xy'], (p.x, p.y)) <= width for p in sc.truth.people)
                near_d = sum(math.dist(e['xy'], (t.x, t.y)) <= width for e in det)
                amb += (near_t >= 2 or near_d >= 2)
                amb_all += (near_t_all >= 2 or near_d >= 2)
            row['amb'], row['amb_all'] = amb, amb_all
            # oracle people of conorc for comparison
            op = o.people or []
            cons = [u['constraint'] for u in v.constraints]
            dropped = {d['number'] for d in det if d['number'] not in m}
            variants = {
                'R1': (True, False, False, False, True),
                'R2': (False, True, False, False, True),
                'R3': (False, False, True, False, True),
                'R4': (False, False, False, True, True),
                'P': (True, True, True, True, True),
                'Ps': (True, True, True, True, False),
            }
            for name, (a, b, c, d4, s) in variants.items():
                ppl = repaired(det, visible, m, th, obs, a, b, c, d4)
                cc = strip(cons, dropped) if (d4 and s) else cons
                rs = H.resolve(v, people=ppl, constraints=cc)
                row['res_' + name] = bool(rs.score.success)
                if name == 'P':
                    row['P_people_eq_o'] = sorted((round(p['xy'][0], 6), round(p['xy'][1], 6)) for p in ppl) == \
                        sorted((round(p['xy'][0], 6), round(p['xy'][1], 6)) for p in op)
            # P at 1.0 m matching
            dropped1 = {d['number'] for d in det if d['number'] not in m1}
            ppl = repaired(det, visible, m1, th, obs, True, True, True, True)
            row['res_P1m'] = bool(H.resolve(v, people=ppl, constraints=strip(cons, dropped1)).score.success)
            row['kinds_v'] = [c['kind'] for c in cons]
            row['kinds_o'] = [u['constraint']['kind'] for u in (o.constraints or [])]
        rows.append(row)
    print(model, 'done', flush=True)

with open(OUT + 'v1_rows.jsonl', 'w') as f:
    for r in rows:
        f.write(json.dumps(r) + '\n')
print(len(rows))
