"""Perception decomposition re-run (claims P1, P2, P3 and the NEW rows), read only through research_harness.

P1  share of missed members that the people call never found (GLM s19 con, realistic; plus s20 DeepSeek, s19 Qwen,
    s23 GLM / DeepSeek con for the new wording).
P2  share of the constraints call's visible raw_text taken by member lists (s19 GLM, s20 DeepSeek; s23 too).
P3  GLM's C line failures: members_recall, people_found_share, unmatched, come_after stated (and DeepSeek; s23).
NEW phantom detections at lines (s23 glm53 con; s19 for comparison): detections more than 1 m from every true person.

Matching rule (the package's own, sgi.run._matched_people, used to make measures.people_found_share and
members_recall): detected people (result.people, xy on the floor) and the ground-truth people that pass the
visibility rule (sgi.gt.derive.is_visible under the run's recorded thresholds) are paired greedily, nearest pair
first, one to one, when centres are within 2 x person_radius_m = 0.5 m. Members = visible true people whose group is
the group at the chosen target (truth.group_for(episode, chosen)); chosen = score.matched_target_id, else the
episode's first target. A member is "named" when some come_after number is matched to them (the package's rule for
members_recall). Recomputed measures are checked against the recorded ones.

Writes nothing but its stdout; run with stdout tee'd to perception_out.txt.
"""
import collections
import math
import re
import sys

sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H  # noqa: E402
from sgi.gt.derive import is_visible  # noqa: E402
from sgi.run import _matched_people  # noqa: E402
from sgi.types import DetectedPerson  # noqa: E402

REAL = ('real', 'staffed', 'plain')
DEV = ('dev_seed0', 'dev_seed1', 'dev_seed2')


def runs_of(stage, model, arms):
    out = []
    for arm in arms:
        rr = H.runs(stage=stage, arm=arm, model=model, method='con')
        assert len(rr) == 1, (stage, arm, model, [r.folder for r in rr])
        out.append(rr[0])
    return out


def recs_of(stage, model, arms):
    for ref in runs_of(stage, model, arms):
        tally = collections.Counter()
        for rec in H.records(ref, tally=tally):
            yield rec


_VIS = {}


def scene_info(rec):
    key = (str(rec.ref.scenes_dir), rec.scene_id, H.recorded_config(rec.ref)[1].thresholds.model_dump_json())
    if key not in _VIS:
        sc = H.scene(rec.ref, rec.scene_id)
        th = H.recorded_config(rec.ref)[1].thresholds
        vis = [p for p in sc.truth.people if is_visible(p, sc.observation, th)]
        _VIS[key] = (sc, th, vis)
    return _VIS[key]


def analyse(rec):
    """Per-record perception facts, or None for a record without people."""
    sc, th, vis = scene_info(rec)
    out = dict(rec=rec, n_true=len(sc.truth.people), n_vis=len(vis))
    people = rec.people
    if people is None:
        return None
    det = [DetectedPerson.model_validate(p) for p in people]
    matched = _matched_people(det, vis, th)                 # number -> true id
    found_ids = set(matched.values())
    out['n_det'] = len(det)
    out['found_share'] = len(found_ids) / len(vis) if vis else None
    out['unmatched'] = len(det) - len(matched)
    ep = sc.truth.episodes[rec.episode_id]
    chosen = rec.score.get('matched_target_id') or ep.target_ids[0]
    group = sc.truth.group_for(rec.episode_id, chosen)
    out['group'] = group
    members = [p for p in vis if group is not None and p.group == group]
    out['members'] = [p.id for p in members]
    cons = rec.constraints
    out['has_cons'] = cons is not None and rec.solver is not None
    ca_nums, any_nums = set(), set()
    kinds = []
    for use in cons or []:
        c = use['constraint']
        kinds.append(c['kind'])
        any_nums |= set(c.get('people') or [])
        if c['kind'] == 'come_after':
            ca_nums |= set(c.get('people') or [])
    out['kinds'] = kinds
    out['come_after_stated'] = 'come_after' in kinds
    named_ca = {matched.get(n) for n in ca_nums} - {None}
    named_any = {matched.get(n) for n in any_nums} - {None}
    out['named_ca'] = named_ca
    out['named_any'] = named_any
    if members:
        out['recall_ca'] = len(named_ca & set(out['members'])) / len(members)
    else:
        out['recall_ca'] = None
    # missed members and whether the people call found them
    out['missed_ca'] = [m for m in out['members'] if m not in named_ca]
    out['missed_any'] = [m for m in out['members'] if m not in named_any]
    out['found_ids'] = found_ids
    # looser "found": any detection within 1.0 m of the member (not one-to-one)
    pos = {p.id: (p.x, p.y) for p in vis}
    out['near1'] = {m for m in out['members'] if any(math.dist(d.xy, pos[m]) <= 1.0 for d in det)}
    # phantom detections: more than 1 m from every true person (all, and visible)
    allp = [(p.x, p.y) for p in sc.truth.people]
    vp = [(p.x, p.y) for p in vis]
    out['det_min_all'] = [min((math.dist(d.xy, q) for q in allp), default=float('inf')) for d in det]
    out['det_min_vis'] = [min((math.dist(d.xy, q) for q in vp), default=float('inf')) for d in det]
    out['det'] = det
    out['matched'] = matched
    out['ca_nums'] = ca_nums
    return out


def check_measures(rows, label):
    bad_f = bad_r = n_f = n_r = 0
    for a in rows:
        m = a['rec'].measures
        if m.get('people_found_share') is not None or a['found_share'] is not None:
            n_f += 1
            if m.get('people_found_share') is None or a['found_share'] is None or abs(m['people_found_share'] - a['found_share']) > 1e-9:
                bad_f += 1
        if m.get('members_recall') is not None:
            n_r += 1
            if a['recall_ca'] is None or abs(m['members_recall'] - a['recall_ca']) > 1e-9:
                bad_r += 1
    print(f'  [{label}] check vs recorded measures: people_found_share {n_f - bad_f}/{n_f} agree; '
          f'members_recall {n_r - bad_r}/{n_r} agree')


def p1(rows, label, sit=('waiting_line',)):
    """Missed members and the share never found."""
    out = []
    for name, sel in [('all records with members_recall measured', lambda a: a['rec'].measures.get('members_recall') is not None),
                      ('line failures (P3 set)', lambda a: a.get('p3fail'))]:
        missed = nf = nf_near = 0
        recs = 0
        for a in rows:
            if a['rec'].situation not in sit or not sel(a):
                continue
            recs += 1
            for m in a['missed_ca']:
                missed += 1
                nf += m not in a['found_ids']
                nf_near += m not in a['near1']
        share = nf / missed if missed else float('nan')
        share2 = nf_near / missed if missed else float('nan')
        line = (f'  [{label}] {name}: records {recs}; missed members {missed}; never found (package match, 0.5 m) {nf} '
                f'= {share:.3f}; no detection within 1.0 m {nf_near} = {share2:.3f}')
        print(line)
        out.append((name, recs, missed, nf, share))
    return out


def p1_groups(rows, label):
    """Variant: conversation-group members (target's group) not named in any constraint's people list."""
    missed = nf = recs = 0
    for a in rows:
        if a['rec'].situation != 'conversation_group' or not a['has_cons'] or not a['members']:
            continue
        recs += 1
        for m in a['missed_any']:
            missed += 1
            nf += m not in a['found_ids']
    print(f'  [{label}] variant groups (conversation_group goal-step records, member = visible person of the group at '
          f'the target, named = in any constraint people list): records {recs}; missed {missed}; never found {nf}'
          + (f' = {nf / missed:.3f}' if missed else ''))
    # pooled lines+groups
    return recs, missed, nf


def is_p3_fail(a):
    r = a['rec']
    s = r.score
    return (r.situation == 'waiting_line' and r.result.get('source') == 'constraints' and r.solver_step
            and s.get('target_correct') and not s.get('success')
            and (not s.get('in_region') or not s.get('reachable')))


def p3(rows, label):
    fails = [a for a in rows if is_p3_fail(a)]
    for a in rows:
        a['p3fail'] = is_p3_fail(a)
    n = len(fails)
    mr = [a['rec'].measures.get('members_recall') for a in fails]
    pf = [a['rec'].measures.get('people_found_share') for a in fails]
    um = [a['rec'].measures.get('people_unmatched') for a in fails]
    print(f'  [{label}] C line failures (waiting_line, solver step, right target, place outside region or unreachable): {n}')
    print(f'      members_recall <1: {sum(x is not None and x < 1 for x in mr)} (None {sum(x is None for x in mr)}), '
          f'=0: {sum(x == 0 for x in mr)}')
    print(f'      people_found_share <1: {sum(x is not None and x < 1 for x in pf)}, =0: {sum(x == 0 for x in pf)}; '
          f'unmatched >0: {sum(x is not None and x > 0 for x in um)}; no come_after stated: '
          f'{sum(not a["come_after_stated"] for a in fails)}')
    unreach = sum(not a['rec'].score.get('reachable') for a in fails)
    print(f'      of them unreachable: {unreach}; outside region: {sum(not a["rec"].score.get("in_region") for a in fails)}')
    return dict(n=n, mr_lt1=sum(x is not None and x < 1 for x in mr), mr0=sum(x == 0 for x in mr),
                pf_lt1=sum(x is not None and x < 1 for x in pf), pf0=sum(x == 0 for x in pf),
                um=sum(x is not None and x > 0 for x in um), noca=sum(not a['come_after_stated'] for a in fails))


PEOPLE_LIST = re.compile(r'"people"\s*:\s*\[[^\]]*\]')
TOKEN = re.compile(r'\w+|[^\w\s]')
# GPT-2-style pre-tokens (a word or number with its leading space, a run of punctuation with its leading space):
# a proxy for BPE tokens, since no tokenizer is installed; BPE merges these further or splits rare words.
PRETOKEN = re.compile(r' ?[^\W\d_]+| ?\d{1,3}| ?[^\s\w]+|\s+(?!\S)|\s+')


def p2(rows, label):
    """Share of the constraints call's visible raw_text taken by the people lists."""
    tot_c = lst_c = val_c = ca_c = 0
    tot_t = lst_t = ca_t = 0
    pt_tot = pt_lst = pt_val = 0
    per = []
    ncalls = nraw = 0
    seen = set()
    for a in rows:
        r = a['rec']
        for c in r.result.get('calls') or []:
            if c.get('stage') != 'constraints':
                continue
            ncalls += 1
            raw = c.get('raw_text')
            if not raw:
                continue
            nraw += 1
            key = raw  # each distinct answer once (cache copies repeat exactly)
            if key in seen:
                continue
            seen.add(key)
            lists = PEOPLE_LIST.findall(raw)
            lc = sum(len(x) for x in lists)
            vc = sum(len(x[x.index('['):]) for x in lists)
            # come_after lists only
            cac = 0
            for m in re.finditer(r'\{[^{}]*"kind"\s*:\s*"come_after"[^{}]*\}', raw):
                cac += sum(len(x) for x in PEOPLE_LIST.findall(m.group(0)))
            tot_c += len(raw); lst_c += lc; val_c += vc; ca_c += cac
            tt = len(TOKEN.findall(raw)); lt = sum(len(TOKEN.findall(x)) for x in lists)
            tot_t += tt; lst_t += lt
            cat = 0
            for m in re.finditer(r'\{[^{}]*"kind"\s*:\s*"come_after"[^{}]*\}', raw):
                cat += sum(len(TOKEN.findall(x)) for x in PEOPLE_LIST.findall(m.group(0)))
            ca_t += cat
            pt_tot += len(PRETOKEN.findall(raw)); pt_lst += sum(len(PRETOKEN.findall(x)) for x in lists)
            pt_val += sum(len(PRETOKEN.findall(x[x.index('['):])) for x in lists)
            per.append(lc / len(raw))
    per.sort()
    med = per[len(per) // 2] if per else float('nan')
    print(f'  [{label}] constraints calls {ncalls}, with raw_text {nraw}, distinct answers {len(seen)}')
    if tot_c:
        print(f'      chars: "people": [...] incl. key {lst_c}/{tot_c} = {lst_c / tot_c:.4f}; list values only {val_c / tot_c:.4f}; '
              f'come_after lists only {ca_c / tot_c:.4f}; per-answer median (incl. key) {med:.4f}')
        print(f'      regex tokens (\\w+ | punct): people lists {lst_t}/{tot_t} = {lst_t / tot_t:.4f}; come_after lists {ca_t / tot_t:.4f}')
        print(f'      GPT-2-style pre-tokens: people lists incl. key {pt_lst}/{pt_tot} = {pt_lst / pt_tot:.4f}; values only {pt_val / pt_tot:.4f}')
        ot = sum(c['usage'].get('output_tokens') or 0 for a in rows for c in a['rec'].result.get('calls') or [] if c.get('stage') == 'constraints' and not c.get('cached'))
        rt = sum(c['usage'].get('reasoning_tokens') or 0 for a in rows for c in a['rec'].result.get('calls') or [] if c.get('stage') == 'constraints')
        print(f'      (reported output_tokens of fresh constraints calls {ot}; reasoning_tokens of all {rt}; pre-tokens per answer '
              f'{pt_tot / len(seen):.0f})')
    return dict(calls=ncalls, distinct=len(seen), share_c=lst_c / tot_c if tot_c else None,
                share_v=val_c / tot_c if tot_c else None, share_t=lst_t / tot_t if tot_t else None,
                share_ca=ca_c / tot_c if tot_c else None)


def poly_dist(pt, poly):
    """Distance from a point to the boundary of a polygon (list of (x, y))."""
    best = float('inf')
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        dx, dy = x2 - x1, y2 - y1
        L = dx * dx + dy * dy
        t = 0 if L == 0 else max(0, min(1, ((pt[0] - x1) * dx + (pt[1] - y1) * dy) / L))
        best = min(best, math.dist(pt, (x1 + t * dx, y1 + t * dy)))
    return best


def ray_to_wall(cam, pt, poly):
    """Signed distance along the camera ray from pt to where the ray (from cam, inside poly) leaves poly: positive
    when pt lies in front of the wall, negative when beyond it. The crossing farthest along the ray is the wall."""
    dx, dy = pt[0] - cam[0], pt[1] - cam[1]
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    ts = []
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        ex, ey = x2 - x1, y2 - y1
        den = ux * ey - uy * ex
        if abs(den) < 1e-12:
            continue
        t = ((x1 - cam[0]) * ey - (y1 - cam[1]) * ex) / den
        u = ((x1 - cam[0]) * uy - (y1 - cam[1]) * ux) / den
        if 0 <= u <= 1 and t > 0:
            ts.append(t)
    return max(ts) - L if ts else float('nan')


def phantoms(rows, label):
    """Detections more than 1 m from every true person, at lines, in records with people."""
    line = [a for a in rows if a['rec'].situation == 'waiting_line']
    goal = [a for a in line if a['rec'].solver_step]
    for name, sel in [('line records with people', line), ('line goal-step records', goal)]:
        dets = sum(a['n_det'] for a in sel)
        far_all = sum(d > 1.0 for a in sel for d in a['det_min_all'])
        far_vis = sum(d > 1.0 for a in sel for d in a['det_min_vis'])
        print(f'  [{label}] {name}: records {len(sel)}; detections {dets}; >1 m from every true person {far_all}; '
              f'>1 m from every visible true person {far_vis}')
    # distinct people answers: (arm, scene, trial) people list identical
    distinct = {}
    for a in line:
        key = (a['rec'].arm, a['rec'].scene_id, tuple((round(d.xy[0], 6), round(d.xy[1], 6)) for d in a['det']))
        distinct.setdefault(key, a)
    dets = sum(a['n_det'] for a in distinct.values())
    far = sum(d > 1.0 for a in distinct.values() for d in a['det_min_all'])
    print(f'  [{label}] distinct line people answers: {len(distinct)}; detections {dets}; >1 m from every true person {far}')
    # where the far ones lie: behind the people (camera range), near the room boundary, and used by the solver
    behind = near_wall = in_ca = in_join = fail_with = 0
    nfar = 0
    ray_gap = []
    true_near_wall = [poly_dist((p.x, p.y), scene_info(a['rec'])[0].truth.bounds) <= 0.5
                      for a in goal for p in scene_info(a['rec'])[0].truth.people if scene_info(a['rec'])[0].truth.bounds]
    rec_with = set()
    for a in goal:
        sc, th, vis = scene_info(a['rec'])
        cam = sc.truth and sc.observation.camera.T_world_camera
        cx, cy = float(cam[0][3]), float(cam[1][3])
        bounds = sc.truth.bounds
        joined = set((a['rec'].solver or {}).get('joined_waiting') or [])
        for d, dm in zip(a['det'], a['det_min_all']):
            if dm <= 1.0:
                continue
            nfar += 1
            rng = math.dist(d.xy, (cx, cy))
            near_true = min(sc.truth.people, key=lambda p: math.dist(d.xy, (p.x, p.y)))
            behind += rng > math.dist((near_true.x, near_true.y), (cx, cy))
            if bounds:
                near_wall += poly_dist(d.xy, bounds) <= 0.5
                ray_gap.append(ray_to_wall((cx, cy), d.xy, bounds))
            in_ca += d.number in a['ca_nums']
            in_join += d.number in joined
            rec_with.add(id(a))
    ray_gap = sorted(g for g in ray_gap if g == g)
    if ray_gap:
        print(f'  [{label}] far detections: distance to the wall along the camera ray through them: median '
              f'{ray_gap[len(ray_gap) // 2]:.2f} m (min {ray_gap[0]:.2f}, max {ray_gap[-1]:.2f}; negative = beyond the wall line), '
              f'|gap| <=0.3 m {sum(abs(g) <= 0.3 for g in ray_gap)}/{len(ray_gap)}; '
              f'true people (per record) within 0.5 m of the boundary {sum(true_near_wall)}/{len(true_near_wall)}')
    fails_with = sum(1 for a in goal if id(a) in rec_with and not a['rec'].score.get('success'))
    print(f'  [{label}] far detections in line goal-step records: {nfar}; farther from the camera than the nearest true '
          f'person {behind}; within 0.5 m of the room boundary {near_wall}; listed under come_after {in_ca}; '
          f'joined by the solver {in_join}; records holding one {len(rec_with)}, of them failed {fails_with}')
    return dict(nfar=nfar)


def block(stage, model, arms, label, do_p2=True):
    print(f'\n=== {label}: {stage} {model} con {arms}')
    rows = [a for a in (analyse(r) for r in recs_of(stage, model, arms)) if a is not None]
    allrecs = list(recs_of(stage, model, arms))
    print(f'  records {len(allrecs)}, with people {len(rows)}, solver steps {sum(r.solver_step for r in allrecs)}, '
          f'success {sum(bool(r.score.get("success")) for r in allrecs)}')
    check_measures(rows, label)
    res = {}
    res['p3'] = p3(rows, label)
    res['p1'] = p1(rows, label)
    res['p1g'] = p1_groups(rows, label)
    # mean measures at lines for context
    lines = [a for a in rows if a['rec'].situation == 'waiting_line' and a['rec'].solver_step]
    mr = [a['rec'].measures['members_recall'] for a in rows if a['rec'].measures.get('members_recall') is not None]
    pf = [a['rec'].measures['people_found_share'] for a in rows if a['rec'].solver_step and a['rec'].measures.get('people_found_share') is not None]
    if mr:
        print(f'  [{label}] context: mean members_recall {sum(mr) / len(mr):.3f} (n {len(mr)}); mean people_found_share '
              f'(goal steps) {sum(pf) / len(pf):.3f} (n {len(pf)})')
    if do_p2:
        res['p2'] = p2(rows, label)
    res['ph'] = phantoms(rows, label)
    return res


if __name__ == '__main__':
    out = {}
    out['glm19'] = block('s19', 'glm53', REAL, 'GLM s19 realistic')
    out['ds20'] = block('s20', 'dsv41', REAL, 'DeepSeek s20 realistic')
    out['qw19'] = block('s19', 'qwen38_low', REAL, 'Qwen s19 realistic (context)')
    out['glm23'] = block('s23', 'glm53', REAL, 'GLM s23 realistic (new wording)')
    out['ds23'] = block('s23', 'dsv41', REAL, 'DeepSeek s23 realistic (new wording)')
    print('\n=== P2 on development seeds (placeholder dev_seed0-2), for range')
    for stage, model in [('s19', 'glm53'), ('s20', 'dsv41')]:
        rows = [a for a in (analyse(r) for r in recs_of(stage, model, DEV)) if a is not None]
        p2(rows, f'{model} {stage} dev seeds')
    # cache-copy check: s23 people answers equal to s19/s20?
    print('\n=== Are s23 C people answers copies of s19/s20 (same people list per arm, scene, episode, trial)?')
    for m19, stage in [('glm53', 's19'), ('dsv41', 's20')]:
        old = {(r.arm, r.scene_id, r.episode_id, r.trial): r for r in recs_of(stage, m19, REAL)}
        same_p = diff_p = same_c = diff_c = fresh_c = cached_c = 0
        for r in recs_of('s23', m19, REAL):
            o = old.get((r.arm, r.scene_id, r.episode_id, r.trial))
            if o is None or r.people is None or o.people is None:
                continue
            if [p['xy'] for p in r.people] == [p['xy'] for p in o.people]:
                same_p += 1
            else:
                diff_p += 1
            rc = [c for c in r.result['calls'] if c['stage'] == 'constraints']
            oc = [c for c in o.result['calls'] if c['stage'] == 'constraints']
            if rc and oc:
                if rc[0].get('raw_text') == oc[0].get('raw_text'):
                    same_c += 1
                else:
                    diff_c += 1
                cached_c += bool(rc[0].get('cached'))
                fresh_c += not rc[0].get('cached')
        print(f'  {m19}: people list same {same_p}, different {diff_p}; constraints raw_text same {same_c}, different '
              f'{diff_c}; s23 constraints calls cached {cached_c}, fresh {fresh_c}')
