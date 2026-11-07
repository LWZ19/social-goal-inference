"""I06 perception-noise model (FROZEN.md section 8): fit on development runs only, through the harness.

Fit data: the old-wording C people answers of Qwen (s19 qwen38_low), GLM (s19 glm53) and DeepSeek (s20 dsv41), arms
real, staffed, plain, ph, dev_seed0-2 (seven runs per model). An answer is the people list of a solver-step record, one
per (run, scene, trial); the other episode's list must be equal, otherwise both are kept (counted).

Per visible person (derive.is_visible under the run's recorded thresholds) one outcome: located (run._matched_people,
0.5 m), mislocated (not matched, paired greedily nearest first, one to one, with an unmatched detection whose recorded
pixel lies within 40 px of the person's mark projection at person_mark_height_m), missed. Unpaired unmatched detections
are false people.

Role groups: member (group at the record's target, truth.group_for(episode, target id)), staff (role set), other.

    noise.py fit     -> score/noise/answers_<model>.jsonl, score/noise_model.json, score/noise/fit.txt
"""
from __future__ import annotations

import collections
import json
import math
import sys
import zlib

import numpy as np

import lib as L
from lib import H
from sgi.geometry import bearing, project, wrap_angle
from sgi.gt.derive import is_visible
from sgi.run import OraclePeople, _matched_people
from sgi.stages.people import FoundPerson
from sgi.types import DetectedPerson

NOISE = L.SCORE / 'noise'
MODELS = [('qwen', 's19', 'qwen38_low', 'Qwen 3.8'), ('glm', 's19', 'glm53', 'GLM-5.3'),
          ('dsv', 's20', 'dsv41', 'DeepSeek V4.1')]
MODEL_INDEX = {'qwen': 0, 'glm': 1, 'dsv': 2}
ARMS = ('real', 'staffed', 'plain', 'ph', 'dev_seed0', 'dev_seed1', 'dev_seed2')
REALISTIC = ('real', 'staffed', 'plain')
PLACEHOLDER = ('ph', 'dev_seed0', 'dev_seed1', 'dev_seed2')
PAIR_PX = 40.0                       # I03 Stage B's registered relocation radius (data-informed; no new radius)
MIN_POOL = 10                        # below this many pairs a model takes the three models' pooled offsets
NOISE_SEED = 20261018
ROLES = ('member', 'other', 'staff')
VIEWS = ('frontal', 'oblique', 'profile', 'back')


def view_class(a):
    # phase1/a0_facing/a0_facing.py view_class, copied (both ends of profile in).
    if a < 45:
        return 'frontal'
    if a < 75:
        return 'oblique'
    if a <= 105:
        return 'profile'
    return 'back'


def ray_frame(cam_xy, p_xy):
    r = np.asarray(p_xy, dtype=float) - np.asarray(cam_xy, dtype=float)
    n = float(np.hypot(*r))
    r = r / n
    return r, np.array([-r[1], r[0]])


def to_offset(cam_xy, p_xy, q_xy):
    r, l = ray_frame(cam_xy, p_xy)
    d = np.asarray(q_xy, dtype=float) - np.asarray(p_xy, dtype=float)
    return [float(d @ r), float(d @ l)]


def from_offset(cam_xy, p_xy, off):
    r, l = ray_frame(cam_xy, p_xy)
    q = np.asarray(p_xy, dtype=float) + off[0] * r + off[1] * l
    return (float(q[0]), float(q[1]))


def runs_of(stage, model):
    out = []
    for arm in ARMS:
        refs = H.runs(stage=stage, arm=arm, model=model, method='con')
        assert len(refs) == 1, (stage, arm, model, refs)
        out.append(refs[0])
    return out


def role_of(p, group):
    if group is not None and p.group == group:
        return 'member'
    if p.role:
        return 'staff'
    return 'other'


def analyse_answer(det, vis, roles, obs, th):
    """One answer against the visible truth: per-person outcomes, offsets, false people."""
    cam_xy = obs.camera.T_world_camera[:2, 3]
    matched = _matched_people(det, vis, th)                     # number -> true id
    by_id = {p.id: p for p in vis}
    num_of = {tid: n for n, tid in matched.items()}
    det_by = {d.number: d for d in det}
    mark = {}
    for p in vis:
        m = project(obs.camera, (p.x, p.y, th.person_mark_height_m))
        mark[p.id] = None if m is None else (float(m[0]), float(m[1]))
    um_people = [p for p in vis if p.id not in num_of]
    um_det = [d for d in det if d.number not in matched]
    pairs = sorted((math.dist(d.pixel, mark[p.id]), d.number, i) for d in um_det for i, p in enumerate(um_people)
                   if mark[p.id] is not None)
    mis, used_d, used_p = {}, set(), set()
    for dist, num, i in pairs:
        if dist <= PAIR_PX and num not in used_d and i not in used_p:
            mis[um_people[i].id] = num
            used_d.add(num)
            used_p.add(i)
    persons = []
    for p in vis:
        a = abs(math.degrees(wrap_angle(bearing((p.x, p.y), cam_xy) - p.yaw)))
        row = {'id': p.id, 'role': roles[p.id], 'view': view_class(a), 'view_deg': round(a, 2)}
        if p.id in num_of:
            d = det_by[num_of[p.id]]
            row.update(outcome='located', offset=to_offset(cam_xy, (p.x, p.y), d.xy),
                       yaw_known=d.yaw is not None,
                       yaw_err=None if d.yaw is None else float(wrap_angle(d.yaw - p.yaw)))
        elif p.id in mis:
            d = det_by[mis[p.id]]
            row.update(outcome='mislocated', offset=to_offset(cam_xy, (p.x, p.y), d.xy),
                       pixel_gap=round(math.dist(d.pixel, mark[p.id]), 2), yaw_known=d.yaw is not None,
                       yaw_err=None if d.yaw is None else float(wrap_angle(d.yaw - p.yaw)))
        else:
            row.update(outcome='missed')
        persons.append(row)
    false = []
    for d in um_det:
        if d.number in used_d:
            continue
        cands = [(math.dist(d.pixel, mark[p.id]), p) for p in vis if mark[p.id] is not None]
        if cands:
            _, anchor = min(cands, key=lambda c: (c[0], c[1].id))
            off = to_offset(cam_xy, (anchor.x, anchor.y), d.xy)
            false.append({'anchor': anchor.id, 'offset': off, 'yaw_none': d.yaw is None,
                          'dist_m': round(math.dist(d.xy, (anchor.x, anchor.y)), 3)})
        else:
            false.append({'anchor': None, 'offset': None, 'yaw_none': d.yaw is None, 'dist_m': None})
    return persons, false


def fit():
    import hashlib
    assert hashlib.sha256((L.I06 / 'FROZEN.md').read_bytes()).hexdigest() == L.FROZEN_SHA
    NOISE.mkdir(exist_ok=True)
    lines = []

    def say(t=''):
        print(t, flush=True)
        lines.append(t)

    all_answers = {}
    i03_check = {}
    for key, stage, model, label in MODELS:
        answers, counts = [], collections.Counter()
        i03_matched = i03_visible = 0
        for ref in runs_of(stage, model):
            th = H.recorded_config(ref)[1].thresholds
            recs = sorted(H.records(ref), key=lambda r: (r.scene_id, str(r.trial), r.episode_id))
            by_key = collections.defaultdict(list)
            for r in recs:
                by_key[(r.scene_id, r.trial)].append(r)
            for (scene_id, trial), group in sorted(by_key.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
                sc = H.scene(ref, scene_id)
                obs, truth = sc.observation, sc.truth
                vis = [p for p in truth.people if is_visible(p, obs, th)]
                # I03's way: union of the records' lists (records with a list), all visible people
                with_list = [r for r in group if r.people is not None]
                if with_list:
                    union = {}
                    for r in with_list:
                        for p in r.people:
                            d = DetectedPerson.model_validate(p)
                            k = (round(d.xy[0], 9), round(d.xy[1], 9), round(d.pixel[0], 6), round(d.pixel[1], 6))
                            union.setdefault(k, d)
                    ppl = [union[k] for k in sorted(union)]
                    dets = [DetectedPerson(number=i + 1, xy=d.xy, yaw=d.yaw, pixel=(0.0, 0.0)) for i, d in enumerate(ppl)]
                    i03_matched += len(_matched_people(dets, vis, th))
                    i03_visible += len(vis)
                steps = [r for r in group if r.solver_step]
                counts['records'] += len(group)
                counts['solver_steps'] += len(steps)
                if not steps:
                    continue
                lists = [r.people for r in steps]
                if all(l == lists[0] for l in lists):
                    use = [steps[0]]
                    counts['answers_shared'] += 1
                else:
                    use = steps
                    counts['answers_unequal_kept_both'] += len(steps)
                groups_eps = {truth.group_for(r.episode_id, sc.truths[r.episode_id].gt.target_ids[0]) for r in steps}
                if len(groups_eps) > 1:
                    counts['answers_with_episode_groups_differing'] += 1
                for r in use:
                    tid = sc.truths[r.episode_id].gt.target_ids[0]
                    counts[f'target_id_{tid}'] += 1
                    g = truth.group_for(r.episode_id, tid)
                    roles = {p.id: role_of(p, g) for p in vis}
                    det = [DetectedPerson.model_validate(p) for p in r.people]
                    persons, false = analyse_answer(det, vis, roles, obs, th)
                    answers.append({'model': key, 'run': ref.folder, 'arm': ref.arm,
                                    'domain': 'realistic' if ref.arm in REALISTIC else 'placeholder',
                                    'scene': scene_id, 'trial': trial, 'episode': r.episode_id,
                                    'situation': r.situation, 'n_det': len(det), 'n_visible': len(vis),
                                    'persons': persons, 'false': false})
        with open(NOISE / f'answers_{key}.jsonl', 'w') as fh:
            for a in answers:
                fh.write(json.dumps(a) + '\n')
        all_answers[key] = answers
        i03_check[key] = (i03_matched, i03_visible, dict(counts))
    # I03 cross-check against phase1b/i03/stageA/vlm_rows.jsonl (same runs)
    vrows = [json.loads(l) for l in (L.RES / 'phase1b/i03/stageA/vlm_rows.jsonl').read_text().splitlines() if l]
    say('Cross-check (FROZEN section 8): recall computed I03\'s way (one answer per (run, scene, trial) = union of the '
        'records\' people lists, all visible people, staff included) against vlm_rows.jsonl\'s pooled recall')
    check_ok = True
    for key, stage, model, label in MODELS:
        runs = {ref.folder for ref in runs_of(stage, model)}
        vr = [r for r in vrows if r['run'] in runs and r['has_list']]
        vm, vv = sum(r['matched'] for r in vr), sum(r['n_visible'] for r in vr)
        m, v, counts = i03_check[key]
        ok = (m, v) == (vm, vv)
        check_ok &= ok
        say(f'  {label:14s} mine {m}/{v} = {m / v:.4f}; vlm_rows {vm}/{vv} = {vm / vv:.4f}  -> {"EQUAL" if ok else "MISMATCH"}')
        say(f'      records {counts["records"]}, solver steps {counts["solver_steps"]}, answers shared by both '
            f'episodes {counts.get("answers_shared", 0)}, unequal lists kept both {counts.get("answers_unequal_kept_both", 0)}, '
            f'answers whose episodes have different groups {counts.get("answers_with_episode_groups_differing", 0)}, '
            f'target ids {[(k, v) for k, v in counts.items() if k.startswith("target_id")]}')
    say(f'  cross-check: {"PASS" if check_ok else "FAIL (noisy arm stopped until explained)"}')
    say('')
    model_doc = {'frozen_sha256': L.FROZEN_SHA, 'pair_px': PAIR_PX, 'min_pool': MIN_POOL,
                 'definitions': {
                     'p_miss': 'missed / visible, by role group (member, other, staff); pooled over both domains',
                     'q_mis': 'mislocated / (located + mislocated), pooled over roles',
                     'loc_offsets': 'located (radial, lateral) offsets in the camera-ray frame, metres',
                     'mis_offsets': 'mislocated (radial, lateral) offsets of the paired detection from its person',
                     'false_counts': 'false people per answer (empirical)',
                     'false_offsets': '(radial, lateral) from the visible person whose mark projects nearest in pixels',
                     'false_yaw_none': 'share of false people with yaw None',
                     'yaw': 'per A0 view class: p_unknown and signed errors (radians) on located pairs'},
                 'models': {}, 'placeholder_only': {}, 'check_i03': check_ok}
    pooled_mis = [p['offset'] for k in all_answers for a in all_answers[k] for p in a['persons'] if p['outcome'] == 'mislocated']
    pooled_false = [f['offset'] for k in all_answers for a in all_answers[k] for f in a['false'] if f['offset'] is not None]
    pooled_yaw = {v: [p['yaw_err'] for k in all_answers for a in all_answers[k] for p in a['persons']
                      if p['outcome'] == 'located' and p['view'] == v and p['yaw_known']] for v in VIEWS}
    pooled_yaw_n = {v: sum(1 for k in all_answers for a in all_answers[k] for p in a['persons']
                           if p['outcome'] == 'located' and p['view'] == v) for v in VIEWS}

    def params(answers, pool_label):
        persons = [p for a in answers for p in a['persons']]
        out = {'n_answers': len(answers), 'n_person_outcomes': len(persons)}
        out['p_miss'], out['n_role'] = {}, {}
        for role in ROLES:
            ps = [p for p in persons if p['role'] == role]
            out['n_role'][role] = {'visible': len(ps), 'missed': sum(p['outcome'] == 'missed' for p in ps),
                                   'mislocated': sum(p['outcome'] == 'mislocated' for p in ps),
                                   'located': sum(p['outcome'] == 'located' for p in ps)}
            out['p_miss'][role] = (out['n_role'][role]['missed'] / len(ps)) if ps else None
        listed = [p for p in persons if p['outcome'] != 'missed']
        mis = [p for p in listed if p['outcome'] == 'mislocated']
        out['q_mis'] = len(mis) / len(listed) if listed else 0.0
        out['n_listed'], out['n_mislocated'] = len(listed), len(mis)
        out['loc_offsets'] = [p['offset'] for p in listed if p['outcome'] == 'located']
        own_mis = [p['offset'] for p in mis]
        out['mis_offsets_source'] = 'own' if len(own_mis) >= MIN_POOL else f'pooled over the three models ({pool_label})'
        out['mis_offsets'] = own_mis if len(own_mis) >= MIN_POOL else pooled_mis
        out['n_mis_offsets_own'] = len(own_mis)
        out['false_counts'] = [len(a['false']) for a in answers]
        own_false = [f['offset'] for a in answers for f in a['false'] if f['offset'] is not None]
        out['false_offsets_source'] = 'own' if len(own_false) >= MIN_POOL else f'pooled over the three models ({pool_label})'
        out['false_offsets'] = own_false if len(own_false) >= MIN_POOL else pooled_false
        out['n_false_offsets_own'] = len(own_false)
        out['n_false_unanchored'] = sum(1 for a in answers for f in a['false'] if f['offset'] is None)
        fy = [f['yaw_none'] for a in answers for f in a['false']]
        out['false_yaw_none'] = (sum(fy) / len(fy)) if fy else 1.0
        out['yaw'] = {}
        for v in VIEWS:
            loc = [p for p in listed if p['outcome'] == 'located' and p['view'] == v]
            known = [p['yaw_err'] for p in loc if p['yaw_known']]
            if len(known) >= MIN_POOL:
                out['yaw'][v] = {'source': 'own', 'n_located': len(loc), 'n_known': len(known),
                                 'p_unknown': 1 - len(known) / len(loc), 'errors': known}
            else:
                pk = pooled_yaw[v]
                out['yaw'][v] = {'source': f'pooled over the three models ({pool_label})', 'n_located_own': len(loc),
                                 'n_known_own': len(known), 'n_located': pooled_yaw_n[v], 'n_known': len(pk),
                                 'p_unknown': (1 - len(pk) / pooled_yaw_n[v]) if pooled_yaw_n[v] else 1.0,
                                 'errors': pk}
        return out

    for key, stage, model, label in MODELS:
        model_doc['models'][key] = params(all_answers[key], 'all seven runs')
        model_doc['models'][key]['label'] = label
        model_doc['models'][key]['by_domain'] = {}
        for dom in ('realistic', 'placeholder'):
            sub = [a for a in all_answers[key] if a['domain'] == dom]
            pd = params(sub, 'all seven runs')
            model_doc['models'][key]['by_domain'][dom] = {k: pd[k] for k in ('n_answers', 'p_miss', 'n_role', 'q_mis',
                                                                             'n_listed', 'n_mislocated')}
            model_doc['models'][key]['by_domain'][dom]['false_per_answer'] = float(np.mean(pd['false_counts'])) if pd['false_counts'] else None
    # secondary arm: rates fitted on the placeholder answers alone (pools also placeholder-only)
    ph_all = {k: [a for a in all_answers[k] if a['domain'] == 'placeholder'] for k in all_answers}
    pooled_mis_ph = [p['offset'] for k in ph_all for a in ph_all[k] for p in a['persons'] if p['outcome'] == 'mislocated']
    pooled_false_ph = [f['offset'] for k in ph_all for a in ph_all[k] for f in a['false'] if f['offset'] is not None]
    saved = (pooled_mis, pooled_false, dict(pooled_yaw), dict(pooled_yaw_n))
    pooled_mis, pooled_false = pooled_mis_ph, pooled_false_ph
    for v in VIEWS:
        pooled_yaw[v] = [p['yaw_err'] for k in ph_all for a in ph_all[k] for p in a['persons']
                         if p['outcome'] == 'located' and p['view'] == v and p['yaw_known']]
        pooled_yaw_n[v] = sum(1 for k in ph_all for a in ph_all[k] for p in a['persons']
                              if p['outcome'] == 'located' and p['view'] == v)
    for key, stage, model, label in MODELS:
        model_doc['placeholder_only'][key] = params(ph_all[key], 'placeholder runs')
    pooled_mis, pooled_false, pooled_yaw, pooled_yaw_n = saved
    (L.SCORE / 'noise_model.json').write_text(json.dumps(model_doc))
    # fit table
    say('Noise-model parameters per model (fit on the seven development runs; n given; pooled over both domains as '
        'ruled; per domain below)')
    for key, stage, model, label in MODELS:
        m = model_doc['models'][key]
        say(f'{label} ({stage} {model}): {m["n_answers"]} answers, {m["n_person_outcomes"]} visible-person outcomes')
        for role in ROLES:
            n = m['n_role'][role]
            pm = m['p_miss'][role]
            say(f'  p_miss[{role}] = {n["missed"]}/{n["visible"]} = {"-" if pm is None else f"{pm:.3f}"}  '
                f'(mislocated {n["mislocated"]}, located {n["located"]})')
        say(f'  q_mis = {m["n_mislocated"]}/{m["n_listed"]} = {m["q_mis"]:.3f}; mislocation offsets: {m["mis_offsets_source"]} '
            f'(own n {m["n_mis_offsets_own"]})')
        lo = np.array(m['loc_offsets'])
        say(f'  located offsets n {len(lo)}: radial median {np.median(lo[:, 0]):+.3f} m, lateral median '
            f'{np.median(lo[:, 1]):+.3f} m, |offset| median {np.median(np.hypot(lo[:, 0], lo[:, 1])):.3f} m, p90 '
            f'{np.quantile(np.hypot(lo[:, 0], lo[:, 1]), 0.9):.3f} m')
        if m['mis_offsets']:
            mo = np.array(m['mis_offsets'])
            say(f'  mislocation offsets used n {len(mo)}: radial median {np.median(mo[:, 0]):+.3f} m, |offset| median '
                f'{np.median(np.hypot(mo[:, 0], mo[:, 1])):.3f} m, max {np.hypot(mo[:, 0], mo[:, 1]).max():.3f} m')
        fc = collections.Counter(m['false_counts'])
        say(f'  false people per answer: mean {np.mean(m["false_counts"]):.3f}, distribution {dict(sorted(fc.items()))}; '
            f'offsets {m["false_offsets_source"]} (own n {m["n_false_offsets_own"]}, unanchored {m["n_false_unanchored"]}); '
            f'yaw None share {m["false_yaw_none"]:.3f}')
        if m['false_offsets']:
            fo = np.array(m['false_offsets'])
            say(f'      false offset |d| median {np.median(np.hypot(fo[:, 0], fo[:, 1])):.3f} m, radial median '
                f'{np.median(fo[:, 0]):+.3f} m')
        for v in VIEWS:
            y = m['yaw'][v]
            e = np.degrees(np.abs(y['errors'])) if y['errors'] else np.array([])
            say(f'  yaw[{v}]: {y["source"]}; located {y["n_located"]}, known {y["n_known"]}, p_unknown '
                f'{y["p_unknown"]:.3f}, |err| median {np.median(e) if len(e) else float("nan"):.1f} deg, within 45 '
                f'{(e <= 45).mean() if len(e) else float("nan"):.3f}')
        for dom in ('realistic', 'placeholder'):
            d = m['by_domain'][dom]
            say(f'  [{dom}] answers {d["n_answers"]}; p_miss ' + ', '.join(
                f'{r} {d["n_role"][r]["missed"]}/{d["n_role"][r]["visible"]}' for r in ROLES) +
                f'; q_mis {d["n_mislocated"]}/{d["n_listed"]}; false/answer {d["false_per_answer"]:.3f}')
        say('')
    say('Secondary arm (placeholder answers alone; descriptive):')
    for key, stage, model, label in MODELS:
        m = model_doc['placeholder_only'][key]
        say(f'  {label}: answers {m["n_answers"]}; p_miss ' + ', '.join(
            f'{r} {m["n_role"][r]["missed"]}/{m["n_role"][r]["visible"]}' for r in ROLES) +
            f'; q_mis {m["n_mislocated"]}/{m["n_listed"]}; false/answer {np.mean(m["false_counts"]):.3f}')
    (NOISE / 'fit.txt').write_text('\n'.join(lines) + '\n')


# ------------------------------------------------------------------------------------------------------- simulation
class Model:
    def __init__(self, params):
        self.p = params
        self.loc = np.array(params['loc_offsets'], dtype=float)
        self.mis = np.array(params['mis_offsets'], dtype=float).reshape(-1, 2)
        self.false_counts = np.array(params['false_counts'], dtype=int)
        self.false_off = np.array(params['false_offsets'], dtype=float).reshape(-1, 2)
        self.yaw = {v: (params['yaw'][v]['p_unknown'], np.array(params['yaw'][v]['errors'], dtype=float)) for v in VIEWS}

    def p_miss(self, role):
        v = self.p['p_miss'].get(role)
        return 0.0 if v is None else v


def scene_rng(model_key, scene_key, draw):
    return np.random.default_rng([NOISE_SEED, MODEL_INDEX[model_key], zlib.crc32(scene_key.encode()), draw])


def simulate(model: Model, rng, truth, obs, th, roles):
    """One draw: [(FoundPerson, source id or None)] in found order (true-sourced first, in OraclePeople order, then
    false people). Fixed draw order per visible person: u_miss, u_mis, u_offset, u_yaw_known, u_yaw_err; then the false
    count index and, per false person, u_anchor, u_offset, u_yaw_none, yaw ~ U(-pi, pi)."""
    cam_xy = obs.camera.T_world_camera[:2, 3]
    found_true = OraclePeople(truth, th)(obs).found
    by_xy = {(p.x, p.y): p for p in truth.people}
    vis = [by_xy[tuple(f.xy)] for f in found_true]
    out = []
    for f, p in zip(found_true, vis):
        u_miss, u_mis, u_off, u_yk, u_ye = rng.random(5)
        if u_miss < model.p_miss(roles[p.id]):
            continue
        offs = model.mis if (u_mis < model.p['q_mis'] and len(model.mis)) else model.loc
        off = offs[min(int(u_off * len(offs)), len(offs) - 1)]
        xy = from_offset(cam_xy, (p.x, p.y), off)
        a = abs(math.degrees(wrap_angle(bearing((p.x, p.y), cam_xy) - p.yaw)))
        p_unk, errs = model.yaw[view_class(a)]
        if u_yk < p_unk or len(errs) == 0:
            yaw = None
        else:
            yaw = float(wrap_angle(p.yaw + errs[min(int(u_ye * len(errs)), len(errs) - 1)]))
        out.append((FoundPerson(xy=xy, yaw=yaw, pixel=tuple(f.pixel), label=None), p.id))
    k = int(model.false_counts[int(rng.integers(len(model.false_counts)))]) if len(model.false_counts) else 0
    for _ in range(k):
        u_a, u_o, u_y = rng.random(3)
        yaw_u = float(rng.uniform(-math.pi, math.pi))
        if not vis or not len(model.false_off):
            continue
        anchor = vis[min(int(u_a * len(vis)), len(vis) - 1)]
        off = model.false_off[min(int(u_o * len(model.false_off)), len(model.false_off) - 1)]
        xy = from_offset(cam_xy, (anchor.x, anchor.y), off)
        m = project(obs.camera, (xy[0], xy[1], th.person_mark_height_m))
        if m is None:
            continue
        yaw = None if u_y < model.p['false_yaw_none'] else yaw_u
        out.append((FoundPerson(xy=xy, yaw=yaw, pixel=(float(m[0]), float(m[1])), label=None), None))
    return out


def load_models(which='models'):
    doc = json.loads((L.SCORE / 'noise_model.json').read_text())
    return {k: Model(v) for k, v in doc[which].items()}


if __name__ == '__main__':
    if sys.argv[1] == 'fit':
        fit()
