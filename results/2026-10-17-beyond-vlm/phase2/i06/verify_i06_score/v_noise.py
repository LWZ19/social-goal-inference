"""Verifier: the perception-noise model (FROZEN 8), fitted on the development runs through the harness, my own code.

    v_noise.py fit   -> out/noise_fit.json, out/noise_fit.txt, out/answers_<model>.jsonl
Also the simulation used by v_validate.py and v_noisy.py (my own draw order; the RNG seed is the frozen one).
"""
from __future__ import annotations

import collections
import importlib.util
import json
import math
import sys
import zlib

import numpy as np

import vlib as V
from vlib import H
from sgi.geometry import bearing, project, wrap_angle
from sgi.gt.derive import is_visible
from sgi.run import OraclePeople, _matched_people
from sgi.stages.people import FoundPerson
from sgi.types import DetectedPerson

FIT = {'qwen': ('s19', 'qwen38_low'), 'glm': ('s19', 'glm53'), 'dsv': ('s20', 'dsv41')}
MODEL_NO = {'qwen': 0, 'glm': 1, 'dsv': 2}
ARMS = ('real', 'staffed', 'plain', 'ph', 'dev_seed0', 'dev_seed1', 'dev_seed2')
REAL = ('real', 'staffed', 'plain')
RADIUS_PX = 40.0
POOL_BELOW = 10
SEED0 = 20261018


def fit_runs(key):
    stage, model = FIT[key]
    refs = []
    for arm in ARMS:
        found = H.runs(stage=stage, arm=arm, model=model, method='con')
        assert len(found) == 1, (key, arm, [r.folder for r in found])
        refs.append(found[0])
    return refs


def view_of(deg):
    return 'frontal' if deg < 45 else 'oblique' if deg < 75 else 'profile' if deg <= 105 else 'back'


def ray_offset(cam, p, q):
    """(radial, lateral) of q from p, in the frame of the ray camera -> p (lateral to its left)."""
    r = np.array(p, float) - np.array(cam, float)
    r /= np.hypot(*r)
    lft = np.array([-r[1], r[0]])
    d = np.array(q, float) - np.array(p, float)
    return [float(d @ r), float(d @ lft)]


def apply_offset(cam, p, off):
    r = np.array(p, float) - np.array(cam, float)
    r /= np.hypot(*r)
    lft = np.array([-r[1], r[0]])
    q = np.array(p, float) + off[0] * r + off[1] * lft
    return float(q[0]), float(q[1])


def role(p, group):
    if group is not None and p.group == group:
        return 'member'
    return 'staff' if p.role else 'other'


def outcomes(det, vis, obs, th, roles):
    cam = obs.camera.T_world_camera[:2, 3]
    matched = _matched_people(det, vis, th)                  # number -> id
    num_of = {v: k for k, v in matched.items()}
    marks = {}
    for p in vis:
        m = project(obs.camera, (p.x, p.y, th.person_mark_height_m))
        marks[p.id] = None if m is None else (float(m[0]), float(m[1]))
    free_d = [d for d in det if d.number not in matched]
    free_p = [p for p in vis if p.id not in num_of and marks[p.id] is not None]
    cand = sorted((math.dist(d.pixel, marks[p.id]), d.number, p.id) for d in free_d for p in free_p)
    pair, used_d, used_p = {}, set(), set()
    for dist, n, pid in cand:
        if dist <= RADIUS_PX and n not in used_d and pid not in used_p:
            pair[pid] = n
            used_d.add(n)
            used_p.add(pid)
    by_n = {d.number: d for d in det}
    persons = []
    for p in vis:
        ang = abs(math.degrees(wrap_angle(bearing((p.x, p.y), cam) - p.yaw)))
        row = {'id': p.id, 'role': roles[p.id], 'view': view_of(ang)}
        n = num_of.get(p.id, pair.get(p.id))
        if n is None:
            row['out'] = 'missed'
        else:
            d = by_n[n]
            row['out'] = 'located' if p.id in num_of else 'mislocated'
            row['off'] = ray_offset(cam, (p.x, p.y), d.xy)
            row['yaw_err'] = None if d.yaw is None else float(wrap_angle(d.yaw - p.yaw))
        persons.append(row)
    false = []
    for d in free_d:
        if d.number in used_d:
            continue
        near = [(math.dist(d.pixel, marks[p.id]), p.id, p) for p in vis if marks[p.id] is not None]
        if not near:
            false.append({'off': None, 'yaw_none': d.yaw is None})
            continue
        _, _, a = min(near, key=lambda z: (z[0], z[1]))
        false.append({'off': ray_offset(cam, (a.x, a.y), d.xy), 'yaw_none': d.yaw is None, 'anchor': a.id})
    return persons, false, len(matched)


def collect(key):
    """Answers of one model's seven fit runs, plus the I03-way union rows for the cross-check."""
    answers, union_rows, tally = [], [], collections.Counter()
    for ref in fit_runs(key):
        th = H.recorded_config(ref)[1].thresholds
        groups = collections.defaultdict(list)
        for r in H.records(ref):
            groups[(r.scene_id, r.trial)].append(r)
        for (sid, trial), recs in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
            sc = H.scene(ref, sid)
            obs, truth = sc.observation, sc.truth
            vis = [p for p in truth.people if is_visible(p, obs, th)]
            listed = [r for r in recs if r.people is not None]
            union_rows.append({'run': ref.folder, 'scene': sid, 'trial': trial, 'has_list': bool(listed),
                               'lists': [r.people for r in listed]})
            steps = [r for r in recs if r.solver_step]
            tally['records'] += len(recs)
            tally['solver_steps'] += len(steps)
            if not steps:
                continue
            same = all(s.people == steps[0].people for s in steps)
            tally['answers_shared' if same else 'answers_unequal'] += 1
            for r in ([steps[0]] if same else steps):
                tid = sc.truths[r.episode_id].gt.target_ids[0]
                grp = truth.group_for(r.episode_id, tid)
                roles = {p.id: role(p, grp) for p in vis}
                det = [DetectedPerson.model_validate(p) for p in r.people]
                persons, false, nm = outcomes(det, vis, obs, th, roles)
                answers.append({'run': ref.folder, 'arm': ref.arm, 'dom': 'realistic' if ref.arm in REAL else 'placeholder',
                                'scene': sid, 'trial': trial, 'episode': r.episode_id, 'situation': r.situation,
                                'persons': persons, 'false': false, 'n_det': len(det), 'n_vis': len(vis)})
    return answers, union_rows, tally


def params(ans, pools):
    P = [p for a in ans for p in a['persons']]
    out = {'n_answers': len(ans), 'n_persons': len(P), 'roles': {}}
    for rl in ('member', 'other', 'staff'):
        q = [p for p in P if p['role'] == rl]
        k = sum(p['out'] == 'missed' for p in q)
        out['roles'][rl] = {'n': len(q), 'missed': k, 'p_miss': (k / len(q)) if q else None}
    lst = [p for p in P if p['out'] != 'missed']
    mis = [p for p in lst if p['out'] == 'mislocated']
    out['n_listed'], out['n_mis'] = len(lst), len(mis)
    out['q_mis'] = len(mis) / len(lst) if lst else 0.0
    out['loc'] = [p['off'] for p in lst if p['out'] == 'located']
    own = [p['off'] for p in mis]
    out['mis'] = own if len(own) >= POOL_BELOW else pools['mis']
    out['mis_src'] = 'own' if len(own) >= POOL_BELOW else 'pooled'
    out['false_counts'] = [len(a['false']) for a in ans]
    fo = [f['off'] for a in ans for f in a['false'] if f['off'] is not None]
    out['false_off'] = fo if len(fo) >= POOL_BELOW else pools['false']
    out['false_src'] = 'own' if len(fo) >= POOL_BELOW else 'pooled'
    fy = [f['yaw_none'] for a in ans for f in a['false']]
    out['false_yaw_none'] = sum(fy) / len(fy) if fy else 1.0
    out['yaw'] = {}
    for v in ('frontal', 'oblique', 'profile', 'back'):
        loc = [p for p in lst if p['out'] == 'located' and p['view'] == v]
        kn = [p['yaw_err'] for p in loc if p['yaw_err'] is not None]
        if len(kn) >= POOL_BELOW:
            out['yaw'][v] = {'src': 'own', 'p_unknown': 1 - len(kn) / len(loc), 'errs': kn, 'n': len(loc)}
        else:
            pl, pk = pools['yaw'][v]
            out['yaw'][v] = {'src': 'pooled', 'p_unknown': (1 - len(pk) / pl) if pl else 1.0, 'errs': pk, 'n': pl}
    return out


def make_pools(by_model):
    allans = [a for k in by_model for a in by_model[k]]
    P = [p for a in allans for p in a['persons']]
    pools = {'mis': [p['off'] for p in P if p['out'] == 'mislocated'],
             'false': [f['off'] for a in allans for f in a['false'] if f['off'] is not None], 'yaw': {}}
    for v in ('frontal', 'oblique', 'profile', 'back'):
        loc = [p for p in P if p['out'] == 'located' and p['view'] == v]
        pools['yaw'][v] = (len(loc), [p['yaw_err'] for p in loc if p['yaw_err'] is not None])
    return pools


def fit():
    spec = importlib.util.spec_from_file_location('people_scorer', V.RES / 'phase1b/i03/people_scorer.py')
    PS = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(PS)
    vrows = [json.loads(x) for x in (V.RES / 'phase1b/i03/stageA/vlm_rows.jsonl').read_text().splitlines() if x]
    by_model, lines, cross = {}, [], {}
    for key in FIT:
        ans, urows, tally = collect(key)
        by_model[key] = ans
        with open(V.OUT / f'answers_{key}.jsonl', 'w') as fh:
            for a in ans:
                fh.write(json.dumps(a) + '\n')
        # I03 cross-check: union of lists per (run, scene, trial), PS.score_people, against vlm_rows
        runs = {r.folder for r in fit_runs(key)}
        vr = {(r['run'], r['scene_id'], r['trial']): r for r in vrows if r['run'] in runs}
        m_mine = v_mine = m_v = v_v = 0
        diff = 0
        for u in urows:
            if not u['has_list']:
                continue
            ref = H.run_ref(u['run'])
            th = H.recorded_config(ref)[1].thresholds
            sc = H.scene(ref, u['scene'])
            uni = {}
            for lst in u['lists']:
                for p in lst:
                    d = DetectedPerson.model_validate(p)
                    uni.setdefault((round(d.xy[0], 9), round(d.xy[1], 9), round(d.pixel[0], 6), round(d.pixel[1], 6)), d)
            ppl = [uni[k] for k in sorted(uni)]
            r = PS.score_people([(d.xy[0], d.xy[1], d.yaw) for d in ppl], sc, th, exempt_xy=PS.exempt_points(sc.truth))
            m_mine += r['matched']
            v_mine += r['n_visible']
            row = vr[(u['run'], u['scene'], u['trial'])]
            m_v += row['matched']
            v_v += row['n_visible']
            diff += (r['matched'], r['n_visible']) != (row['matched'], row['n_visible'])
        # and my answer-level matched/visible (solver-step answers only)
        am = sum(sum(p['out'] == 'located' for p in a['persons']) for a in ans)
        av = sum(len(a['persons']) for a in ans)
        cross[key] = {'union_mine': [m_mine, v_mine], 'vlm_rows': [m_v, v_v], 'rows_differing': diff,
                      'answers_located': [am, av], 'tally': dict(tally)}
    pools = make_pools(by_model)
    ph = {k: [a for a in by_model[k] if a['dom'] == 'placeholder'] for k in by_model}
    pools_ph = make_pools(ph)
    doc = {'models': {}, 'placeholder_only': {}, 'by_domain': {}, 'cross': cross}
    for key in FIT:
        doc['models'][key] = params(by_model[key], pools)
        doc['placeholder_only'][key] = params(ph[key], pools_ph)
        doc['by_domain'][key] = {}
        for dom in ('realistic', 'placeholder'):
            sub = params([a for a in by_model[key] if a['dom'] == dom], pools)
            doc['by_domain'][key][dom] = {'n_answers': sub['n_answers'], 'roles': sub['roles'], 'n_listed': sub['n_listed'],
                                          'n_mis': sub['n_mis'],
                                          'false_mean': float(np.mean(sub['false_counts'])) if sub['false_counts'] else None}
    (V.OUT / 'noise_fit.json').write_text(json.dumps(doc))
    for key in FIT:
        m = doc['models'][key]
        c = cross[key]
        lines.append(f"{key}: answers {m['n_answers']} (tally {c['tally']}); persons {m['n_persons']}")
        lines.append('  p_miss ' + ', '.join(f"{rl} {m['roles'][rl]['missed']}/{m['roles'][rl]['n']}" for rl in m['roles']))
        lines.append(f"  q_mis {m['n_mis']}/{m['n_listed']} = {m['q_mis']:.4f}; mis offsets {m['mis_src']} n {len(m['mis'])}; "
                     f"false/answer {np.mean(m['false_counts']):.4f} {dict(collections.Counter(m['false_counts']))}; "
                     f"false offsets {m['false_src']} n {len(m['false_off'])}; false yaw None {m['false_yaw_none']:.3f}")
        lo = np.array(m['loc'])
        mo = np.array(m['mis'])
        lines.append(f"  located n {len(lo)} radial med {np.median(lo[:, 0]):+.3f}; mislocated radial med {np.median(mo[:, 0]):+.3f}"
                     f" |off| med {np.median(np.hypot(*mo.T)):.3f}")
        lines.append('  yaw ' + '; '.join(f"{v} {y['src']} n {y['n']} p_unk {y['p_unknown']:.3f}" for v, y in m['yaw'].items()))
        for dom, d in doc['by_domain'][key].items():
            lines.append(f"  [{dom}] answers {d['n_answers']}; " + ', '.join(
                f"{rl} {d['roles'][rl]['missed']}/{d['roles'][rl]['n']}" for rl in d['roles']) +
                f"; q_mis {d['n_mis']}/{d['n_listed']}; false/answer {d['false_mean']}")
        lines.append(f"  I03 cross-check: union mine {c['union_mine']}, vlm_rows {c['vlm_rows']}, rows differing "
                     f"{c['rows_differing']}; answer-level located/visible {c['answers_located']}")
    (V.OUT / 'noise_fit.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


# ------------------------------------------------------------------------------------------------------- simulation
class Noise:
    def __init__(self, m):
        self.pm = {rl: (m['roles'][rl]['p_miss'] or 0.0) for rl in m['roles']}
        self.q = m['q_mis']
        self.loc = np.array(m['loc'], float).reshape(-1, 2)
        self.mis = np.array(m['mis'], float).reshape(-1, 2)
        self.fc = np.array(m['false_counts'], int)
        self.fo = np.array(m['false_off'], float).reshape(-1, 2)
        self.fyn = m['false_yaw_none']
        self.yaw = {v: (y['p_unknown'], np.array(y['errs'], float)) for v, y in m['yaw'].items()}


def rng_for(key, scene_key, draw):
    return np.random.default_rng([SEED0, MODEL_NO[key], zlib.crc32(scene_key.encode()), draw])


def simulate(nz: Noise, rng, truth, obs, th, roles):
    """One draw. My order: per visible person, a block of 4 uniforms (miss, mislocated, yaw unknown, spare) and two
    integer indices (offset, yaw error); then the false count index and per false person (anchor, offset) indices,
    a yaw-None uniform and a uniform yaw. Returns [(FoundPerson, source id or None)], true-sourced first."""
    cam = obs.camera.T_world_camera[:2, 3]
    idx = {(p.x, p.y): p for p in truth.people}
    found = OraclePeople(truth, th)(obs).found
    out = []
    for f in found:
        p = idx[(f.xy[0], f.xy[1])]
        u = rng.random(4)
        i_off = int(rng.integers(1 << 30))
        i_yaw = int(rng.integers(1 << 30))
        if u[0] < nz.pm[roles[p.id]]:
            continue
        pool = nz.mis if (u[1] < nz.q and len(nz.mis)) else nz.loc
        off = pool[i_off % len(pool)]
        xy = apply_offset(cam, (p.x, p.y), off)
        ang = abs(math.degrees(wrap_angle(bearing((p.x, p.y), cam) - p.yaw)))
        p_unk, errs = nz.yaw[view_of(ang)]
        yaw = None if (u[2] < p_unk or len(errs) == 0) else float(wrap_angle(p.yaw + errs[i_yaw % len(errs)]))
        out.append((FoundPerson(xy=xy, yaw=yaw, pixel=(float(f.pixel[0]), float(f.pixel[1])), label=None), p.id))
    vis = [idx[(f.xy[0], f.xy[1])] for f in found]
    k = int(nz.fc[int(rng.integers(len(nz.fc)))]) if len(nz.fc) else 0
    for _ in range(k):
        ia = int(rng.integers(1 << 30))
        io = int(rng.integers(1 << 30))
        uy = rng.random()
        yu = float(rng.uniform(-math.pi, math.pi))
        if not vis or not len(nz.fo):
            continue
        a = vis[ia % len(vis)]
        xy = apply_offset(cam, (a.x, a.y), nz.fo[io % len(nz.fo)])
        m = project(obs.camera, (xy[0], xy[1], th.person_mark_height_m))
        if m is None:
            continue
        out.append((FoundPerson(xy=xy, yaw=None if uy < nz.fyn else yu, pixel=(float(m[0]), float(m[1])), label=None),
                    None))
    return out


def load_noise(which='models'):
    doc = json.loads((V.OUT / 'noise_fit.json').read_text())
    return {k: Noise(m) for k, m in doc[which].items()}


if __name__ == '__main__':
    if sys.argv[1] == 'fit':
        fit()
