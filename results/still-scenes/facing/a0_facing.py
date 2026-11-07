"""A0 (I09's step A0, inside I01): how well the VLM's people call gets each person's facing, against ground truth.

Read only through research_harness (H.runs, H.records, H.scene, H.recorded_config). No model call, nothing tuned.

What the people call reports (WT/src/sgi/schemas.py FACINGS, PersonEntry; stages/people.py _heading, _place):
  every person gets `facing`, one of 8 image-relative bins a step of 45 deg apart (towards_camera, towards_right,
  right, away_right, away_from_camera, away_left, left, towards_left) or `unclear`, and an optional `faced_point`.
  The pipeline's world yaw (DetectedPerson.yaw, what the solver uses) is the bearing from the person's centre to the
  lifted faced point when that lifts at least person_radius_m (0.25 m) away; otherwise the bearing from the centre
  to the camera turned by the bin's row of _TURN_DEG (0, 45, 90, 135, 180, -135, -90, -45 deg, anticlockwise);
  otherwise (unclear and no usable faced point) None = unknown.

Unit (deduplicated): one people answer per (arm, scene, trial). The answer is cached per scene and trial, so all
  episodes of a scene within a trial share it. Its people = the union over that key's records of their people lists
  (result.people, after the package's merge; the cut at max_people = 40 never binds here), identical entries once.
  Each listed person is tied back to its raw entry by re-placing every entry of the call's raw_text with the
  package's own _place (same observation, recorded thresholds); xy, pixel and yaw must agree exactly.

Matching (the package's own, sgi.run._matched_people): the union's people and the ground-truth people that pass
  sgi.gt.derive.is_visible under the run's recorded thresholds, paired greedily nearest first, one to one, within
  2 x person_radius_m = 0.5 m. Unmatched detections are not scored for facing (they are counted).

Measures per matched pair:
  err      = |wrap(reported world yaw - true yaw)| in deg, [0, 180]; reported unknown -> no err.
  within45 = err <= 45. Two denominators: of pairs with a known yaw ("known"), and of all pairs (unknown = miss).
  label-only err: the bin alone turned into a world yaw with the package's table and the bearing from the reported
           centre to the camera (what the pipeline does without a faced point); unclear -> unknown.
  angle to camera = |wrap(bearing(true person -> camera optical centre xy) - true yaw)|: frontal < 45,
           oblique 45-75, profile 75-105 (both ends in), back > 105 (as RERUN R55 / profile_angles.py).
  true bin = the bin whose turn is nearest wrap(true yaw - bearing(true person -> camera)); confusion is true bin x
           reported label (9 columns), and true bin x the reported world yaw's bin (relative to the same bearing).
  front/back correct (I09's profile criterion read for the VLM): err < 90 among known.
  situation = the situation of the scene's allowed records (each realistic scene has one).
  role (as rerun/profile/profile_angles.py): line member / group member / staff / other.

Models (realistic arms real, staffed, plain; method con): primary rows Qwen 3.8 s19, GLM s19, DeepSeek s20, Haiku,
  Sonnet, Gemini s23. Check rows: Qwen s21, GLM s23, DeepSeek s23 (their people answers are cache copies of s19/s20;
  identity is checked and they are not new samples). Context rows: placeholder panel s8 convlm (VLM people), all models.

CIs: percentile cluster bootstrap over layouts (scene id without its arm; 2000 draws, seed 0), resampling whole
  layouts with all their arms, trials and people.

Writes a0_tables.txt (and prints it).
"""
from __future__ import annotations

import collections
import json
import math
import random
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H  # noqa: E402
from sgi.geometry import bearing, wrap_angle  # noqa: E402
from sgi.gt.derive import is_visible  # noqa: E402
from sgi.run import _matched_people  # noqa: E402
from sgi.schemas import FACINGS, people_from_wire  # noqa: E402
from sgi.stages.people import _TURN_DEG, _place, _apart, _EPS  # noqa: E402
from sgi.stages.goal import _lift_point  # noqa: E402
from sgi.types import DetectedPerson  # noqa: E402

OUT = Path(__file__).resolve().parent
REAL = ('real', 'staffed', 'plain')
BINS8 = FACINGS[:8]
VIEW_CLASSES = ('frontal', 'oblique', 'profile', 'back')
LINE_SITS = ('waiting_line',)
GROUP_SITS = ('conversation_group', 'audience', 'group_photo')
N_BOOT = 2000

PRIMARY = [  # (row label, stage, model)
    ('Qwen 3.8 (s19)', 's19', 'qwen38_low'),
    ('GLM-5.3 (s19)', 's19', 'glm53'),
    ('DeepSeek V4.1 (s20)', 's20', 'dsv41'),
    ('Haiku 5.5 (s23)', 's23', 'haiku55'),
    ('Sonnet 5.5 low (s23)', 's23', 'sonnet55_low'),
    ('Gemini 3.8 low (s23)', 's23', 'gemini38_low'),
]
COPIES = [  # (row label, stage, model, the primary it copies)
    ('Qwen 3.8 (s21)', 's21', 'qwen38_low', 'Qwen 3.8 (s19)'),
    ('GLM-5.3 (s23)', 's23', 'glm53', 'GLM-5.3 (s19)'),
    ('DeepSeek V4.1 (s23)', 's23', 'dsv41', 'DeepSeek V4.1 (s20)'),
]

lines_out: list[str] = []


def say(*parts):
    text = " ".join(str(p) for p in parts)
    print(text)
    lines_out.append(text)


def deg(a):
    return math.degrees(a)


def angdiff(a, b):
    """|wrap(a - b)| in degrees, radians in."""
    return abs(deg(wrap_angle(a - b)))


def nearest_bin(rel_rad):
    """The FACINGS bin whose turn (package table) is nearest to a relative angle (radians)."""
    best = min(BINS8, key=lambda f: abs(wrap_angle(rel_rad - math.radians(_TURN_DEG[f]))))
    return best


def view_class(a):
    if a < 45:
        return 'frontal'
    if a < 75:
        return 'oblique'
    if a <= 105:
        return 'profile'
    return 'back'


def groups_by_situation(truth, allowed_sits):
    out = collections.defaultdict(set)
    for ep in truth.episodes.values():
        if ep.situation not in allowed_sits:
            continue
        g = ep.group_at_target
        ids = list(g.values()) if isinstance(g, dict) else [g]
        for gid in ids:
            if gid is not None:
                out[gid].add(ep.situation)
    return out


def role_of(person, gsits):
    if person.role:
        return 'staff'
    if person.group is not None:
        sits = gsits.get(person.group, set())
        if sits & set(LINE_SITS):
            return 'line'
        if sits & set(GROUP_SITS):
            return 'group'
    return 'other'


_SCENE = {}


def scene_info(ref, scene_id, allowed_sits):
    key = (str(ref.scenes_dir), scene_id, H.recorded_config(ref)[1].thresholds.model_dump_json())
    if key not in _SCENE:
        sc = H.scene(ref, scene_id)
        th = H.recorded_config(ref)[1].thresholds
        vis = [p for p in sc.truth.people if is_visible(p, sc.observation, th)]
        _SCENE[key] = (sc, th, vis, groups_by_situation(sc.truth, allowed_sits))
    return _SCENE[key]


def yaw_source(entry, obs, th, centre):
    """Which channel gave the pipeline's yaw (replicates sgi.stages.people._heading's branches)."""
    faced = None if entry.faced_point is None else _lift_point(obs, entry.faced_point, th)[1]
    if faced is not None and _apart(centre, faced[:2]) >= th.person_radius_m - _EPS:
        return 'faced'
    if entry.facing in _TURN_DEG:
        return 'label'
    return 'none'


def collect(stage, model, arms=REAL, method='con'):
    """Per (arm, scene, trial): the deduplicated answer with its matched pairs. Returns (answers, checks)."""
    checks = collections.Counter()
    answers = {}
    for arm in arms:
        refs = H.runs(stage=stage, arm=arm, model=model, method=method)
        if not refs:
            continue
        assert len(refs) == 1, (stage, arm, model, [r.folder for r in refs])
        ref = refs[0]
        recs = list(H.records(ref))
        allowed_sits = {r.situation for r in recs}
        scene_sits = collections.defaultdict(set)
        for r in recs:
            scene_sits[r.scene_id].add(r.situation)
        by_key = collections.defaultdict(list)
        for r in recs:
            if r.people is None:
                continue
            by_key[(arm, r.scene_id, r.trial)].append(r)
        for key, rs in sorted(by_key.items()):
            arm_, scene_id, trial = key
            sc, th, vis, gsits = scene_info(ref, scene_id, allowed_sits)
            obs = sc.observation
            cam_xy = obs.camera.T_world_camera[:2, 3]
            raw_texts, union = set(), {}
            call_meta = None
            for r in rs:
                for c in r.result.get('calls') or []:
                    if c['stage'] == 'people':
                        raw_texts.add(c['raw_text'])
                        call_meta = (c['point_format'], tuple(c['image_size']), c['status'])
                for p in r.people:
                    d = DetectedPerson.model_validate(p)
                    k = (round(d.xy[0], 9), round(d.xy[1], 9), round(d.pixel[0], 6), round(d.pixel[1], 6))
                    if k in union:
                        prev = union[k]
                        if (prev.yaw is None) != (d.yaw is None) or (prev.yaw is not None and abs(prev.yaw - d.yaw) > 1e-9):
                            checks['union: same place, different yaw'] += 1
                        continue
                    union[k] = d
            checks['answers'] += 1
            if len(raw_texts) != 1:
                checks[f'answers with {len(raw_texts)} distinct raw texts'] += 1
            raw = sorted(raw_texts)[0] if raw_texts else None
            # Re-place every raw entry with the package's code to recover its label and yaw channel.
            entries = []
            if raw is not None and call_meta and call_meta[2] == 'ok':
                pf, size, _ = call_meta
                try:
                    ans = people_from_wire(json.loads(raw), pf, size)
                except Exception as exc:  # an older schema: label unknown, the pipeline yaw is still scored
                    checks[f'raw answer not readable ({type(exc).__name__})'] += 1
                    ans = None
                for e in (ans.people if ans is not None else []):
                    fp = _place(e, obs, th)
                    entries.append((e, fp))
            checks['raw entries'] += len(entries)
            checks['raw entries not lifted'] += sum(fp is None for _, fp in entries)
            dets = []
            for n, (k, d) in enumerate(sorted(union.items()), 1):
                hit = [(e, fp) for e, fp in entries if fp is not None
                       and math.dist(fp.xy, d.xy) < 1e-6 and math.dist(fp.pixel, d.pixel) < 1e-6]
                if len(hit) != 1:
                    checks[f'listed person tied to {len(hit)} raw entries'] += 1
                    e = None
                else:
                    e, fp = hit[0]
                    same = (fp.yaw is None and d.yaw is None) or (
                        fp.yaw is not None and d.yaw is not None and abs(wrap_angle(fp.yaw - d.yaw)) < 1e-6)
                    checks['re-placed yaw agrees' if same else 're-placed yaw DISAGREES'] += 1
                dets.append((DetectedPerson(number=n, xy=d.xy, yaw=d.yaw, pixel=d.pixel, label=d.label), e))
            checks['listed people (union)'] += len(dets)
            checks['raw entries dropped by merge'] += sum(fp is not None for _, fp in entries) - len(dets)
            matched = _matched_people([d for d, _ in dets], vis, th)   # number -> true id
            by_id = {p.id: p for p in vis}
            pairs = []
            for d, e in dets:
                tid = matched.get(d.number)
                if tid is None:
                    continue
                p = by_id[tid]
                true_yaw = p.yaw
                b_true = math.atan2(cam_xy[1] - p.y, cam_xy[0] - p.x)
                cam_ang = angdiff(b_true, true_yaw)
                rel_true = wrap_angle(true_yaw - b_true)
                tbin = nearest_bin(rel_true)
                err = None if d.yaw is None else angdiff(d.yaw, true_yaw)
                rep_bin = None if d.yaw is None else nearest_bin(wrap_angle(d.yaw - b_true))
                label = None if e is None else e.facing
                if label is None or label not in _TURN_DEG:
                    lab_err = None
                else:
                    lab_yaw = wrap_angle(bearing(np.array(d.xy), cam_xy) + math.radians(_TURN_DEG[label]))
                    lab_err = angdiff(lab_yaw, true_yaw)
                src = None if e is None else yaw_source(e, obs, th, np.array(d.xy))
                # signed turn toward the camera: + = the reported yaw points more toward the camera than the truth
                to_cam = None if d.yaw is None else cam_ang - angdiff(b_true, d.yaw)
                lab_to_cam = None if lab_err is None else cam_ang - angdiff(b_true, lab_yaw)
                pairs.append(dict(arm=arm_, scene=scene_id, layout=scene_id.split('_', 1)[1], trial=trial,
                                  pid=p.id, role=role_of(p, gsits), situation='+'.join(sorted(scene_sits[scene_id])),
                                  cam_ang=cam_ang, view=view_class(cam_ang), tbin=tbin, err=err, rep_bin=rep_bin,
                                  label=label, lab_err=lab_err, src=src, gap=math.dist(d.xy, (p.x, p.y)),
                                  to_cam=to_cam, lab_to_cam=lab_to_cam))
            answers[key] = dict(raw=raw, n_vis=len(vis), n_det=len(dets), n_matched=len(pairs),
                                pairs=pairs, situation='+'.join(sorted(scene_sits[scene_id])))
    return answers, checks


# ---------------------------------------------------------------- summaries

def metrics(pairs):
    n = len(pairs)
    known = [p for p in pairs if p['err'] is not None]
    nk = len(known)
    w = sum(p['err'] <= 45 for p in known)
    lab_known = [p for p in pairs if p['lab_err'] is not None]
    lw = sum(p['lab_err'] <= 45 for p in lab_known)
    return dict(
        n=n, nk=nk,
        unk=(n - nk) / n if n else float('nan'),
        w45k=w / nk if nk else float('nan'),
        w45a=w / n if n else float('nan'),
        med=statistics.median([p['err'] for p in known]) if known else float('nan'),
        fb=sum(p['err'] < 90 for p in known) / nk if nk else float('nan'),
        lab_n=len(lab_known),
        lab_unk=sum(p['label'] == 'unclear' for p in pairs) / n if n else float('nan'),
        lab_w45=lw / len(lab_known) if lab_known else float('nan'),
        lab_med=statistics.median([p['lab_err'] for p in lab_known]) if lab_known else float('nan'),
        faced=sum(p['src'] == 'faced' for p in pairs) / n if n else float('nan'),
        n_people=len({(p['layout'], p['pid']) for p in pairs}),
        n_layouts=len({p['layout'] for p in pairs}),
    )


def boot_ci(pairs, stat, seed=0, n_boot=N_BOOT):
    """Percentile 95% CI of stat(pairs) resampling whole layouts."""
    by_layout = collections.defaultdict(list)
    for p in pairs:
        by_layout[p['layout']].append(p)
    layouts = sorted(by_layout)
    if len(layouts) < 2:
        return (float('nan'), float('nan'))
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        draw = [p for _ in layouts for p in by_layout[rng.choice(layouts)]]
        v = stat(draw)
        if not math.isnan(v):
            vals.append(v)
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))])


def st_w45k(ps):
    k = [p for p in ps if p['err'] is not None]
    return sum(p['err'] <= 45 for p in k) / len(k) if k else float('nan')


def st_w45a(ps):
    return sum(p['err'] is not None and p['err'] <= 45 for p in ps) / len(ps) if ps else float('nan')


def st_med(ps):
    k = [p['err'] for p in ps if p['err'] is not None]
    return statistics.median(k) if k else float('nan')


HEAD = (f"{'row':34s} {'n':>4s} {'nk':>4s} {'ppl':>4s} {'unk':>5s} {'w45k':>5s} {'w45a':>5s} {'med':>6s} "
        f"{'f/b':>5s} {'faced':>5s} | {'Lw45':>5s} {'Lmed':>6s} {'Lunc':>5s}")


def row(label, pairs):
    m = metrics(pairs)
    if not m['n']:
        return f"{label:34s} {0:4d}"
    return (f"{label:34s} {m['n']:4d} {m['nk']:4d} {m['n_people']:4d} {m['unk']:5.3f} {m['w45k']:5.3f} {m['w45a']:5.3f} "
            f"{m['med']:6.1f} {m['fb']:5.3f} {m['faced']:5.3f} | {m['lab_w45']:5.3f} {m['lab_med']:6.1f} "
            f"{m['lab_unk']:5.3f}")


def confusion(pairs, col_key, cols):
    tab = collections.Counter((p['tbin'], p[col_key]) for p in pairs)
    short = {'towards_camera': 'tc', 'towards_right': 'tr', 'right': 'r', 'away_right': 'ar',
             'away_from_camera': 'afc', 'away_left': 'al', 'left': 'l', 'towards_left': 'tl', 'unclear': 'unc',
             None: 'unk'}
    out = [f"    {'true \\ reported':16s} " + " ".join(f"{short[c]:>4s}" for c in cols) + "   n  diag"]
    for t in BINS8:
        cnt = [tab[(t, c)] for c in cols]
        n = sum(cnt)
        diag = tab[(t, t)] / n if n else float('nan')
        out.append(f"    {t:16s} " + " ".join(f"{x:4d}" for x in cnt) + f" {n:4d} {diag:5.2f}")
    return out


def main():
    say("A0: VLM facing accuracy (I09 step A0, inside I01). Realistic arms real/staffed/plain, method con.")
    say("Unit: one people answer per (arm, scene, trial); people = union of the key's records' people lists;")
    say("matched to visible true people one-to-one within 0.5 m (sgi.run._matched_people).")
    say("")
    data = {}
    all_checks = {}
    for label, stage, model in PRIMARY:
        data[label], all_checks[label] = collect(stage, model)
    for label, stage, model, _ in COPIES:
        data[label], all_checks[label] = collect(stage, model)

    say("=== 0. Integrity checks per row ===")
    for label in [r[0] for r in PRIMARY] + [r[0] for r in COPIES]:
        say(f"{label:24s}", dict(all_checks[label]))
    say("")
    say("Copy check (people answers of the check rows against the primary they copy, by (arm, scene, trial)):")
    for label, stage, model, src in COPIES:
        a, b = data[label], data[src]
        shared = set(a) & set(b)
        same = sum(a[k]['raw'] == b[k]['raw'] for k in shared)
        say(f"  {label:22s} vs {src:22s}: keys {len(a)} vs {len(b)}; shared {len(shared)}; byte-identical raw "
            f"{same}; only in check row {len(set(a) - set(b))}; only in primary {len(set(b) - set(a))}")
    say("")

    say("=== 1. Detection context (answers, people) ===")
    say(f"{'row':24s} {'answers':>7s} {'visible':>7s} {'listed':>6s} {'matched':>7s} {'unmatched':>9s} "
        f"{'recall':>6s} {'median gap m':>12s}")
    for label in [r[0] for r in PRIMARY]:
        ans = data[label]
        nv = sum(a['n_vis'] for a in ans.values())
        nd = sum(a['n_det'] for a in ans.values())
        nm = sum(a['n_matched'] for a in ans.values())
        gaps = [p['gap'] for a in ans.values() for p in a['pairs']]
        say(f"{label:24s} {len(ans):7d} {nv:7d} {nd:6d} {nm:7d} {nd - nm:9d} {nm / nv:6.3f} "
            f"{statistics.median(gaps):12.3f}")
    say("")

    say("=== 2. Facing accuracy per model, all matched people (primary rows) ===")
    say("n = matched pairs (answer x true person); nk = with a known yaw; ppl = distinct true people (layout, id);")
    say("unk = share with no yaw (unclear and no usable faced point); w45k = share err <= 45 of nk; w45a = of n")
    say("(unknown = miss); med = median err (deg, nk); f/b = share err < 90 of nk; faced = share whose yaw came from")
    say("the faced point; L* = the 8-bin label alone (Lw45 of known labels, Lmed, Lunc = share 'unclear').")
    say(HEAD)
    allp = {}
    for label in [r[0] for r in PRIMARY]:
        allp[label] = [p for a in data[label].values() for p in a['pairs']]
        say(row(label, allp[label]))
    say("check rows (copies, not new samples):")
    for label, *_ in COPIES:
        say(row(label, [p for a in data[label].values() for p in a['pairs']]))
    say("")
    say("Cluster-bootstrap 95% CIs over layouts (2000 draws, seed 0):")
    for label in [r[0] for r in PRIMARY]:
        ps = allp[label]
        say(f"  {label:24s} w45k {st_w45k(ps):.3f} {boot_ci(ps, st_w45k)}  w45a {st_w45a(ps):.3f} "
            f"{boot_ci(ps, st_w45a)}  med {st_med(ps):.1f} {boot_ci(ps, st_med)}  layouts {metrics(ps)['n_layouts']}")
    say("")
    say("=== 2b. Common support: only true people (arm, scene, person) that all six primary models matched in at")
    say("least one trial (removes the survivor effect of different recall; trials are not paired across models) ===")
    common = None
    for label in allp:
        ids = {(p['arm'], p['scene'], p['pid']) for p in allp[label]}
        common = ids if common is None else common & ids
    say(f"common true-person instances: {len(common)}")
    say(HEAD)
    for label in allp:
        say(row(label, [p for p in allp[label] if (p['arm'], p['scene'], p['pid']) in common]))
    say("")

    say("=== 3. Per situation (the scene's situation) ===")
    sits = sorted({p['situation'] for ps in allp.values() for p in ps})
    for s in sits:
        say(f"--- situation {s}")
        say(HEAD)
        for label in allp:
            say(row(label, [p for p in allp[label] if p['situation'] == s]))
    say("")

    say("=== 4. Per role of the true person ===")
    for r in ('line', 'group', 'staff', 'other'):
        say(f"--- role {r}")
        say(HEAD)
        for label in allp:
            say(row(label, [p for p in allp[label] if p['role'] == r]))
    say("")

    say("=== 5. Per angle to the camera (true person; frontal <45, oblique 45-75, profile 75-105, back >105) ===")
    for v in VIEW_CLASSES:
        say(f"--- {v}")
        say(HEAD)
        for label in allp:
            say(row(label, [p for p in allp[label] if p['view'] == v]))
    say("")
    say("--- per situation x angle class (w45k / med / n)")
    for s in sits:
        for label in allp:
            parts = []
            for v in VIEW_CLASSES:
                ps = [p for p in allp[label] if p['situation'] == s and p['view'] == v]
                m = metrics(ps)
                parts.append(f"{v} {m['w45k']:.2f}/{m['med']:.0f}/{m['n']}" if m['n'] else f"{v} -/-/0")
            say(f"  {s:20s} {label:24s} " + "  ".join(parts))
    say("")
    say("--- non-profile (frontal + oblique + back) and profile, with bootstrap CIs on w45k")
    for label in allp:
        npf = [p for p in allp[label] if p['view'] != 'profile']
        pf = [p for p in allp[label] if p['view'] == 'profile']
        mn, mp = metrics(npf), metrics(pf)
        say(f"  {label:24s} non-profile n {mn['n']} w45k {mn['w45k']:.3f} {boot_ci(npf, st_w45k)} med {mn['med']:.1f}"
            f" | profile n {mp['n']} w45k {mp['w45k']:.3f} {boot_ci(pf, st_w45k)} med {mp['med']:.1f}"
            f" front/back {mp['fb']:.3f}")
    say("")

    say("=== 6. Confusion by 45-deg bin: true bin (rows) x reported 8-bin label (cols), pooled over arms and trials ===")
    cols = list(FACINGS) + [None]
    for label in allp:
        say(f"--- {label} (label; 'unk' = listed person not tied to a raw entry)")
        for line in confusion(allp[label], 'label', cols):
            say(line)
    say("")
    say("=== 7. Confusion: true bin x the pipeline yaw's bin (relative to the true person's camera bearing) ===")
    cols2 = list(BINS8) + [None]
    for label in allp:
        say(f"--- {label} (pipeline yaw; 'unk' = no yaw)")
        for line in confusion(allp[label], 'rep_bin', cols2):
            say(line)
    say("")

    say("=== 8. Per arm (w45k / med / n) ===")
    for label in allp:
        parts = []
        for arm in REAL:
            m = metrics([p for p in allp[label] if p['arm'] == arm])
            parts.append(f"{arm} {m['w45k']:.3f}/{m['med']:.1f}/{m['n']}")
        say(f"  {label:24s} " + "  ".join(parts))
    say("")

    say("=== 9. Error histogram (pipeline yaw, deg; 15-deg bins; known only) ===")
    edges = list(range(0, 181, 15))
    for label in allp:
        errs = [p['err'] for p in allp[label] if p['err'] is not None]
        cnt = [sum(edges[i] <= e < edges[i + 1] or (i == len(edges) - 2 and e == 180) for e in errs)
               for i in range(len(edges) - 1)]
        say(f"  {label:24s} " + " ".join(f"{c:3d}" for c in cnt) + f"   (>135: {sum(e > 135 for e in errs)})")
    say("")

    say("=== 9b. The two channels on the same pairs: faced-point yaw (what the pipeline used) vs the label alone ===")
    say("Pairs whose pipeline yaw came from the faced point and whose label is not unclear. w45 of each; paired:")
    say("'label only' = label within 45 and faced not; 'faced only' = the reverse. to_cam = median signed turn toward")
    say("the camera (deg; + = the reported yaw points more toward the camera than the truth).")
    say(f"{'row':24s} {'n':>4s} {'w45 faced':>9s} {'w45 label':>9s} {'label only':>10s} {'faced only':>10s} "
        f"{'med faced':>9s} {'med label':>9s} {'to_cam faced':>12s} {'to_cam label':>12s}")
    for label in allp:
        ps = [p for p in allp[label] if p['src'] == 'faced' and p['lab_err'] is not None]
        if not ps:
            continue
        wf = sum(p['err'] <= 45 for p in ps)
        wl = sum(p['lab_err'] <= 45 for p in ps)
        lo = sum(p['lab_err'] <= 45 < p['err'] for p in ps)
        fo = sum(p['err'] <= 45 < p['lab_err'] for p in ps)
        say(f"{label:24s} {len(ps):4d} {wf / len(ps):9.3f} {wl / len(ps):9.3f} {lo:10d} {fo:10d} "
            f"{statistics.median(p['err'] for p in ps):9.1f} {statistics.median(p['lab_err'] for p in ps):9.1f} "
            f"{statistics.median(p['to_cam'] for p in ps):12.1f} {statistics.median(p['lab_to_cam'] for p in ps):12.1f}")
    say("by angle class (faced-source pairs; w45 faced / w45 label / n):")
    for label in allp:
        parts = []
        for v in VIEW_CLASSES:
            ps = [p for p in allp[label] if p['src'] == 'faced' and p['lab_err'] is not None and p['view'] == v]
            if ps:
                parts.append(f"{v} {sum(p['err'] <= 45 for p in ps) / len(ps):.2f}/"
                             f"{sum(p['lab_err'] <= 45 for p in ps) / len(ps):.2f}/{len(ps)}")
        say(f"  {label:24s} " + "  ".join(parts))
    say("label channel over ALL matched pairs (as if the faced point were not used; unclear = unknown), with CIs:")
    for label in allp:
        ps = allp[label]
        def st_lab(qs):
            k = [q for q in qs if q['lab_err'] is not None]
            return sum(q['lab_err'] <= 45 for q in k) / len(k) if k else float('nan')
        def st_lab_a(qs):
            return sum(q['lab_err'] is not None and q['lab_err'] <= 45 for q in qs) / len(qs) if qs else float('nan')
        npf = [p for p in ps if p['view'] != 'profile']
        pf = [p for p in ps if p['view'] == 'profile' and p['lab_err'] is not None]
        say(f"  {label:24s} Lw45 known {st_lab(ps):.3f} {boot_ci(ps, st_lab)}  all {st_lab_a(ps):.3f}; non-profile "
            f"Lw45 {st_lab(npf):.3f} Lmed {statistics.median(p['lab_err'] for p in npf if p['lab_err'] is not None):.1f}; "
            f"profile front/back {sum(p['lab_err'] < 90 for p in pf) / len(pf):.3f}")
    say("")

    # Placeholder context: s8 convlm (VLM people) on the placeholder dev scenes, old panel.
    say("=== 10. Context: placeholder panel s8 convlm (VLM people; arm ph), same unit and matching ===")
    say(HEAD)
    for ref in H.runs(stage='s8', method='convlm'):
        ans, chk = collect('s8', ref.model, arms=('ph',), method='convlm')
        ps = [p for a in ans.values() for p in a['pairs']]
        nv = sum(a['n_vis'] for a in ans.values())
        say(row(f"s8 {ref.model}", ps) + f"   answers {len(ans)} recall {len(ps) / nv if nv else float('nan'):.3f}"
            f" checks {dict(chk)}")
        for v in VIEW_CLASSES:
            sub = [p for p in ps if p['view'] == v]
            if sub:
                m = metrics(sub)
                say(f"      {v:8s} n {m['n']:4d} w45k {m['w45k']:.3f} med {m['med']:.1f}")

    # Summary for I09
    say("")
    say("=== 11. I09 comparison targets ===")
    best = max(allp, key=lambda k: st_w45k(allp[k]))
    best_a = max(allp, key=lambda k: st_w45a(allp[k]))
    say(f"best VLM by w45k: {best} {st_w45k(allp[best]):.3f}; I09 bar 'more than 10 points below the best VLM' = "
        f"w45 below {st_w45k(allp[best]) - 0.10:.3f}; absolute bar 0.70; the binding bar = "
        f"{max(0.70, st_w45k(allp[best]) - 0.10):.3f}")
    say(f"best VLM by w45a (unknown = miss): {best_a} {st_w45a(allp[best_a]):.3f}")
    for label in allp:
        npf = [p for p in allp[label] if p['view'] != 'profile']
        pf = [p for p in allp[label] if p['view'] == 'profile']
        say(f"  {label:24s} w45k {st_w45k(allp[label]):.3f}; non-profile med {st_med(npf):.1f} w45k {st_w45k(npf):.3f}"
            f"; profile front/back {metrics(pf)['fb']:.3f}")

    (OUT / 'a0_tables.txt').write_text("\n".join(lines_out) + "\n", encoding='utf-8')
    # per-pair dump for later steps (A1 comparison)
    with open(OUT / 'a0_pairs.jsonl', 'w', encoding='utf-8') as f:
        for label in list(allp):
            for p in allp[label]:
                f.write(json.dumps(dict(row=label, **p)) + "\n")


if __name__ == '__main__':
    main()
