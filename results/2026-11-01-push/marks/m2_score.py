"""Marks probe M2 scorer (M2_FROZEN.md). Zero calls.

    m2_score.py --precheck                 coverage of the M2 candidate set, by the 'number' rule with the right number
    m2_score.py --calls runs/<name>/calls.jsonl [--out runs/<name>]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np

import m1_coverage as M
import m2lib as L

X, H = M.X, M.H
HERE = Path(__file__).resolve().parent
RULES = ('both', 'number', 'point')
HARM_SHARE = 0.02          # judged: right -> wrong at most 2% of A0-right records, per model (rule 'both')
PLAN_SHARE = 0.02          # judged: plan exact-match not lower by more than 2% of records, per model
MODEL_OF = {'soclaas_qwen38_low': 'qwen38_low', 'soclaas_glm53': 'glm53', 'soclaas_dsv41': 'dsv41',
            'gemini_fast': 'gemini38_low', 'soclaas_qwen38_next_low': None}
STAGE_OF = {'qwen38_low': 's21'}


def _scene_ctx(cache, wl, scenes_dir, scene_id):
    key = (scenes_dir, scene_id)
    if key not in cache:
        sref = next(s for s in H.distinct_scenes(wl=wl) if str(s.scenes_dir) == scenes_dir and s.scene_id == scene_id)
        sc = H.scene(sref.runs[0], scene_id, wl=wl)
        g = X.Geom(sc.observation.depth, sc.observation.camera)
        cache[key] = (sref, sc, g, L.candidates(str(sref.folder), g))
    return cache[key]


def precheck():
    from sgi.evaluate import match_target
    wl = H.whitelist()
    cache, n, ok, by_kind = {}, 0, 0, collections.Counter()
    tot_kind = collections.Counter()
    for sref in H.distinct_scenes(wl=wl):
        if not M.is_realistic(sref.scene_id):
            continue
        _, sc, g, cands = _scene_ctx(cache, wl, str(sref.scenes_dir), sref.scene_id)
        th = H.recorded_config(sref.runs[0])[1].thresholds
        nums = list(range(1, len(cands) + 1))
        for eid, ep in sc.truth.episodes.items():
            for tid in ep.target_ids:
                kind = sc.truth.targets[tid].kind
                n += 1
                tot_kind[kind] += 1
                hit = any(match_target(t.world, sc.truth, ep, th) == tid
                          for k in nums for t in [L.resolve(g, cands, nums, k, None, 'number')[0]] if t is not None)
                ok += hit
                by_kind[kind] += hit
    print(f'M2 precheck: realistic target instances covered by some mark: {ok}/{n} = {ok / n:.3f}; ' +
          ', '.join(f'{k} {by_kind[k]}/{tot_kind[k]}' for k in sorted(tot_kind)))
    return ok, n


def first_nav(subs):
    for i, s in enumerate(subs):
        if s['type'] == 'navigate' and (s.get('point') is not None or s.get('mark') is not None):
            return i, s
    return None, None


def score(calls_path: Path, out_dir: Path):
    from sgi.evaluate import decomposition_accuracy, match_target, reduce_plan
    from sgi.types import Plan, Subtask
    wl = H.whitelist()
    rows = [json.loads(l) for l in calls_path.read_text().splitlines() if l.strip()]
    cache, recs = {}, {}
    pairs = collections.defaultdict(dict)
    for r in rows:
        pairs[(r['model'], r['scenes_dir'], r['scene_id'], r['episode'], r['trial'])][r['arm']] = r
    results = []
    for (model, sdir, sid, eid, t), arms in sorted(pairs.items()):
        if 'A0' not in arms or 'A2' not in arms:
            continue
        sref, sc, g, cands = _scene_ctx(cache, wl, sdir, sid)
        ep = sc.truth.episodes[eid]
        short = MODEL_OF.get(model)
        arm_name = sid.split('_')[0]
        rec = None
        if short is not None:
            stage = STAGE_OF.get(short, 's23')
            key = (short, arm_name)
            if key not in recs:
                run = H.runs(stage=stage, method='con', model=short, arm=arm_name, wl=wl)[0]
                recs[key] = {(x.scene_id, x.episode_id, x.trial): x for x in H.records(run, wl=wl)}
            rec = recs[key].get((sid, eid, t - 3))
        th = (H.recorded_config(rec.ref)[1].thresholds if rec is not None
              else H.recorded_config(sref.runs[0])[1].thresholds)
        assert [c['mark'] for c in arms['A2']['cands']] == [list(c.mark) for c in cands], 'candidates changed'
        nums = arms['A2']['nums']
        row = dict(model=model, scene=sid, episode=eid, trial=t, n_marks=len(cands))
        for arm in ('A0', 'A2'):
            a = arms[arm]
            if a['status'] != 'ok':
                row[f'{arm}_status'] = a['status']
                continue
            i, s = first_nav(a['subtasks'])
            for rule in (('point',) if arm == 'A0' else RULES):
                tgt = None
                if s is not None:
                    tgt, how = L.resolve(g, cands, nums, s.get('mark') if arm == 'A2' else None,
                                         None if s.get('point') is None else tuple(s['point']), rule)
                right = tgt is not None and match_target(tgt.world, sc.truth, ep, th) is not None
                ok = None
                if rec is not None and tgt is not None and right:
                    ok = bool(X.replay_success(rec, tgt)[0])
                elif rec is not None:
                    ok = False
                name = 'A0' if arm == 'A0' else f'A2_{rule}'
                row[f'{name}_right'] = right
                row[f'{name}_replay'] = ok
            # plan decomposition (navigate steps matched by their own points, as now; A2 by the 'both' rule)
            subs = a['subtasks']
            matched = []
            for s2 in subs:
                if s2['type'] != 'navigate':
                    matched.append(None)
                    continue
                tg = None
                if s2.get('point') is not None or (arm == 'A2' and s2.get('mark') is not None):
                    tg, _ = L.resolve(g, cands, nums, s2.get('mark') if arm == 'A2' else None,
                                      None if s2.get('point') is None else tuple(s2['point']),
                                      'both' if arm == 'A2' else 'point')
                matched.append('ungrounded' if tg is None else match_target(tg.world, sc.truth, ep, th))
            plan = Plan(subtasks=[Subtask(type=s2['type'], description=s2.get('target') or '-') for s2 in subs])
            exact, recall = (decomposition_accuracy(reduce_plan(plan, matched), ep.reference_plan)
                             if ep.reference_plan else (None, None))
            row[f'{arm}_plan_exact'] = exact
            if arm == 'A2' and s is not None:
                row['mark'] = s.get('mark')
                if s.get('mark') is not None:
                    k = nums.index(s['mark'])
                    row['mark_kind'] = cands[k].kind
                    cx = [math.hypot(c.mark[0] - 320, c.mark[1] - 240) for c in cands]
                    row['mark_is_centre'] = k == int(np.argmin(cx))
                    comp_idx = [j for j, c in enumerate(cands) if c.kind == 'component']
                    row['mark_is_largest_component'] = bool(comp_idx) and k == comp_idx[0]
                    # binding: the click lies on (within 15 px of) another candidate but not the numbered one
                    if s.get('point') is not None:
                        u, v = s['point']
                        near = [j for j, c in enumerate(cands) if L._dist_to_cand(g, c, u, v) <= L.NEAR_PX]
                        row['binding_conflict'] = bool(near) and k not in near
        results.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'm2_rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in results))
    lines = []
    say = lines.append
    say(f'M2 scored pairs: {len(results)}')
    verdict_ok = True
    for model in sorted({r['model'] for r in results}):
        rr = [r for r in results if r['model'] == model]
        say(f'\n== {model}: {len(rr)} pairs')
        a0r = sum(bool(r.get('A0_right')) for r in rr)
        say(f'  target right: A0 {a0r}' + ''.join(
            f', A2-{rule} {sum(bool(r.get(f"A2_{rule}_right")) for r in rr)}' for rule in RULES))
        for rule in RULES:
            gain = sum(1 for r in rr if not r.get('A0_right') and r.get(f'A2_{rule}_right'))
            harm = sum(1 for r in rr if r.get('A0_right') and not r.get(f'A2_{rule}_right'))
            say(f'  {rule:<7} gain {gain:>3}  harm {harm:>3}  (harm share of A0-right {harm / max(a0r, 1):.3f})')
        rep = [r for r in rr if r.get('A0_replay') is not None]
        if rep:
            say(f'  replayed success (recorded people and constraints): A0 {sum(bool(r["A0_replay"]) for r in rep)}' +
                ''.join(f', A2-{rule} {sum(bool(r.get(f"A2_{rule}_replay")) for r in rep)}' for rule in RULES) +
                f'  of {len(rep)}')
        pe = [r for r in rr if r.get('A0_plan_exact') is not None and r.get('A2_plan_exact') is not None]
        p0, p2 = sum(bool(r['A0_plan_exact']) for r in pe), sum(bool(r['A2_plan_exact']) for r in pe)
        say(f'  plan exact: A0 {p0}, A2 {p2} of {len(pe)}')
        marks = [r for r in rr if r.get('mark') is not None]
        say(f'  answers with a mark: {len(marks)}/{len(rr)}; person marks {sum(r.get("mark_kind") == "person" for r in marks)}; '
            f'centre-nearest {sum(bool(r.get("mark_is_centre")) for r in marks)}; largest component '
            f'{sum(bool(r.get("mark_is_largest_component")) for r in marks)}; binding conflicts '
            f'{sum(bool(r.get("binding_conflict")) for r in marks)}')
        say('  number histogram: ' + ', '.join(f'{k}:{v}' for k, v in sorted(collections.Counter(r['mark'] for r in marks).items())))
        harm = sum(1 for r in rr if r.get('A0_right') and not r.get('A2_both_right'))
        gain = sum(1 for r in rr if not r.get('A0_right') and r.get('A2_both_right'))
        ok_harm = harm <= HARM_SHARE * max(a0r, 1)
        ok_plan = (p0 - p2) <= PLAN_SHARE * max(len(pe), 1)
        say(f'  JUDGED (rule both): harm {harm} <= {HARM_SHARE:.0%} of {a0r} -> {ok_harm}; plan drop {p0 - p2} <= '
            f'{PLAN_SHARE:.0%} of {len(pe)} -> {ok_plan}; gain {gain} vs harm {harm}')
        verdict_ok &= ok_harm and ok_plan
    say(f'\nVERDICT: {"harm bars met: proceed to the Gemini arm (next step)" if verdict_ok else "a harm bar failed: stop before Gemini"}')
    text = '\n'.join(lines)
    print(text)
    (out_dir / 'm2_out.txt').write_text(text + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--precheck', action='store_true')
    ap.add_argument('--calls')
    ap.add_argument('--out')
    a = ap.parse_args()
    M.check_frozen()
    if a.precheck:
        precheck()
    else:
        score(Path(a.calls), Path(a.out or Path(a.calls).parent))
