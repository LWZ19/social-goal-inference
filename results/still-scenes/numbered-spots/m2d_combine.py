"""POST HOC (M2 follow-up, zero calls): what a two-call design would give: the target's number from the marked call
(A2) and the location from the unmarked call's click (A0), same (episode, trial). Rule 'both' with A0's click.
    m2d_combine.py runs/m2_<model>"""
import collections, json, sys
from pathlib import Path
import m1_coverage as M
import m2lib as L
from m2_score import _scene_ctx, MODEL_OF, STAGE_OF, first_nav
X, H = M.X, M.H
run = Path(sys.argv[1])
M.check_frozen()
from sgi.evaluate import match_target
wl = H.whitelist()
rows = [json.loads(l) for l in (run / 'calls.jsonl').read_text().splitlines() if l.strip()]
pairs = collections.defaultdict(dict)
for r in rows:
    pairs[(r['model'], r['scenes_dir'], r['scene_id'], r['episode'], r['trial'])][r['arm']] = r
cache, recs = {}, {}
tot = collections.Counter()
for (model, sdir, sid, eid, t), arms in sorted(pairs.items()):
    if any(arms.get(a, {}).get('status') != 'ok' for a in ('A0', 'A2')):
        continue
    sref, sc, g, cands = _scene_ctx(cache, wl, sdir, sid)
    ep = sc.truth.episodes[eid]
    short = MODEL_OF.get(model); arm_name = sid.split('_')[0]
    rec = None
    if short:
        key = (short, arm_name)
        if key not in recs:
            run_ = H.runs(stage=STAGE_OF.get(short, 's23'), method='con', model=short, arm=arm_name, wl=wl)[0]
            recs[key] = {(x.scene_id, x.episode_id, x.trial): x for x in H.records(run_, wl=wl)}
        rec = recs[key].get((sid, eid, t - 3))
    th = H.recorded_config(rec.ref if rec else sref.runs[0])[1].thresholds
    _, s0 = first_nav(arms['A0']['subtasks']); _, s2 = first_nav(arms['A2']['subtasks'])
    nums = arms['A2']['nums']
    click0 = None if s0 is None or s0.get('point') is None else tuple(s0['point'])
    mark = None if s2 is None else s2.get('mark')
    for name, (m, pt, rule) in {'A0': (None, click0, 'point'), 'combo': (mark, click0, 'both')}.items():
        tgt, _ = L.resolve(g, cands, nums, m, pt, rule) if (pt is not None or m is not None) else (None, None)
        right = tgt is not None and match_target(tgt.world, sc.truth, ep, th) is not None
        tot[f'{name}_right'] += right
        if rec is not None:
            tot[f'{name}_replay'] += bool(right and X.replay_success(rec, tgt)[0])
    tot['n'] += 1
print(f"{run.name}: pairs {tot['n']}; target right A0 {tot['A0_right']}, combo {tot['combo_right']}; "
      f"replay A0 {tot.get('A0_replay', '-')}, combo {tot.get('combo_replay', '-')}")
