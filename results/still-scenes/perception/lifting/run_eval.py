"""I11 Stage 0 (phase 1b) evaluation: one pass over the primary runs, or over one extra set, exactly as FROZEN.md
sections 1-9 define it. Writes rows_<set>.jsonl (one row per record) and records_<set>.tsv (one row per record x
fix); report.py turns them into out_primary.txt, out_extras.txt and tables.txt.

    run_eval.py --set primary|xnew|xph|xph2 [--replay all|fixes]

--replay all replays every moved record of every fix (full M3, net successes); --replay fixes replays only the
records a fix turns from wrong to right and the recorded successes it moves (enough for M3's fixed successes and
losses; gains among moved right-target failures are then not counted and are reported as not computed).
"""
import argparse
import collections
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import i11lib as I  # noqa: E402
import frontsnap as FS  # noqa: E402
from sgi.evaluate import match_target  # noqa: E402
from sgi.types import GroundedTarget  # noqa: E402

H, L = I.H, I.L
REAL = ('real', 'staffed', 'plain')
PRIMARY_FOLDERS = {
    ('s19', 'qwen38_low', 'real'): '20261011-160334', ('s19', 'qwen38_low', 'staffed'): '20261011-160344',
    ('s19', 'qwen38_low', 'plain'): '20261011-160527',
    ('s19', 'glm53', 'real'): '20261011-160500', ('s19', 'glm53', 'staffed'): '20261011-160352',
    ('s19', 'glm53', 'plain'): '20261011-160452',
    ('s20', 'dsv41', 'real'): '20261013-191337', ('s20', 'dsv41', 'staffed'): '20261013-191345',
    ('s20', 'dsv41', 'plain'): '20261013-191402',
}


def select_runs(which: str):
    """The runs of a set, each asserted to be exactly one listed run (FROZEN.md section 1)."""
    out = []

    def one(stage, arm, model, method):
        rs = H.runs(stage=stage, arm=arm, model=model, method=method)
        assert len(rs) == 1, (stage, arm, model, method, rs)
        out.append(rs[0])

    if which == 'primary':
        for stage, model in (('s19', 'qwen38_low'), ('s19', 'glm53'), ('s20', 'dsv41')):
            for arm in REAL:
                one(stage, arm, model, 'con')
                assert out[-1].folder.startswith(PRIMARY_FOLDERS[(stage, model, arm)]), out[-1].folder
    elif which == 'xnew':
        for arm in REAL:
            one('s21', arm, 'qwen38_low', 'con')
        for model in ('glm53', 'dsv41', 'haiku55', 'sonnet55_low', 'gemini38_low'):
            for arm in REAL:
                one('s23', arm, model, 'con')
    elif which == 'xph':
        for stage, model in (('s19', 'qwen38_low'), ('s19', 'glm53'), ('s20', 'dsv41')):
            for arm in ('ph', 'dev_seed0', 'dev_seed1', 'dev_seed2'):
                one(stage, arm, model, 'con')
    elif which == 'xph2':
        out.extend(H.runs(stage='s10', method='con'))
        out.extend(H.runs(stage='s8', method='convlm'))
        out.extend(H.runs(stage='s8', method='conorc'))
        assert len(out) == 15, len(out)
    else:
        raise ValueError(which)
    return out


# ---------------------------------------------------------------- per-scene caches
_GEOM = collections.OrderedDict()


def geom_for(rec, sd):
    key = (str(rec.ref.scenes_dir), rec.scene_id)
    if key in _GEOM:
        _GEOM.move_to_end(key)
        return _GEOM[key]
    g = I.Geom(sd['depth'], sd['cam'])
    assert np.array_equal(g.W, sd['W']), 'per-pixel lift differs from lifting.scene_data'
    g.comps('G1')
    g.comps('G2')
    _GEOM[key] = g
    while len(_GEOM) > 6:
        _GEOM.popitem(last=False)
    return g


def front_target(rec, w):
    """The front snap returns a world point (lifted at its chosen pixel's centre); as a GroundedTarget its pixel
    is that point projected, which is the chosen pixel's centre."""
    return H.target_at(rec, world=tuple(float(c) for c in w))


# ---------------------------------------------------------------- one record
def process(rec, setname, rng, replay_mode, strict):
    sc_ref = rec.ref
    row = dict(set=setname, stage=sc_ref.stage, arm=rec.arm, model=rec.model, method=rec.method, run=rec.run,
               scene=rec.scene_id, layout=I.layout(rec.scene_id), episode=rec.episode_id, trial=rec.trial,
               situation=rec.situation, status=rec.result.get('status'), source=rec.result.get('source'),
               solver_step=bool(rec.solver_step), rec_success=bool(rec.score.get('success')),
               rec_target_correct=bool(rec.score.get('target_correct')))
    th = H.recorded_config(rec.ref)[1].thresholds
    I.check_thresholds(th)
    sd = L.scene_data(rec)
    sc = sd['sc']
    gt = sc.truths[rec.episode_id].gt
    row['ntargets'] = len(gt.target_ids)
    tgt = rec.target
    if tgt is None:
        tid = gt.target_ids[0]
        t = sc.truth.targets[tid]
        row.update(has_point=False, tid=tid, kind=('parcel' if t.kind == 'object' else t.kind), label=t.label)
        return row
    if tgt.get('pixel') is None:
        tid = gt.target_ids[0]
        t = sc.truth.targets[tid]
        row.update(has_point=False, no_pixel=True, tid=tid, kind=('parcel' if t.kind == 'object' else t.kind),
                   label=t.label)
        return row
    row['has_point'] = True
    u, v = (float(c) for c in tgt['pixel'])
    row['u'], row['v'] = u, v
    world_rec = np.asarray(tgt['world'], dtype=np.float64)
    again = H.target_at(rec, pixel=(u, v))
    rep = math.inf if again is None else float(np.linalg.norm(np.asarray(again.world) - world_rec))
    row['lift_repro_m'] = rep
    if rep > 1e-9:
        if strict:
            raise AssertionError(f'lift does not reproduce: {rec.run} {rec.scene_id} {rec.episode_id} {rep}')
        row['skip'] = 'lift does not reproduce'
        return row
    tid = I.true_target(sd, gt, th, u, v)
    t = sc.truth.targets[tid]
    row.update(tid=tid, kind=('parcel' if t.kind == 'object' else t.kind), label=t.label)
    right = match_target(world_rec, sc.truth, gt, th) is not None
    row['right_before'] = bool(right)
    row['tc_disagree'] = bool(right) != row['rec_target_correct']
    if strict:
        assert not row['tc_disagree'], (rec.run, rec.scene_id, rec.episode_id, rec.trial)
    g = geom_for(rec, sd)
    i1 = I.point_info(g, u, v, 'G1')
    i2 = I.point_info(g, u, v, 'G2')
    row.update(s_cls=i1['s_cls'], win_labels=i1['win_labels'], T1=i1['T1'], T2=i1['T2'], s_cls_G2=i2['s_cls'],
               T1_G2=i2['T1'])
    if not right:
        ec = I.error_class(sd, tid, th, u, v, world_rec)
        row.update(ec)
        row['surface'] = L.surface(sd, world_rec, tid)
        if row['ntargets'] == 1:
            bad, nchk = L.check_vectorised(rec, sd, tid, th, gt, rng)
            row['vec_mismatch'], row['vec_checked'] = int(bad), int(nchk)
    body = I.body_union(sd, gt, th)
    fx = {}
    for name in I.FIXES:
        info = {}
        if name == 'oracle':
            nt, db = I.oracle_fix(sd, body, u, v)
            info['d'] = None if not math.isfinite(db) else float(db)
        elif name == 'front':
            w = FS.front_snap(sd, u, v)
            nt = None if w is None else front_target(rec, w)
            if w is not None:
                info['world_err'] = float(np.linalg.norm(np.asarray(nt.world) - np.asarray(w)))
        else:
            nt, info = I.variant(g, u, v, name)
            info = {k: (float(x) if isinstance(x, (float, np.floating)) else x) for k, x in info.items()}
        moved = nt is not None
        right_after = right if not moved else (match_target(nt.world, sc.truth, gt, th) is not None)
        d = dict(moved=moved, right_after=bool(right_after), **{k: x for k, x in info.items() if k != 'c'})
        if moved:
            d['pixel'] = [float(nt.pixel[0]), float(nt.pixel[1])]
            d['world'] = [float(c) for c in nt.world]
            need = replay_mode == 'all' or (not right and right_after) or row['rec_success']
            if need:
                succ, path = I.replay_success(rec, nt)
                d['succ_after'], d['path'] = succ, path
                if row['rec_success'] and not succ:
                    base, _ = I.replay_success(rec, GroundedTarget.model_validate(tgt))
                    d['baseline_reproduces'] = bool(base)
            else:
                d['succ_after'] = None
        else:
            d['succ_after'] = row['rec_success']
        fx[name] = d
    row['fx'] = fx
    return row


def to_jsonable(x):
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def m4_rows_for(runs, setname):
    """M4 over the distinct (scene folder, scene, true target) of a set's records' episodes."""
    want = collections.defaultdict(set)
    first_rec = {}
    for run in runs:
        for rec in H.records(run):
            key = (str(rec.ref.scenes_dir), rec.scene_id)
            sc = H.scene(rec.ref, rec.scene_id)
            want[key] |= set(sc.truths[rec.episode_id].gt.target_ids)
            first_rec.setdefault(key, rec)
    rows = []
    for key in sorted(want):
        rec = first_rec[key]
        sd = L.scene_data(rec)
        th = H.recorded_config(rec.ref)[1].thresholds
        g = geom_for(rec, sd)
        for r in I.m4_scene(g, sd, th, sorted(want[key])):
            r.update(set=setname, scenes_dir=key[0], scene=key[1], layout=I.layout(key[1]))
            rows.append(r)
    return rows


def m4_wrong(rows_by_rec, setname):
    """M4 for the true target of each wrong-target record with a point, tagged with its error class."""
    out = []
    for rec, row in rows_by_rec:
        if not row.get('has_point') or row.get('skip') or row.get('right_before'):
            continue
        sd = L.scene_data(rec)
        th = H.recorded_config(rec.ref)[1].thresholds
        g = geom_for(rec, sd)
        for r in I.m4_scene(g, sd, th, [row['tid']]):
            r.update(set=setname, model=row['model'], scene=row['scene'], episode=row['episode'],
                     trial=row['trial'], lift_class=row['lift_class'])
            out.append(r)
    return out


def m4_fresh():
    """M4 on fresh seeds 100-109, read directly (FROZEN.md section 1; DEVIATIONS.md D6, D8): only seed folders
    named seed100..seed109 that carry DONE.json, and only the scenes DONE.json lists. Seeds 3-99 are never opened."""
    import hashlib
    base = I.OUTDIR.parent / 'fresh_seeds'
    if not base.exists():
        return None, f'{base} absent'
    from sgi.config import load_config
    from sgi.scene import load_scene
    from sgi.gt.derive import prepare
    cfg = load_config()
    I.check_thresholds(cfg.thresholds)
    held = set(cfg.held_out_situations)
    rows, seen, notes = [], [], []
    for seed in range(100, 110):
        folder = base / f'seed{seed}'
        done = folder / 'DONE.json'
        if not done.exists():
            notes.append(f'seed{seed}: no DONE.json, skipped')
            continue
        listing = json.loads(done.read_text())
        assert listing.get('seed') == seed, (seed, listing.get('seed'))
        for ent in listing['scenes']:
            sfolder = folder / ent['split'] / ent['scene_id']
            if set(ent.get('situations') or []) & held:
                notes.append(f"seed{seed} {ent['scene_id']}: held-out situation, skipped")
                continue
            H.guard_tree(sfolder)
            inputs, truth = load_scene(sfolder)
            truths = prepare(inputs, truth, cfg)
            truths = {k: v for k, v in truths.items() if v.gt.situation not in held}
            sc = H.Scene(inputs, inputs.observation, truth, truths)
            sd = I.sd_from_scene(sc)
            g = I.Geom(sc.observation.depth, sc.observation.camera)
            tids = sorted({tid for et in truths.values() for tid in et.gt.target_ids})
            for r in I.m4_scene(g, sd, cfg.thresholds, tids):
                r.update(set='fresh', scene=f"seed{seed}/{ent['split']}/{ent['scene_id']}", layout=ent['scene_id'],
                         seed=seed, subset=ent['split'])
                rows.append(r)
            seen.append(f"seed{seed}/{ent['split']}/{ent['scene_id']}")
    return rows, f"{len(seen)} fresh scenes from {len({x.split('/')[0] for x in seen})} seeds; " + '; '.join(notes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', required=True)
    ap.add_argument('--replay', default='all', choices=('all', 'fixes'))
    ap.add_argument('--no-m4', action='store_true')
    a = ap.parse_args()
    for line in I.check_pins():
        print(line, flush=True)
    if a.set == 'fresh':
        fr, why = m4_fresh()
        print(why, flush=True)
        with open(I.OUTDIR / 'm4_fresh.jsonl', 'w') as f:
            for r in fr or []:
                f.write(json.dumps(to_jsonable(dict(table='m4', **r))) + '\n')
        (I.OUTDIR / 'meta_fresh.json').write_text(json.dumps(dict(set='fresh', note=why), indent=1))
        return
    runs = select_runs(a.set)
    strict = a.set == 'primary'
    rng = np.random.default_rng(0)
    recs = []
    tallies = {}
    for run in runs:
        tally = collections.Counter()
        rs = list(H.records(run, tally=tally))
        tallies[run.folder] = dict(tally)
        recs.extend(rs)
    print(f'{a.set}: {len(runs)} runs, {len(recs)} records', flush=True)
    recs.sort(key=lambda r: (str(r.ref.scenes_dir), r.scene_id, r.run, r.episode_id, r.trial if r.trial is not None else -1))
    t0 = time.time()
    out_rows, pairs = [], []
    for k, rec in enumerate(recs):
        row = process(rec, a.set, rng, a.replay, strict)
        out_rows.append(row)
        pairs.append((rec, row))
        if (k + 1) % 100 == 0:
            print(f'  {k + 1}/{len(recs)} records, {time.time() - t0:.0f} s, replays cached {len(I._REPLAY)}', flush=True)
    print(f'records done in {time.time() - t0:.0f} s', flush=True)
    with open(I.OUTDIR / f'rows_{a.set}.jsonl', 'w') as f:
        for r in out_rows:
            f.write(json.dumps(to_jsonable(r)) + '\n')
    keys = ['set', 'stage', 'arm', 'model', 'method', 'run', 'scene', 'episode', 'trial', 'situation', 'kind',
            'has_point', 'right_before', 'lift_class', 's_cls', 'T1', 'T2', 'rec_success']
    fkeys = ['moved', 'right_after', 'succ_after', 'path', 'd', 'why']
    with open(I.OUTDIR / f'records_{a.set}.tsv', 'w') as f:
        f.write('\t'.join(keys + ['fix'] + fkeys) + '\n')
        for r in out_rows:
            if not r.get('has_point') or r.get('skip'):
                continue
            for name in I.FIXES:
                d = r['fx'][name]
                f.write('\t'.join([str(r.get(k, '')) for k in keys] + [name] + [str(d.get(k, '')) for k in fkeys]) + '\n')
    meta = dict(set=a.set, replay=a.replay, runs=[r.folder for r in runs], tallies=tallies,
                n_records=len(recs), seconds=time.time() - t0)
    if not a.no_m4 and a.set in ('primary', 'xph'):
        t1 = time.time()
        m4 = m4_rows_for(runs, a.set)
        m4w = m4_wrong(pairs, a.set)
        meta['m4_seconds'] = time.time() - t1
        with open(I.OUTDIR / f'm4_{a.set}.jsonl', 'w') as f:
            for r in m4:
                f.write(json.dumps(to_jsonable(dict(table='m4', **r))) + '\n')
            for r in m4w:
                f.write(json.dumps(to_jsonable(dict(table='m4_wrong', **r))) + '\n')
    (I.OUTDIR / f'meta_{a.set}.json').write_text(json.dumps(meta, indent=1))
    print('done', json.dumps({k: v for k, v in meta.items() if k != 'tallies'}), flush=True)


if __name__ == '__main__':
    main()
