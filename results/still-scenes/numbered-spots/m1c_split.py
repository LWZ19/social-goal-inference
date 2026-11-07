"""POST HOC (questioner's Q1-Q3 on M1): where does M1b-snap's +98 come from?

For every new-wording C record whose replay changes between recorded and M1b-snap, the click is classed by its pixel
distance to the right component (the one M1b used): inside (0 px), near miss (<= 15 px, I11's R_SNAP), or wrong
object (> 15 px). For each record it also replays the right component's interior pixel and 20 random pixels of it
(seeded), to see whether the success comes from the model's location or from the component's geometry. Recall@K of
"the K components nearest the model's own click" contains the right component is reported for K = 1..8 over the
wrong-target records. Zero calls, CPU only.
"""
from __future__ import annotations

import collections
import json
import random
from pathlib import Path

import numpy as np

import m1_coverage as M
from m1b_diagnose import snap_into

X, H = M.X, M.H
HERE = Path(__file__).resolve().parent
NEAR_PX = 15.0


def main():
    M.check_frozen()
    from sgi.evaluate import match_target
    wl = H.whitelist()
    runs = H.runs(stage='s21', method='con', wl=wl) + H.runs(stage='s23', method='con', wl=wl)
    rows, cache = [], {}
    recall = collections.Counter()
    n_wrong = 0
    for run in runs:
        for rec in H.records(run, wl=wl):
            if rec.target is None:
                continue
            sc = H.scene(rec.ref, rec.scene_id, wl=wl)
            ep = sc.truth.episodes[rec.episode_id]
            th = H.recorded_config(rec.ref)[1].thresholds
            key = (str(rec.ref.scenes_dir), rec.scene_id)   # was id(sc.observation): ids are reused after the harness evicts a scene
            if key not in cache:
                g = X.Geom(sc.observation.depth, sc.observation.camera)
                cache[key] = (g, M.proposer(g))
            g, cands = cache[key]
            cp = g.comps('G1')
            u, v = rec.target['pixel']
            right = [(rank, c) for rank, (c, cls, size, q, tgt) in enumerate(cands[:M.JUDGED_CAP])
                     if match_target(tgt.world, sc.truth, ep, th) is not None]
            if not right:
                continue
            # the component M1b chose: the first right candidate whose replay succeeds, else the first right one
            chosen = right[0][1]
            for rank, c in right:
                if X.replay_success(rec, cands[rank][4])[0]:
                    chosen = c
                    break
            ii, jj = np.nonzero(cp.label == chosen)
            dpx = float(np.sqrt(((ii + 0.5 - v) ** 2 + (jj + 0.5 - u) ** 2).min()))
            kind = 'inside' if dpx < 1.0 else ('near' if dpx <= NEAR_PX else 'wrong_object')
            rec_ok = bool(rec.score.get('success'))
            rec_right = match_target(rec.target['world'], sc.truth, ep, th) is not None
            st = snap_into(g, cp, chosen, u, v)
            snap_ok = match_target(st.world, sc.truth, ep, th) is not None and X.replay_success(rec, st)[0]
            interior = next(t for (c, _, _, _, t) in cands if c == chosen)
            int_ok = X.replay_success(rec, interior)[0]
            r_ok, picks = None, [0]
            if bool(snap_ok) != rec_ok:            # random pixels only where the outcome changes (cost)
                rnd = random.Random(f'{rec.scene_id}|{rec.episode_id}|{rec.trial}|{rec.model}')
                picks = rnd.sample(range(len(ii)), min(20, len(ii)))
                r_ok = 0
                for k in picks:
                    t = X.comp_lift(g, cp, chosen, (int(ii[k]), int(jj[k])))
                    r_ok += match_target(t.world, sc.truth, ep, th) is not None and X.replay_success(rec, t)[0]
            if not rec_right:
                n_wrong += 1
                # recall@K of the components nearest the model's own click
                dist = []
                for (c, cls, size, q, tgt) in cands:
                    mi, mj = np.nonzero(cp.label == c)
                    dist.append((float(np.sqrt(((mi + 0.5 - v) ** 2 + (mj + 0.5 - u) ** 2).min())), c))
                dist.sort()
                rights = {c for _, c in right}
                for K in range(1, 9):
                    if any(c in rights for _, c in dist[:K]):
                        recall[K] += 1
            rows.append(dict(model=rec.model, scene=rec.scene_id, episode=rec.episode_id, trial=rec.trial,
                             situation=rec.situation, rec_ok=rec_ok, rec_right=rec_right, click=kind, dpx=dpx,
                             snap_ok=bool(snap_ok), interior_ok=bool(int_ok), random_ok=None if r_ok is None else r_ok / len(picks)))
    out = []
    say = out.append
    say('M1c (POST HOC): the M1b-snap gain split by click type')
    gains = [r for r in rows if r['snap_ok'] and not r['rec_ok']]
    losses = [r for r in rows if r['rec_ok'] and not r['snap_ok']]
    say(f'records {len(rows)}; snap gains {len(gains)}, snap losses {len(losses)}, net {len(gains) - len(losses):+d}')
    for name, rr in (('GAINS', gains), ('LOSSES', losses)):
        say(f'\n== {name} by click type and model')
        by = collections.defaultdict(collections.Counter)
        for r in rr:
            by[r['click']][r['model']] += 1
        for kind in ('inside', 'near', 'wrong_object'):
            c = by[kind]
            say(f'  {kind:<13} {sum(c.values()):>4}   ' + ', '.join(f'{m} {n}' for m, n in sorted(c.items())))
    say('\n== For the snap gains: does the success come from the model\'s location or from the component?')
    for kind in ('inside', 'near', 'wrong_object'):
        rr = [r for r in gains if r['click'] == kind]
        if rr:
            say(f'  {kind:<13} n={len(rr):>3}  interior pixel succeeds {sum(r["interior_ok"] for r in rr):>3}; '
                f'random pixels of the right component succeed {np.mean([r["random_ok"] for r in rr]):.2f} on average')
    say(f'\n== Recall@K: the K components nearest the model\'s own click contain the right one '
        f'(wrong-target records, n={n_wrong})')
    say('  ' + '  '.join(f'K={K}: {recall[K]}/{n_wrong}={recall[K] / max(n_wrong, 1):.2f}' for K in range(1, 9)))
    say('\n== Gemini wrong-target records by click type: ' +
        ', '.join(f'{k} {v}' for k, v in collections.Counter(r['click'] for r in rows
                                                             if r['model'] == 'gemini38_low' and not r['rec_right']).items()))
    text = '\n'.join(out)
    print(text)
    (HERE / 'm1c_out.txt').write_text(text + '\n')
    (HERE / 'm1c_rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))


if __name__ == '__main__':
    main()
