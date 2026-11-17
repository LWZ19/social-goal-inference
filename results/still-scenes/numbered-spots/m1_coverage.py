"""Marks probe M1 (M1_FROZEN.md): coverage of the depth-component proposer P0, and the perfect-pick ceiling.

Zero model calls, CPU only. Runs and scenes through research_harness; components and replays through phase1b/i11.
    .venv/bin/python -B m1_coverage.py            (from the research worktree's venv)
"""
from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / '_lib' / 'paths.py').exists()) / '_lib'))
import paths  # noqa: E402  (results/_lib/paths.py)
I11 = paths.STILL / 'perception/lifting'
sys.path.insert(0, str(I11))
import i11lib as X  # noqa: E402  (imports research_harness as X.H, lifting as X.L)

H = X.H
FROZEN_SHA = '04c23e25c648430d9611e380bc215f7fdff061cd3332e6aa65ac2e39c2b9ecb5'
CAPS = (10, 15, 20, None)
JUDGED_CAP = 20
R6_GAIN = {'qwen38_low': 11, 'glm53': 14, 'dsv41': 12, 'haiku55': 1, 'sonnet55_low': 7, 'gemini38_low': 26}


def check_frozen():
    got = hashlib.sha256((HERE / 'M1_FROZEN.md').read_bytes()).hexdigest()
    assert got == FROZEN_SHA, f'M1_FROZEN.md changed: {got}'
    lines = X.check_pins()
    bad = [ln for ln in lines if not ln.endswith(' OK')]
    assert not bad, f'i11 pins: {bad}'


def interior_pixel(mask: np.ndarray) -> tuple[int, int]:
    """The most interior pixel of a boolean mask: erode by 4-neighbourhood until the next erosion empties it, then
    the remaining pixel nearest the mask's pixel centroid (lower row, then column, on a tie)."""
    ii, jj = np.nonzero(mask)
    ci, cj = ii.mean(), jj.mean()
    i0, i1, j0, j1 = ii.min(), ii.max() + 1, jj.min(), jj.max() + 1
    m = np.zeros((i1 - i0 + 2, j1 - j0 + 2), bool)
    m[1:-1, 1:-1] = mask[i0:i1, j0:j1]
    while True:
        e = m.copy()
        e[1:-1, 1:-1] &= m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
        e[0, :] = e[-1, :] = False
        e[:, 0] = e[:, -1] = False
        if not e.any():
            break
        m = e
    ki, kj = np.nonzero(m)
    ki, kj = ki - 1 + i0, kj - 1 + j0
    d = (ki - ci) ** 2 + (kj - cj) ** 2
    order = np.lexsort((kj, ki, d))
    return int(ki[order[0]]), int(kj[order[0]])


def proposer(g: X.Geom):
    """P0: one candidate per G1 object or structure component, largest first: (component, class, size, pixel,
    grounded target)."""
    cp = g.comps('G1')
    out = []
    for c, k in cp.ccls.items():
        if k not in (X.OBJECT, X.STRUCT):
            continue
        q = interior_pixel(cp.label == c)
        out.append((c, X.CLS_NAME[k], cp.size[c], q, X.comp_lift(g, cp, c, q)))
    out.sort(key=lambda t: (-t[2], t[0]))
    return out


def is_realistic(scene_id: str) -> bool:
    return scene_id.split('_')[0] in ('real', 'staffed', 'plain')


def main():
    check_frozen()
    from sgi.evaluate import match_target
    wl = H.whitelist()
    inst_rows, cand_counts, per_scene = [], [], {}
    for sref in H.distinct_scenes(wl=wl):
        run = sref.runs[0]
        sid = sref.listed[0][1] if sref.scene_id is None else sref.scene_id
        sc = H.scene(run, sid, wl=wl)
        th = H.recorded_config(run)[1].thresholds
        g = X.Geom(sc.observation.depth, sc.observation.camera)
        cands = proposer(g)
        per_scene[(str(sref.folder))] = cands
        cand_counts.append((sid, len(cands)))
        for eid, ep in sc.truth.episodes.items():
            for tid in ep.target_ids:
                t = sc.truth.targets[tid]
                hits = []
                for rank, (c, cls, size, q, tgt) in enumerate(cands):
                    m = match_target(tgt.world, sc.truth, ep, th)
                    if m == tid:
                        hits.append((rank, cls, size))
                inst_rows.append(dict(scene=sid, realistic=is_realistic(sid), episode=eid, target=tid, kind=t.kind,
                                      situation=ep.situation, n_cands=len(cands),
                                      first_rank=hits[0][0] if hits else None,
                                      hit_classes=[h[1] for h in hits], n_hits=len(hits)))

    out = []
    say = out.append
    say(f'M1 coverage (P0, G1 depth components); frozen {FROZEN_SHA[:12]}')
    n = np.array([k for _, k in cand_counts])
    say(f'candidates per image: median {np.median(n):.0f}, p90 {np.percentile(n, 90):.0f}, max {n.max()}, '
        f'scenes {len(n)}')
    for subset, rows in (('realistic', [r for r in inst_rows if r['realistic']]),
                         ('placeholder', [r for r in inst_rows if not r['realistic']])):
        say(f'\n== {subset}: {len(rows)} target instances')
        kinds = sorted({r['kind'] for r in rows})
        header = 'cap   ' + '  '.join(f'{k:>16}' for k in kinds) + f'  {"pooled":>16}'
        say(header)
        for cap in CAPS:
            cells = []
            for k in kinds + ['*']:
                rr = [r for r in rows if k == '*' or r['kind'] == k]
                cov = sum(1 for r in rr if r['first_rank'] is not None and (cap is None or r['first_rank'] < cap))
                cells.append(f'{cov:>4}/{len(rr):<4}={cov / max(len(rr), 1):.3f}')
            say(f'{str(cap or "all"):>5} ' + '  '.join(f'{c:>16}' for c in cells))
        miss = [r for r in rows if r['first_rank'] is None or r['first_rank'] >= JUDGED_CAP]
        say(f'not covered at N={JUDGED_CAP}: {len(miss)}')
        for r in miss:
            say(f"  {r['scene']:<28} {r['episode']:<6} {r['target']:<26} {r['kind']:<7} {r['situation']:<20} "
                f"rank {r['first_rank']}  of {r['n_cands']}")

    # ---------------------------------------------------------------- ceiling over the 18 new-wording C runs
    runs = H.runs(stage='s21', method='con', wl=wl) + H.runs(stage='s23', method='con', wl=wl)
    say(f'\n== ceiling: perfect pick among the first {JUDGED_CAP} candidates, replayed ({len(runs)} runs)')
    by_model = collections.defaultdict(lambda: [0, 0, 0, 0])   # records, recorded successes, ceiling, ceiling-all
    cache = {}
    for run in runs:
        for rec in H.records(run, wl=wl):
            m = rec.model
            by_model[m][0] += 1
            rec_ok = bool(rec.score.get('success'))
            by_model[m][1] += rec_ok
            if rec.target is None:
                by_model[m][2] += rec_ok
                by_model[m][3] += rec_ok
                continue
            sc = H.scene(rec.ref, rec.scene_id, wl=wl)
            key = (str(rec.ref.scenes_dir), rec.scene_id)   # was id(sc.observation): ids are reused after the harness evicts a scene
            if key not in cache:
                cache[key] = proposer(X.Geom(sc.observation.depth, sc.observation.camera))
            cands = cache[key]
            ep = sc.truth.episodes[rec.episode_id]
            th = H.recorded_config(rec.ref)[1].thresholds
            best, best_all = False, False
            for rank, (_, _, _, _, tgt) in enumerate(cands):
                if match_target(tgt.world, sc.truth, ep, th) is None:
                    continue                       # success needs the right target (evaluate.score)
                ok, _ = X.replay_success(rec, tgt)
                if ok:
                    best_all = True
                    if rank < JUDGED_CAP:
                        best = True
                        break
            by_model[m][2] += best
            by_model[m][3] += best_all
    total_gain = 0
    say(f'{"model":<14}{"records":>8}{"recorded":>10}{"ceil@20":>9}{"gain":>6}{"ceil@all":>10}{"R6 gain":>9}')
    for m, (nr, ok, ce, ca) in sorted(by_model.items()):
        total_gain += ce - ok
        say(f'{m:<14}{nr:>8}{ok:>10}{ce:>9}{ce - ok:>+6}{ca:>10}{R6_GAIN.get(m, 0):>+9}')
    say(f'pooled ceiling gain at N={JUDGED_CAP}: {total_gain:+d} (gate: >= +36; R6 +71)')

    real = [r for r in inst_rows if r['realistic']]
    cov20 = sum(1 for r in real if r['first_rank'] is not None and r['first_rank'] < JUDGED_CAP) / len(real)
    kinds_ok = all(
        sum(1 for r in real if r['kind'] == k and r['first_rank'] is not None and r['first_rank'] < JUDGED_CAP)
        / len([r for r in real if r['kind'] == k]) >= 0.90
        for k in {r['kind'] for r in real} if len([r for r in real if r['kind'] == k]) >= 5)
    say(f'\nVERDICT: H1 {"holds" if cov20 >= 0.95 and kinds_ok else "fails"} '
        f'(realistic pooled {cov20:.3f}, every kind >= 0.90: {kinds_ok}); '
        f'ceiling gate {"passes" if total_gain >= 36 else "fails"} ({total_gain:+d})')
    text = '\n'.join(out)
    print(text)
    (HERE / 'm1_out.txt').write_text(text + '\n')
    (HERE / 'm1_instances.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in inst_rows))


if __name__ == '__main__':
    main()
