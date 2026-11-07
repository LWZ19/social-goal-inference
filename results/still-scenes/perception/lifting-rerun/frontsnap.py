"""A zero-call (no ground truth) snap, to test the steelman "so a zero-call fix exists". Parameters fixed before
running, never varied: R=15 px, JUMP=0.5 m, AGREE=0.3 m.

front-surface snap: if some pixel within R px of the model's point has depth at least JUMP nearer than the
pipeline's 5x5 median depth at the point (the point sits on a farther surface beside a nearer one), move to the
nearest such pixel and lift it with the median of its 5x5 window's depths within AGREE of its own; otherwise keep
the pipeline's lift. Applied to every C record with a point (s19 Qwen/GLM, s20 DeepSeek, realistic sets): fixes =
wrong -> right, breaks = right -> wrong, by the package's match_target. Reads only through research_harness.
"""
import collections
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_harness as H  # noqa: E402
import lifting as L  # noqa: E402
from sgi.evaluate import match_target  # noqa: E402

R, JUMP, AGREE = 15.0, 0.5, 0.3
OUT = Path(__file__).resolve().parent
LOG = []


def out(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def front_snap(sd, u, v):
    Z = sd['Z']
    h, w = Z.shape
    i, j = math.floor(v), math.floor(u)
    win = L.window(Z, i, j)
    zpt = float(np.median(win[np.isfinite(win) & (win > 0)]))
    r = int(math.ceil(R))
    i0, i1, j0, j1 = max(i - r, 0), min(i + r + 1, h), max(j - r, 0), min(j + r + 1, w)
    sub = Z[i0:i1, j0:j1]
    vv, uu = np.meshgrid(np.arange(i0, i1) + 0.5, np.arange(j0, j1) + 0.5, indexing='ij')
    dpx = np.hypot(uu - u, vv - v)
    cand = (dpx <= R) & (sub < zpt - JUMP) & np.isfinite(sub) & (sub > 0)
    if not cand.any():
        return None
    k = np.argmin(np.where(cand, dpx, np.inf))
    qi, qj = np.unravel_index(k, sub.shape)
    qi, qj = qi + i0, qj + j0
    zq = Z[qi, qj]
    ww = L.window(Z, qi, qj)
    zm = float(np.median(ww[np.abs(ww - zq) <= AGREE]))
    return L.lift_at_depth(sd['cam'], qj + 0.5, qi + 0.5, zm)


def main():
    tally = collections.Counter()
    per_class = collections.defaultdict(collections.Counter)
    for stage, model in [('s19', 'qwen38_low'), ('s19', 'glm53'), ('s20', 'dsv41')]:
        for arm in L.REAL_ARMS:
            run = H.runs(stage=stage, arm=arm, model=model, method='con')[0]
            recs = sorted(H.records(run), key=lambda r: r.scene_id)
            for rec in recs:
                if rec.target is None:
                    continue
                sd = L.scene_data(rec)
                th = H.recorded_config(rec.ref)[1].thresholds
                gt = sd['sc'].truths[rec.episode_id].gt
                tid = gt.target_ids[0]
                t = sd['sc'].truth.targets[tid]
                cls = 'parcel' if t.kind == 'object' else t.kind
                before = match_target(rec.target['world'], sd['sc'].truth, gt, th) == tid
                assert before == bool(rec.score['target_correct'])
                u, v = rec.target['pixel']
                w = front_snap(sd, u, v)
                moved = w is not None
                after = before if w is None else (match_target(w, sd['sc'].truth, gt, th) == tid)
                key = ('right' if before else 'wrong') + '->' + ('right' if after else 'wrong') + (' (moved)' if moved else '')
                tally[(model, key)] += 1
                per_class[cls][key] += 1
    out('front-surface snap (zero-call), C records with a point, realistic sets; R=15 px, JUMP=0.5 m, AGREE=0.3 m')
    for m in ('qwen38_low', 'glm53', 'dsv41'):
        out(f"  {m:11s} " + str({k[1]: v for k, v in sorted(tally.items()) if k[0] == m}))
    for c in ('person', 'parcel', 'place'):
        out(f"  {c:7s} " + str(dict(sorted(per_class[c].items()))))
    fixes = sum(v for k, v in tally.items() if k[1].startswith('wrong->right'))
    breaks = sum(v for k, v in tally.items() if k[1].startswith('right->wrong'))
    moved_ok = sum(v for k, v in tally.items() if k[1] == 'right->right (moved)')
    out(f"  total fixes {fixes}, breaks {breaks}, correct records moved but still right {moved_ok}")
    (OUT / 'frontsnap_out.txt').write_text('\n'.join(LOG) + '\n')


if __name__ == '__main__':
    main()
