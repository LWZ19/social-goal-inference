"""C5 flow pilot: example figures (figs/). Records are picked by a fixed rule, not by outcome: for each situation in
(waiting_line, conversation_group) the first goal-step record (file order) of the s19 Qwen real arm and of the s20
DeepSeek real arm, and for audience the first record of s19 dev_seed0 Qwen and s20 dev_seed0 DeepSeek; plus the first
validation unit of each situation of seed 190. Left: the raster the flow saw (setting b). Right: the world top-down
view: F(rec) floor cells, the acceptable region, people, the 64 raw samples, the valid snapped samples, the flow's top
pose and hit@5 places, and the solver's recorded place (C). No metric is computed here.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import research_harness as H  # noqa: E402
from score_dev import records_by_key  # noqa: E402

OUT = Path(C.OUT)
FIGS = OUT / 'figs'


def raster_img(r):
    img = np.ones((C.N, C.N, 3))
    img[r[0] > 0.5] = (0.78, 0.78, 0.78)
    img[r[1] > 0.5] = (1.0, 0.45, 0.45)
    img[r[4] > 0.5] = (0.2, 0.3, 1.0)
    img[r[5] > 0.5] = (0.1, 0.7, 0.1)
    img[r[6] > 0.5] = (0.9, 0.1, 0.9)
    return img


def arrow(ax, x, y, th, color, L=0.35, **kw):
    ax.arrow(x, y, L * math.cos(th), L * math.sin(th), color=color, head_width=0.08, length_includes_head=True, **kw)


def dev_fig(rec, i, dv, s, row, title):
    sc = H.scene(rec.ref, rec.scene_id)
    gt = sc.truths[rec.episode_id].gt
    fig, ax = plt.subplots(1, 2, figsize=(14, 6.6))
    ax[0].imshow(raster_img(dv['raster_b'][i]), origin='lower', extent=(-3.2, 3.2, -3.2, 3.2),
                 interpolation='nearest')
    ax[0].set_title('raster (setting b): floor grey, people red, target blue, seen face green\n'
                    'x = robot -> target direction', fontsize=9)
    a = ax[1]
    F = dv['F_b'][dv['off_b'][i]:dv['off_b'][i + 1]]
    ox, oy, yaw = dv['frame'][i]
    near = np.hypot(F[:, 0] - ox, F[:, 1] - oy) < 4.5
    a.plot(F[near, 0], F[near, 1], ',', color='0.8')
    for g in gt.acceptable_regions or []:
        v = np.array(g.polygon + [g.polygon[0]])
        a.fill(v[:, 0], v[:, 1], color='gold', alpha=0.45)
    for g in gt.forbidden_regions or []:
        v = np.array(g.polygon + [g.polygon[0]])
        a.plot(v[:, 0], v[:, 1], color='k', lw=0.6, ls=':')
    for p in sc.truth.people:
        a.plot(p.x, p.y, 'x', color='darkred', ms=6)
    for p in rec.people:
        a.plot(p['xy'][0], p['xy'][1], 'o', color='red', ms=6, mfc='none')
        if p.get('yaw') is not None:
            arrow(a, p['xy'][0], p['xy'][1], p['yaw'], 'red', L=0.3)
    raw = s['raw_world'][i]
    a.plot(raw[:, 0], raw[:, 1], '.', color='tab:purple', ms=3, alpha=0.6, label='64 raw samples')
    v = s['valid'][i].astype(bool)
    sp = s['sp'][i]
    a.plot(sp[v, 0], sp[v, 1], 'o', color='tab:blue', ms=4, mfc='none', label='valid snapped')
    top = s['top'][i]
    a.plot(top[0], top[1], '*', color='tab:blue', ms=16, label='flow top pose')
    arrow(a, top[0], top[1], top[2], 'tab:blue')
    for j in range(1, int(s['n_places'][i])):
        pl = s['places'][i][j]
        a.plot(pl[0], pl[1], 's', color='tab:cyan', ms=6)
    pw = rec.pose_world
    a.plot(pw[0], pw[1], '^', color='black', ms=10, label='solver place (C)')
    arrow(a, pw[0], pw[1], pw[2], 'black')
    tw = rec.target['world']
    a.plot(tw[0], tw[1], 'D', color='blue', ms=6, label='target point')
    a.set_xlim(ox - 4.0, ox + 4.0)
    a.set_ylim(oy - 4.0, oy + 4.0)
    a.set_aspect('equal')
    a.legend(fontsize=7, loc='upper right')
    a.set_title(f"{title}\nflow top-1 {row['top1']}, C top-1 {row['C_top1']}, P(succ) {row['psucc']:.2f}, "
                f"valid {row['valid_share']:.2f}, fallback {row['fallback']}", fontsize=9)
    fig.tight_layout()
    return fig


def main():
    FIGS.mkdir(exist_ok=True)
    dv = dict(np.load(OUT / 'data' / 'dev.npz'))
    s = np.load(OUT / 'samples' / 'dev_b_main.npz')
    kidx = {str(k): i for i, k in enumerate(dv['keys'])}
    rows = {}
    for f in OUT.glob('rows_b_*.jsonl'):
        if 'loso' in f.name:
            continue
        for x in f.read_text().splitlines():
            r = json.loads(x)
            rows[r['key']] = r
    recs = records_by_key()
    wanted = [('s19', 'real', 'qwen38_low', 'waiting_line'), ('s20', 'real', 'dsv41', 'waiting_line'),
              ('s19', 'real', 'qwen38_low', 'conversation_group'), ('s20', 'real', 'dsv41', 'conversation_group'),
              ('s19', 'dev_seed0', 'qwen38_low', 'audience'), ('s20', 'dev_seed0', 'dsv41', 'audience')]
    for stage, arm, model, sit in wanted:
        (ref,) = H.runs(stage=stage, arm=arm, model=model, method='con')
        key = next(k for k, r in recs.items() if r.ref == ref and r.situation == sit)
        rec, i = recs[key], kidx[key]
        fig = dev_fig(rec, i, dv, s, rows[key], f"{stage} {arm} {model}: {rec.scene_id} {rec.episode_id} "
                                                f"t{rec.trial} ({sit})")
        out = FIGS / f"dev_{sit}_{stage}_{arm}_{model}.png"
        fig.savefig(out, dpi=85)
        plt.close(fig)
        print(out)
    # validation: first unit of each situation of seed 190, main model
    vs = np.load(OUT / 'samples' / 'val_main.npz')
    vk = {str(k): i for i, k in enumerate(vs['keys'])}
    z = np.load(OUT / 'data' / 'seeds' / 'seed190.npz')
    info = json.loads((OUT / 'data' / 'seeds' / 'seed190.json').read_text())
    vrows = {json.loads(x)['key']: json.loads(x) for x in (OUT / 'rows_val_main.jsonl').read_text().splitlines()}
    done = set()
    for ui, u in enumerate(info['units']):
        if u['situation'] in done:
            continue
        done.add(u['situation'])
        i = vk[u['key']]
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.5))
        ax.imshow(raster_img(z['rasters'][ui]), origin='lower', extent=(-3.2, 3.2, -3.2, 3.2), interpolation='nearest')
        ext = z['ext_xy'][z['ext_off'][ui]:z['ext_off'][ui + 1]]
        ax.plot(ext[:, 0], ext[:, 1], 's', color='gold', ms=3, alpha=0.6, label='region cells')
        pr = vs['raw_raster'][i]
        x = vs['x'][i]
        ax.plot(pr[:, 0], pr[:, 1], '.', color='tab:purple', ms=5, label='64 samples')
        for j in range(len(pr)):
            ax.plot([pr[j, 0], pr[j, 0] + 0.2 * x[j, 2] / max(np.hypot(x[j, 2], x[j, 3]), 1e-9)],
                    [pr[j, 1], pr[j, 1] + 0.2 * x[j, 3] / max(np.hypot(x[j, 2], x[j, 3]), 1e-9)],
                    color='tab:purple', lw=0.5)
        t = int(vs['top'][i])
        ax.plot(pr[t, 0], pr[t, 1], '*', color='tab:blue', ms=15, label='top pose')
        lab = z['labels'][ui]
        if z['has_label'][ui]:
            ax.plot(lab[:, 0] * 3.2, lab[:, 1] * 3.2, 'x', color='darkorange', ms=4, label='label poses (not seen)')
        r = vrows[u['key']]
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.legend(fontsize=7)
        ax.set_title(f"validation {u['key']} ({u['situation']})\nsample success {r['share_success']:.2f}, "
                     f"coverage {r['coverage'] if r['coverage'] is None else round(r['coverage'], 2)}, "
                     f"top {r['top_success']}", fontsize=8)
        fig.tight_layout()
        out = FIGS / f"val_{u['situation']}_{u['scene']}_{u['episode']}.png"
        fig.savefig(out, dpi=85)
        plt.close(fig)
        print(out)


if __name__ == '__main__':
    main()
