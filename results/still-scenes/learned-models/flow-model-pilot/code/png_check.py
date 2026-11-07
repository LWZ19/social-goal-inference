"""C5 flow pilot: visual check of a few training rasters (data/png/). Left: the scene top-down in the world frame
(people with facing, target anchor, acceptable region, robot, the raster's square). Right: the raster as built
(floor grey, people red with facing ticks, target blue, seen face green, robot magenta) with the label poses (yellow,
heading ticks), plotted with x to the right and y up, so the robot should lie to the left. No metric.

    .venv/bin/python code/png_check.py SEED SPLIT/SCENE/EPISODE [...]
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
from sgi.config import load_config  # noqa: E402
from sgi.gt.derive import prepare  # noqa: E402
from sgi.scene import load_scene  # noqa: E402

PNG = Path(C.OUT) / 'data' / 'png'


def draw(seed, want):
    cfg = load_config()
    z = np.load(Path(C.OUT) / 'data' / 'seeds' / f'seed{seed}.npz')
    info = json.loads((Path(C.OUT) / 'data' / 'seeds' / f'seed{seed}.json').read_text())
    keys = list(z['keys'])
    for w in want:
        key = f'seed{seed}/{w}'
        i = keys.index(key)
        u = info['units'][i]
        r, lab = z['rasters'][i], z['labels'][i]
        split, scene, ep = w.split('/')
        folder = Path(C.SEEDS_DIR) / f'seed{seed}' / split / scene
        H.guard_scene(folder)
        inputs, truth = load_scene(folder)
        gt = prepare(inputs, truth, cfg)[ep].gt
        obs = inputs.observation
        fr = C.Frame((u['frame']['ox'], u['frame']['oy']), (obs.robot.x, obs.robot.y))
        fig, ax = plt.subplots(1, 2, figsize=(13, 6.3))
        a = ax[0]
        for g in gt.acceptable_regions or []:
            v = np.array(g.polygon + [g.polygon[0]])
            a.fill(v[:, 0], v[:, 1], color='gold', alpha=0.4, label='acceptable region')
        for g in gt.forbidden_regions or []:
            v = np.array(g.polygon + [g.polygon[0]])
            a.plot(v[:, 0], v[:, 1], color='k', lw=0.6, ls=':')
        for p in truth.people:
            a.plot(p.x, p.y, 'o', color='red', ms=6)
            a.arrow(p.x, p.y, 0.4 * math.cos(p.yaw), 0.4 * math.sin(p.yaw), color='red', head_width=0.08)
        t = truth.targets[u['target_id']]
        a.plot(t.anchor[0], t.anchor[1], 's', color='blue', ms=8, label='target anchor')
        a.plot(obs.robot.x, obs.robot.y, 'D', color='magenta', ms=8, label='robot start')
        a.arrow(obs.robot.x, obs.robot.y, 0.5 * math.cos(obs.robot.yaw), 0.5 * math.sin(obs.robot.yaw),
                color='magenta', head_width=0.1)
        sq = np.array([[-3.2, -3.2], [3.2, -3.2], [3.2, 3.2], [-3.2, 3.2], [-3.2, -3.2]])
        wx, wy = fr.to_world(sq[:, 0], sq[:, 1])
        a.plot(wx, wy, 'k-', lw=1, label='raster')
        ex, ey = fr.to_world([0, 1.0], [0, 0])
        a.annotate('', xy=(ex[1], ey[1]), xytext=(ex[0], ey[0]), arrowprops=dict(color='k', arrowstyle='->'))
        a.set_aspect('equal')
        a.set_title(f'world: {scene} {ep} ({u["situation"]}); black arrow = raster +x')
        a.legend(loc='best', fontsize=7)
        b = ax[1]
        img = np.ones((C.N, C.N, 3))
        img[r[0] > 0.5] = (0.75, 0.75, 0.75)
        img[r[1] > 0.5] = (1.0, 0.45, 0.45)
        img[r[4] > 0.5] = (0.2, 0.3, 1.0)
        img[r[5] > 0.5] = (0.1, 0.7, 0.1)
        img[r[6] > 0.5] = (0.9, 0.1, 0.9)
        ext = 3.2
        b.imshow(img, origin='lower', extent=(-ext, ext, -ext, ext), interpolation='nearest')
        pos = np.argwhere(r[1] > 0.5)
        for rr, cc in pos[::3]:
            x, y = C.OFFS[cc], C.OFFS[rr]
            b.plot([x, x + 0.12 * r[2][rr, cc]], [y, y + 0.12 * r[3][rr, cc]], color='darkred', lw=0.8)
        if lab.any():
            x, y = lab[:, 0] * C.SCALE, lab[:, 1] * C.SCALE
            b.plot(x, y, '.', color='orange', ms=5)
            for k in range(len(x)):
                b.plot([x[k], x[k] + 0.25 * lab[k, 2]], [y[k], y[k] + 0.25 * lab[k, 3]], color='darkorange', lw=0.8)
        b.set_title(f'raster (x right = towards target; robot at x={-fr.dist:.1f} m); '
                    f'region cells {u["n_region_cells_raster"]}, labels {"yes" if lab.any() else "none"}', fontsize=9)
        b.set_xlabel('x (m)')
        b.set_ylabel('y (m)')
        fig.tight_layout()
        PNG.mkdir(parents=True, exist_ok=True)
        out = PNG / f'seed{seed}_{scene}_{ep}.png'
        fig.savefig(out, dpi=90)
        plt.close(fig)
        print(out)


if __name__ == '__main__':
    draw(int(sys.argv[1]), sys.argv[2:])
