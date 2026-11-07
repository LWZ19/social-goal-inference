"""Independent verifier re-implementation of the depth-blob pilot (I03 / I33), development scenes only.

Written from the REPORT's I03 description, before reading the first agent's script. Everything is read through
research_harness (H.distinct_scenes, H.scene). Outputs: verify/per_scene/<scene>.json and verify/v_blobs.txt.

Pipeline per scene:
  sgi.geometry.lift_all(depth, camera) -> world points; keep 0.15 < z < 2.1 (floor is z = 0).
  height map = max z per 5 cm cell; occupied cells grouped into 8-connected components.
  split at local maxima: markers = cells equal to the max of the component's heights within a disk of radius r
  (touching marker cells merged into one marker). Two assignment rules:
    'ws'  : marker watershed (priority flood from markers, highest cells first, 8-neighbours, inside the component)
    'vor' : each cell to its nearest marker cell (Voronoi inside the component)
  blob kept when 1.2 <= top <= 2.0 and 0.2 <= footprint <= 0.8, footprint = extent of the cell centres along their
  first principal axis + one cell (alternative 'bbox': the larger of the x / y extents + one cell).
  blob position = mean of its cell centres.
  facing axis = minor principal axis of the xy of the kept points in the blob's cells (and of the cell centres).
"""
import heapq
import json
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H  # noqa: E402
from sgi.geometry import lift_all, project  # noqa: E402
from sgi.gt.derive import is_visible  # noqa: E402

OUT = Path(__file__).resolve().parent
(OUT / 'per_scene').mkdir(exist_ok=True)
CELL = 0.05
ZLO, ZHI = 0.15, 2.1
TOP = (1.2, 2.0)
FOOT = (0.2, 0.8)
RADII = (0.15, 0.25, 0.35)
RULES = ('ws', 'vor')
NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def set_of(sref):
    d = str(sref.scenes_dir)
    if '/realistic_final/plain/' in d + '/':
        return 'plain'
    if '/realistic_final/real/' in d + '/':
        return 'real'
    if '/realistic_final/staffed/' in d + '/':
        return 'staffed'
    if d.endswith('data/placeholder/eval'):
        return 'ph'
    for k in ('seed0', 'seed1', 'seed2'):
        if f'/dev_new/{k}/' in d + '/':
            return k
    return 'other'


def components(occ):
    lab = -np.ones(occ.shape, dtype=np.int64)
    n = 0
    nx, ny = occ.shape
    for i, j in zip(*np.nonzero(occ)):
        if lab[i, j] >= 0:
            continue
        lab[i, j] = n
        q = deque([(i, j)])
        while q:
            a, b = q.popleft()
            for da, db in NB8:
                c, d = a + da, b + db
                if 0 <= c < nx and 0 <= d < ny and occ[c, d] and lab[c, d] < 0:
                    lab[c, d] = n
                    q.append((c, d))
        n += 1
    return lab, n


def disk_max(hm_comp, r_cells):
    """Max over a disk of radius r_cells of hm_comp (-inf outside)."""
    nx, ny = hm_comp.shape
    pad = np.full((nx + 2 * r_cells, ny + 2 * r_cells), -np.inf)
    pad[r_cells:r_cells + nx, r_cells:r_cells + ny] = hm_comp
    out = np.full(hm_comp.shape, -np.inf)
    for da in range(-r_cells, r_cells + 1):
        for db in range(-r_cells, r_cells + 1):
            if da * da + db * db > r_cells * r_cells:
                continue
            out = np.maximum(out, pad[r_cells + da:r_cells + da + nx, r_cells + db:r_cells + db + ny])
    return out


def split_component(cells, hm, r_cells, rule):
    """cells: list of (i, j) of one component; returns list of lists of cells (sub-blobs)."""
    ci = np.array([c[0] for c in cells])
    cj = np.array([c[1] for c in cells])
    i0, j0 = ci.min(), cj.min()
    sub = np.full((ci.max() - i0 + 1, cj.max() - j0 + 1), -np.inf)
    sub[ci - i0, cj - j0] = hm[ci, cj]
    mx = disk_max(sub, r_cells)
    inside = np.isfinite(sub)
    mark = inside & (sub >= mx)
    # merge touching marker cells
    mlab, nm = components(mark)
    lab = -np.ones(sub.shape, dtype=np.int64)
    lab[mark] = mlab[mark]
    if nm <= 1:
        return [cells]
    if rule == 'ws':
        heap = []
        for a, b in zip(*np.nonzero(mark)):
            heapq.heappush(heap, (-sub[a, b], a, b))
        while heap:
            _, a, b = heapq.heappop(heap)
            for da, db in NB8:
                c, d = a + da, b + db
                if 0 <= c < sub.shape[0] and 0 <= d < sub.shape[1] and inside[c, d] and lab[c, d] < 0:
                    lab[c, d] = lab[a, b]
                    heapq.heappush(heap, (-sub[c, d], c, d))
    else:
        ma, mb = np.nonzero(mark)
        mk = mlab[ma, mb]
        ia, ib = np.nonzero(inside & ~mark)
        for s0 in range(0, len(ia), 2000):
            a_, b_ = ia[s0:s0 + 2000], ib[s0:s0 + 2000]
            d2 = (a_[:, None] - ma[None, :]) ** 2 + (b_[:, None] - mb[None, :]) ** 2
            lab[a_, b_] = mk[np.argmin(d2, axis=1)]
    out = []
    for k in range(nm):
        a, b = np.nonzero(lab == k)
        if len(a):
            out.append(list(zip((a + i0).tolist(), (b + j0).tolist())))
    return out


def axis_minor(xy):
    if len(xy) < 3:
        return None
    c = xy - xy.mean(axis=0)
    w, v = np.linalg.eigh(c.T @ c / len(xy))
    return float(math.atan2(v[1, 0], v[0, 0]))  # eigenvector of the smallest eigenvalue


def blobs_of_scene(P):
    keep = (P[:, 2] > ZLO) & (P[:, 2] < ZHI)
    Q = P[keep]
    x0, y0 = math.floor(Q[:, 0].min() / CELL), math.floor(Q[:, 1].min() / CELL)
    ix = np.floor(Q[:, 0] / CELL).astype(np.int64) - x0
    iy = np.floor(Q[:, 1] / CELL).astype(np.int64) - y0
    nx, ny = ix.max() + 1, iy.max() + 1
    hm = np.full((nx, ny), -np.inf)
    np.maximum.at(hm, (ix, iy), Q[:, 2])
    occ = np.isfinite(hm)
    clab, nc = components(occ)
    flat = ix * ny + iy
    order = np.argsort(flat, kind='stable')
    fs = flat[order]
    res = {}
    comp_cells = [[] for _ in range(nc)]
    for a, b in zip(*np.nonzero(occ)):
        comp_cells[clab[a, b]].append((int(a), int(b)))
    for r in RADII:
        rc = int(round(r / CELL))
        for rule in RULES:
            blobs = []
            for cells in comp_cells:
                for sub in split_component(cells, hm, rc, rule):
                    ca = np.array(sub)
                    cen = np.stack([(ca[:, 0] + x0 + 0.5) * CELL, (ca[:, 1] + y0 + 0.5) * CELL], axis=1)
                    top = float(hm[ca[:, 0], ca[:, 1]].max())
                    if len(ca) >= 2:
                        c = cen - cen.mean(axis=0)
                        w, v = np.linalg.eigh(c.T @ c / len(cen))
                        major = v[:, 1]
                        proj = c @ major
                        foot_pca = float(proj.max() - proj.min() + CELL)
                    else:
                        foot_pca = CELL
                    foot_bbox = float(max(np.ptp(cen[:, 0]), np.ptp(cen[:, 1])) + CELL)
                    body_pca = TOP[0] <= top <= TOP[1] and FOOT[0] <= foot_pca <= FOOT[1]
                    body_bbox = TOP[0] <= top <= TOP[1] and FOOT[0] <= foot_bbox <= FOOT[1]
                    if not (body_pca or body_bbox):
                        blobs.append({'pos': cen.mean(axis=0).round(4).tolist(), 'top': round(top, 3),
                                      'foot_pca': round(foot_pca, 3), 'foot_bbox': round(foot_bbox, 3),
                                      'kept_pca': False, 'kept_bbox': False, 'n_cells': len(ca)}
                                     if (top > 1.0 and len(ca) >= 4) else None)
                        continue
                    # points of these cells
                    keys = (ca[:, 0] * ny + ca[:, 1])
                    lo = np.searchsorted(fs, keys, 'left')
                    hi = np.searchsorted(fs, keys, 'right')
                    idx = np.concatenate([order[l:h] for l, h in zip(lo, hi)])
                    pts = Q[idx, :2]
                    zz = Q[idx, 2]
                    band = pts[(zz > 1.0) & (zz < 1.6)]
                    blobs.append({'pos': cen.mean(axis=0).round(4).tolist(), 'top': round(top, 3),
                                  'foot_pca': round(foot_pca, 3), 'foot_bbox': round(foot_bbox, 3),
                                  'kept_pca': bool(body_pca), 'kept_bbox': bool(body_bbox), 'n_cells': len(ca),
                                  'n_pts': int(len(idx)),
                                  'ax_pts': axis_minor(pts), 'ax_cells': axis_minor(cen),
                                  'ax_band': axis_minor(band) if len(band) >= 3 else None})
            res[f'{rule}_{r}'] = [b for b in blobs if b is not None]
    return res


def in_frustum(cam, x, y):
    for z in (0.2, 0.6, 1.0, 1.4, 1.7):
        pr = project(cam, (x, y, z))
        if pr is not None and 0 <= pr[0] < cam.width and 0 <= pr[1] < cam.height:
            return True
    return False


def main():
    ds = H.distinct_scenes()
    lines = []
    for s in ds:
        st = set_of(s)
        ref = s.runs[0]
        sc = H.scene(ref, s.scene_id)
        obs = sc.observation
        cam = obs.camera
        th = H.recorded_config(ref)[1].thresholds
        P = lift_all(obs.depth, cam)
        res = blobs_of_scene(P)
        people = []
        for p in sc.truth.people:
            people.append({'id': p.id, 'x': p.x, 'y': p.y, 'yaw': p.yaw, 'group': p.group, 'role': p.role,
                           'in_img': in_frustum(cam, p.x, p.y), 'visible': bool(is_visible(p, obs, th))})
        # occlusion of lower body (z 0.1..0.8): depth image nearer than the axis point by > 0.3 m
        for pp in people:
            hid = seen = 0
            for z in np.linspace(0.1, 0.8, 8):
                pr = project(cam, (pp['x'], pp['y'], float(z)))
                if pr is None or not (0 <= pr[0] < cam.width and 0 <= pr[1] < cam.height):
                    continue
                dv = float(obs.depth[int(pr[1]), int(pr[0])])
                if dv < pr[2] - 0.3:
                    hid += 1
                else:
                    seen += 1
            pp['low_hidden'], pp['low_seen'] = hid, seen
        sit = s.scene_id
        out = {'scene_id': s.scene_id, 'set': st, 'folder': str(s.folder), 'cam': cam.T_world_camera[:3, 3].tolist(),
               'targets': {k: list(v.anchor) for k, v in sc.truth.targets.items()},
               'people': people, 'blobs': res}
        (OUT / 'per_scene' / f'{st}__{s.scene_id}.json').write_text(json.dumps(out))
        nk = len([b for b in res['ws_0.25'] if b.get('kept_pca')])
        line = f'{st:8s} {s.scene_id:38s} people={len(people)} in_img={sum(p["in_img"] for p in people)} ' \
               f'visible={sum(p["visible"] for p in people)} kept(ws,0.25,pca)={nk}'
        print(line, flush=True)
        lines.append(line)
    (OUT / 'v_blobs.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
