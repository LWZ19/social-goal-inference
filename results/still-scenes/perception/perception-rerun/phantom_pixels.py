"""Phantom detections at lines (>1 m from every true person): is the marked body pixel on a true person in the image
(a real person lifted onto the wall behind) or on empty background (an invented person)?

For each far detection, the image distance from its recorded pixel to the image segment of each true person's
vertical axis (z = 0.0 to 1.8 m, projected with sgi.geometry.project); the nearest person, whether they are a member
of the line at the target, and whether that member was never found (package matching) in the record."""
import math
import sys

sys.path.insert(0, 'results/still-scenes/perception/perception-rerun')
import perception as P  # noqa: E402
from sgi.geometry import project  # noqa: E402


def seg_dist(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = dx * dx + dy * dy
    t = 0 if L == 0 else max(0, min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L))
    return math.dist(p, (a[0] + t * dx, a[1] + t * dy))


for stage, model in [('s19', 'glm53'), ('s23', 'glm53'), ('s20', 'dsv41')]:
    rows = [a for a in (P.analyse(r) for r in P.recs_of(stage, model, P.REAL)) if a is not None]
    goal = [a for a in rows if a['rec'].situation == 'waiting_line' and a['rec'].solver_step]
    seen = set()
    dists, on_member, on_missed = [], 0, 0
    n = 0
    for a in goal:
        sc, th, vis = P.scene_info(a['rec'])
        cam = sc.observation.camera
        for d, dm in zip(a['det'], a['det_min_all']):
            if dm <= 1.0:
                continue
            key = (a['rec'].arm, a['rec'].scene_id, a['rec'].trial, d.number, round(d.xy[0], 5))
            if key in seen:                      # each distinct people answer once
                continue
            seen.add(key)
            n += 1
            best = None
            for p in sc.truth.people:
                lo, hi = project(cam, (p.x, p.y, 0.0)), project(cam, (p.x, p.y, 1.8))
                if lo is None or hi is None:
                    continue
                g = seg_dist(d.pixel, lo[:2], hi[:2])
                if best is None or g < best[0]:
                    best = (g, p)
            dists.append(best[0])
            on_member += best[0] <= 25 and best[1].id in a['members']
            on_missed += best[0] <= 25 and best[1].id in a['members'] and best[1].id not in a['found_ids']
    dists.sort()
    print(f'{stage} {model}: distinct far detections at line goal steps {n}; pixel to nearest true person axis: median '
          f'{dists[len(dists) // 2]:.1f} px, <=10 px {sum(x <= 10 for x in dists)}, <=25 px {sum(x <= 25 for x in dists)}, '
          f'>50 px {sum(x > 50 for x in dists)}; within 25 px of a line member {on_member}, of a member never found in '
          f'that record {on_missed}')
