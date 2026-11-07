"""N6: phantoms (detections >1 m from every true person) at s23 GLM line goal steps: farthest listed from target?
distance of the solved place; image origin (pixel to the projected axis of true people)."""
import sys, math, statistics
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
from sgi.gt.derive import is_visible
from sgi.geometry import project
sys.path.insert(0, 'results/2026-10-17-beyond-vlm/rerun/perception/verify')
from v_main import my_match
OUT = __file__.replace('.py', '_out.txt')
L = []
def P(*a):
    s = ' '.join(map(str, a)); print(s); L.append(s)

def segd(p, a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]; LL = dx*dx+dy*dy
    t = 0 if LL == 0 else max(0, min(1, ((p[0]-a[0])*dx+(p[1]-a[1])*dy)/LL))
    return math.dist(p, (a[0]+t*dx, a[1]+t*dy))

for stage, model in (('s23', 'glm53'), ('s23', 'dsv41')):
    rec_listed = farthest = 0; place_d = []
    rec_hold = rec_fail = 0
    pix = {}; found_pix = []
    for r in H.runs(stage=stage, model=model, method='con'):
        if r.arm not in ('real', 'staffed', 'plain'): continue
        th = H.recorded_config(r)[1].thresholds
        for rec in H.records(r):
            if not rec.solver_step or rec.situation != 'waiting_line': continue
            sc = H.scene(r, rec.scene_id)
            tp = sc.truth.people
            vis = [p for p in tp if is_visible(p, sc.observation, th)]
            m = my_match(rec.people, vis, th.person_radius_m)
            ep = sc.truth.episodes[rec.episode_id]
            chosen = rec.score.get('matched_target_id') or ep.target_ids[0]
            g = sc.truth.group_for(rec.episode_id, chosen)
            members = {p.id for p in vis if p.group == g}
            found = set(m.values())
            far = [d for d in rec.people if min(math.dist(d['xy'], (p.x, p.y)) for p in tp) > 1.0]
            if far:
                rec_hold += 1; rec_fail += not rec.score.get('success')
            listed = {n for c in rec.constraints if c['constraint']['kind'] == 'come_after' for n in c['constraint']['people']}
            tw = rec.target['world'][:2]
            ld = [d for d in rec.people if d['number'] in listed]
            farl = [d for d in far if d['number'] in listed]
            if farl:
                rec_listed += 1
                fl = max(ld, key=lambda d: math.dist(d['xy'], tw))
                if fl['number'] in {d['number'] for d in farl}:
                    farthest += 1
                    place_d.append(math.dist(rec.pose_world[:2], fl['xy']))
            cam = sc.observation.camera
            # frame check: matched detections to own person's axis
            for n, pid in m.items():
                d = next(x for x in rec.people if x['number'] == n)
                p = next(x for x in tp if x.id == pid)
                lo, hi = project(cam, (p.x, p.y, 0.0)), project(cam, (p.x, p.y, 1.8))
                if lo and hi and d.get('pixel'): found_pix.append(segd(d['pixel'], lo[:2], hi[:2]))
            for d in far:
                key = (r.arm, rec.scene_id, tuple(round(v, 6) for v in d['xy']))
                best = None
                for p in tp:
                    lo, hi = project(cam, (p.x, p.y, 0.0)), project(cam, (p.x, p.y, 1.8))
                    if lo is None or hi is None: continue
                    gd = segd(d['pixel'], lo[:2], hi[:2])
                    if best is None or gd < best[0]: best = (gd, p)
                nfm = best[1].id in members and best[1].id not in found
                pix.setdefault(key, []).append((best[0], nfm))
    P(f'== {stage} {model}: records holding a phantom {rec_hold}, failing {rec_fail}; records listing a phantom under come_after {rec_listed}; phantom is the listed detection farthest from target {farthest}; place-to-phantom median {statistics.median(place_d):.2f} m' if place_d else '')
    dd = [v[0][0] for v in pix.values()]
    nf25 = sum(1 for v in pix.values() if v[0][0] <= 25 and v[0][1])
    nf25_any = sum(1 for v in pix.values() if any(x[0] <= 25 and x[1] for x in v))
    P(f'   distinct phantoms (arm, scene, xy) {len(pix)}; pixel to nearest true axis median {statistics.median(dd):.1f} px; <=25 px {sum(x <= 25 for x in dd)}; >50 px {sum(x > 50 for x in dd)}; <=25 px and a never-found member (first record) {nf25}, (any record) {nf25_any}')
    P(f'   frame check: matched detections pixel-to-own-axis median {statistics.median(found_pix):.1f} px (n {len(found_pix)})')
open(OUT, 'w').write('\n'.join(L) + '\n')
