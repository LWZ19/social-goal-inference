"""Extra checks of side statements (verifier): moved counts, camera-side counts over all failures, Haiku anchor
reading, Sonnet dv against the region's nearest point."""
import sys, json, math, collections, statistics as st, io
import numpy as np
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
from sgi.geometry import project
V = 'results/2026-10-17-beyond-vlm/rerun/snap/verify/'
rows = [json.loads(l) for l in open(V + 'v2_snap_records.jsonl')]
OUT = io.StringIO()
def p(*a):
    print(*a); print(*a, file=OUT)
L = lambda m: [r for r in rows if r['model'] == m and r['sit'] == 'waiting_line' and r['tc'] and not r['ok'] and r['source'] == 'direct_marking']
p('sonnet line fails moved by snapD:', sum(1 for r in L('sonnet55_low') if r['movedD'] > 0), 'of', len(L('sonnet55_low')))
for m in ['glm53', 'sonnet55_low', 'haiku55', 'qwen38_low', 'dsv41', 'gemini38_low']:
    x = L(m)
    p(m, 'line fails (all, before snap) >0.3 m camera side:', sum(1 for r in x if r['lat_camside'] > 0.3), 'of', len(x),
      '| still-failing after snapD:', sum(1 for r in x if not r['okD'] and r['lat_camside'] > 0.3), 'of', sum(1 for r in x if not r['okD']),
      '| behind (not_ahead, recorded pose):', sum(1 for r in x if r['not_ahead']))
def nearest_on_poly(pt, poly):
    best, bq = 1e9, None
    P = list(poly)
    for a, b in zip(P, P[1:] + P[:1]):
        ax, ay = a; bx, by = b; dx, dy = bx - ax, by - ay
        t = max(0.0, min(1.0, ((pt[0] - ax) * dx + (pt[1] - ay) * dy) / (dx * dx + dy * dy + 1e-18)))
        q = (ax + t * dx, ay + t * dy); d = math.dist(pt, q)
        if d < best: best, bq = d, q
    return bq
# Haiku: pose nearer the anchor than the first member is (radius reading)
for meth in ['cot', 'fsc']:
    (dummy) = None
dv_near = collections.defaultdict(list)
hk = collections.Counter()
for meth in ['cot', 'fsc']:
    for arm in ['real', 'staffed', 'plain']:
        for model, stage in [('haiku55', 's23'), ('sonnet55_low', 's23')]:
            (ref,) = H.runs(stage=stage, arm=arm, model=model, method=meth)
            for rec in H.records(ref):
                if rec.situation != 'waiting_line' or not rec.score.get('target_correct') or rec.score.get('success'): continue
                if rec.result.get('source') != 'direct_marking': continue
                sc = H.scene(ref, rec.scene_id)
                tid = rec.score['matched_target_id']
                gid = sc.truth.group_for(rec.episode_id, tid)
                mem = [q for q in sc.truth.people if q.group == gid]
                a = sc.truth.targets[tid].anchor[:2]
                first = min(((m.x, m.y) for m in mem), key=lambda q: math.dist(a, q))
                x, y, _ = rec.pose_world
                if model == 'haiku55':
                    hk[(meth, 'n')] += 1
                    hk[(meth, 'd_anchor<|anchor-first|')] += math.dist((x, y), a) < math.dist(a, first)
                    hk[(meth, 'd_anchor<d_first')] += math.dist((x, y), a) < math.dist((x, y), first)
                else:
                    gt = sc.truths[rec.episode_id].gt
                    reg = [r for r in gt.acceptable_regions if r.target_id in (None, tid)][0]
                    q = nearest_on_poly((x, y), reg.polygon)
                    pix = project(sc.observation.camera, (q[0], q[1], 0.0))
                    gp = rec.result['goal_pixel']
                    dv_near[meth].append(gp[1] - pix[1])
p('haiku', dict(hk))
for k, v in dv_near.items():
    p('sonnet', k, 'fails n', len(v), 'median dv vs nearest region point', f'{st.median(v):.1f}')
open(V + 'v4_extra.txt', 'w').write(OUT.getvalue())
