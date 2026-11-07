"""I03 Stage A scorer self-check on fit data (FROZEN.md section 9 step 2): true people given as detections must
give recall 1 and 0 false. Also: true people with true yaw give facing 1 (full and axis); an empty list gives
recall 0; a detection 0.6 m off a person is false. Worked examples only (and fresh seeds if present)."""
import sys
import time
sys.path.insert(0, 'results/2026-10-17-beyond-vlm/phase1b/i03/stageA')
import fitdata as FD  # noqa: E402
import people_scorer as PS  # noqa: E402
import blobs as B  # noqa: E402

only_examples = '--examples' in sys.argv
n = bad = 0
vis_by_domain = {}
t0 = time.monotonic()
refs = FD.fit_refs(require_fresh=False) if not only_examples else [r for r in FD.fit_refs(require_fresh=False) if r[3][0] == 'example']
for domain, sset, scene_id, loader in refs:
    sc, th = FD.load(loader)
    truth = sc.truth
    ex = PS.exempt_points(truth)
    vis = PS.truth_info(sc, th)['visible']
    r = PS.score_people([(p.x, p.y, p.yaw) for p in vis], sc, th, exempt_xy=ex)
    ok = (r['recall'] in (1.0, None) and r['false'] == 0 and r['exempt'] == 0 and r['merges'] == 0
          and r['merged_pairs'] == 0 and r['facing']['within45_full'] == r['matched'] == r['n_visible']
          and r['facing']['within45_axis'] == r['matched'])
    r0 = PS.score_people([], sc, th, exempt_xy=ex)
    ok &= r0['matched'] == 0 and r0['false'] == 0
    if vis:
        p = vis[0]
        r1 = PS.score_people([(p.x + 0.6, p.y, None)], sc, th)
        others_near = any(abs(q.x - p.x - 0.6) ** 2 + (q.y - p.y) ** 2 <= 0.25 for q in vis[1:])
        ok &= others_near or (r1['matched'] == 0 and r1['false_raw'] == 1)
    vis_by_domain.setdefault(domain, [0, 0, 0])
    vis_by_domain[domain][0] += 1
    vis_by_domain[domain][1] += len(vis)
    vis_by_domain[domain][2] += len(truth.people)
    n += 1
    bad += not ok
    if not ok:
        print('FAIL', domain, scene_id, r['recall'], r['false'], r['merges'], r['facing'])
print(f'self-check: {n - bad} of {n} fit scenes pass; {time.monotonic() - t0:.1f} s')
for d, (ns, nv, np_) in vis_by_domain.items():
    print(f'  {d}: {ns} scenes, {nv} visible true people of {np_} in truth.people')
# timing of the detector on one realistic and one placeholder example
for domain, sset, scene_id, loader in refs[:1] + [r for r in refs if r[0] == 'F-ph'][:1]:
    sc, th = FD.load(loader)
    t = time.monotonic()
    pts = B.lifted(sc.observation)
    for z_lo in B.GRID['z_lo']:
        for z_hi in B.GRID['z_hi']:
            B.blobs_band(pts, z_lo, z_hi, B.GRID['split_r'])
    print(f'  detector, 6 bands x 3 splits on {scene_id} ({sc.observation.depth.shape}, {len(pts)} points): '
          f'{time.monotonic() - t:.1f} s')
