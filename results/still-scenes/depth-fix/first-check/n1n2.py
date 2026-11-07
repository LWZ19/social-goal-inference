"""N1 and N2: E near misses at groups (judge near.py) and the median distance of failed marked points at lines.

Reads only through research_harness. Recorded scores (as the judge read them) and the package's score now
(score_pose of the recorded pose, the freespace cached per scene) are both shown.
"""
import sys, collections, statistics as st
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H

OUT = 'results/2026-10-17-beyond-vlm/rerun/snap/n1n2.txt'
lines = []
def p(*a):
    s = ' '.join(str(x) for x in a); print(s); lines.append(s)

ARMS = ('real', 'staffed', 'plain')
PANEL = [('qwen38_low', 's16b'), ('glm53', 's16b'), ('dsv41', 's20')]

def med(v):
    return f"{st.median(v):.2f}" if v else '-'

rows = []   # one dict per record
for model, stage in PANEL:
    for method in ('cot', 'fsc'):
        for arm in ARMS:
            refs = H.runs(stage=stage, arm=arm, model=model, method=method)
            assert len(refs) == 1, (stage, arm, model, method, refs)
            ref = refs[0]
            tally = collections.Counter()
            n = 0
            for rec in H.records(ref, tally=tally):
                n += 1
                s = rec.score
                now = H.score_pose(rec, rec.pose_world) if (rec.pose_world is not None and rec.result.get('status') == 'ok') else H.rescore(rec)
                rows.append(dict(model=model, stage=stage, method=method, arm=arm, run=ref.folder, sit=rec.situation,
                                 source=rec.result.get('source'), status=rec.result.get('status'),
                                 rec=s, now=now.model_dump(), scene=rec.scene_id, ep=rec.episode_id, trial=rec.trial))
            p(f"# {ref.folder}: {dict(tally)}")

def view(r, which):
    return r['rec'] if which == 'rec' else r['now']

for which in ('rec', 'now'):
    p(f"\n=== N1 part 1 (near.py block 1), scores {'recorded' if which=='rec' else 'now'}: failures with the right target and a distance to the region; all sources ===")
    for model, stage in PANEL:
        for method in ('cot', 'fsc'):
            by = collections.defaultdict(list)
            for r in rows:
                if (r['model'], r['method']) != (model, method):
                    continue
                s = view(r, which)
                if s.get('success') or not s.get('target_correct'):
                    continue
                if s.get('distance_to_region_m') is None:
                    continue
                by[r['sit']].append((s['distance_to_region_m'], s.get('min_body_gap_m'), s.get('in_region'), s.get('reachable'), s.get('heading_ok')))
            for sit, v in sorted(by.items()):
                ds = [x[0] for x in v]
                p(f"{model:11s} {method} {sit:19s} n {len(v):3d} med {med(ds)} <=0.15: {sum(x<=0.15 for x in ds):3d} <=0.3: {sum(x<=0.3 for x in ds):3d} <=0.5: {sum(x<=0.5 for x in ds):3d} inreg_fail {sum(1 for x in v if x[2])}")

    p(f"\n=== N1 part 2 (near.py block 2), scores {'recorded' if which=='rec' else 'now'}: fsc failures with the right target, people scenes ===")
    p("key: (situation, near = distance to region <= 0.3 m / far, reachable or not, min_body_gap < 0.3 m or not)")
    for model, stage in PANEL:
        c = collections.Counter()
        for r in rows:
            if (r['model'], r['method']) != (model, 'fsc') or r['sit'] == 'no_one':
                continue
            s = view(r, which)
            if s.get('success') or not s.get('target_correct'):
                continue
            dd = s.get('distance_to_region_m')
            if dd is None:
                continue
            g = s.get('min_body_gap_m')
            c[(r['sit'], 'near' if dd <= 0.3 else 'far', 'unreach' if not s.get('reachable') else 'reach',
               'gap<0.3' if (g is not None and g < 0.3) else 'gap>=0.3')] += 1
        for k, v in sorted(c.items()):
            p(model, k, v)

    p(f"\n=== N1 summary, scores {'recorded' if which=='rec' else 'now'}: E (fsc) failures at groups with the right target ===")
    p("near = distance to region <= 0.3 m; on a body (judge) = min_body_gap_m < 0.3 (robot's edge within 0.3 m of a body disc,"
      " i.e. personal-space intrusion); overlap = min_body_gap_m < 0 (robot disc overlaps a body disc of radius 0.25 m)")
    tot = collections.Counter()
    for model, stage in PANEL:
        c = collections.Counter()
        for r in rows:
            if (r['model'], r['method'], r['sit']) != (model, 'fsc', 'conversation_group'):
                continue
            s = view(r, which)
            if s.get('success') or not s.get('target_correct') or s.get('distance_to_region_m') is None:
                continue
            g = s.get('min_body_gap_m'); dd = s['distance_to_region_m']
            near = dd <= 0.3; body = g is not None and g < 0.3; over = g is not None and g < 0
            c['n'] += 1; c['near'] += near; c['body'] += body; c['near&body'] += near and body
            c['overlap'] += over; c['near&overlap'] += near and over; c['unreach'] += not s.get('reachable')
            c['marked'] += r['source'] == 'direct_marking'
        tot.update(c)
        p(f"{model:11s} n {c['n']:3d}  marked-point {c['marked']:3d}  near {c['near']:3d}  on-body(gap<0.3) {c['body']:3d}  near&on-body {c['near&body']:3d}  overlap(gap<0) {c['overlap']:3d}  near&overlap {c['near&overlap']:3d}  unreachable {c['unreach']:3d}")
    p(f"{'pooled':11s} n {tot['n']:3d}  marked-point {tot['marked']:3d}  near {tot['near']:3d}  on-body(gap<0.3) {tot['body']:3d}  near&on-body {tot['near&body']:3d}  overlap(gap<0) {tot['overlap']:3d}  near&overlap {tot['near&overlap']:3d}  unreachable {tot['unreach']:3d}")

# ---------------------------------------------------------------- N2: failed marked points at lines
p("\n=== N2: median distance to the region of failed records with the right target at waiting lines ===")
for which in ('rec', 'now'):
    for only_marked in (True, False):
        p(f"-- scores {'recorded' if which=='rec' else 'now'}; {'marked points only (source direct_marking)' if only_marked else 'all sources'}")
        for model, stage in PANEL:
            out = []
            for methods in (('cot',), ('fsc',), ('cot', 'fsc')):
                ds = []
                for r in rows:
                    if r['model'] != model or r['method'] not in methods or r['sit'] != 'waiting_line':
                        continue
                    if only_marked and r['source'] != 'direct_marking':
                        continue
                    s = view(r, which)
                    if s.get('success') or not s.get('target_correct') or s.get('distance_to_region_m') is None:
                        continue
                    ds.append(s['distance_to_region_m'])
                out.append(f"{'+'.join(methods)}: med {med(ds)} m (n {len(ds)})")
            p(f"{model:11s} {stage:5s} " + ';  '.join(out))

# recorded vs now disagreements
mis = [r for r in rows if bool(r['rec'].get('success')) != bool(r['now'].get('success'))]
p(f"\nrecorded vs now: {len(mis)} of {len(rows)} records differ in success")
for r in mis[:20]:
    p(f"  {r['run']} {r['scene']} {r['ep']} t{r['trial']} rec {r['rec'].get('success')} now {r['now'].get('success')}")
open(OUT, 'w').write('\n'.join(lines) + '\n')
