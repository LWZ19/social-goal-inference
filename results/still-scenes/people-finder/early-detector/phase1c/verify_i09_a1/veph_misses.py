"""Verifier (descriptive, post hoc): the E-ph and E-real misses: view class and the best raw box score over the
person's projected chest point (z = 1.3 m)."""
import json, math, sys
sys.path.insert(0, 'results/2026-10-17-beyond-vlm/phase1b/i03')
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H, people_scorer as PS
from sgi.geometry import project, bearing, wrap_angle
D = 'results/2026-10-17-beyond-vlm/phase1c/i09/verify_i09_a1/'
man = {e['folder']: e['rgb'] for e in json.load(open(D + 'manifest_verify.json'))['images'] if e['part'] == 'eval'}
raw = {json.loads(l)['rgb']: json.loads(l) for l in open(D + 'raw/rtdetr_v2_r50vd.jsonl')}
rows = [json.loads(l) for l in open(D + 'veval_scenes.jsonl')]
by = {s.scene_id + '|' + str(s.folder): s for s in H.distinct_scenes()}
for r in rows:
    if not r['misses']:
        continue
    s = by[r['scene_id'] + '|' + r['folder']]
    sc = H.scene(s.runs[0], s.scene_id)
    cam = sc.observation.camera
    vis = {p.id: p for p in sc.truth.people}
    for m in r['misses']:
        p = vis[m['id']]
        u, v, z = project(cam, (p.x, p.y, 1.3))
        best = max([b[4] for b in raw[man[r['folder']]]['boxes'] if b[0] <= u <= b[2] and b[1] <= v <= b[3]], default=None)
        va = abs(math.degrees(wrap_angle(bearing((p.x, p.y), cam.T_world_camera[:2, 3]) - p.yaw)))
        print(r['domain'], r['folder'].split('/')[-3] if r['domain'] == 'E-ph' else '', r['scene_id'], m['id'], m['role_class'], f'view {va:.0f} deg', f'best box score over chest {best}')
