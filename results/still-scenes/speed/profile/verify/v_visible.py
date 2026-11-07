"""Visibility of every realistic-set person in the observation image (projection + depth-occlusion test), and the
facing angle restricted to visible people. One arm per layout is enough for staff (real); all three for the rest."""
import sys, math, collections, statistics
import numpy as np
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H
REAL = ("real", "staffed", "plain")
out = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)
def wrap(a): return math.pi - ((math.pi - a) % (2 * math.pi))
rows = []
sits = collections.Counter()
for sref in H.distinct_scenes():
    rr = [r for r in sref.runs if r.arm in REAL]
    if not rr: continue
    arm = rr[0].arm
    sc = H.scene(rr[0], sref.scene_id)
    obs = sc.observation; cam = obs.camera
    T = np.asarray(cam.T_world_camera); Ti = np.linalg.inv(T)
    depth = obs.depth
    line, conv = set(), set()
    for ep in sc.truth.episodes.values():
        sits[ep.situation] += 1
        g = ep.group_at_target
        gs = (set(g.values()) - {None}) if isinstance(g, dict) else ({g} if g else set())
        (line if ep.situation == "waiting_line" else conv if ep.situation == "conversation_group" else set()).update(gs)
    for p in sc.truth.people:
        cat = "line" if p.group in line else "group" if p.group in conv else "staff" if p.role else "other"
        pts_in = 0; pts_vis = 0
        for h in (0.2, 0.6, 1.0, 1.4, 1.6):
            pc = Ti @ np.array([p.x, p.y, h, 1.0]); Z = pc[2]
            if Z <= 0.3: continue
            u = cam.fx * pc[0] / Z + cam.cx; v = cam.fy * pc[1] / Z + cam.cy
            if 0 <= u < cam.width and 0 <= v < cam.height:
                pts_in += 1
                d = float(depth[int(v), int(u)])
                if not math.isfinite(d) or d >= Z - 0.4:   # nothing much nearer than the person's centre line
                    pts_vis += 1
        a = abs(math.degrees(wrap(math.atan2(T[1, 3] - p.y, T[0, 3] - p.x) - p.yaw)))
        rows.append(dict(arm=arm, scene=sref.scene_id, pid=p.id, cat=cat, group=p.group, ang=a,
                         inimg=pts_in > 0, vis=pts_vis > 0, pin=pts_in, pvis=pts_vis,
                         d=math.hypot(p.x - T[0, 3], p.y - T[1, 3])))
P("episode situations (all realistic scenes):", dict(sits))
for cat in ("line", "group", "staff", "other"):
    sel = [r for r in rows if r['cat'] == cat]
    P(f"\n== {cat}: instances {len(sel)}, in image (any of 5 body points) {sum(r['inimg'] for r in sel)}, "
      f"visible after depth test {sum(r['vis'] for r in sel)}")
    for lab, sub in (("all", sel), ("in image", [r for r in sel if r['inimg']]), ("visible", [r for r in sel if r['vis']])):
        xs = [r['ang'] for r in sub]
        if not xs: P(f"  {lab}: n=0"); continue
        P(f"  {lab}: n={len(xs)} min={min(xs):.1f} q25={np.percentile(xs,25):.1f} median={statistics.median(xs):.1f} "
          f"q75={np.percentile(xs,75):.1f} max={max(xs):.1f}; <45 {sum(x<45 for x in xs)}; [75,107] "
          f"{sum(75<=x<=107 for x in xs)}; >135 {sum(x>135 for x in xs)}; dist {min(r['d'] for r in sub):.2f}-{max(r['d'] for r in sub):.2f}")
    if cat in ("group", "other", "staff"):
        for r in sorted(sel, key=lambda r: (r['scene'], r['pid'])):
            if r['arm'] == ('real' if cat == 'staff' else 'plain'):
                P(f"   {r['scene']:36s} {r['pid']:4s} grp={r['group']} ang={r['ang']:6.1f} d={r['d']:5.2f} "
                  f"in={r['pin']}/5 vis={r['pvis']}/5")
    if cat == "line":
        P("   line members not fully visible:", [(r['scene'], r['pid'], r['pin'], r['pvis']) for r in sel if r['pvis'] < 5][:20])
open('results/still-scenes/speed/profile/verify/v_visible.txt', 'w').write("\n".join(out) + "\n")
