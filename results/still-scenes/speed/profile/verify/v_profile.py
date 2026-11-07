"""Independent verifier: facing angle to camera for line members, group members, staff, others on realistic sets."""
import sys, math, collections, statistics
import numpy as np
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H

REAL = ("real", "staffed", "plain")
out = []
def P(*a):
    s = " ".join(str(x) for x in a)
    print(s); out.append(s)

def wrap(a):
    return math.pi - ((math.pi - a) % (2 * math.pi))

def ang(px, py, yaw, vx, vy):
    return abs(math.degrees(wrap(math.atan2(vy - py, vx - px) - yaw)))

def q(xs, p):
    return float(np.percentile(xs, p))

def summ(name, xs):
    if not xs:
        P(f"{name}: n=0"); return
    P(f"{name}: n={len(xs)} min={min(xs):.2f} q25={q(xs,25):.2f} median={statistics.median(xs):.2f} "
      f"q75={q(xs,75):.2f} max={max(xs):.2f} mean={statistics.mean(xs):.2f}")

rows = []   # (arm, scene, pid, category, ang_robot, ang_cam, dist_robot, dist_cam, axis_angle_cam, u, v, Z, x, y, yaw)
lines = []  # (arm, scene, group, n, side-on angle)
seen = 0
for sref in H.distinct_scenes():
    arms = {r.arm for r in sref.runs}
    real_runs = [r for r in sref.runs if r.arm in REAL]
    if not real_runs:
        continue
    assert arms <= set(REAL), (sref.scene_id, arms)
    arm = real_runs[0].arm
    assert len({r.arm for r in real_runs}) == 1, (sref.scene_id, {r.arm for r in real_runs})
    sc = H.scene(real_runs[0], sref.scene_id)
    seen += 1
    truth = sc.truth
    obs = sc.observation
    T = np.asarray(obs.camera.T_world_camera)
    cx, cy, cz = T[:3, 3]
    rx, ry = obs.robot.x, obs.robot.y
    cam_yaw = math.atan2(T[1, 2], T[0, 2])   # optical axis (camera z) projected
    line_groups, conv_groups, other_sit = set(), set(), collections.Counter()
    for eid, ep in truth.episodes.items():
        gat = ep.group_at_target
        gs = set(gat.values()) - {None} if isinstance(gat, dict) else ({gat} if gat else set())
        if ep.situation == "waiting_line":
            line_groups |= gs
        elif ep.situation == "conversation_group":
            conv_groups |= gs
        elif gs:
            other_sit[ep.situation] += 1
    assert not (line_groups & conv_groups), sref.scene_id
    Tinv = np.linalg.inv(T)
    for p in truth.people:
        if p.group in line_groups:
            cat = "line"
        elif p.group in conv_groups:
            cat = "group"
        elif p.role:
            cat = "staff:" + p.role
        else:
            cat = "other"
        pc = Tinv @ np.array([p.x, p.y, 1.0, 1.0])
        Z = pc[2]
        u = obs.camera.fx * pc[0] / Z + obs.camera.cx
        v = obs.camera.fy * pc[1] / Z + obs.camera.cy
        rows.append(dict(arm=arm, scene=sref.scene_id, folder=str(sref.folder), pid=p.id, cat=cat, group=p.group,
                         a_rob=ang(p.x, p.y, p.yaw, rx, ry), a_cam=ang(p.x, p.y, p.yaw, cx, cy),
                         a_axis=abs(math.degrees(wrap(cam_yaw + math.pi - p.yaw))),
                         d_rob=math.hypot(p.x - rx, p.y - ry), d_cam=math.hypot(p.x - cx, p.y - cy),
                         u=u, v=v, Z=Z, W=obs.camera.width, Hh=obs.camera.height, x=p.x, y=p.y, yaw=p.yaw,
                         cam=(cx, cy, cz), rob=(rx, ry, obs.robot.yaw)))
    for g in sorted(line_groups):
        mem = truth.group(g)
        xy = np.array([[m.x, m.y] for m in mem])
        c = xy.mean(0)
        if len(mem) >= 2:
            w, V = np.linalg.eigh(np.cov((xy - c).T))
            axis = V[:, -1]
        else:
            axis = np.array([math.cos(mem[0].yaw), math.sin(mem[0].yaw)])
        ray = c - np.array([cx, cy])
        cosv = abs(axis @ ray) / (np.linalg.norm(axis) * np.linalg.norm(ray))
        side = math.degrees(math.acos(min(1.0, cosv)))
        lines.append((arm, sref.scene_id, g, len(mem), side))
    if other_sit:
        P(f"note {arm} {sref.scene_id}: other situations with groups {dict(other_sit)}")

P(f"realistic scenes loaded: {seen}; per arm:", dict(collections.Counter(r['arm'] for r in {(x['arm'], x['scene']): x for x in rows}.values())))
P("people per category:", dict(collections.Counter(r['cat'] for r in rows)))
for cat in ["line", "group", "staff:staff", "other"]:
    sel = [r for r in rows if r['cat'] == cat]
    P(f"\n== {cat} ==")
    summ("angle, robot viewpoint", [r['a_rob'] for r in sel])
    summ("angle, camera viewpoint", [r['a_cam'] for r in sel])
    summ("angle vs optical axis", [r['a_axis'] for r in sel])
    summ("dist xy, robot", [r['d_rob'] for r in sel])
    summ("dist xy, camera", [r['d_cam'] for r in sel])
    summ("depth Z (ground point)", [r['Z'] for r in sel])
    for key in ("a_rob", "a_cam"):
        xs = [r[key] for r in sel]
        if not xs: continue
        P(f"  {key}: in [75,107]: {sum(75 <= x <= 107 for x in xs)}/{len(xs)}; in rounded band [74.5,107.5): "
          f"{sum(74.5 <= x < 107.5 for x in xs)}; <45: {sum(x < 45 for x in xs)}; >135: {sum(x > 135 for x in xs)}; "
          f"in [45,135]: {sum(45 <= x <= 135 for x in xs)}; <90: {sum(x < 90 for x in xs)}; >90: {sum(x > 90 for x in xs)}")
        b15 = collections.Counter(int(x // 15) * 15 for x in xs)
        P(f"  {key} 15-deg bins:", {f"[{k},{k+15})": b15[k] for k in sorted(b15)})
        b5 = collections.Counter(int(x // 5) * 5 for x in xs)
        P(f"  {key} 5-deg bins:", {f"[{k},{k+5})": b5[k] for k in sorted(b5)})
    P("  max |angle-90| (cam):", f"{max(abs(r['a_cam']-90) for r in sel):.2f}" if sel else "-",
      "median:", f"{statistics.median(abs(r['a_cam']-90) for r in sel):.2f}" if sel else "-")
    P("  arms:", dict(collections.Counter(r['arm'] for r in sel)))

P("\n== line members, per row ==")
for r in sorted([r for r in rows if r['cat'] == 'line'], key=lambda r: (r['arm'], r['scene'], r['pid'])):
    inside = 0 <= r['u'] < r['W'] and 0 <= r['v'] < r['Hh']
    P(f"{r['arm']:8s} {r['scene']:10s} {r['group']:8s} {r['pid']:10s} rob={r['a_rob']:7.2f} cam={r['a_cam']:7.2f} "
      f"axis={r['a_axis']:6.2f} d_rob={r['d_rob']:.3f} d_cam={r['d_cam']:.3f} Z={r['Z']:.2f} u={r['u']:.0f} v={r['v']:.0f} in={inside}")

P("\n== lines: side-on angle (PCA axis vs camera ray to centroid; 90 = side-on) ==")
for l in lines:
    P(f"{l[0]:8s} {l[1]:10s} {l[2]:8s} n={l[3]} side={l[4]:.2f}")
summ("side-on angle", [l[4] for l in lines])

# identity across arms: per scene id, compare person poses and camera
P("\n== identity across arms ==")
by_scene = collections.defaultdict(dict)
for r in rows:
    by_scene[r['scene']].setdefault(r['arm'], []).append(r)
diff = 0; cmpd = 0
for sid, d in sorted(by_scene.items()):
    arms = sorted(d)
    base = {r['pid']: r for r in d[arms[0]]}
    for a in arms[1:]:
        for r in d[a]:
            b = base.get(r['pid'])
            if b is None: continue
            cmpd += 1
            if abs(b['x'] - r['x']) > 1e-6 or abs(b['y'] - r['y']) > 1e-6 or abs(b['yaw'] - r['yaw']) > 1e-6 \
               or max(abs(i - j) for i, j in zip(b['cam'], r['cam'])) > 1e-6:
                diff += 1
P(f"scene ids: {len(by_scene)}; arms per scene id: {dict(collections.Counter(len(d) for d in by_scene.values()))}; "
  f"compared person pairs {cmpd}, differing {diff}")
for cat in ["line", "group", "staff:staff", "other"]:
    sel = [r for r in rows if r['cat'] == cat]
    P(f"{cat}: instances {len(sel)}, distinct (scene,pid) {len({(r['scene'], r['pid']) for r in sel})}, "
      f"distinct line groups {len({(r['scene'], r['group']) for r in sel}) if cat=='line' else '-'}")

open('results/still-scenes/speed/profile/verify/v_profile.txt', 'w').write("\n".join(out) + "\n")
