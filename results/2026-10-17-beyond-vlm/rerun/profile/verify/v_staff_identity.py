"""Staff in-image check and identity of layouts across arms (scene ids carry an arm prefix)."""
import sys, math, collections
import numpy as np
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
REAL = ("real", "staffed", "plain")
out = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)
def wrap(a): return math.pi - ((math.pi - a) % (2 * math.pi))
layouts = collections.defaultdict(dict)
for sref in H.distinct_scenes():
    rr = [r for r in sref.runs if r.arm in REAL]
    if not rr: continue
    arm = rr[0].arm
    sc = H.scene(rr[0], sref.scene_id)
    T = np.asarray(sc.observation.camera.T_world_camera); Ti = np.linalg.inv(T)
    cam = sc.observation.camera
    key = sref.scene_id[len(arm) + 1:] if sref.scene_id.startswith(arm + "_") else sref.scene_id
    layouts[key][arm] = (T.copy(), (sc.observation.robot.x, sc.observation.robot.y, sc.observation.robot.yaw),
                        {p.id: (p.x, p.y, p.yaw, p.group, p.role) for p in sc.truth.people},
                        {e: (ep.situation, ep.group_at_target) for e, ep in sc.truth.episodes.items()})
    if arm == "plain": P(f"{sref.scene_id}: people {len(sc.truth.people)}")
    for p in sc.truth.people:
        if not p.role: continue
        for h in (0.0, 1.0):
            pc = Ti @ np.array([p.x, p.y, h, 1.0])
            Z = pc[2]
            u = cam.fx * pc[0] / Z + cam.cx if Z > 0 else float('nan')
            v = cam.fy * pc[1] / Z + cam.cy if Z > 0 else float('nan')
            if h == 1.0:
                a = abs(math.degrees(wrap(math.atan2(T[1,3]-p.y, T[0,3]-p.x) - p.yaw)))
                P(f"staff {sref.scene_id:34s} {p.id:6s} xy=({p.x:.2f},{p.y:.2f}) yaw={math.degrees(p.yaw):7.1f} "
                  f"cam=({T[0,3]:.2f},{T[1,3]:.2f},{T[2,3]:.2f}) Z@1m={Z:.2f} u={u:.0f} v={v:.0f} "
                  f"inimg={(0<=u<cam.width and 0<=v<cam.height) if Z>0 else False} ang={a:.1f}")
P("\nT_world_camera of one scene:\n", np.round(next(iter(next(iter(layouts.values())).values()))[0], 3))
P("\n== identity across arms by layout ==")
P("layouts:", len(layouts), "arms per layout:", dict(collections.Counter(tuple(sorted(d)) for d in layouts.values())))
npair = ndiff = cam_diff = 0
for k, d in sorted(layouts.items()):
    arms = sorted(d); b = d[arms[0]]
    for a in arms[1:]:
        o = d[a]
        if np.abs(o[0] - b[0]).max() > 1e-9 or max(abs(i - j) for i, j in zip(o[1], b[1])) > 1e-9: cam_diff += 1
        for pid, v in o[2].items():
            if v[4]: continue          # staff only in real/staffed; compare non-staff people
            w = b[2].get(pid)
            npair += 1
            if w is None or max(abs(v[i] - w[i]) for i in range(3)) > 1e-9 or v[3] != w[3]: ndiff += 1
        # staff across real vs staffed
        sv = {p: v for p, v in o[2].items() if v[4]}; sw = {p: v for p, v in b[2].items() if v[4]}
        if sv and sw and (set(sv) != set(sw) or any(max(abs(sv[p][i]-sw[p][i]) for i in range(3)) > 1e-9 for p in sv)):
            P(f"  staff differ in {k} {arms[0]} vs {a}")
        if o[3] != b[3]: P(f"  episodes differ in {k} {arms[0]} vs {a}")
    P(f"{k:28s} arms={arms} people={[len(d[a][2]) for a in arms]} staff={[sum(1 for v in d[a][2].values() if v[4]) for a in arms]}")
P(f"non-staff person pairs compared {npair}, differing {ndiff}; camera/robot differ in {cam_diff} arm pairs")
open('results/2026-10-17-beyond-vlm/rerun/profile/verify/v_staff_identity.txt', 'w').write("\n".join(out) + "\n")
