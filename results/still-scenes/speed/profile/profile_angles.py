"""I09 re-check: how far from facing the camera do people stand in the realistic eval scenes?

Reads scenes only through research_harness (H.distinct_scenes(), H.scene()). No model call, nothing tuned.

Definitions
- Realistic eval scenes: the distinct scenes whose runs have arm real, staffed or plain (3 sets x 14 = 42 images).
- Role of a person (ground truth, scene.json):
    line member  : person.group is the group_at_target of an episode whose situation is "waiting_line"
    group member : person.group is the group_at_target of an episode whose situation is "conversation_group"
                   (or "audience"/"group_photo", none of which occur in the realistic sets)
    staff        : person.role is set (role "staff")
    other        : everyone else (bystanders, group not used by any episode)
- Camera: the observation camera, position = T_world_camera[:3, 3]; forward = T_world_camera[:3, 2] (OpenCV, z
  forward, as sgi.geometry.lift uses it).
- Angle to camera (primary): |wrap(atan2(cam_y - p_y, cam_x - p_x) - person.yaw)| in degrees, in [0, 180];
  0 = facing the camera, 90 = profile, 180 = back to the camera. person.yaw is the world facing (atan2 convention,
  as sgi.gt.regions uses it).
- Angle to optical axis (alternative): the same with the ground-plane direction -forward instead of the ray to
  the camera (ignores the person's bearing off the image centre).
- Distance: horizontal (xy) distance camera -> person (primary); also the optical depth Z (camera frame) of the
  person's point at 1.0 m height, and the horizontal distance from the robot pose.
- Line axis (side-on check): for lines of >= 2 members the principal axis (PCA) of member xy; the angle between
  that axis and the horizontal ray from the camera to the line centroid, folded to [0, 90] (90 = side-on). For a
  1-member line the member's facing is the axis.
"""
from __future__ import annotations

import collections
import math
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H  # noqa: E402

OUT = Path(__file__).resolve().parent
REALISTIC = ("real", "staffed", "plain")
LINE_SITS = ("waiting_line",)
GROUP_SITS = ("conversation_group", "audience", "group_photo")
TORSO_Z = 1.0

lines_out: list[str] = []


def say(*parts):
    text = " ".join(str(p) for p in parts)
    print(text)
    lines_out.append(text)


def wrap(a: float) -> float:
    return math.pi - ((math.pi - a) % (2 * math.pi))


def groups_by_situation(truth):
    """{group id: set of situations whose episode names it as the group at a target}."""
    out = collections.defaultdict(set)
    for ep in truth.episodes.values():
        g = ep.group_at_target
        ids = list(g.values()) if isinstance(g, dict) else [g]
        for gid in ids:
            if gid is not None:
                out[gid].add(ep.situation)
    return out


def role_of(person, gsits) -> str:
    if person.role:
        return "staff"
    if person.group is not None:
        sits = gsits.get(person.group, set())
        if sits & set(LINE_SITS):
            return "line"
        if sits & set(GROUP_SITS):
            return "group"
    return "other"


def project(cam, p_world):
    """(u, v, Z) of a world point in the observation camera."""
    T_inv = np.linalg.inv(cam.T_world_camera)
    pc = T_inv @ np.array([*p_world, 1.0])
    Z = pc[2]
    u = cam.fx * pc[0] / Z + cam.cx
    v = cam.fy * pc[1] / Z + cam.cy
    return u, v, Z


def summary(vals):
    if not vals:
        return "n=0"
    s = sorted(vals)
    q = np.percentile(s, [25, 75])
    return (f"n={len(s)} min={s[0]:.1f} q25={q[0]:.1f} median={statistics.median(s):.1f} q75={q[1]:.1f} "
            f"max={s[-1]:.1f} mean={statistics.fmean(s):.1f}")


def hist(vals, edges):
    counts = [0] * (len(edges) - 1)
    for v in vals:
        for i in range(len(edges) - 1):
            if edges[i] <= v < edges[i + 1] or (i == len(edges) - 2 and v == edges[-1]):
                counts[i] += 1
                break
    return "  ".join(f"[{edges[i]:.0f},{edges[i + 1]:.0f}):{counts[i]}" for i in range(len(counts)))


def main():
    rows = []           # one per (scene, person)
    n_scenes = 0
    cam_vs_robot = []
    line_rows = []      # one per (scene, line)
    for sref in H.distinct_scenes():
        arms = {r.arm for r in sref.runs}
        if not arms & set(REALISTIC):
            continue
        assert len(arms) == 1, (sref.scene_id, arms)
        arm = next(iter(arms))
        run = sref.runs[0]
        sc = H.scene(run, sref.scene_id)
        cam = sc.observation.camera
        robot = sc.observation.robot
        T = cam.T_world_camera
        cx_w, cy_w, cz_w = T[0, 3], T[1, 3], T[2, 3]
        fwd = T[:3, 2]
        fwd_yaw = math.atan2(fwd[1], fwd[0])
        gsits = groups_by_situation(sc.truth)
        n_scenes += 1
        cam_vs_robot.append((math.hypot(cx_w - robot.x, cy_w - robot.y),
                             abs(math.degrees(wrap(fwd_yaw - robot.yaw)))))
        for p in sc.truth.people:
            role = role_of(p, gsits)
            bearing = math.atan2(cy_w - p.y, cx_w - p.x)
            ang = abs(math.degrees(wrap(bearing - p.yaw)))
            ang_axis = abs(math.degrees(wrap((fwd_yaw + math.pi) - p.yaw)))
            ang_robot = abs(math.degrees(wrap(math.atan2(robot.y - p.y, robot.x - p.x) - p.yaw)))
            dist = math.hypot(p.x - cx_w, p.y - cy_w)
            dist_robot = math.hypot(p.x - robot.x, p.y - robot.y)
            u, v, Z = project(cam, (p.x, p.y, TORSO_Z))
            in_view = 0 <= u < cam.width and Z > 0
            rows.append(dict(arm=arm, scene=sref.scene_id, pid=p.id, role=role, group=p.group, rolename=p.role,
                             x=p.x, y=p.y, yaw=math.degrees(p.yaw), ang=ang, ang_axis=ang_axis, ang_robot=ang_robot, dist=dist,
                             dist_robot=dist_robot, Z=Z, u=u, in_view=in_view, cam_h=cz_w,
                             cam=(cx_w, cy_w)))
        # lines: per waiting_line group
        for gid, sits in gsits.items():
            if not sits & set(LINE_SITS):
                continue
            members = sc.truth.group(gid)
            xy = np.array([[m.x, m.y] for m in members])
            cen = xy.mean(axis=0)
            if len(members) >= 2:
                c = xy - cen
                w, vecs = np.linalg.eigh(c.T @ c)
                axis = vecs[:, np.argmax(w)]
                axis_src = "pca"
            else:
                axis = np.array([math.cos(members[0].yaw), math.sin(members[0].yaw)])
                axis_src = "facing"
            ray = cen - np.array([cx_w, cy_w])
            cosang = abs(float(axis @ ray) / (np.linalg.norm(axis) * np.linalg.norm(ray)))
            side_on = math.degrees(math.acos(min(1.0, cosang)))
            dists = [math.hypot(m.x - cx_w, m.y - cy_w) for m in members]
            line_rows.append(dict(arm=arm, scene=sref.scene_id, group=gid, n=len(members), side_on=side_on,
                                  axis_src=axis_src, dmin=min(dists), dmax=max(dists),
                                  dcen=float(np.linalg.norm(ray))))

    say("I09 profile re-check: realistic eval scenes (real, staffed, plain) via research_harness")
    say(f"scenes: {n_scenes} (with people: {len({(r['arm'], r['scene']) for r in rows})}) ; people instances: {len(rows)}")
    say("camera offset from robot pose (m), camera yaw minus robot yaw (deg):",
        sorted({(round(d, 3), round(a, 2)) for d, a in cam_vs_robot}))
    say("role counts (instances over 42 scenes):",
        dict(collections.Counter(r['role'] for r in rows)))
    say("role counts per arm:",
        {a: dict(collections.Counter(r['role'] for r in rows if r['arm'] == a)) for a in REALISTIC})
    say("camera height(s) m:", sorted({round(r['cam_h'], 3) for r in rows}))
    say("")

    # V1: line members
    say("=== V1. Line members: angle between facing and direction to the camera (deg; 90 = profile) ===")
    line = [r for r in rows if r['role'] == 'line']
    say("all instances :", summary([r['ang'] for r in line]))
    for a in REALISTIC:
        say(f"  {a:8s}    :", summary([r['ang'] for r in line if r['arm'] == a]))
    uniq = {}
    for r in line:
        key = (r['scene'].split('_', 1)[1], r['pid'], round(r['x'], 3), round(r['y'], 3), round(r['yaw'], 2),
               tuple(round(c, 3) for c in r['cam']))
        uniq.setdefault(key, r)
    say("unique (layout, person, pose, camera) across sets:", summary([r['ang'] for r in uniq.values()]))
    say("within [75,107] deg:", sum(75 <= r['ang'] <= 107 for r in line), "of", len(line),
        "; within [74.5,107.5):", sum(74.5 <= r['ang'] < 107.5 for r in line))
    say("histogram (15-deg bins):", hist([r['ang'] for r in line], list(range(0, 181, 15))))
    say("histogram (5-deg bins, 60-120):", hist([r['ang'] for r in line], list(range(60, 121, 5))))
    say("|angle - 90| :", summary([abs(r['ang'] - 90) for r in line]))
    say("facing toward camera side (<90):", sum(r['ang'] < 90 for r in line),
        "; away (>90):", sum(r['ang'] > 90 for r in line))
    say("alternative (vs optical axis instead of ray):", summary([r['ang_axis'] for r in line]))
    say("alternative (ray to robot pose xy instead of camera xy):", summary([r['ang_robot'] for r in line]))
    _rv = sorted(r['ang_robot'] for r in line)
    _cv = sorted(r['ang'] for r in line)
    say(f"   precise extremes: camera ray min={_cv[0]:.3f} max={_cv[-1]:.3f}; robot-pose ray min={_rv[0]:.3f} "
        f"max={_rv[-1]:.3f}; robot-pose dist min={min(r['dist_robot'] for r in line):.3f} "
        f"max={max(r['dist_robot'] for r in line):.3f}; camera dist min={min(r['dist'] for r in line):.3f} "
        f"max={max(r['dist'] for r in line):.3f}")
    say("   robot-pose variant within [75,107]:", sum(75 <= r['ang_robot'] <= 107 for r in line), "of", len(line))
    say("all line members in view (u inside image, Z>0):", all(r['in_view'] for r in line))
    say("")
    say("per line member (arm scene pid group ang ang_axis ang_robot dist_xy Z@1m dist_robot):")
    for r in sorted(line, key=lambda r: (r['scene'].split('_', 1)[1], r['arm'], r['pid'])):
        say(f"  {r['arm']:8s} {r['scene']:38s} {r['pid']:10s} {str(r['group']):10s} {r['ang']:6.1f} "
            f"{r['ang_axis']:6.1f} {r['ang_robot']:6.1f} {r['dist']:5.2f} {r['Z']:5.2f} {r['dist_robot']:5.2f}")
    say("")

    say("=== The note: all five lines seen side-on at 4.4-6.2 m ===")
    say("distinct line layouts:", sorted({r['scene'].split('_', 1)[1] for r in line_rows}),
        "; line instances:", len(line_rows))
    say("line member horizontal distance camera->person (m):", summary([r['dist'] for r in line]))
    say("line member optical depth Z at 1.0 m height (m):", summary([r['Z'] for r in line]))
    say("line member horizontal distance robot pose->person (m):", summary([r['dist_robot'] for r in line]))
    say("per line (arm scene group n axis side_on_deg dmin dmax dcentroid):")
    for r in sorted(line_rows, key=lambda r: (r['scene'].split('_', 1)[1], r['arm'])):
        say(f"  {r['arm']:8s} {r['scene']:38s} {r['group']:10s} n={r['n']} {r['axis_src']:6s} "
            f"{r['side_on']:5.1f} {r['dmin']:5.2f} {r['dmax']:5.2f} {r['dcen']:5.2f}")
    say("side-on angle of line axis to camera ray (deg, 90 = side-on):", summary([r['side_on'] for r in line_rows]))
    say("")

    # NEW: group members and staff
    say("=== NEW. Same angle for group members, staff, others (context) ===")
    for role in ("group", "staff", "other"):
        sub = [r for r in rows if r['role'] == role]
        say(f"{role:6s}:", summary([r['ang'] for r in sub]))
        if sub:
            say("        hist 15-deg:", hist([r['ang'] for r in sub], list(range(0, 181, 15))))
            say("        within [75,107]:", sum(75 <= r['ang'] <= 107 for r in sub), "of", len(sub),
                "; within 45 of profile [45,135]:", sum(45 <= r['ang'] <= 135 for r in sub),
                "; frontal <45:", sum(r['ang'] < 45 for r in sub), "; back >135:", sum(r['ang'] > 135 for r in sub))
            say("        dist_xy:", summary([r['dist'] for r in sub]))
            say("        robot-pose variant angle:", summary([r['ang_robot'] for r in sub]))
            for a in REALISTIC:
                s2 = [r['ang'] for r in sub if r['arm'] == a]
                if s2:
                    say(f"        {a:8s}:", summary(s2))
    say("")
    say("per group member / staff / other (arm scene pid role group ang dist_xy):")
    for r in sorted([r for r in rows if r['role'] != 'line'],
                    key=lambda r: (r['role'], r['scene'].split('_', 1)[1], r['arm'], r['pid'])):
        say(f"  {r['arm']:8s} {r['scene']:38s} {r['pid']:10s} {r['role']:6s} {str(r['group'] or r['rolename']):10s} "
            f"{r['ang']:6.1f} {r['dist']:5.2f}")

    (OUT / "profile_angles.txt").write_text("\n".join(lines_out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
