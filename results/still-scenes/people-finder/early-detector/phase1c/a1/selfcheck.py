"""I09 A1 step 2 (FROZEN.md sections 4 and 12): synthetic checks of the facing sign convention and the pipeline's
geometry, and the scorer check on fit data (true people as detections: recall 1, 0 false, facing 1). Project venv;
fit data only. Writes selfcheck.txt."""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, 'results/still-scenes/perception/depth-blob-detector/stageA')
sys.path.insert(0, 'results/still-scenes/perception/depth-blob-detector')
import pipeline as P  # noqa: E402
from sgi.geometry import Camera, RobotPose, camera_on_robot, project, wrap_angle  # noqa: E402

L = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    L.append(s)


def deg(a):
    return None if a is None else round(math.degrees(a), 3)


def check(name, cond):
    say(("PASS " if cond else "FAIL ") + name)
    assert cond, name


# ------------------------------------------------------------------ synthetic scene: camera at the origin along +y
cam = Camera(fx=554.3, fy=554.3, cx=320.0, cy=240.0, width=640, height=480,
             T_world_camera=camera_on_robot(RobotPose(0.0, 0.0, math.pi / 2), 0.0, 1.5, 0.0))
cam_xy = cam.T_world_camera[:2, 3]


def px(p):
    u, v, _ = project(cam, p)
    return u, v


def synthetic_depth(person_xy, half_w=0.25):
    """A box-shaped person (a slab facing the camera) on an empty background (no depth)."""
    depth = np.full((480, 640), np.nan, np.float32)
    x, y = person_xy
    for z0, z1 in ((0.0, 1.75),):
        u0, v0 = px((x - half_w, y, z1))
        u1, v1 = px((x + half_w, y, z0))
        depth[int(v0):int(v1), int(u0):int(u1)] = y
    return depth


def kp_list(points):
    """17 keypoints, all unconfident except those given {index: (world xyz, score)}."""
    kp = [[0.0, 0.0, 0.0] for _ in range(17)]
    for j, (p, s) in points.items():
        u, v = px(p)
        kp[j] = [u, v, s]
    return kp


say("== synthetic: lateral vector and sign ==")
check("L at +y, R at -y gives yaw 0", abs(P.yaw_from_lateral(np.array([0.0, 0.0]) + np.array([0.0, 2.0]))) < 1e-12)
check("L at +x, R at -x gives yaw -90 deg (faces -y)", abs(P.yaw_from_lateral(np.array([2.0, 0.0])) + math.pi / 2) < 1e-12)
check("lateral shorter than 0.05 m gives None", P.yaw_from_lateral(np.array([0.03, 0.0])) is None)

person = (0.0, 3.0)
depth = synthetic_depth(person)
# camera-facing person (true yaw -90 deg, faces -y towards the camera): own left at +x = image right
kp = kp_list({P.LS: ((0.2, 3.0, 1.45), 0.9), P.RS: ((-0.2, 3.0, 1.45), 0.9),
              P.LH: ((0.15, 3.0, 0.95), 0.9), P.RH: ((-0.15, 3.0, 0.95), 0.9),
              P.NOSE: ((0.0, 3.0, 1.62), 0.9), P.LEYE: ((0.03, 3.0, 1.65), 0.9), P.REYE: ((-0.03, 3.0, 1.65), 0.9)})
check("camera-facing person: L_Shoulder image-right of R_Shoulder (probe convention)", kp[P.LS][0] > kp[P.RS][0])
y, d = P.facing(kp, depth, cam, 3.0, person, cam_xy)
say("  yaws", {k: deg(v) for k, v in y.items()}, d)
check("camera-facing person: F1-F4 all -90 deg", all(abs(wrap_angle(y[v] + math.pi / 2)) < 1e-6 for v in P.VARIANTS))
# same keypoints but L/R swapped (a back view read as a front view) with no face: F1-F3 say +90 deg, F3 does not flip
kp_back = [list(k) for k in kp]
kp_back[P.LS], kp_back[P.RS] = kp[P.RS], kp[P.LS]
kp_back[P.LH], kp_back[P.RH] = kp[P.RH], kp[P.LH]
for j in (P.NOSE, P.LEYE, P.REYE):
    kp_back[j] = [0.0, 0.0, 0.0]
y, d = P.facing(kp_back, depth, cam, 3.0, person, cam_xy)
check("back view (own left image-left), no face: F3 = +90 deg, no flip", abs(wrap_angle(y["F3"] - math.pi / 2)) < 1e-6
      and not d["f3_flip"])
# a face seen while the shoulders say 'away': F3 flips to the camera half-plane
kp_flip = [list(k) for k in kp_back]
for j in (P.NOSE, P.LEYE, P.REYE):
    kp_flip[j] = list(kp[j])
y, d = P.facing(kp_flip, depth, cam, 3.0, person, cam_xy)
check("face confident while shoulders say away: F3 flips to -90 deg", abs(wrap_angle(y["F3"] + math.pi / 2)) < 1e-6
      and d["f3_flip"])
check("F3 never creates a yaw", P.facing(kp_list({P.NOSE: ((0, 3, 1.6), 0.9), P.LEYE: ((0.03, 3, 1.65), 0.9),
                                                  P.REYE: ((-0.03, 3, 1.65), 0.9)}), depth, cam, 3.0, person,
                                         cam_xy)[0]["F3"] is None)
# profile, nose image-right: F4 gives yaw 0 (FROZEN.md section 4's synthetic case); nose image-left: 180 deg
for side, nose_x, want in (("right", 0.10, 0.0), ("left", -0.10, math.pi)):
    kpp = kp_list({P.LS: ((0.002, 3.0, 1.45), 0.9), P.RS: ((-0.002, 3.0, 1.45), 0.9),
                   P.NOSE: ((nose_x, 3.0, 1.62), 0.9), P.LEAR: ((0.0, 3.0, 1.63), 0.9)})
    y, d = P.facing(kpp, depth, cam, 3.0, person, cam_xy)
    say(f"  profile nose image-{side}:", {k: deg(v) for k, v in y.items()}, d)
    check(f"profile, nose image-{side}: F4 = {math.degrees(want):.0f} deg, edge-on", d["edge_on"]
          and abs(wrap_angle(y["F4"] - want)) < 1e-6)
# dead zone: nose within 1 px of the ear keeps F3
kpp = kp_list({P.LS: ((0.002, 3.0, 1.45), 0.9), P.RS: ((-0.002, 3.0, 1.45), 0.9),
               P.NOSE: ((0.001, 3.0, 1.62), 0.9), P.LEAR: ((0.0, 3.0, 1.63), 0.9)})
y, d = P.facing(kpp, depth, cam, 3.0, person, cam_xy)
check("profile, nose within 1 px of the ear: F4 keeps F3", y["F4"] == y["F3"] and d["f4_rule"] == "dead zone")
# off-axis profile person: F4 is +-90 deg from the camera-to-person ray
pers2 = (2.0, 4.0)
depth2 = synthetic_depth(pers2)
kpp = kp_list({P.LS: ((2.001, 4.0, 1.45), 0.9), P.RS: ((1.999, 4.0, 1.45), 0.9),
               P.NOSE: ((2.06, 4.0 - 0.06, 1.62), 0.9), P.LEAR: ((2.0, 4.0, 1.63), 0.9)})
y, d = P.facing(kpp, depth2, cam, 4.0, pers2, cam_xy)
ray = math.atan2(4.0, 2.0)
check("off-axis profile: F4 = ray - 90 deg for nose image-right", abs(wrap_angle(y["F4"] - (ray - math.pi / 2))) < 1e-6)

say("== synthetic: box features ==")
u0, v0 = px((-0.25, 3.0, 1.75))
u1, v1 = px((0.25, 3.0, 0.0))
f = P.box_features([u0, v0, u1, v1, 0.9], kp, depth, cam, cam_xy)
say("  ", {k: f[k] for k in ("region", "region_valid", "zmed", "kept", "body_xy", "z_top", "top_touch", "n_Q",
                             "flat_share", "flat")})
check("slab person: hull region, Z_med 3.0, position (0, 3), z_top 1.75", f["region"] == "hull" and
      abs(f["zmed"] - 3.0) < 1e-6 and math.dist(f["body_xy"], (0, 3)) < 0.02 and abs(f["z_top"] - 1.75) < 0.01)
check("slab person with no background is flat (a picture would be)", f["flat"])
check("push 0.2 m along the camera ray", math.dist(P.push((0.0, 3.0), cam_xy, 0.2), (0.0, 3.2)) < 1e-12)
# a person in front of a wall 1 m behind: not flat
depth3 = np.full((480, 640), 4.0, np.float32)
depth3[np.isfinite(depth)] = 3.0
f3 = P.box_features([u0, v0, u1, v1, 0.9], kp, depth3, cam, cam_xy)
say("   with a wall 1 m behind:", f3["flat_share"], f3["flat"])
check("slab person 1 m in front of a wall, box wider than the body: flat share below 0.80 only when the wall fills "
      "> 20% of Q; here the box equals the body, so it stays flat", f3["flat"])
fw = P.box_features([u0 - 40, v0, u1 + 40, v1, 0.9], kp, depth3, cam, cam_xy)
say("   wider box with the wall:", fw["flat_share"], fw["flat"])
check("wider box, wall behind: not flat", not fw["flat"])
check("convex hull of 4 collinear points is degenerate", len(P.convex_hull([(0, 0), (1, 1), (2, 2), (3, 3)])) < 3)
say("synthetic checks: all pass")

# ------------------------------------------------------------------ scorer on fit data (I03 self-check, repeated)
if "--no-scorer" not in sys.argv:
    import fitdata as FD
    import people_scorer as PS
    say("== scorer: true people as detections on every fit scene ==")
    t0 = time.monotonic()
    n = ok_n = 0
    vis_n = {"F-real": 0, "F-ph": 0}
    for domain, sset, scene_id, loader in FD.fit_refs():
        sc, th = FD.load(loader)
        vis = PS.truth_info(sc, th)["visible"]
        r = PS.score_people([(p.x, p.y, p.yaw) for p in vis], sc, th, exempt_xy=PS.exempt_points(sc.truth))
        ok = (r["recall"] in (1.0, None) and r["false"] == 0 and r["exempt"] == 0 and r["merges"] == 0
              and r["facing"]["within45_full"] == r["matched"] == len(vis))
        n += 1
        ok_n += ok
        vis_n[domain] += len(vis)
        if not ok:
            say("FAIL", domain, scene_id, r["recall"], r["false"], r["facing"])
    say(f"scorer check: {ok_n} of {n} fit scenes give recall 1 (or nobody visible), 0 false, 0 merges, facing 1; "
        f"visible people {vis_n}; {time.monotonic() - t0:.0f} s")
    assert ok_n == n

(HERE / "selfcheck.txt").write_text("\n".join(L) + "\n")
