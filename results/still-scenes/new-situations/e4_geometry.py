"""E4 pre-call validity check (I13_FROZEN.md section 3): zero-call geometry baselines on each scene, scored by sgi's
own scorer (sgi.evaluate.score on the prepared episode truth), with a perfect click on the target. Not methods: they
say whether people change the goal (G0), whether the bend is tested (G2 vs G1), and whether the success area can be
stood in at all (G1, G3). Zero model calls. the run runs it on e4/scenes (held-out scenes are scored by the
controller); it was tested on development waiting lines and on a synthetic machine case (`--selftest`).

    PYTHONPATH= .venv/bin/python e4_geometry.py SCENES_DIR [SCENES_DIR ...]
    PYTHONPATH= .venv/bin/python e4_geometry.py --selftest

Baselines (positions from scene truth; the waiting-line order is distance from the target, as region_waiting_line):
- G0 standard goal: C's people-blind stop (sgi.stages.goal.standard_goal) at the perfect click.
- G1 local extension (waiting line): 0.85 m behind the last member, along the last two members' direction, facing
  up the line. 0.85 m is the middle of the rule's 0.5-1.2 m band: geometry that knows the rule.
- G2 straight extension (waiting line): on the axis of the first leg (the two members nearest the target), 0.85 m
  past the farthest member's projection on it, facing up that axis: what a straight-line assumption gives.
- G3 behind the user (machine use): 1.5 m straight behind the user (opposite their yaw), facing the machine; 1.5 m is
  the middle of the rule's 1.0-2.0 m band.
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

G_LINE_M, G_MACHINE_M = 0.85, 1.5


def to_robot(robot, x, y, yaw):
    dx, dy = x - robot.x, y - robot.y
    c, s = math.cos(robot.yaw), math.sin(robot.yaw)
    return (c * dx + s * dy, -s * dx + c * dy, yaw - robot.yaw)


def baselines(obs, truth, eid, ep, th):
    from sgi.geometry import lift, point_to_pixel
    from sgi.gt.derive import oracle_point
    from sgi.stages.goal import standard_goal
    from sgi.types import GroundedTarget
    tid = ep.target_ids[0]
    target = truth.targets[tid]
    pt = oracle_point(obs, target.anchor)
    if pt is None:
        return None, {}
    cam = obs.camera
    u, v = point_to_pixel(pt.x, pt.y, cam.width, cam.height)
    world = lift(obs.depth, cam, u, v, window=th.lift_window)
    if world is None:
        return None, {}
    grounded = GroundedTarget(point=pt, pixel=(u, v), world=(float(world[0]), float(world[1]), float(world[2])))
    out = {"G0 standard goal": standard_goal(obs, grounded, th).model_copy(update={"target": grounded})}
    gid = truth.group_for(eid, tid)
    members = truth.group(gid) if gid else []
    a = target.anchor[:2]
    poses = {}
    if ep.situation == "waiting_line" and len(members) >= 2:
        line = sorted(((m.x, m.y) for m in members), key=lambda p: math.dist(a, p))
        last, before = line[-1], line[-2]
        d = ((last[0] - before[0]) / math.dist(last, before), (last[1] - before[1]) / math.dist(last, before))
        poses["G1 local extension"] = (last[0] + G_LINE_M * d[0], last[1] + G_LINE_M * d[1], math.atan2(-d[1], -d[0]))
        first, second = line[0], line[1]
        d1 = ((second[0] - first[0]) / math.dist(first, second), (second[1] - first[1]) / math.dist(first, second))
        far = max((p[0] - first[0]) * d1[0] + (p[1] - first[1]) * d1[1] for p in line) + G_LINE_M
        poses["G2 straight extension"] = (first[0] + far * d1[0], first[1] + far * d1[1], math.atan2(-d1[1], -d1[0]))
    if ep.situation == "machine_use" and len(members) == 1:
        u_ = members[0]
        gx, gy = u_.x - G_MACHINE_M * math.cos(u_.yaw), u_.y - G_MACHINE_M * math.sin(u_.yaw)
        poses["G3 behind the user"] = (gx, gy, math.atan2(a[1] - gy, a[0] - gx))
    from sgi.types import GoalResult
    for name, (x, y, yaw) in poses.items():
        out[name] = GoalResult(status="ok", source="baseline", pose_world=(x, y, yaw),
                               pose_robot=to_robot(obs.robot, x, y, yaw), target=grounded)
    return grounded, out


def main(dirs):
    from sgi.config import load_config
    from sgi.evaluate import score
    from sgi.gt.derive import prepare
    from sgi.scene import load_scene
    cfg = load_config()
    th = cfg.thresholds
    tally = defaultdict(lambda: [0, 0])
    for folder in sorted({f.parent for d in dirs for f in Path(d).rglob("scene.json")}):
        inputs, truth = load_scene(folder)
        ets = prepare(inputs, truth, cfg)
        for eid, et in ets.items():
            grounded, out = baselines(inputs.observation, truth, eid, et.gt, th)
            if grounded is None:
                print(f"{inputs.scene_id} {eid}: target not in the image or lift failed: no baseline")
                continue
            parts = []
            for name, res in out.items():
                sc = score(res, inputs.observation, truth, et.gt, th)
                key = (et.gt.situation, name)
                tally[key][0] += sc.success
                tally[key][1] += 1
                parts.append(f"{name.split()[0]} {'ok' if sc.success else 'x'}(r{int(sc.in_region)}h{int(sc.heading_ok)}"
                             f"t{int(sc.target_correct)}u{int(sc.reachable)})")
            print(f"{inputs.scene_id} {eid} [{et.gt.situation}]: " + "; ".join(parts))
    print("\n| situation | baseline | success |\n|---|---|---|")
    for (sit, name), (ok, n) in sorted(tally.items()):
        print(f"| {sit} | {name} | {ok}/{n} |")
    print("\nValidity rules (I13_FROZEN.md section 3, fixed before any model call):")
    for sit in sorted({s for s, _ in tally}):
        g0 = tally.get((sit, "G0 standard goal"), [0, 0])
        print(f"- {sit}: G0 passes {g0[0]}/{g0[1]}: {'PEOPLE DO NOT CHANGE THE GOAL here (>= 50%)' if g0[1] and g0[0] >= g0[1] / 2 else 'people matter'}")
        g2 = tally.get((sit, "G2 straight extension"))
        if g2:
            print(f"  G2 passes {g2[0]}/{g2[1]}: {'THE BEND IS NOT TESTED (>= 50%)' if g2[0] >= g2[1] / 2 else 'the bend is tested'}")
        for name in ("G1 local extension", "G3 behind the user"):
            g = tally.get((sit, name))
            if g:
                print(f"  {name} passes {g[0]}/{g[1]}: {'AREA BLOCKED OR UNREACHABLE in some scenes (> 10% fail): fix before calls' if g[0] < 0.9 * g[1] else 'the success area can be stood in'}")


def selftest():
    """Synthetic machine case: G3 must lie inside region_machine_use's area and outside its forbidden zones."""
    from sgi.gt.regions import region_machine_use
    from sgi.polygons import point_in_polygon
    from sgi.scene import Person, Target
    for yaw in (0.0, 1.0, 2.5, -2.0):
        ax, ay = 3.0, 1.0
        t = Target(id="t0", kind="object", label="ticket machine", anchor=(ax, ay, 1.0), front_yaw=yaw + math.pi,
                   footprint=[(ax - 0.3, ay - 0.3), (ax + 0.3, ay - 0.3), (ax + 0.3, ay + 0.3), (ax - 0.3, ay + 0.3)])
        ux, uy = ax + 0.8 * math.cos(yaw + math.pi), ay + 0.8 * math.sin(yaw + math.pi)
        user = Person(id="p0", x=ux, y=uy, yaw=yaw)
        regions, forbidden = region_machine_use(t, [user])
        gx, gy = ux - G_MACHINE_M * math.cos(yaw), uy - G_MACHINE_M * math.sin(yaw)
        inside = point_in_polygon((gx, gy), regions[0].polygon)
        clear = not any(point_in_polygon((gx, gy), f.polygon) for f in forbidden)
        print(f"yaw {yaw:+.1f}: G3 in the area {inside}, outside the forbidden zones {clear}")
        assert inside and clear


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main(sys.argv[1:])
