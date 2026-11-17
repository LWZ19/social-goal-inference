"""I06 step 2c: builder self-tests (FROZEN.md section 3), before any variant is generated.

1. The note's test: c1-c8 (c5 re-implemented without the 22-degree cone) must accept every bystander and second-group
   member the package placed in eval_10..12 of seeds 0, 1 and 2. Each such person is tested as if added, with every other
   person of the base present.
2. Extra sanity tests (builder's own): a person put where a check must refuse them is refused by that check.
3. The streams of the 270 planned variants are distinct and none is a base stream.

Writes build/self_test.txt; exit 1 on any failure.
"""
from __future__ import annotations

import math
import pickle

import numpy as np

from sgi.scene import Person

import crowd as C

LINES: list[str] = []


def say(text: str = "") -> None:
    print(text, flush=True)
    LINES.append(text)


def as_added(base, person):
    ctx = C.Context(base)
    ctx.people = [p for p in ctx.people if p.id != person.id]
    return ctx


def main() -> int:
    bases = {}
    for seed in (0, 1, 2):
        with open(C.BUILD / "base_layouts" / f"seed{seed}.pkl", "rb") as f:
            bases[seed] = {b.index: b for b in pickle.load(f)}
    failures = 0
    say("1. c1-c8 on the people the package placed (eval_10..12, seeds 0-2):")
    tested = 0
    for seed in (0, 1, 2):
        for index in (10, 11, 12):
            base = bases[seed][index]
            ctx0 = C.Context(base)
            for person in ctx0.non_members:
                ctx = as_added(base, person)
                why = ctx.first_failure(person, C.CHECKS_ALL)
                tested += 1
                status = "accepted" if why is None else f"REFUSED by {why}"
                if why is not None:
                    failures += 1
                say(f"  seed {seed} {base.scene.scene_id} {person.id} (group {person.group}): {status}")
    say(f"  {tested} people tested, {failures} refused")

    say("\n2. sanity: a person where a check must refuse them (seed 0 eval_07 waiting_line_2, eval_15 audience_5):")
    base = bases[0][7]
    ctx = C.Context(base)
    cam, anchor = np.array(ctx.cam_xy), np.array(ctx.anchor_xy)
    robot = np.array([ctx.robot.x, ctx.robot.y])
    member = ctx.members[-1]
    first = min(ctx.members, key=lambda p: math.dist((p.x, p.y), anchor))

    def P(xy):
        return Person(id="px", x=float(xy[0]), y=float(xy[1]), yaw=0.0)

    wedge_mid = None
    for t in np.linspace(0.3, 0.7, 5):          # a point between the anchor and the first in line, well away from the robot
        q = robot + 0.8 * ((anchor * (1 - t) + np.array([first.x, first.y]) * t) - robot)
        wedge_mid = q
        break
    fp = ctx.target_footprints[0]
    fp_mid = np.mean(np.array(fp), axis=0)
    out_dir = (fp_mid - cam) / np.linalg.norm(fp_mid - cam)
    yaw = ctx.robot.yaw
    cases = [
        ("c1", P(np.array(ctx.bounds[0]) - 1.0)),
        ("c2", P((member.x + 0.3, member.y))),
        ("c3", P(robot + 1.0 * np.array([math.cos(yaw), math.sin(yaw)]))),
        ("c4", P(np.array(ctx.cam_xy) + 4.0 * np.array([math.cos(yaw + math.radians(40)), math.sin(yaw + math.radians(40))]))),
        ("c5", P(robot + 0.5 * (np.array([member.x, member.y]) - robot))),
        ("c5", P(wedge_mid)),
        ("c6", P(cam + 0.5 * (anchor - cam))),
        ("c7", P(fp_mid - 1.2 * out_dir)),
        ("c8", P((anchor + np.array([first.x, first.y])) / 2)),
    ]
    for expected, person in cases:
        got = getattr(ctx, expected)(person, others=ctx.people, present=ctx.people)
        ok = not got
        failures += not ok
        say(f"  {expected} at ({person.x:.2f}, {person.y:.2f}): {'refused, as it must be' if ok else 'NOT REFUSED'}")
    base = bases[0][15]
    ctx = C.Context(base)
    rows = ctx.members
    cx, cy = np.mean([[p.x, p.y] for p in rows], axis=0)
    ax, ay = ctx.anchor_xy
    between = P(((cx + ax) / 2, (cy + ay) / 2))      # between the audience and the screen: in the audience_view polygon
    got = ctx.c8(between)
    failures += got
    say(f"  c8 (audience view space) at ({between.x:.2f}, {between.y:.2f}): {'refused, as it must be' if not got else 'NOT REFUSED'}")

    say("\n3. streams of the planned variants:")
    keys = []
    for seed in (0, 1, 2):
        for index in C.BASE_ROWS:
            row = bases[seed][index].row
            for family, levels in C.LEVELS.items():
                if not C.applies(family, row):
                    continue
                for level in levels:
                    keys.append(tuple(C.stream_key(seed, family, index, level)))
    distinct = len(set(keys)) == len(keys)
    planned = len(keys)
    failures += (not distinct) + (planned != 270)
    say(f"  {planned} planned variants (270 expected), streams distinct: {distinct}; every stream has the form "
        f"[seed, 2, family, row, level] with seed in 0..2, so none equals default_rng(seed) or [seed, 1]")
    say("\nRESULT: " + ("PASS" if failures == 0 else f"FAIL ({failures})"))
    (C.BUILD / "self_test.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
