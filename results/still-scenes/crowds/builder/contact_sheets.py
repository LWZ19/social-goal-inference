"""I06: contact sheets of kept variants, one PNG per family (build/contact_<family>.png), to look at the additions.

Each row: the base picture, the variant picture (added people marked by a red triangle over the head), and a top-down
map (room, obstacles, members blue, the base's other people grey, added people red, with facing ticks; e0's acceptable
region green with its representative point; the robot and the camera's view edges).
"""
from __future__ import annotations

import json
import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sgi.geometry import project  # noqa: E402
from sgi.scene import load_scene  # noqa: E402

import crowd as C  # noqa: E402


def entries():
    out = []
    for seed in (0, 1, 2):
        p = C.BUILD / f"variants_seed{seed}.jsonl"
        if p.exists():
            out += [json.loads(l) for l in p.read_text().splitlines() if l]
    return out


def pick(es, family):
    kept = [e for e in es if e["family"] == family and e["accepted"]]
    if family == "A":
        for seed, row in ((0, 7), (1, 8), (2, 13), (0, 13), (1, 7)):
            chosen = sorted([e for e in kept if e["seed"] == seed and e["row"] == row], key=lambda e: e["level"])
            if len(chosen) >= 4:
                return chosen[:6]
        return kept[:6]
    chosen, rows = [], set()
    for e in sorted(kept, key=lambda e: (e["row"], e["seed"], e["level"])):
        if (e["row"], e["level"]) in rows:
            continue
        rows.add((e["row"], e["level"]))
        chosen.append(e)
    step = max(1, len(chosen) // 6)
    return chosen[::step][:6]


def draw_map(ax, truth, inputs, added_ids, gid):
    xs, ys = zip(*truth.bounds)
    ax.fill(xs, ys, color="#f4f1ea", zorder=0)
    for poly in truth.obstacles:
        px, py = zip(*poly)
        ax.fill(px, py, color="#999999", zorder=1)
    e0 = truth.episodes["e0"]
    for r in e0.acceptable_regions or []:
        px, py = zip(*r.polygon)
        ax.fill(px, py, color="#7fc97f", alpha=0.5, zorder=2)
        ax.plot(*r.representative_point, "gx", ms=8, mew=2, zorder=6)
    for f in e0.forbidden_regions or []:
        px, py = zip(*f.polygon)
        ax.fill(px, py, facecolor="none", edgecolor="#d95f02", hatch="//", lw=0.5, zorder=2)
    for p in truth.people:
        color = "#d7191c" if p.id in added_ids else ("#2c7bb6" if p.group == gid else "#777777")
        ax.add_patch(plt.Circle((p.x, p.y), 0.25, color=color, zorder=5))
        ax.plot([p.x, p.x + 0.45 * math.cos(p.yaw)], [p.y, p.y + 0.45 * math.sin(p.yaw)], color="k", lw=1, zorder=6)
    robot = inputs.observation.robot
    ax.plot(robot.x, robot.y, "k^", ms=8, zorder=6)
    for side in (-1, 1):
        a = robot.yaw + side * math.radians(30)
        ax.plot([robot.x, robot.x + 8 * math.cos(a)], [robot.y, robot.y + 8 * math.sin(a)], "k:", lw=0.7, zorder=4)
    t = list(truth.targets.values())[0]
    ax.plot(t.anchor[0], t.anchor[1], "m*", ms=10, zorder=7)
    pts = [(p.x, p.y) for p in truth.people] + [(robot.x, robot.y)]
    px, py = zip(*pts)
    ax.set_xlim(min(px) - 1.5, max(px) + 1.5)
    ax.set_ylim(min(py) - 1.5, max(py) + 1.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def sheet(family: str, es) -> None:
    chosen = pick(es, family)
    if not chosen:
        print(f"{family}: nothing kept")
        return
    fig, axes = plt.subplots(len(chosen), 3, figsize=(15, 3.9 * len(chosen)), squeeze=False)
    for row, e in zip(axes, chosen):
        vin, vtruth = load_scene(C.I06 / e["folder"])
        bin_, _ = load_scene(C.SCENES / f"seed{e['seed']}" / e["base_scene_id"])
        row[0].imshow(bin_.observation.rgb)
        row[0].set_title(f"base s{e['seed']} {e['base_scene_id']}", fontsize=9)
        row[1].imshow(vin.observation.rgb)
        cam = vin.observation.camera
        for a in e["added"]:
            q = project(cam, (a["x"], a["y"], 1.95))
            if q is not None:
                row[1].plot(q[0], q[1], "v", color="red", ms=9, mec="white")
        geo = "; ".join(f"{g['id']}: member {g['nearest_member_m']:.2f} m" for g in e.get("added_geometry", []))
        row[1].set_title(f"{e['scene_id']} (attempts {e['attempts']})\n{geo}", fontsize=8)
        gid = vtruth.episodes["e0"].group_at_target
        draw_map(row[2], vtruth, vin, {a["id"] for a in e["added"]}, gid)
        row[2].set_title("top-down: members blue, added red, region green", fontsize=8)
        for ax in row[:2]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    path = C.BUILD / f"contact_{family}.png"
    fig.savefig(path, dpi=70)
    plt.close(fig)
    print(f"{family}: {len(chosen)} rows -> {path}")


if __name__ == "__main__":
    es = entries()
    for family in (sys.argv[1:] or ["A", "B", "C", "D", "E"]):
        sheet(family, es)
