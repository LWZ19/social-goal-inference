"""I06 step 2b: the base reproduction checks (FROZEN.md section 2), before any variant is built.

Compares each reproduced base scene (scenes/seed<N>/<id>, loaded with sgi.scene.load_scene, and the in-memory arrays
of build/base_layouts/seed<N>.pkl) with its whitelisted development scene, loaded through the step-0 harness only:
- seed 0, eval_06..13: data/placeholder/eval (an s10 run lists it);
- seeds 0, 1, 2, eval_14..15: data/dev_scenes/dev_new/seed<N>/scenes (an s17 run of that seed lists it).
Exact equality of people (id, x, y, yaw, group, role), targets (every field but tour_frames, which the bases do not
have: no tour is built), T_world_camera, robot pose, both episodes' acceptable polygons and representative points (and
forbidden regions, bounds and obstacles as extras), and the rgb and depth arrays. Seeds 1 and 2 eval_06..13 have no
whitelisted copy and are listed as not checkable.

Writes builder/base_check.txt and build/base_check.txt; exit 1 on any difference.
"""
from __future__ import annotations

import pickle
import sys

import numpy as np

sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H  # noqa: E402

from sgi.scene import load_scene  # noqa: E402

import crowd as C  # noqa: E402

LINES: list[str] = []


def say(text: str = "") -> None:
    print(text, flush=True)
    LINES.append(text)


def maxdiff(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return float("inf")
    return float(np.abs(a - b).max()) if a.size else 0.0


def compare(mine_inputs, mine_truth, base, sc) -> list[str]:
    """Differences between a reproduced base and its whitelisted scene; empty when they are equal."""
    diffs = []
    pa = [(p.id, p.x, p.y, p.yaw, p.group, p.role) for p in mine_truth.people]
    pb = [(p.id, p.x, p.y, p.yaw, p.group, p.role) for p in sc.truth.people]
    if pa != pb:
        diffs.append(f"people differ: {pa} vs {pb}")
    for tid in sorted(set(mine_truth.targets) | set(sc.truth.targets)):
        ta, tb = mine_truth.targets.get(tid), sc.truth.targets.get(tid)
        if ta is None or tb is None:
            diffs.append(f"target {tid} missing on one side")
            continue
        da, db = ta.model_dump(exclude={"tour_frames"}), tb.model_dump(exclude={"tour_frames"})
        if da != db:
            diffs.append(f"target {tid} differs: {da} vs {db}")
    if not np.array_equal(mine_inputs.observation.camera.T_world_camera, sc.observation.camera.T_world_camera):
        diffs.append("T_world_camera differs")
    if mine_inputs.observation.robot != sc.observation.robot:
        diffs.append(f"robot differs: {mine_inputs.observation.robot} vs {sc.observation.robot}")
    for eid in sorted(set(mine_truth.episodes) | set(sc.truth.episodes)):
        ea, eb = mine_truth.episodes[eid], sc.truth.episodes[eid]
        ra, rb = ea.acceptable_regions or [], eb.acceptable_regions or []
        if len(ra) != len(rb):
            diffs.append(f"{eid}: {len(ra)} vs {len(rb)} acceptable regions")
        for i, (x, y) in enumerate(zip(ra, rb)):
            if list(map(tuple, x.polygon)) != list(map(tuple, y.polygon)):
                diffs.append(f"{eid} region {i}: polygon differs (max {maxdiff(x.polygon, y.polygon):.3e})")
            if tuple(x.representative_point) != tuple(y.representative_point):
                diffs.append(f"{eid} region {i}: representative point {x.representative_point} vs {y.representative_point}")
            if x.heading != y.heading:
                diffs.append(f"{eid} region {i}: heading differs")
        fa, fb = ea.forbidden_regions or [], eb.forbidden_regions or []
        if [(f.kind, list(map(tuple, f.polygon))) for f in fa] != [(f.kind, list(map(tuple, f.polygon))) for f in fb]:
            diffs.append(f"{eid}: forbidden regions differ")
        for field in ("situation", "group_at_target", "target_ids", "request_type"):
            if getattr(ea, field) != getattr(eb, field):
                diffs.append(f"{eid}: {field} {getattr(ea, field)} vs {getattr(eb, field)}")
    if list(map(tuple, mine_truth.bounds)) != list(map(tuple, sc.truth.bounds)):
        diffs.append("bounds differ")
    if [list(map(tuple, o)) for o in mine_truth.obstacles] != [list(map(tuple, o)) for o in sc.truth.obstacles]:
        diffs.append("obstacles differ")
    for name, a, b in (("rgb", base.rgb, sc.observation.rgb), ("depth", base.depth, sc.observation.depth),
                       ("rgb (written)", mine_inputs.observation.rgb, sc.observation.rgb),
                       ("depth (written)", mine_inputs.observation.depth, sc.observation.depth)):
        a, b = np.asarray(a), np.asarray(b)
        same = a.shape == b.shape and np.array_equal(np.nan_to_num(a, nan=-1.0), np.nan_to_num(b, nan=-1.0))
        if not same:
            diffs.append(f"{name} array differs")
    return diffs


def main() -> int:
    ph_run = H.runs(stage="s10", model="qwen38_low", method="con")[0]
    dev_runs = {n: H.runs(arm=f"dev_seed{n}")[0] for n in (0, 1, 2)}
    say("I06 base reproduction check (FROZEN.md section 2). Whitelisted scenes loaded through the step-0 harness:")
    say(f"  seed 0 eval_06..13 via run {ph_run.folder} (scenes {ph_run.scenes_dir})")
    for n, r in dev_runs.items():
        say(f"  seed {n} eval_14..15 via run {r.folder} (scenes {r.scenes_dir})")
    failures, checked, unchecked = 0, 0, []
    for seed in (0, 1, 2):
        with open(C.BUILD / "base_layouts" / f"seed{seed}.pkl", "rb") as f:
            bases = pickle.load(f)
        say(f"\nseed {seed}: {len(bases)} bases reproduced: {', '.join(b.scene.scene_id for b in bases)}")
        for base in bases:
            folder = C.SCENES / f"seed{seed}" / base.scene.scene_id
            mine_inputs, mine_truth = load_scene(folder)
            if base.index <= C.FIRST_PASS_LAST:
                run = ph_run if seed == 0 else None
            else:
                run = dev_runs[seed]
            rej = base.rejected_before or "none"
            if run is None:
                unchecked.append(f"seed {seed} {base.scene.scene_id}")
                say(f"  {base.scene.scene_id}: no whitelisted copy on disk (not checkable); people "
                    f"{len(mine_truth.people)}, rejected draws before it {rej}")
                continue
            sc = H.scene(run, base.scene.scene_id)
            diffs = compare(mine_inputs, mine_truth, base, sc)
            checked += 1
            if diffs:
                failures += 1
                say(f"  {base.scene.scene_id}: DIFFERENT from the whitelisted scene:")
                for d in diffs:
                    say(f"    - {d}")
            else:
                say(f"  {base.scene.scene_id}: EQUAL (people {len(mine_truth.people)}, targets, camera, robot, both "
                    f"episodes' regions and representative points, forbidden regions, bounds, obstacles, rgb and depth "
                    f"arrays; max difference 0); rejected draws before it {rej}")
    say(f"\n{checked} bases checked, {failures} with a difference; {len(unchecked)} not checkable "
        f"(no whitelisted copy): {', '.join(unchecked)}")
    say("RESULT: " + ("PASS" if failures == 0 and checked == 14 else "FAIL"))
    text = "\n".join(LINES) + "\n"
    (C.I06 / "builder" / "base_check.txt").write_text(text, encoding="utf-8")
    (C.BUILD / "base_check.txt").write_text(text, encoding="utf-8")
    return 0 if failures == 0 and checked == 14 else 1


if __name__ == "__main__":
    raise SystemExit(main())
