"""I06 pre-freeze format check (written before FROZEN.md; generates no crowd variant and scores nothing).

1. Reproduces the base layout eval_06 (waiting_line_1) of seed 0 with the package's own accept step, drawing rows
   eval_00..eval_06 of the main stream np.random.default_rng(0) in order (as generate() does), in a temporary
   folder under this directory that is removed afterwards. Compares its people, targets, camera and robot pose,
   and its derived acceptable regions, with the whitelisted scene data/placeholder/eval/eval_06_waiting_line_1,
   loaded through the step-0 harness.
2. Compares the whitelisted realistic real_eval_06 / staffed / plain scenes' members and targets with the placeholder
   eval_06 (which seed the realistic set's layouts come from).
3. Lists the distinct whitelisted development scenes by folder (which on-disk sets belong to which seed).
4. Lists the solver note and flag fields.
5. For the reproduced eval_06 only: how far beyond the last member, along the line's axis, a body axis is still in
   the image (>= 3 of the 9 visibility sample heights project inside the image; projection only, no occlusion).
"""
from __future__ import annotations

import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H  # noqa: E402

from sgi.config import load_config  # noqa: E402
from sgi.geometry import Camera, in_image, project  # noqa: E402
from sgi.scene import load_scene  # noqa: E402
from sgi.synth import generate as G  # noqa: E402
from sgi.types import Flags, Keys, SolverNote  # noqa: E402

HERE = Path(__file__).resolve().parent


def say(*a):
    print(*a, flush=True)


def people_rows(truth):
    return [(p.id, round(p.x, 9), round(p.y, 9), round(p.yaw, 9), p.group, p.role) for p in truth.people]


def max_diff(a, b):
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def main():
    cfg = load_config()
    th = cfg.thresholds
    say("thresholds:", {k: getattr(th, k) for k in ("person_radius_m", "robot_radius_m", "standoff_m", "personal_gap_m",
                                                    "relevant_radius_m", "visibility_min_samples", "max_people",
                                                    "view_half_angle_deg", "max_keep_m", "grid_resolution_m")})

    # 1. reproduce seed 0, rows eval_00..eval_06 of the main stream
    work = Path(tempfile.mkdtemp(prefix="work_", dir=HERE))
    try:
        rng = np.random.default_rng(0)
        counts = dict.fromkeys(G._REASONS, 0)
        split_dir = work / "eval"
        layout = scene = None
        for index, row in enumerate(G._EVAL[:7]):
            layout, scene, _, _ = G._accept(split_dir, index, row, rng, cfg, counts)
            say(f"reproduced seed 0 eval_{index:02d}: {scene.scene_id}")
        rejected = {k: v for k, v in counts.items() if v}
        say("rejections while reproducing:", rejected or "none")
        folder = split_dir / scene.scene_id
        scene_r = G._with_regions(folder, scene, cfg)
        inputs_r, truth_r = load_scene(folder)
    finally:
        pass

    ph_run = H.runs(stage="s10", model="qwen38_low", method="con")[0]
    sc = H.scene(ph_run, "eval_06_waiting_line_1")
    say("whitelisted scene via", ph_run.folder, "->", sc.inputs.scene_id)
    a, b = people_rows(truth_r), people_rows(sc.truth)
    same_ids = [r[0] for r in a] == [r[0] for r in b] and [r[4:] for r in a] == [r[4:] for r in b]
    dp = max((max_diff(r[1:4], s[1:4]) for r, s in zip(a, b)), default=0.0)
    say(f"people: n {len(a)} vs {len(b)}; ids/groups/roles equal {same_ids}; max |dx,dy,dyaw| {dp:.3e}")
    ta, tb = list(truth_r.targets.values()), list(sc.truth.targets.values())
    for t1, t2 in zip(ta, tb):
        fp = max_diff([c for p in (t1.footprint or []) for c in p], [c for p in (t2.footprint or []) for c in p])
        say(f"target {t1.id}/{t2.id}: kind {t1.kind}/{t2.kind}; anchor diff {max_diff(t1.anchor, t2.anchor):.3e}; "
            f"footprint diff {fp:.3e}; front_yaw {t1.front_yaw} / {t2.front_yaw}")
    Ta = np.array(inputs_r.observation.camera.T_world_camera)
    Tb = np.array(sc.observation.camera.T_world_camera)
    say(f"camera T_world_camera max diff {np.abs(Ta - Tb).max():.3e}; robot {inputs_r.observation.robot} vs "
        f"{sc.observation.robot}")
    for ep in scene_r.episodes:
        mine = ep.ground_truth.acceptable_regions
        theirs = sc.truths[ep.episode_id].gt.acceptable_regions
        d = max((max_diff([c for p in r1.polygon for c in p], [c for p in r2.polygon for c in p])
                 for r1, r2 in zip(mine, theirs)), default=0.0)
        rp = max((max_diff(r1.representative_point, r2.representative_point) for r1, r2 in zip(mine, theirs)), default=0.0)
        say(f"episode {ep.episode_id}: regions {len(mine)} vs {len(theirs)}; polygon max diff {d:.3e}; "
            f"representative point max diff {rp:.3e}; situation {ep.ground_truth.situation}, "
            f"group_at_target {ep.ground_truth.group_at_target}")

    # 5. in-view extent beyond the last member (projection only)
    obs = inputs_r.observation
    cam = obs.camera
    t0 = truth_r.targets["t0"]
    members = [p for p in truth_r.people if p.group == "g0"]
    ax = np.array(t0.anchor[:2])
    by_d = sorted(members, key=lambda p: math.dist((p.x, p.y), ax))
    last = np.array([by_d[-1].x, by_d[-1].y])
    before = np.array([by_d[-2].x, by_d[-2].y]) if len(by_d) > 1 else ax
    u = (last - before) / np.linalg.norm(last - before)
    best = None
    for g in np.arange(0.0, 4.001, 0.01):
        p = last + g * u
        inside = sum(1 for z in (0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7)
                     if (q := project(cam, (p[0], p[1], z))) is not None and in_image(cam, q[0], q[1]))
        if inside >= th.visibility_min_samples:
            best = g
    say(f"eval_06 (seed 0): a body axis on the line's axis is in the image up to {best:.2f} m beyond the last member")
    shutil.rmtree(work, ignore_errors=True)

    # 2. realistic arms vs placeholder eval_06
    for arm in ("real", "staffed", "plain"):
        run = H.runs(stage="s19", arm=arm, model="qwen38_low", method="con")[0]
        sid = f"{arm}_eval_06_waiting_line_1"
        rs = H.scene(run, sid)
        mem = [(p.id, p.x, p.y, p.yaw) for p in rs.truth.people if p.group is not None]
        ph = [(p.id, p.x, p.y, p.yaw) for p in sc.truth.people if p.group is not None]
        d = max((max_diff(r[1:], s[1:]) for r, s in zip(mem, ph)), default=float("nan"))
        roles = sorted({p.role for p in rs.truth.people if p.role})
        say(f"{sid}: members {len(mem)} (placeholder {len(ph)}), max |dx,dy,dyaw| vs placeholder eval_06 {d:.3e}; "
            f"people {len(rs.truth.people)}, roles {roles}; anchor diff "
            f"{max_diff(rs.truth.targets['t0'].anchor, sc.truth.targets['t0'].anchor):.3e}")

    # 3. distinct whitelisted scenes by folder
    by = {}
    for s in H.distinct_scenes():
        by.setdefault(str(s.scenes_dir), []).append(s.scene_id)
    for k, v in by.items():
        say(f"{k}: {len(v)} scenes: {', '.join(sorted(v))}")

    # 4. solver note and flags
    say("SolverNote fields:", list(SolverNote.model_fields))
    say("Keys fields:", list(Keys.model_fields))
    say("Flags fields:", list(Flags.model_fields))


if __name__ == "__main__":
    main()
