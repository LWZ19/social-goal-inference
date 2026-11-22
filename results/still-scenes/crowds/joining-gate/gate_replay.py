"""Joining gate: recorded answers re-solved with the place-in-turn step gated, zero calls.
Arms: now (the solver as it is), G (facing and spacing), F (facing only), S (spacing only), and copy (the gated code
with both tests off, which must equal now). Development runs from the harness's replay_runs.txt; crowd runs are the
latest constraint-method run of each of the five crowd models and seeds 0 to 2.
    PYTHONPATH=src python results/still-scenes/crowds/joining-gate/gate_replay.py"""
import collections
import json
import math
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir() and (p / "src").is_dir())
HERE = Path(__file__).resolve().parent
HARNESS = REPO / "results/still-scenes/analysis-scripts/harness"
sys.path.insert(0, str(HARNESS))

import sgi.solver as S  # noqa: E402
from replay_solver import HELD_OUT_SCENE, _scenes_folder, solve_again  # noqa: E402
from sgi.config import Config  # noqa: E402
from sgi.evaluate import score  # noqa: E402
from sgi.gt.derive import prepare  # noqa: E402
from sgi.rescore import _recorded_config, _records  # noqa: E402
from sgi.scene import load_scene  # noqa: E402
from sgi.types import GoalResult  # noqa: E402

ARMS = ("G", "F", "S", "copy")
CROWD_MODELS = ("claude_haiku55", "claude_sonnet55_low", "gemini_fast", "soclaas_dsv41", "soclaas_glm53")
ORIGINAL = S._turn
ARM = "now"


def gated_turn(waiting, apart, bodies, target_number, target_xy, robot_xy, th, line=None):
    """sgi.solver._turn with the gate of the arm in ARM; `now` is the original itself."""
    if ARM == "now":
        return ORIGINAL(waiting, apart, bodies, target_number, target_xy, robot_xy, th, line)
    waiting, joined = list(waiting), []
    last, before = S._line_end(waiting, bodies, target_xy, robot_xy, th, line)
    while True:
        place, along = S._behind(bodies[last], None if before is None else bodies[before], target_xy, robot_xy, th,
                                 line)
        taken = []
        for number, body in bodies.items():
            if number in waiting or number == target_number or number in apart:
                continue
            gap, back = float(np.hypot(*(body.xy - place))), body.xy - bodies[last].xy
            behind = float(back @ along) >= math.cos(math.radians(th.view_half_angle_deg)) * float(np.hypot(*back)) - S._EPS
            if not (gap < th.standoff_m - S._EPS and behind):
                continue
            if ARM in ("G", "F") and body.yaw is not None:          # faces within 90 degrees of the line's front
                heading = np.array([math.cos(body.yaw), math.sin(body.yaw)])
                if float(heading @ (-along)) < -S._EPS:
                    continue
            if ARM in ("G", "S") and before is not None:            # no farther back than 1.5 times the line's spacing
                spacing = float(np.hypot(*(bodies[last].xy - bodies[before].xy)))
                if float(np.hypot(*back)) > 1.5 * spacing + S._EPS:
                    continue
            taken.append((round(gap, S._MM), number))
        if not taken:
            from_target = float(np.hypot(*(bodies[last].xy - target_xy)))
            return S._Turn(last=last, last_from_target=from_target, place=place, joined=joined)
        last, before = min(taken)[1], last
        waiting.append(last)
        joined.append(last)


S._turn = gated_turn


def replay(job):
    try:
        return _replay(job)
    except Exception as error:
        return [], collections.Counter({f"run failed ({job[1].name}): {type(error).__name__}": 1})


def _replay(job):
    """Rows for one run: (data set, run folder, scenes folder or None, {scene_id: (family, level)} or None)."""
    global ARM
    dataset, run_dir, scenes, labels = job
    doc, cfg = _recorded_config(run_dir / "config.yaml")
    th, folder = cfg.thresholds, _scenes_folder(run_dir, doc, scenes)
    held = set(cfg.held_out_situations) | set(Config.model_fields["held_out_situations"].default)
    loaded, rows, skipped = {}, [], collections.Counter()
    for record in _records(run_dir / "results.jsonl"):
        situation, scene_id = (record.get("labels") or {}).get("situation"), record["scene_id"]
        if situation is None or situation in held or HELD_OUT_SCENE.search(str(scene_id)):
            skipped["held out"] += 1
            continue
        result = GoalResult.model_validate(record["result"])
        if not (result.source == "constraints" and result.pose_world is not None and result.people is not None
                and result.constraints is not None):
            skipped["no solver step"] += 1
            continue
        if scene_id not in loaded:
            try:
                inputs, truth = load_scene(folder / scene_id)
                loaded[scene_id] = (inputs.observation, truth, prepare(inputs, truth, cfg))
            except Exception as error:                  # a scene folder not unpacked on this machine
                loaded[scene_id] = error
        if isinstance(loaded[scene_id], Exception):
            skipped[f"scene missing ({folder.relative_to(REPO) if folder.is_relative_to(REPO) else folder})"] += 1
            continue
        observation, truth, truths = loaded[scene_id]
        gt = truths[record["episode_id"]].gt
        ARM = "now"
        solved, again = solve_again(result, observation, th)
        ok = {"now": bool(score(again, observation, truth, gt, th).success)}
        joins = list(solved.note.joined_waiting)
        for arm in ARMS:
            if not joins:
                ok[arm] = ok["now"]
                continue
            ARM = arm
            _, again = solve_again(result, observation, th)
            ok[arm] = bool(score(again, observation, truth, gt, th).success)
        ARM = "now"
        family, level = (labels or {}).get(scene_id, (None, None))
        rows.append({"set": dataset, "run": run_dir.name, "scene": scene_id, "episode": record["episode_id"],
                     "trial": record.get("trial"), "situation": situation, "family": family, "level": level,
                     "joined_now": joins, "ok": ok})
    return rows, skipped


def jobs():
    out = []
    for line in (HARNESS / "replay_runs.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            run, scenes = line.split("=", 1)
            out.append(("development", REPO / run, REPO / scenes, None))
    manifest = {}
    for line in open(REPO / "results/still-scenes/crowds/build/manifest.jsonl"):
        x = json.loads(line)
        if x["accepted"]:
            manifest[(x["seed"], x["scene_id"])] = (x["family"], x.get("level"))
    latest = {}
    for p in sorted((REPO / "runs").glob("*_crowd_vlm_*_con")):
        m = re.match(r"^\d{8}-\d{6}_crowd_vlm_seed(\d)_(.+)_con$", p.name)
        if m and m.group(2) in CROWD_MODELS and (p / "results.jsonl").exists():
            latest[(m.group(2), int(m.group(1)))] = p
    for (model, seed), p in sorted(latest.items()):
        labels = {sid: fl for (s, sid), fl in manifest.items() if s == seed}
        out.append(("crowd", p, None, labels))
    return out


def table(rows, key):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    lines = []
    for name in sorted(groups, key=str):
        rs = groups[name]
        cells = [f"{str(name):24s} n {len(rs):5d}  now {sum(r['ok']['now'] for r in rs):5d}"]
        for arm in ("G", "F", "S"):
            got = sum(r["ok"][arm] for r in rs)
            gained = sum(r["ok"][arm] and not r["ok"]["now"] for r in rs)
            lost = sum(r["ok"]["now"] and not r["ok"][arm] for r in rs)
            cells.append(f"{arm} {got:5d} (+{gained} -{lost})")
        lines.append("  ".join(cells))
    return lines


if __name__ == "__main__":
    work = jobs()
    with Pool(8) as pool:
        results = pool.map(replay, work, chunksize=1)
    rows = [r for rs, _ in results for r in rs]
    skipped = sum((s for _, s in results), collections.Counter())
    with open(HERE / "gate_rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    copy_bad = sum(r["ok"]["copy"] != r["ok"]["now"] for r in rows)
    print(f"runs {len(work)}; records re-solved {len(rows)}")
    for reason, count in sorted(skipped.items()):
        print(f"skipped, {reason}: {count}")
    print(f"check: the gated code with both tests off differs from the solver on {copy_bad} records (must be 0)")
    for dataset in ("development", "crowd"):
        rs = [r for r in rows if r["set"] == dataset]
        print(f"\n## {dataset}: {len(rs)} solver steps, someone joins under now in {sum(bool(r['joined_now']) for r in rs)}")
        print("\n".join(table(rs, lambda r: "all")))
        if dataset == "development":
            print("by situation")
            print("\n".join(table(rs, lambda r: r["situation"])))
        else:
            print("by family and level")
            print("\n".join(table(rs, lambda r: f"{r['family']}{r['level'] or ''}")))
