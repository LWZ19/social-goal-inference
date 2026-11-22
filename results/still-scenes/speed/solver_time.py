"""How long the constraint method's solver takes on recorded answers, next to an answer of the model. Zero calls.
Re-solves the recorded people and rules of the development runs listed in the harness's replay_runs.txt, in order,
until 120 solves, timing the floor from depth and the whole solve (floor and ranking) with time.perf_counter.
    PYTHONPATH=src python results/still-scenes/speed/solver_time.py"""
import statistics
import sys
import time
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "results").is_dir() and (p / "src").is_dir())
HARNESS = REPO / "results/still-scenes/analysis-scripts/harness"
sys.path.insert(0, str(HARNESS))
from replay_solver import _scenes_folder  # noqa: E402
from sgi.rescore import _recorded_config, _records  # noqa: E402
from sgi.scene import load_scene  # noqa: E402
from sgi.solver import floor, solve  # noqa: E402
from sgi.stages.goal import standard_goal  # noqa: E402
from sgi.types import GoalResult  # noqa: E402

WANTED = 120
lines = [line for line in (HARNESS / "replay_runs.txt").read_text().splitlines() if line.strip() and not line.startswith("#")]
whole, ground, cells, runs = [], [], [], 0
for line in lines:
    run_dir, scenes = (REPO / part for part in line.split("=", 1))
    doc, cfg = _recorded_config(run_dir / "config.yaml")
    th, folder, loaded = cfg.thresholds, _scenes_folder(run_dir, doc, scenes), {}
    runs += 1
    for rec in _records(run_dir / "results.jsonl"):
        r = GoalResult.model_validate(rec["result"])
        if not (r.source == "constraints" and r.pose_world is not None and r.people is not None
                and r.constraints is not None):
            continue
        if rec["scene_id"] not in loaded:
            loaded[rec["scene_id"]] = load_scene(folder / rec["scene_id"])[0].observation
        obs = loaded[rec["scene_id"]]
        standard = standard_goal(obs, r.target, th)
        t0 = time.perf_counter()
        grid = floor(obs, r.people, th)
        t1 = time.perf_counter()
        solve(obs, r.target, r.people, [use.constraint for use in r.constraints], standard, th)
        t2 = time.perf_counter()
        ground.append(t1 - t0)
        whole.append(t2 - t1)
        cells.append(int(grid.selectable.sum()))
        if len(whole) >= WANTED:
            break
    if len(whole) >= WANTED:
        break
ms = lambda xs: f"median {1000 * statistics.median(xs):.0f} ms, max {1000 * max(xs):.0f} ms"
print(f"{len(whole)} solves from {runs} runs; grid {len(grid.offsets)} x {len(grid.offsets)} cells of "
      f"{th.grid_resolution_m} m")
print(f"whole solve, floor and ranking: {ms(whole)}")
print(f"floor from depth alone: {ms(ground)}")
print(f"ranking, whole minus floor: median {1000 * (statistics.median(whole) - statistics.median(ground)):.0f} ms")
print(f"reachable cells: median {statistics.median(cells):.0f}, max {max(cells)}")
