"""Offline replay of the solver over recorded runs of the constraint method. No model is called, no network used.

A record of a run (results.jsonl) holds what the solver was given: the target point, the people found and the
constraints as the model wrote them. For each run folder named here, every recorded goal step of the constraint
method (`source: constraints`, with a pose) is solved again by the solver of the code that is imported, on the scene
the run used and under the thresholds the run recorded, and the new place is scored by the package's scorer against
the scene's ground truth as it is derived now. Nothing but the solver is run again: the people and the constraints
are the recorded ones. A record without a solver step (the standard goal was taken, or no target was found) is
carried as recorded. A record of a held-out situation, or of no known situation, is left out, and its scene is not
opened. Held out is every situation of the recorded config's `held_out_situations` and every one of that list as
the imported code has it by default: a run recorded under another list does not open what is held out now. A record
of a held-out generated test scene (eval_16 to eval_19, the pattern research_harness.py beside this file refuses too)
is left out as well, whatever its situation, before its scene is opened. The solver is run again by solve_again,
which research_harness.resolve uses as well. This file needs no other file beside it unless --whitelist is given.

Printed for each run, per situation and in all: the records; the solver steps among them; how many of those give
the recorded place again; how many moved by more than a cell; the successes recorded; the recorded places scored
again now (other than recorded only where the scorer or the scene has changed since); the successes with the
replayed places; and how many records gained and lost their success. Then the same over all the runs, and every
record that succeeded as recorded and fails as replayed. Where the solver the runs were made with gives every
recorded place again, "ok recorded" is the success before a change to the solver and "ok replayed" the success
after it.

    replay_solver.py [--src SRC] [--root ROOT] [--gate] [--accepted FILE] [--expect-reproduced] [--known-moved FILE]
                     [--rows OUT.json] [--whitelist] RUN_DIR[=SCENES_DIR] ... [@FILE]

    RUN_DIR      a run folder: config.yaml and results.jsonl. The scenes are read from the folder its config.yaml
                 records, a relative one from the checkout the run folder lies in; or from SCENES_DIR.
    @FILE        a file of further arguments, one a line. An empty line and a line that starts with # are passed
                 over. replay_runs.txt beside this file lists every recorded run with solver steps, each with its
                 scenes folder, from the repository root.
    --root       the folder a RUN_DIR and a SCENES_DIR that are not absolute are taken from; by default the
                 current folder, and with --whitelist the repository root (where the runs are).
    --src        the `src` folder whose `sgi` is imported; by default that of the checkout this file is in.
    --expect-reproduced
                 ends with exit code 1 unless every replayed place is the recorded one. That is the check to make
                 first, with --src on the code as it was when the runs were made: only then is a difference under
                 the code as it is now the solver's doing. A place that --known-moved names does not count.
    --known-moved
                 a file of the places known not to be given again, with why, one a line in the form of --accepted;
                 each is still printed, and marked. research_moved.txt beside this file names those of the runs on
                 research_runs.txt under the code as it is.
    --gate       the regression check for a change to the solver: the tables of the single runs are not printed,
                 only the one over all the runs and every record that lost its success, and the exit code is 1
                 when a record lost it that --accepted does not name.
    --accepted   a file of the records whose loss is known and stated, one a line: the run's name (its folder's,
                 or that without the time before it), the scene, the episode and the trial. An empty line and
                 what follows a # are passed over. Such a record is still printed as lost, and marked.
                 replay_accepted.txt beside this file names them, with the reason.
    --rows       writes one row per record, with both places and both verdicts, as JSON.
    --whitelist  refuses, before anything is opened, every RUN_DIR that is not on research_runs.txt beside this file
                 (the listed runs), a SCENES_DIR or recorded scenes folder
                 that is not the one that list gives for the run, every path that is held out or excluded where it
                 is written or where its links lead (research_harness.guard; a run's config.yaml and results.jsonl
                 whichever form is given), a scene not on research_scenes.txt or with a file in it or named by its
                 scene.json that is refused or lies outside it, before the scene is opened, and a --rows file inside
                 the repository; the error names the path refused.
                 Run it from the repository root: the runs and scenes are read as the lists say.

The regression check over every recorded run, from the repository root (the runs and their scenes are local, not in git):

    .venv/bin/python results/still-scenes/analysis-scripts/harness/replay_solver.py --gate \
        --accepted results/still-scenes/analysis-scripts/harness/replay_accepted.txt @results/still-scenes/analysis-scripts/harness/replay_runs.txt

The reproduction check over the listed runs with solver steps, from the repository root (RUNS holds those lines
of research_runs.txt):

    PYTHONPATH= .venv/bin/python results/still-scenes/analysis-scripts/harness/replay_solver.py --whitelist --gate --expect-reproduced \
        --known-moved results/still-scenes/analysis-scripts/harness/research_moved.txt @RUNS
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

SAME_PLACE_M = 1e-6                                # a replayed place this near the recorded one is the recorded one
HELD_OUT_SCENE = re.compile(r"(?:^|_)eval_1[6-9](?![0-9])", re.I)  # eval_16 to eval_19, as research_harness has it
COUNTS = ("records", "solver steps", "reproduced", "moved", "ok recorded", "ok rescored", "ok replayed", "gained", "lost")


def _harness():
    """research_harness.py beside this file, for --whitelist only: the guard and the written lists. It imports
    nothing of sgi at its top."""
    if "research_harness" not in sys.modules:
        spec = importlib.util.spec_from_file_location("research_harness", Path(__file__).with_name("research_harness.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules["research_harness"] = module
        spec.loader.exec_module(module)
    return sys.modules["research_harness"]


class _Parser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line):
        """A line of an @FILE is one argument; an empty line and a comment are none."""
        line = arg_line.strip()
        return [] if not line or line.startswith("#") else [line]


def _arguments(argv):
    parser = _Parser(prog="replay_solver.py", fromfile_prefix_chars="@",
                     description="Solve the recorded goal steps of runs again, with no model call.")
    parser.add_argument("runs", nargs="+", metavar="RUN_DIR[=SCENES_DIR]")
    parser.add_argument("--src", type=Path, default=Path(__file__).resolve().parents[3] / "src")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--accepted", type=Path, metavar="FILE")
    parser.add_argument("--expect-reproduced", action="store_true")
    parser.add_argument("--known-moved", type=Path, metavar="FILE")
    parser.add_argument("--rows", type=Path, metavar="OUT.json")
    parser.add_argument("--whitelist", action="store_true")
    return parser.parse_args(argv)


def _accepted(path: Path | None) -> set[tuple[str, str, str, str]]:
    """The records a file names (as lost, or as moved), with reason: (run, scene, episode, trial), each as written."""
    named = set()
    for number, line in enumerate([] if path is None else path.read_text(encoding="utf-8").splitlines(), start=1):
        words = line.split("#")[0].split()
        if words and len(words) != 4:
            raise SystemExit(f"error: {path}: line {number}: a run, a scene, an episode and a trial are expected")
        if words:
            named.add(tuple(words))
    return named


def _is_accepted(row: dict, named: set[tuple[str, str, str, str]]) -> bool:
    """Whether a file of accepted losses names the record of a row: by its run folder's name, or by that name
    without the time the folder's name starts with."""
    rest = (row["scene"], row["episode"], str(row["trial"]))
    return (row["run"], *rest) in named or (row["run"].partition("_")[2], *rest) in named


def _scenes_folder(run_dir: Path, doc: dict, given: Path | None) -> Path:
    """Where the scenes of a run are: the folder given, or the one its config.yaml records."""
    if given is not None:
        return given
    recorded = Path(_harness().relocated(doc["scenes_dir"]))   # the development sets now lie in data/dev_scenes/
    return recorded if recorded.is_absolute() else run_dir.resolve().parent.parent / recorded


def solve_again(result, observation, th, *, target=None, people=None, constraints=None):
    """(solved, again): the solver of the imported code run on a recorded GoalResult under the thresholds th, with
    the target, the people and the constraints (each as the answer stated it) that it records, or those given;
    `again` is the result with the solver's place and what the solver made of the answer, as the constraints stage
    writes them (status ok, source constraints). The people are taken as given, already numbered: a target given in
    place of the recorded one does not number them again. ValueError where the record holds none of an input and
    none is given."""
    from sgi.geometry import in_image, project
    from sgi.solver import solve
    from sgi.stages.goal import standard_goal
    target = result.target if target is None else target
    people = result.people if people is None else list(people)
    if constraints is None:
        constraints = None if result.constraints is None else [use.constraint for use in result.constraints]
    missing = [name for name, value in (("target", target), ("people", people), ("constraints", constraints)) if value is None]
    if missing:
        raise ValueError(f"the solver needs {', '.join(missing)}: the record holds none and none was given")
    solved = solve(observation, target, people, list(constraints), standard_goal(observation, target, th), th)
    shown = project(observation.camera, (*solved.pose_world[:2], 0.0))
    in_picture = shown is not None and in_image(observation.camera, *shown[:2])
    again = result.model_copy(update={
        "status": "ok", "source": "constraints", "pose_world": solved.pose_world, "pose_robot": solved.pose_robot,
        "goal_pixel": shown[:2] if in_picture else None, "target": target, "people": people,
        "flags": solved.flags.model_copy(update={"people_cut": result.flags.people_cut}),
        "constraints": solved.uses, "candidates": solved.candidates, "solver": solved.note})
    return solved, again


def replay(run_dir: Path, scenes_dir: Path | None, allow=None) -> tuple[list[dict], int]:
    """One row per record of the run that is not left out, and the number left out. `allow`, where given, is called
    with each scene id before that scene is opened and may refuse it by raising (--whitelist)."""
    from sgi.config import Config
    from sgi.evaluate import score
    from sgi.gt.derive import prepare
    from sgi.rescore import _recorded_config, _records
    from sgi.scene import load_scene
    from sgi.types import GoalResult

    doc, cfg = _recorded_config(run_dir / "config.yaml")
    th, folder = cfg.thresholds, _scenes_folder(run_dir, doc, scenes_dir)
    # Held out by the run's own list or by the list as it is by default now: either keeps a record's scene shut.
    held_out = set(cfg.held_out_situations) | set(Config.model_fields["held_out_situations"].default)
    loaded: dict[str, tuple] = {}                  # the scenes of the records that are replayed, and no others
    rows, left_out = [], 0
    for record in _records(run_dir / "results.jsonl"):
        situation = (record.get("labels") or {}).get("situation")
        if situation is None or situation in held_out or HELD_OUT_SCENE.search(str(record.get("scene_id"))):
            left_out += 1
            continue
        scene_id, episode_id = record["scene_id"], record["episode_id"]
        if scene_id not in loaded:
            if allow is not None:
                allow(scene_id)
            inputs, truth = load_scene(folder / scene_id)
            if inputs.scene_id != scene_id:
                raise SystemExit(f"error: {folder / scene_id} holds the scene {inputs.scene_id}, not {scene_id}")
            loaded[scene_id] = (inputs.observation, truth, prepare(inputs, truth, cfg))
        observation, truth, truths = loaded[scene_id]
        gt = truths[episode_id].gt
        if gt.situation != situation:
            raise SystemExit(f"error: {scene_id} {episode_id}: recorded as {situation}, and the scene says {gt.situation}")
        result = GoalResult.model_validate(record["result"])
        row = {"run": run_dir.name, "scene": scene_id, "episode": episode_id, "trial": record.get("trial"),
               "situation": situation, "source": result.source, "status": result.status, "solver_step": False,
               "ok_recorded": bool(record["score"].get("success")),
               "ok_rescored": bool(score(result, observation, truth, gt, th).success),
               "place_recorded": None if result.pose_world is None else list(result.pose_world)}
        row.update(ok_replayed=row["ok_rescored"], place_replayed=row["place_recorded"], moved_m=0.0)
        if (result.source == "constraints" and result.pose_world is not None and result.people is not None
                and result.constraints is not None):
            solved, again = solve_again(result, observation, th)
            note = solved.note.model_dump()
            row.update(solver_step=True, ok_replayed=bool(score(again, observation, truth, gt, th).success),
                       place_replayed=list(solved.pose_world), moved_m=math.dist(solved.pose_world[:2], result.pose_world[:2]),
                       kinds=sorted({use.constraint.kind + ("(target)" if use.constraint.target else "")
                                     for use in solved.uses if use.status == "used"}),
                       keys=note["keys"], not_ahead_of=note.get("not_ahead_of"), not_ahead_reason=note.get("not_ahead_reason"),
                       joined_waiting=note["joined_waiting"],
                       target_distance_m=note["target_distance_m"],
                       flags=sorted(name for name, value in solved.flags.model_dump().items() if value))
        rows.append(row)
    return rows, left_out


def _counts(rows: list[dict], cell_m: float) -> list[int]:
    steps = [row for row in rows if row["solver_step"]]
    return [len(rows), len(steps), sum(row["moved_m"] <= SAME_PLACE_M for row in steps), sum(row["moved_m"] > cell_m for row in steps),
            sum(row["ok_recorded"] for row in rows), sum(row["ok_rescored"] for row in rows), sum(row["ok_replayed"] for row in rows),
            sum(row["ok_replayed"] and not row["ok_recorded"] for row in rows),
            sum(row["ok_recorded"] and not row["ok_replayed"] for row in rows)]


def _table(title: str, rows: list[dict], cell_m: float) -> None:
    """The counts of some rows, per situation and in all."""
    print(title)
    print(f"  {'situation':20s}" + "".join(f"{name:>13s}" for name in COUNTS))
    by_situation = collections.defaultdict(list)
    for row in rows:
        by_situation[row["situation"]].append(row)
    for situation, some in [*sorted(by_situation.items()), ("all", rows)]:
        print(f"  {situation:20s}" + "".join(f"{count:13d}" for count in _counts(some, cell_m)))


def _whitelisted(given: list[str], root: Path, wl=None) -> list[tuple[Path, Path, object]]:
    """With --whitelist: (run folder, its listed scenes folder, the check of a scene before it is opened) for each
    RUN_DIR[=SCENES_DIR] given, every run folder on research_runs.txt, every scenes folder the one that list gives for
    the run (as given or as its config.yaml records it, both resolved), and nothing held out or excluded where it is
    written or where its links lead (the run folder, its results.jsonl and its config.yaml, whether SCENES_DIR is
    given or not); otherwise an error that names the path refused. Only a listed run's config.yaml is read, and sgi
    is not imported here, so that --src still chooses it. The check of a scene guards it as research_harness does
    (guard_scene: what is in it and what its scene.json names)."""
    import yaml
    harness = _harness()
    wl = wl or harness.whitelist()
    listed = {ref.path: ref for ref in wl.runs}
    checked = []
    for argument in given:
        run_dir, _, scenes_dir = argument.partition("=")
        path = harness.guard(root / run_dir)
        ref = listed.get(harness.guard(path.resolve()))
        if ref is None:
            raise harness.Forbidden(f"error: refused, not on {harness.RUNS_FILE.name}: {path}")
        harness.guard_file(path / "results.jsonl")
        harness.guard_file(path / "config.yaml")   # replay() reads it, whichever form is given
        if scenes_dir:
            folder = root / scenes_dir
        else:
            doc = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))
            folder = _scenes_folder(path, doc, None)
        harness.check_scenes_dir(ref, folder)
        checked.append((ref.path, ref.scenes_dir, lambda scene_id, ref=ref: harness.check_scene(ref, scene_id, wl)))
    return checked


def main(argv=None) -> int:
    args = _arguments(argv)
    if args.whitelist:
        harness = _harness()
        if args.rows is not None and harness.guard(args.rows.resolve()).is_relative_to(harness.main_checkout()):
            raise harness.Forbidden(f"error: refused, --rows inside the repository: {args.rows.resolve()}")
        root = harness.main_checkout() if args.root is None else args.root
        given = _whitelisted(args.runs, root)
    else:
        root = Path.cwd() if args.root is None else args.root
        given = []
        for argument in args.runs:
            run_dir, _, scenes_dir = argument.partition("=")
            given.append((root / run_dir, root / scenes_dir if scenes_dir else None, None))
    sys.path.insert(0, str(args.src.resolve()))
    import sgi
    from sgi.config import Thresholds
    if Path(sgi.__file__).resolve().parents[1] != args.src.resolve():
        raise SystemExit(f"error: sgi was imported from {sgi.__file__}, not from {args.src}")
    cell_m = Thresholds().grid_resolution_m * math.sqrt(2.0)
    print(f"the solver of {Path(sgi.__file__).resolve().parents[1]}")
    every, left_out = [], 0
    for run_dir, scenes_dir, allow in given:
        rows, skipped = replay(run_dir, scenes_dir, allow)
        if not args.gate:
            _table(f"\n{Path(run_dir).name}" + (f"   ({skipped} records of a held-out or unknown situation, or of a "
                                                 f"held-out scene, left out)" if skipped else ""), rows, cell_m)
        every += rows
        left_out += skipped
    _table(f"\nall {len(args.runs)} runs" + (f"   ({left_out} records left out)" if left_out else ""), every, cell_m)
    lost = [row for row in every if row["ok_recorded"] and not row["ok_replayed"]]
    named = _accepted(args.accepted)
    known = [row for row in lost if _is_accepted(row, named)]
    print(f"\nsucceeded as recorded and fails as replayed: {len(lost) or 'none'}"
          + (f", of which {len(known)} named in {args.accepted.name}" if known else ""))
    for row in lost:
        was, now = row["place_recorded"], row["place_replayed"]
        print(f"  {row['run']} {row['scene']} {row['episode']} trial {row['trial']} ({row['situation']}): "
              f"({was[0]:.2f}, {was[1]:.2f}) -> ({now[0]:.2f}, {now[1]:.2f}), moved {row['moved_m']:.2f} m"
              + ("" if row["ok_rescored"] else "; the recorded place fails too as it is scored now")
              + ("; a loss that is known and stated" if row in known else ""))
    steps = [row for row in every if row["solver_step"]]
    apart = [row for row in steps if row["moved_m"] > SAME_PLACE_M]
    named_moved = _accepted(args.known_moved)
    known_moved = [row for row in apart if _is_accepted(row, named_moved)]
    print(f"replayed places that are the recorded ones: {len(steps) - len(apart)} of {len(steps)}"
          + (f"; the farthest of the others moved {max(row['moved_m'] for row in apart):.3f} m" if apart else "")
          + (f"; {len(known_moved)} of those named in {args.known_moved.name}" if args.known_moved else ""))
    for row in apart:
        print(f"  moved: {row['run']} {row['scene']} {row['episode']} trial {row['trial']} ({row['situation']}): "
              f"{row['moved_m']:.3f} m" + ("; known and stated" if row in known_moved else ""))
    if args.rows is not None:
        args.rows.write_text(json.dumps(every, indent=1), encoding="utf-8")
    return 1 if (args.expect_reproduced and len(apart) > len(known_moved)) or (args.gate and len(lost) > len(known)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
