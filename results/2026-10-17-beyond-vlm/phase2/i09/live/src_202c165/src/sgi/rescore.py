"""Scores the records of a finished run again (spec sections 10, 11 and 13).

    python -m sgi.rescore RUN_DIR [--scenes DIR]

A finished run folder holds config.yaml and results.jsonl. The command loads the scenes the run used, derives
their ground truth with the code as it is now under the configuration the run recorded, and scores every
recorded result against it. It writes RUN_DIR/rescored/ with results.jsonl and summary.csv in the formats of a
run, and note.txt, which says when that was, at which commit the code stood, and in how many records `success`
changed. Nothing else in the run folder is touched. No model is called and no client is built.

--scenes names the folder of the scenes, in place of the one config.yaml records. A scene that brings its own
acceptable regions keeps them here as it does in a run. The placeholder scenes are written with theirs
(sgi.synth.generate), so a change of the reference rules shows once they have been generated again.

Scored again in a record is what follows from the ground truth and the recorded result: `score`,
`hidden_relevant_people`, and in `measures` the reference of the context check with the two measures taken
against it. All else is carried over as it was recorded, a field this version of the code does not know
included; a field it knows that the record lacks is read as its default. No stage is run again. Where the
harness gave a stage its answer from the ground truth (`context_check: oracle`, `goal: oracle`), the recorded
result is therefore still the one made from the ground truth of the time.

Exit codes: 0 when rescored/ is written; 2, with one `error:` line and nothing written, for a folder that is
not a run's, a scenes folder that is missing or holds no scene, and a record whose scene or episode is not
among the scenes.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml
from pydantic import ValidationError

from sgi.config import REPO_ROOT, Config, ConfigError
from sgi.evaluate import match_target, score
from sgi.gt.derive import EpisodeTruth, prepare, reference_people_matter
from sgi.inputs import ObservationData
from sgi.run import _one_line, _scene_folders, summarise
from sgi.scene import GroundTruth, SceneError, load_scene, read_scene_file
from sgi.types import GoalResult

_OUT = "rescored"                                  # the folder written inside the run folder
_GIT = ["git", "rev-parse", "--short", "HEAD"]     # asked in the repository this code is in
_UNKNOWN = "unknown"                               # the commit, where git cannot say
_RECORD_KEYS = ("scene_id", "episode_id", "result", "score", "measures")     # what a record is read by here

# Three things below are sgi.run's own, and are written again here because sgi.run has each of them inline: the
# context measures of a result (_rescored), the writing of summary.csv (_rescore) and the last line printed
# (main). Until sgi.run offers them as functions, tests/test_rescore.py holds each against what sgi.run's own
# command gives for the same records: with and without a success rate, a cost and a model's context answer.


def _recorded_config(path: Path) -> tuple[dict, Config]:
    """A run's config.yaml as it was written, and the configuration it records.

    A ConfigError for a file that is missing or is not one a run wrote. A setting added since the run takes
    its default."""
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"{path}: {error}") from error
    if not isinstance(doc, dict) or not isinstance(doc.get("config"), dict):
        raise ConfigError(f"{path}: not the config.yaml of a run: it holds no configuration under `config`")
    try:
        return doc, Config.model_validate(doc["config"])
    except ValidationError as error:
        raise ConfigError(f"{path}: {error}") from error


def _records(path: Path) -> list[dict]:
    """The records of a run's results.jsonl, in order. A ConfigError names the file, and the line that is no record."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ConfigError(f"{path}: {error}") from error
    records = []
    for number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ConfigError(f"{path}: line {number}: not a record: {error}") from error
        missing = [key for key in _RECORD_KEYS if not isinstance(record, dict) or key not in record]
        if missing:
            raise ConfigError(f"{path}: line {number}: not a record of a run: it has no {', '.join(missing)}")
        records.append(record)
    return records


def _scene_dirs(folder: Path, setting: str) -> dict[str, Path]:
    """The scene folders under `folder` by scene id. Two with one id are refused, as a run refuses them."""
    found: dict[str, Path] = {}
    for scene_dir in _scene_folders(folder, setting):
        scene_id = read_scene_file(scene_dir).scene_id
        if scene_id in found:
            raise SceneError(f"{scene_id}: scene_id is used by two folders: {found[scene_id].name} and "
                             f"{scene_dir.name}")
        found[scene_id] = scene_dir
    return found


def _rescored(record: dict, observation: ObservationData, truth: GroundTruth, et: EpisodeTruth, cfg: Config) -> dict:
    """The record with its recorded result scored against the episode's ground truth as it is derived now."""
    result = GoalResult.model_validate(record["result"])
    ep, th, run = et.gt, cfg.thresholds, cfg.run
    scored = score(result, observation, truth, ep, th)
    # The reference is that of the target the context answer was about, chosen as run_episode chooses it: the
    # target the result's target point falls on, and the episode's first one under oracle grounding, for a
    # point on no target and for a result without one.
    chosen = None
    if run.grounding != "oracle" and result.target is not None:
        chosen = match_target(result.target.world, truth, ep, th)
    reference = reference_people_matter(et, chosen)
    said = result.context_check if run.context_check == "vlm" else None
    measures = {**record["measures"], "reference_people_matter": reference,
                "context_missed": None if said is None else (not said.people_matter and reference),
                "context_unnecessary": None if said is None else (said.people_matter and not reference)}
    return {**record, "hidden_relevant_people": len(et.hidden_relevant_people),
            "score": {**record["score"], **scored.model_dump(mode="json")}, "measures": measures}


def _commit() -> str:
    """The short hash of the commit this code stands at, or "unknown" where git cannot say."""
    try:
        done = subprocess.run(_GIT, cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return _UNKNOWN
    return done.stdout.strip() or _UNKNOWN


def _changes(count: int, gained: int, lost: int) -> str:
    """How many of `count` records changed their success, and which way."""
    return f"success changed in {gained + lost} of {count} records ({gained} now a success, {lost} now a failure)"


def _rescore(run_dir: Path, scenes_dir: Path | None) -> tuple[Path, int, int]:
    """`rescore`, with the number of records it read and the number whose success changed. Everything that can
    refuse comes before the folder is made."""
    config_path = run_dir / "config.yaml"
    doc, cfg = _recorded_config(config_path)
    records = _records(run_dir / "results.jsonl")
    if scenes_dir is not None:
        folder, setting = Path(scenes_dir), "--scenes"
    elif isinstance(doc.get("scenes_dir"), str):
        folder, setting = Path(doc["scenes_dir"]), f"scenes_dir of {config_path} (--scenes names another folder)"
    else:
        raise ConfigError(f"{config_path}: it records no scenes_dir; --scenes names the folder of the scenes")
    scene_dirs = _scene_dirs(folder, setting)

    loaded: dict[str, tuple[ObservationData, GroundTruth, dict[str, EpisodeTruth]]] = {}     # the scenes of the records
    rescored = []
    for number, record in enumerate(records, start=1):
        scene_id, episode_id = record["scene_id"], record["episode_id"]
        if scene_id not in scene_dirs:
            raise SceneError(f"{scene_id}: a record of {run_dir / 'results.jsonl'} is of this scene, and {folder} "
                             "holds no scene of that id (--scenes names another folder)")
        if scene_id not in loaded:
            inputs, truth = load_scene(scene_dirs[scene_id])
            loaded[scene_id] = (inputs.observation, truth, prepare(inputs, truth, cfg))
        observation, truth, truths = loaded[scene_id]
        if episode_id not in truths:
            raise SceneError(f"{scene_id}: episode {episode_id}: a record of {run_dir / 'results.jsonl'} is of this "
                             f"episode, and the scene in {folder} has none of that id")
        try:
            rescored.append(_rescored(record, observation, truth, truths[episode_id], cfg))
        except ValidationError as error:           # a result that is not one: the file was not written by a run
            raise ConfigError(f"{run_dir / 'results.jsonl'}: line {number}: {error}") from error

    was = [bool(record["score"].get("success")) for record in records]
    now = [record["score"]["success"] for record in rescored]
    gained = sum(new and not old for old, new in zip(was, now))
    lost = sum(old and not new for old, new in zip(was, now))

    out = run_dir / _OUT
    out.mkdir(exist_ok=True)                       # an earlier rescoring is written over
    with open(out / "results.jsonl", "w", encoding="utf-8") as lines:
        lines.writelines(json.dumps(record) + "\n" for record in rescored)
    rows = summarise(rescored)
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as table:      # as sgi.run writes its own
        writer = csv.DictWriter(table, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "note.txt").write_text(
        f"rescored {time.strftime('%Y-%m-%d %H:%M:%S %z')} with the code at commit {_commit()}\n"
        f"scenes: {folder.resolve()}\n"
        f"{_changes(len(records), gained, lost)}\n", encoding="utf-8")
    return out, len(records), gained + lost


def rescore(run_dir: Path, scenes_dir: Path | None = None) -> Path:
    """Score the records of the finished run in `run_dir` again and write `run_dir/rescored/`, which is returned.

    The scenes are read from `scenes_dir`, or from the folder the run's config.yaml records. The folder written
    holds results.jsonl and summary.csv in the formats of a run, and note.txt. A ConfigError (a folder that is
    not a run's, no scenes) or a SceneError (a record whose scene or episode is not among the scenes) is raised
    before anything is written.
    """
    return _rescore(Path(run_dir), scenes_dir)[0]


def main(argv=None) -> int:
    """The command: rescore, then print the line sgi.run prints, with the number of records whose success
    changed after it. Returns the exit code."""
    parser = argparse.ArgumentParser(prog="python -m sgi.rescore",
                                     description="Score the records of a finished run again, with no model call.")
    parser.add_argument("run_dir", type=Path, metavar="RUN_DIR", help="a run folder: config.yaml and results.jsonl")
    parser.add_argument("--scenes", type=Path, metavar="DIR",
                        help="folder of the scenes the run used, in place of the one its config.yaml records")
    args = parser.parse_args(argv)
    try:
        out, count, changed = _rescore(args.run_dir, args.scenes)
    except (ConfigError, SceneError) as error:
        print(f"error: {_one_line(error)}", file=sys.stderr)
        return 2
    with open(out / "summary.csv", newline="", encoding="utf-8") as table:
        overall = next(csv.DictReader(table))
    rate = f"{float(overall['success_rate']):.3f}" if overall["success_rate"] else "n/a"     # no answered record
    cost = f", about ${float(overall['cost_usd']):.2f}" if overall["cost_usd"] else ""       # empty: no prices known
    print(f"{out}: success rate {rate} ({overall['records']} records, {overall['vlm_errors']} ended in vlm_error; "
          f"{overall['calls']} calls, {overall['cached_calls']} from the cache{cost}); "
          f"success changed in {changed} of {count} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
