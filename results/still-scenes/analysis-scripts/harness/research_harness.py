"""The shared offline harness of the replay analyses. No model is called, no
network used, and nothing is written.

Two written lists beside this file say what may be read: research_runs.txt, the recorded runs (RUN_DIR=SCENES_DIR
from the repository root, where the runs and the scenes are local), and research_scenes.txt,
the development scenes (SCENES_DIR/SCENE_ID): those the runs were evaluated on, and the worked-example scenes, which
are those of a folder named `examples`. A run or a scene that is not on its list is refused, and so is every path or
name that is held out or excluded, whatever list names it (`guard`, case-insensitive, in every component of a path,
as written and where its links lead): an s18 or s20_held run (stage18 and s18b alike), anything with `heldout` or
`held_out` in a component, a `.cache` or `.env` component, an s21_replay or s21_sonnet probe, an s22_replay run, a
pre-flight `pf_` run, and the held-out generated test scenes eval_16 to eval_19. Before a scene is loaded, every file
and folder in it and every file its scene.json names are guarded, and a named file must lie in the scene's folder.
Inside a listed run, a record of a held-out or unknown situation, or of a held-out scene, is skipped and counted
before any of its other fields is used, as replay_solver.py does: held out is every situation of the run's recorded
`held_out_situations` and of that list as the imported code has it by default.

Count or look at runs through this tool, never with a glob or a find over the repository: `--no-rescore
--no-resolve` prints every listed run's records, allowed and skipped, in seconds.

    runs(stage=, arm=, model=, method=)  the listed runs, as RunRef: path, folder, name, stage, arm, model, method,
                                         scenes_dir. model is the model entry without its provider (glm53,
                                         qwen38_low); method is the name's last word (cot, fsc, fsc3, con; for s8
                                         none, convlm or conorc); arm is real, staffed, plain, ph or dev_seed0..2
                                         (an s17 name's seedN is dev_seedN). The stages are those the list holds.
    records(run, tally=)                 the allowed records of a run, as Record; `tally` counts allowed and skipped.
    scene(run, scene_id)                 (inputs, observation, truth, truths) as sgi loads and derives them.
    distinct_scenes()                    each listed evaluation scene once, links resolved, as SceneRef: folder,
                                         scenes_dir, scene_id (the listing it is known by), listed (every listing
                                         that leads to it), runs (the listed runs that load it with scene()).
    fingerprint(folder)                  the sha256 of a scene folder's content; one scene in two places has one.
    examples(examples_dir=)              the listed worked-example scenes, as (examples folder, scene id).
    run_examples(run)                    the worked examples a listed run showed: (examples folder, scene id, episode).
    example(scene_id, examples_dir=, run=, cfg=)   a listed worked-example scene, loaded without a run, its truths
                                         derived under a run's recorded config, cfg, or the default config.
    score_pose(rec, (x, y, theta), target=)   the package's score of the record's result with that world pose, and
    score_poses(rec, poses, target=)          that target if one is given, put in; many poses share one freespace.
    target_at(rec, point=|pixel=|world=) a target grounded as the pipeline grounds one (GroundedTarget).
    resolve(rec, target=, people=, constraints=)  the solver of the imported code run again on the record, with the
                                         recorded inputs or those given, and the package's score of its place.
    rescore(rec)                         the package's score of the recorded result, now.

As a script it prints, per listed run, the records allowed and skipped, the solver steps, the recorded pose of
every allowed record scored again now against its recorded success (a mismatch can be legitimate where the scorer
or a scene changed since the run; they are printed, not failed), the same pose through score_pose against rescore
(they must agree), and the reproduction check: every solver step solved again by the solver of the imported code
must give the recorded place, or be named with its reason in research_moved.txt. It also checks the worked
examples: every one a listed run showed is on the list of example scenes, and every listed one loads; and it
counts the listed scenes once each (distinct_scenes), and checks that no two of them, nor two worked examples, are
one scene by content. Exit code 1 when the check fails, score_pose and rescore disagree, a worked example shown has
no development truth, or two listed scenes are one by content:

    PYTHONPATH=src python results/still-scenes/analysis-scripts/harness/research_harness.py [--stage S] [--arm A] [--model M]
                                                [--method K] [--no-rescore] [--no-resolve] [--moved FILE]
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import dataclasses
import functools
import importlib.util
import math
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECKOUT = HERE.parents[3]                         # results/still-scenes/analysis-scripts/harness lies four levels below the root
RUNS_FILE = HERE / "research_runs.txt"
SCENES_FILE = HERE / "research_scenes.txt"
MOVED_FILE = HERE / "research_moved.txt"
STAGE = re.compile(r"s[0-9]+[a-z]*")             # the first word of a run's name: s8, s10, s16b, s22 ...
ARMS = ("real", "staffed", "plain", "ph", "dev_seed0", "dev_seed1", "dev_seed2")
SEED_ARM = re.compile(r"seed[0-9]+")                # s17 names a development seed seedN, the others dev_seedN
EXAMPLES_FOLDER = "examples"                       # a listed scene of a folder of this name is a worked example
SAME_PLACE_M = 1e-6                                # a place solved again this near the recorded one is the recorded one
# Held out or excluded wherever they stand in a name or a path: each is matched, ignoring case, in every component.
FORBIDDEN = {
    "an s18 run": re.compile(r"(?:^|_)s(?:tage)?18(?![0-9])", re.I),
    "an s20_held run": re.compile(r"(?:^|_)s(?:tage)?20[a-z]*_held", re.I),
    "held out": re.compile(r"held_?out", re.I),
    "an s21_replay probe": re.compile(r"(?:^|_)s21_replay", re.I),
    "an s21_sonnet probe": re.compile(r"(?:^|_)s21_sonnet", re.I),
    "an s22_replay run": re.compile(r"(?:^|_)s22_replay", re.I),
    "a pre-flight run": re.compile(r"^(?:[0-9]{8}-[0-9]{6}_)?pf_", re.I),
    "a held-out scene (eval_16 to eval_19)": re.compile(r"(?:^|_)eval_1[6-9](?![0-9])", re.I),
}
HELD_OUT_SCENE = FORBIDDEN["a held-out scene (eval_16 to eval_19)"]
# The recorded runs' config.yaml files name the development scene sets by the folder that held them when the runs
# were made. In this repository those sets lie in data/dev_scenes/ (the dev-scenes-v1 release), one to one.
RECORDED_SETS = "/keep/"
DEV_SCENES = "data/dev_scenes/"
SCENE_CACHE = 24                                   # scenes kept loaded, the least recently used dropped first


class Forbidden(SystemExit):
    """A path or a name that is held out, excluded, or not on the written lists."""


def forbidden(path_or_name) -> str | None:
    """What makes a path or a name held out or excluded, or None when nothing does."""
    for part in re.split(r"[/\\]", str(path_or_name)):
        low = part.lower()
        if low in (".cache", ".env") or low.startswith(".env.") or low.endswith(".env"):
            return f"the component {part}"
        for what, pattern in FORBIDDEN.items():
            if pattern.search(part):
                return what
    return None


def guard(path_or_name):
    """The path or name as given; Forbidden for one that is held out or excluded."""
    why = forbidden(path_or_name)
    if why is not None:
        raise Forbidden(f"error: refused, {why}: {path_or_name}")
    return path_or_name


def guard_file(path):
    """A path guarded as written and where its links lead; the path as given."""
    guard(path)
    guard(Path(path).resolve())
    return path


def guard_tree(folder):
    """A folder guarded with everything in it, as written and where every link leads (links to folders are
    followed, once each); the folder as given. Only names are read."""
    guard_file(folder)
    seen = set()
    for top, dirs, files in os.walk(Path(folder).resolve(), followlinks=True):
        real = Path(top).resolve()
        if real in seen:
            dirs[:] = []
            continue
        seen.add(real)
        for name in dirs + files:
            guard_file(Path(top) / name)
    return folder


def relocated(recorded) -> str:
    """A scenes or examples folder as a run's config.yaml records it, as it lies in this repository."""
    text = str(recorded)
    return DEV_SCENES + text.split(RECORDED_SETS, 1)[1] if RECORDED_SETS in text else text


def main_checkout(checkout: Path = CHECKOUT) -> Path:
    """The repository root that holds the runs and the scenes (for a linked checkout, the main repository's root)."""
    dot_git = checkout / ".git"
    if dot_git.is_file():
        text = dot_git.read_text(encoding="utf-8").strip()
        if text.startswith("gitdir:"):
            return Path(text.partition(":")[2].strip()).resolve().parents[2]
    return checkout


def _module(name: str):
    """A tool beside this file, loaded once under its own name."""
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@dataclasses.dataclass(frozen=True)
class RunRef:
    path: Path                                     # the run folder, absolute
    folder: str                                    # its name, with the time it starts with
    name: str                                      # without the time
    stage: str
    arm: str
    model: str
    method: str
    scenes_dir: Path                               # the scenes folder the list gives, absolute


@dataclasses.dataclass(frozen=True)
class Whitelist:
    repo: Path
    runs: tuple[RunRef, ...]
    scenes: frozenset[tuple[Path, str]]            # (scenes folder, scene id) of the scenes the runs use
    examples: frozenset[tuple[Path, str]] = frozenset()   # (examples folder, scene id) of the worked examples


def describe(name: str) -> tuple[str, str, str, str]:
    """(stage, arm, model, method) of a run's name without its time. s8 and s10 ran on the generated test scenes; an
    s17 name's seedN is the arm dev_seedN. Which stages are taken is the list's to say, not this function's."""
    stage, _, rest = guard(name).partition("_")
    if not STAGE.fullmatch(stage) or rest.count("_") < 1:
        raise Forbidden(f"error: refused, not the name of a run of a stage: {name}")
    if stage == "s8":                              # s8_<none|convlm|conorc>_<model>
        method, _, model = rest.partition("_")
        return stage, "ph", model, method
    rest, _, method = rest.rpartition("_")
    first = rest.partition("_")[0]
    if SEED_ARM.fullmatch(first):                  # s17_seedN_<model>_<method>
        return stage, "dev_" + first, rest.partition("_")[2], method
    arm = next((arm for arm in ARMS if rest.startswith(arm + "_")), None)
    if arm is None:                                # s10_<model>_<effort>_<method>: effort none is the bare entry
        return stage, "ph", rest.removesuffix("_none"), method
    return stage, arm, rest[len(arm) + 1:], method


def _lines(path: Path) -> list[str]:
    """The lines of a written list that are not empty and not a comment, each guarded."""
    guard_file(path)
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [guard(line) for line in lines if line and not line.startswith("#")]


def _absolute(repo: Path, given: str) -> Path:
    """A listed path from the repository root, absolute and guarded as written and as resolved."""
    path = Path(guard(given))
    path = path if path.is_absolute() else repo / path
    guard(path)
    return guard(path.resolve())


@functools.lru_cache(maxsize=None)
def whitelist(repo: Path | None = None, runs_file: Path = RUNS_FILE, scenes_file: Path = SCENES_FILE) -> Whitelist:
    """The two written lists, read once. By default from the root of the repository this file lies in."""
    repo = main_checkout() if repo is None else Path(repo)
    refs = []
    for line in _lines(Path(runs_file)):
        run_dir, sep, scenes_dir = line.partition("=")
        if not sep:
            raise SystemExit(f"error: {runs_file}: RUN_DIR=SCENES_DIR is expected, not {line}")
        path = _absolute(repo, run_dir)
        folder = path.name
        name = folder.partition("_")[2]
        refs.append(RunRef(path, folder, name, *describe(name), _absolute(repo, scenes_dir)))
    scenes, examples = set(), set()
    for line in _lines(Path(scenes_file)):          # a scene folder may be a link to another: its own folder counts
        _absolute(repo, line)
        kind = examples if Path(line).parent.name == EXAMPLES_FOLDER else scenes
        kind.add((_absolute(repo, str(Path(line).parent)), Path(line).name))
    return Whitelist(repo, tuple(refs), frozenset(scenes), frozenset(examples))


def runs(stage=None, arm=None, model=None, method=None, *, wl: Whitelist | None = None) -> list[RunRef]:
    """The listed runs, in the order of the list, that match every filter given."""
    wl = wl or whitelist()
    wanted = {"stage": stage, "arm": arm, "model": model, "method": method}
    return [ref for ref in wl.runs if all(value is None or getattr(ref, key) == value for key, value in wanted.items())]


def run_ref(run, wl: Whitelist | None = None) -> RunRef:
    """A listed run by its RunRef, its folder's name, its name or its path; Forbidden for any run not listed."""
    wl = wl or whitelist()
    if isinstance(run, RunRef):
        if run in wl.runs:
            return run
        raise Forbidden(f"error: refused, not on the list of runs: {run.path}")
    text = str(guard(run))
    found = [ref for ref in wl.runs if text in (ref.folder, ref.name)]
    if not found:
        path = Path(text) if Path(text).is_absolute() else wl.repo / text
        resolved = guard(guard(path).resolve())
        found = [ref for ref in wl.runs if ref.path == resolved]
    if len(found) != 1:
        raise Forbidden(f"error: refused, not on the list of runs: {text}")
    return found[0]


def check_scenes_dir(ref: RunRef, folder) -> Path:
    """The listed scenes folder of a listed run when `folder` is it (resolved); Forbidden otherwise."""
    guard(folder)
    if guard(Path(folder).resolve()) != ref.scenes_dir:
        raise Forbidden(f"error: refused, {folder} is not the scenes folder of {ref.folder} on {RUNS_FILE.name}: "
                        f"{ref.scenes_dir}")
    return ref.scenes_dir


def guard_scene(folder) -> Path:
    """A scene folder, guarded before it is loaded: everything in it (guard_tree, names only), and every file its
    scene.json names (the images, the depth, the tour and the occupancy), as written and where its links lead,
    which must lie in the scene's folder. scene.json is read only once it is guarded itself."""
    from sgi.scene import read_scene_file
    guard_tree(folder)
    scene = read_scene_file(Path(folder))
    named = [scene.observation.rgb, scene.observation.depth, *(frame.rgb for frame in scene.tour)]
    if scene.ground_truth.occupancy is not None:
        named.append(scene.ground_truth.occupancy.image)
    inside = Path(folder).resolve()
    for name in named:
        path = guard_file(Path(folder) / guard(name))
        if not path.resolve().is_relative_to(inside):
            raise Forbidden(f"error: refused, {Path(folder) / 'scene.json'} names a file outside its folder: {name}")
    return Path(folder)


def check_scene(ref: RunRef, scene_id, wl: Whitelist | None = None) -> Path:
    """The folder of a scene of a listed run, when the scene is on the list of scenes for the run's scenes folder
    and nothing in it or named by it is held out or excluded (guard_scene); Forbidden otherwise."""
    wl = wl or whitelist()
    folder = guard(ref.scenes_dir / guard(str(scene_id)))
    if (ref.scenes_dir, scene_id) not in wl.scenes:
        raise Forbidden(f"error: refused, not on the list of scenes: {folder}")
    return guard_scene(folder)


@dataclasses.dataclass(frozen=True)
class SceneRef:
    folder: Path                                   # the scene's own folder, every link resolved
    scenes_dir: Path                               # the listing it is known by: the folder it lies in, where that
    scene_id: str                                  # one is listed (else its first listing)
    listed: tuple[tuple[Path, str], ...]           # every (scenes folder, scene id) on the list that leads to it
    runs: tuple[RunRef, ...]                       # the listed runs evaluated from one of those, in the list's order


def distinct_scenes(*, wl: Whitelist | None = None) -> list[SceneRef]:
    """Each listed evaluation scene once, however many listings lead to it: keyed by its folder with every link
    resolved, so the 14 links of data/dev_scenes/placeholder_first14 and the 14 generated test scenes they lead to are 14 scenes,
    not 28. A scene-level pass over the list (a re-run of a method on the development scenes) takes these. Each
    run of `runs` loads the scene with scene(run, scene_id); listings of one scene by different names are refused.
    Sorted by the listing each is known by. Paths only: nothing is opened (fingerprint() compares contents)."""
    wl = wl or whitelist()
    by_folder: dict[Path, list] = collections.defaultdict(list)
    for key in sorted(wl.scenes):
        by_folder[guard(guard(key[0] / key[1]).resolve())].append(key)
    found = []
    for folder, keys in by_folder.items():
        if len({scene_id for _, scene_id in keys}) != 1:
            raise SystemExit(f"error: {folder} is listed under more than one name: {keys}")
        own = (folder.parent, folder.name)
        scenes_dir, scene_id = own if own in keys else keys[0]
        dirs = {scenes_dir for scenes_dir, _ in keys}
        found.append(SceneRef(folder, scenes_dir, scene_id, tuple(keys),
                              tuple(ref for ref in wl.runs if ref.scenes_dir in dirs)))
    return sorted(found, key=lambda one: (str(one.scenes_dir), one.scene_id))


def fingerprint(folder) -> str:
    """The sha256 of a scene folder's content: every file in it by its path in the folder and its bytes, links
    followed (each linked folder once). Guarded first, as a scene is before it is loaded (guard_scene). Two scenes
    with one fingerprint are one scene in two places."""
    import hashlib
    guard_scene(folder)
    root = Path(folder).resolve()
    digest, seen = hashlib.sha256(), set()
    for top, dirs, files in os.walk(root, followlinks=True):
        real = Path(top).resolve()
        if real in seen:
            dirs[:] = []
            continue
        seen.add(real)
        dirs.sort()
        for name in sorted(files):
            path = Path(top) / name
            data = path.read_bytes()
            digest.update(f"{path.relative_to(root).as_posix()}\0{len(data)}\0".encode())
            digest.update(data)
    return digest.hexdigest()


def examples(examples_dir=None, *, wl: Whitelist | None = None) -> list[tuple[Path, str]]:
    """The listed worked-example scenes, as (examples folder, scene id), sorted; of one folder where it is given."""
    wl = wl or whitelist()
    wanted = None if examples_dir is None else guard(Path(guard(examples_dir)).resolve())
    return sorted(key for key in wl.examples if wanted is None or key[0] == wanted)


def check_example(scene_id, examples_dir=None, wl: Whitelist | None = None) -> Path:
    """The folder of a listed worked-example scene, of `examples_dir` where it is given (a scene id is listed in one
    examples folder only otherwise), guarded as check_scene guards a scene; Forbidden for any other."""
    wl = wl or whitelist()
    folder, _ = _example_key(scene_id, examples_dir, wl)
    return guard_scene(guard(folder / scene_id))


def _example_key(scene_id, examples_dir, wl: Whitelist) -> tuple[Path, str]:
    """(examples folder, scene id) of a listed worked-example scene; Forbidden for a scene not listed."""
    guard(str(scene_id))
    found = [key for key in examples(examples_dir, wl=wl) if key[1] == scene_id]
    if len(found) != 1:
        raise Forbidden(f"error: refused, not on the list of example scenes"
                        f"{'' if examples_dir is None else ' of ' + str(examples_dir)}: {scene_id}")
    return found[0]


@functools.lru_cache(maxsize=None)
def recorded_config(ref: RunRef):
    """(config.yaml as written, the Config it records) of a listed run."""
    from sgi.rescore import _recorded_config
    return _recorded_config(guard_file(ref.path / "config.yaml"))


def held_out(ref: RunRef) -> frozenset[str]:
    """The situations held out for a run: by its recorded list and by the list as it is by default now."""
    from sgi.config import Config
    return frozenset(recorded_config(ref)[1].held_out_situations) | frozenset(Config.model_fields["held_out_situations"].default)


TIMING_FIELDS = ("calls", "cached", "latency_s", "latency_total_s", "transient_retries", "validation_retries",
                 "input_tokens", "output_tokens", "reasoning_tokens", "fresh_latency_s", "fresh_latency_total_s",
                 "fresh_input_tokens", "fresh_output_tokens", "fresh_reasoning_tokens")


def timings(calls) -> dict[str, dict]:
    """Per stage of a record's calls, summed over its calls; each `fresh_` figure over the calls not cached.

    latency_s is the call's last back-end attempt alone: no wait for a turn of the pacer, no retry, no earlier
    attempt. latency_total_s is the call's whole time from the client's start, the pacer's waits and the retries'
    sleeps in it: the time the package puts into a stage (pipeline `_timed`). A cached call carries both latencies,
    its retries and its tokens from the call that first gave it. reasoning_tokens are billed as output tokens.
    measures.latency_s of a record is neither sum: it is the sum of result.timings_s (Record.timings_s), the stages'
    wall time with each call's latency_total_s in place of the time `generate` took, the geometry included. The people
    and constraints calls have no key of their own there: they count under goal."""
    out: dict[str, dict] = {}
    for call in calls or []:
        stage = out.setdefault(call["stage"], dict.fromkeys(TIMING_FIELDS, 0))
        usage = call.get("usage") or {}
        cached = bool(call.get("cached"))
        figures = {"latency_s": call.get("latency_s") or 0.0, "latency_total_s": call.get("latency_total_s") or 0.0,
                   "input_tokens": usage.get("input_tokens") or 0, "output_tokens": usage.get("output_tokens") or 0,
                   "reasoning_tokens": usage.get("reasoning_tokens") or 0}
        stage["calls"] += 1
        stage["cached"] += cached
        stage["transient_retries"] += call.get("transient_retries") or 0
        stage["validation_retries"] += call.get("validation_retries") or 0
        for key, value in figures.items():
            stage[key] += value
            stage["fresh_" + key] += 0 if cached else value
    return out


@dataclasses.dataclass
class Record:
    ref: RunRef
    scene_id: str
    episode_id: str
    trial: int | None
    situation: str
    labels: dict
    result: dict                                   # the recorded GoalResult, as written
    score: dict                                    # the recorded score
    raw: dict = dataclasses.field(repr=False)      # the whole record
    timings: dict = dataclasses.field(repr=False)  # per stage of its calls (timings())
    wl: Whitelist = dataclasses.field(repr=False)

    run = property(lambda self: self.ref.folder)
    arm = property(lambda self: self.ref.arm)
    model = property(lambda self: self.ref.model)
    method = property(lambda self: self.ref.method)
    target = property(lambda self: self.result.get("target"))
    people = property(lambda self: self.result.get("people"))
    constraints = property(lambda self: self.result.get("constraints"))
    candidates = property(lambda self: self.result.get("candidates"))
    solver = property(lambda self: self.result.get("solver"))
    pose_world = property(lambda self: self.result.get("pose_world"))
    measures = property(lambda self: self.raw.get("measures") or {})
    timings_s = property(lambda self: self.result.get("timings_s") or {},
                         doc="per stage wall time as the run recorded it (plan, locate, context, goal, geometry)")
    latency_s = property(lambda self: self.measures.get("latency_s"), doc="measures.latency_s: sum of timings_s")

    @property
    def solver_step(self) -> bool:
        """Whether the record holds a goal step of the constraint method's solver, as replay_solver.py counts one."""
        r = self.result
        return (r.get("source") == "constraints" and r.get("pose_world") is not None and r.get("people") is not None
                and r.get("constraints") is not None)


def records(run, *, wl: Whitelist | None = None, tally: collections.Counter | None = None):
    """The allowed records of a listed run, in order. A record of a held-out or unknown situation, of a held-out
    scene, or of a scene not on the list is skipped before any other field of it is used; `tally` counts each. A run
    not on the list is refused here, not at the first record."""
    wl = wl or whitelist()
    ref = run_ref(run, wl)
    return _allowed(ref, wl, held_out(ref), collections.Counter() if tally is None else tally)


def _allowed(ref: RunRef, wl: Whitelist, held: frozenset[str], tally: collections.Counter):
    from sgi.rescore import _records
    for raw in _records(guard_file(ref.path / "results.jsonl")):
        situation = (raw.get("labels") or {}).get("situation")
        if situation is None or situation in held:
            tally["skipped: held-out or unknown situation"] += 1
            continue
        scene_id = raw.get("scene_id")
        if not isinstance(scene_id, str) or forbidden(scene_id) is not None:
            tally["skipped: held-out scene"] += 1
            continue
        if (ref.scenes_dir, scene_id) not in wl.scenes:
            tally["skipped: scene not listed"] += 1
            continue
        tally["allowed"] += 1
        result = raw["result"]
        yield Record(ref, scene_id, raw["episode_id"], raw.get("trial"), situation, raw["labels"], result,
                     raw["score"], raw, timings(result.get("calls")), wl)


Scene = collections.namedtuple("Scene", "inputs observation truth truths")
_SCENES: collections.OrderedDict = collections.OrderedDict()     # (scenes folder, scene id) -> (inputs, truth, {})


def _loaded(ref: RunRef, scene_id: str, wl: Whitelist):
    """(inputs, truth, per thresholds: {"truths", "reach"}) of a listed scene of a listed run, loaded once."""
    key = (ref.scenes_dir, scene_id)
    return _load(key, key in wl.scenes, lambda: check_scene(ref, scene_id, wl))


def _load(key: tuple[Path, str], listed: bool, check):
    """(inputs, truth, per thresholds: {"truths", "reach"}) of the scene `key` names, loaded once; `check` gives
    its folder, guarded, or refuses it, and is called whenever the scene is not listed on the list in use or not
    loaded yet."""
    from sgi.scene import load_scene
    scene_id = key[1]
    if not listed or key not in _SCENES:           # listed for this list, whatever list loaded it before
        folder = check()
    if key not in _SCENES:
        inputs, truth = load_scene(folder)
        if inputs.scene_id != scene_id:
            raise SystemExit(f"error: {folder} holds the scene {inputs.scene_id}, not {scene_id}")
        _SCENES[key] = (inputs, truth, {})
        while len(_SCENES) > SCENE_CACHE:
            _SCENES.popitem(last=False)
    _SCENES.move_to_end(key)
    return _SCENES[key]


def _derived(ref: RunRef, scene_id: str, wl: Whitelist) -> tuple:
    """(inputs, truth, the run's thresholds, the per-thresholds cache) of a listed scene of a listed run."""
    from sgi.gt.derive import prepare
    run_ref(ref, wl)
    inputs, truth, derived = _loaded(ref, scene_id, wl)
    cfg = recorded_config(ref)[1]
    cache = derived.setdefault(cfg.thresholds.model_dump_json(), {})
    if "truths" not in cache:
        cache["truths"] = prepare(inputs, truth, cfg)
    return inputs, truth, cfg.thresholds, cache


def scene(run, scene_id: str, *, wl: Whitelist | None = None) -> Scene:
    """A listed scene of a listed run, loaded as replay_solver.py loads it, with its episodes' truths derived under
    the run's recorded thresholds. Loaded once per (scenes folder, scene id); derived once per thresholds."""
    wl = wl or whitelist()
    ref = run_ref(run, wl)
    inputs, truth, _, cache = _derived(ref, scene_id, wl)
    return Scene(inputs, inputs.observation, truth, cache["truths"])


def _examples_dir(ref: RunRef, wl: Whitelist) -> Path | None:
    """The examples folder a listed run's config.yaml records, absolute (a relative one from the repository root,
    as sgi.run takes it) and guarded as written and resolved; None where it records none."""
    recorded = recorded_config(ref)[1].run.examples_dir
    if recorded is None:
        return None
    path = Path(guard(relocated(recorded)))
    return guard(guard(path if path.is_absolute() else wl.repo / path).resolve())


def run_examples(run, *, wl: Whitelist | None = None) -> list[tuple[Path, str, str]]:
    """The worked examples a listed run showed, as its config.yaml records them: (examples folder, scene id,
    episode id), in order; each scene on the list of example scenes, Forbidden otherwise. None shown, none given."""
    wl = wl or whitelist()
    ref = run_ref(run, wl)
    shown = []
    for example_id in recorded_config(ref)[0].get("example_ids") or []:
        scene_id, _, episode_id = guard(str(example_id)).partition("/")
        folder = _examples_dir(ref, wl)
        if folder is None or (folder, scene_id) not in wl.examples:
            raise Forbidden(f"error: refused, {ref.folder} shows {example_id} of {folder}, not on the list of "
                            f"example scenes")
        shown.append((folder, scene_id, episode_id))
    return shown


def example(scene_id: str, examples_dir=None, *, run=None, cfg=None, wl: Whitelist | None = None) -> Scene:
    """A listed worked-example scene, loaded without a run as sgi loads it, with the truths of its episodes derived
    under the recorded config of `run` (a listed run), under `cfg`, or under the default config; an episode of a
    held-out situation (of that config's list and of the default one) is left out of the truths. Loaded once."""
    from sgi.config import Config, load_config
    from sgi.gt.derive import prepare
    wl = wl or whitelist()
    if run is not None:
        if cfg is not None:
            raise ValueError("example takes run or cfg, not both")
        cfg = recorded_config(run_ref(run, wl))[1]
    cfg = load_config() if cfg is None else cfg
    key = _example_key(scene_id, examples_dir, wl)
    inputs, truth, derived = _load(key, True, lambda: check_example(scene_id, examples_dir, wl))
    held = frozenset(cfg.held_out_situations) | frozenset(Config.model_fields["held_out_situations"].default)
    cache = derived.setdefault(cfg.thresholds.model_dump_json(), {})
    if "truths" not in cache:
        cache["truths"] = prepare(inputs, truth, cfg)
    truths = {episode: et for episode, et in cache["truths"].items() if et.gt.situation not in held}
    return Scene(inputs, inputs.observation, truth, truths)


def _against(rec: Record):
    """(scene, episode ground truth, thresholds, the per-thresholds cache) a record is scored against."""
    inputs, truth, th, cache = _derived(rec.ref, rec.scene_id, rec.wl)
    gt = cache["truths"][rec.episode_id].gt
    if gt.situation != rec.situation:
        raise SystemExit(f"error: {rec.scene_id} {rec.episode_id}: recorded as {rec.situation}, and the scene says {gt.situation}")
    return Scene(inputs, inputs.observation, truth, cache["truths"]), gt, th, cache


@contextlib.contextmanager
def _reach_once(sc: Scene, th, cache: dict):
    """The package's scorer with the robot's freespace of one scene computed once: inside, sgi.evaluate's
    robot_grid and reachable give the very values they compute, from a cache, for this scene's truth, robot and
    thresholds, and compute them as ever for anything else."""
    import sgi.evaluate as evaluate
    real = evaluate.freespace
    robot = sc.observation.robot
    if "reach" not in cache:
        grid = real.robot_grid(sc.truth, robot, th)
        cache["reach"] = (grid, real.connected(grid, (robot.x, robot.y)))
    grid, joined = cache["reach"]

    class Cached:
        def __getattr__(self, name):
            return getattr(real, name)

        @staticmethod
        def robot_grid(truth, at, thresholds):
            if truth is sc.truth and at is robot and thresholds == th:
                return grid
            return real.robot_grid(truth, at, thresholds)

        @staticmethod
        def reachable(on, start_xy, goal_xy):
            if on is grid and tuple(start_xy) == (robot.x, robot.y):
                cell = on.index(*goal_xy)
                return cell is not None and bool(joined[cell])
            return real.reachable(on, start_xy, goal_xy)

    evaluate.freespace = Cached()
    try:
        yield
    finally:
        evaluate.freespace = real


def target_at(rec: Record, *, point=None, pixel=None, world=None):
    """A GroundedTarget for the record's scene, made as the pipeline makes one: from a point (x, y on the 0..1000
    grid) or a pixel (u, v) of the full frame, lifted from the depth image with the run's lift window (None where
    the lift fails); or from a world point (x, y, z), its pixel projected and its point the nearest on the grid."""
    from sgi.geometry import lift, pixel_to_point, point_to_pixel, project
    from sgi.types import GroundedTarget, Point
    if sum(given is not None for given in (point, pixel, world)) != 1:
        raise ValueError("target_at takes one of point, pixel and world")
    sc, _, th, _ = _against(rec)
    camera = sc.observation.camera
    if world is not None:
        shown = project(camera, world)
        if shown is None:
            raise ValueError(f"the world point {tuple(world)} is behind the camera")
        u, v = shown[:2]
        x, y = pixel_to_point(u, v, camera.width, camera.height)
        return GroundedTarget(point=Point(x=x, y=y), pixel=(u, v), world=tuple(float(c) for c in world))
    if point is not None:
        u, v = point_to_pixel(point[0], point[1], camera.width, camera.height)
        x, y = point
    else:
        u, v = float(pixel[0]), float(pixel[1])
        x, y = pixel_to_point(u, v, camera.width, camera.height)
    lifted = lift(sc.observation.depth, camera, u, v, window=th.lift_window)
    if lifted is None:
        return None
    return GroundedTarget(point=Point(x=x, y=y), pixel=(u, v), world=tuple(float(c) for c in lifted))


def _targets(target, count: int) -> list:
    """One target or None (the recorded one) per pose: a list gives one for each pose, anything else is one for all
    of them; a dict is read as a GroundedTarget. A world point is made a target with target_at first."""
    from sgi.types import GroundedTarget
    some = list(target) if isinstance(target, list) else [target] * count
    if len(some) != count:
        raise ValueError(f"{len(some)} targets for {count} poses")
    return [None if one is None else GroundedTarget.model_validate(one) for one in some]


def score_poses(rec: Record, poses, *, target=None, force_ok: bool = False) -> list:
    """The package's Score of the record's recorded result with each world pose (x, y, theta) put in its place,
    pose_robot derived from it as the package derives it (sgi.geometry.make_poses), under the run's recorded
    thresholds. `target` puts a target in place of the recorded one: a GroundedTarget (target_at) or its dict for
    every pose, or a list of them (None for the recorded one) with one for each pose; success needs a target that
    matches. A result whose status is not ok fails whatever its pose, as the scorer has it; force_ok scores it as an
    ok result, which can succeed only with a target (no record without one has a target recorded). The robot's
    freespace is computed once per scene and thresholds, not once per pose; the scores are the package's."""
    from sgi.evaluate import score
    from sgi.geometry import make_poses
    from sgi.types import GoalResult
    sc, gt, th, cache = _against(rec)
    result = GoalResult.model_validate(rec.result)
    poses = list(poses)
    scores = []
    with _reach_once(sc, th, cache):
        for pose, put in zip(poses, _targets(target, len(poses))):
            pose_world, pose_robot = make_poses(sc.observation.robot, pose[:2], pose[2])
            update = {"pose_world": pose_world, "pose_robot": pose_robot}
            update |= ({} if put is None else {"target": put}) | ({"status": "ok"} if force_ok else {})
            scores.append(score(result.model_copy(update=update), sc.observation, sc.truth, gt, th))
    return scores


def score_pose(rec: Record, pose_world, *, target=None, force_ok: bool = False):
    """The package's Score of the record's result with one world pose (x, y, theta) put in; see score_poses."""
    return score_poses(rec, [pose_world], target=None if target is None else [target], force_ok=force_ok)[0]


def rescore(rec: Record):
    """The package's Score of the record's result as recorded, scored now (no cache)."""
    from sgi.evaluate import score
    from sgi.types import GoalResult
    sc, gt, th, _ = _against(rec)
    return score(GoalResult.model_validate(rec.result), sc.observation, sc.truth, gt, th)


@dataclasses.dataclass
class Resolved:
    result: object                                 # the GoalResult with the solver's place, as the constraints stage writes it
    score: object                                  # the package's Score of it
    pose_world: tuple
    pose_robot: tuple
    flags: object                                  # Flags
    note: object                                   # SolverNote
    uses: list                                     # ConstraintUse, one per constraint, in order
    candidates: list                               # the runners-up
    moved_m: float | None                          # from the recorded place; None where none is recorded

    @property
    def reproduced(self) -> bool:
        return self.moved_m is not None and self.moved_m <= SAME_PLACE_M


def resolve(rec: Record, *, target=None, people=None, constraints=None) -> Resolved:
    """The solver of the imported code run again on a record, on its scene and under the run's thresholds, and the
    package's score of the place; replay_solver.solve_again is the one code path both tools use. Each input is the
    recorded one unless given: target a GroundedTarget (target_at) or its dict; people a list of DetectedPerson or
    their dicts, numbered as the model was shown them (a target given does not number them again); constraints a
    list of Constraint or their dicts, as the answer states them. A record holding no target, people or constraints
    needs them given."""
    from sgi.evaluate import score
    from sgi.types import Constraint, DetectedPerson, GoalResult, GroundedTarget
    sc, gt, th, cache = _against(rec)
    result = GoalResult.model_validate(rec.result)
    solved, again = _module("replay_solver").solve_again(
        result, sc.observation, th,
        target=None if target is None else GroundedTarget.model_validate(target),
        people=None if people is None else [DetectedPerson.model_validate(p) for p in people],
        constraints=None if constraints is None else [Constraint.model_validate(c) for c in constraints])
    with _reach_once(sc, th, cache):
        scored = score(again, sc.observation, sc.truth, gt, th)
    moved = None if result.pose_world is None else math.dist(solved.pose_world[:2], result.pose_world[:2])
    return Resolved(again, scored, solved.pose_world, solved.pose_robot, solved.flags, solved.note, solved.uses,
                    solved.candidates, moved)


def moved_known(path: Path = MOVED_FILE) -> dict[tuple[str, str, str, str], str]:
    """The solver places known not to be given again by the solver of the code as it is: (run, scene, episode,
    trial) -> the reason, from a file of lines `RUN SCENE EPISODE TRIAL  # reason`, the run by its name with or
    without the time its folder's name starts with. An empty line and a line that is only a comment are passed over;
    a place with no reason is an error."""
    named = {}
    for number, line in enumerate(Path(guard_file(path)).read_text(encoding="utf-8").splitlines(), start=1):
        words, _, reason = line.partition("#")
        words = words.split()
        if not words:
            continue
        if len(words) != 4 or not reason.strip():
            raise SystemExit(f"error: {path}: line {number}: a run, a scene, an episode, a trial and # a reason are expected")
        named[tuple(words)] = reason.strip()
    return named


COLUMNS = ("records", "allowed", "skipped", "solver", "same", "moved", "ok rec", "ok now", "mismatch")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="research_harness.py", description="Summarise the listed runs, offline.")
    for option in ("--stage", "--arm", "--model", "--method"):
        parser.add_argument(option)
    parser.add_argument("--no-rescore", action="store_true", help="score nothing again")
    parser.add_argument("--no-resolve", action="store_true", help="solve nothing again (no reproduction check)")
    parser.add_argument("--moved", type=Path, default=MOVED_FILE, help="the places known to move, with reasons")
    args = parser.parse_args(argv)
    started = time.monotonic()
    chosen = runs(args.stage, args.arm, args.model, args.method)
    known = {} if args.no_resolve else moved_known(args.moved)
    print(f"{len(chosen)} listed runs of {RUNS_FILE.name}, from {whitelist().repo}")
    print(f"{'run':48s}" + "".join(f"{name:>9s}" for name in COLUMNS))
    totals, every = collections.Counter(), collections.Counter()
    mismatches, disagree, moved, unnamed = [], [], [], []
    for ref in chosen:
        tally, row = collections.Counter(), collections.Counter()
        for rec in records(ref, tally=tally):
            label = f"{rec.run} {rec.scene_id} {rec.episode_id} trial {rec.trial} ({rec.situation})"
            row["solver"] += rec.solver_step
            row["ok rec"] += bool(rec.score.get("success"))
            if not args.no_rescore:
                now = rescore(rec)
                row["ok now"] += bool(now.success)
                if bool(now.success) != bool(rec.score.get("success")):
                    row["mismatch"] += 1
                    mismatches.append(f"{label}: recorded {bool(rec.score.get('success'))}, now {now.success}")
                if rec.pose_world is not None and score_pose(rec, rec.pose_world) != now:
                    disagree.append(label)
            if rec.solver_step and not args.no_resolve:
                again = resolve(rec)
                if again.reproduced:
                    row["same"] += 1
                    continue
                row["moved"] += 1
                key = (rec.ref.name, rec.scene_id, rec.episode_id, str(rec.trial))
                why = known.get(key) or known.get((rec.run, *key[1:]))
                moved.append(f"{label}: moved {again.moved_m:.3f} m" + (f"; known: {why}" if why else "; NOT NAMED"))
                if not why:
                    unnamed.append(label)
        skipped = sum(count for key, count in tally.items() if key.startswith("skipped"))
        row.update(records=tally["allowed"] + skipped, allowed=tally["allowed"], skipped=skipped)
        print(f"{ref.folder:48s}" + "".join(f"{row[name]:9d}" for name in COLUMNS))
        totals.update(row)
        every.update(tally)
    print(f"{'all':48s}" + "".join(f"{totals[name]:9d}" for name in COLUMNS))
    for key in sorted(every):
        print(f"  {key}: {every[key]}")
    shown = {ref.folder: run_examples(ref) for ref in chosen}
    episodes = sorted({one for some in shown.values() for one in some})
    print(f"worked examples: {sum(bool(some) for some in shown.values())} of the {len(chosen)} runs showed them, "
          f"{len(episodes)} distinct episodes of {len({one[:2] for one in episodes})} scenes, every one on "
          f"{SCENES_FILE.name} ({len(whitelist().examples)} example scenes listed)")
    missing = []
    if not args.no_rescore:                        # every listed example loads, and each one shown has its truth
        for folder, scene_id in examples():
            example(scene_id, folder)
        missing = [f"{scene_id}/{episode}" for folder, scene_id, episode in episodes
                   if episode not in example(scene_id, folder).truths]
        print(f"  every listed example scene loads; {len(missing) or 'no'} shown episode(s) without a development "
              f"truth" + (f": {', '.join(missing)}" if missing else ""))
    wl, distinct = whitelist(), distinct_scenes()
    print(f"scenes: {len(wl.scenes)} listed for the runs, {len(distinct)} distinct with links resolved "
          f"({sum(len(one.listed) > 1 for one in distinct)} listed more than once); {len(wl.examples)} worked-example "
          f"scenes")
    same = []
    if not args.no_rescore:                        # no two of the distinct scenes and worked examples are one by content
        prints = collections.defaultdict(list)
        for folder in [one.folder for one in distinct] + [folder / scene_id for folder, scene_id in examples()]:
            prints[fingerprint(folder)].append(os.path.relpath(folder, wl.repo))
        same = [", ".join(group) for group in prints.values() if len(group) > 1]
        print(f"  by content: {'every one of the ' + str(len(prints)) + ' distinct' if not same else 'SAME'}"
              + "".join(f"\n  the same: {line}" for line in same))
    if not args.no_rescore:
        print(f"rescore self-check: {len(mismatches) or 'no'} record(s) whose recorded pose scored now differs from "
              f"the recorded success, of {totals['allowed']}")
        for line in mismatches:
            print(f"  {line}")
        print(f"score_pose against rescore: {'agree' if not disagree else 'DISAGREE'} on every record with a pose"
              + (f" but {len(disagree)}" if disagree else ""))
        for line in disagree:
            print(f"  {line}")
    if not args.no_resolve:
        print(f"reproduction check: {'PASS' if not unnamed else 'FAIL'}: {totals['same']} of {totals['solver']} solver "
              f"places given again by the solver of {Path(sys.modules['sgi'].__file__).resolve().parents[1]}; "
              f"{totals['moved']} moved, {totals['moved'] - len(unnamed)} of them named in {args.moved.name}")
        for line in moved:
            print(f"  {line}")
    print(f"{time.monotonic() - started:.1f} s")
    return 1 if unnamed or disagree or missing or same else 0


if __name__ == "__main__":
    raise SystemExit(main())
