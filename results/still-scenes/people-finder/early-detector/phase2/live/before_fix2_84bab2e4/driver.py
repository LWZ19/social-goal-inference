"""I09 live check (phase 2), arm D-live: the recorded C runs at t0 with the frozen A1 detector people as the people
source, so the constraints call sees the detector's numbered people and no VLM people call is made (FROZEN.md).

    driver.py --model {glm53,dsv41} --trial 0 --out DIR --cache-dir ABS [--count-only | --mock]
              [--max-calls 120] [--max-calls-per-model 60] [--env-file PATH] [--arms real,staffed,plain]
              [--only SCENE_ID ...] [--prefix s24_i09_]

Modes:
  live (default)  sends the cache misses to the gateway, through the package's own client (validation, retries,
                  pacing, the disk cache). Every request to the back end (each attempt, retries included) is counted
                  in DIR/ledger.jsonl before it is sent; the 121st in total and the 61st of one model are refused
                  (CapReached), across invocations that share DIR, even concurrent ones (each reservation re-counts
                  the ledger under a file lock; a lock file also refuses a second live invocation on DIR). A plan,
                  locate or context request that misses the cache is refused unsent (UpstreamMiss): only constraints
                  calls are sent. Needs --env-file: the key is read by sgi.vlm.env
                  alone, from that file, as docs/model_notes/tools/exp_run.py has it; this script never opens it.
  --count-only    nothing is sent and no back end is built: a cache hit is answered from the cache exactly as the
                  package's client answers it; a miss is counted and recorded, and answered locally as vlm_error.
                  Writes DIR/count_only_<model>_<time>.json and .jsonl; no run folder, no cache write.
  --mock          the package's mock back end answers every request (no cache): the end-to-end dry run. Writes run
                  folders named ..._con_mock.

What it runs: for each arm, the recorded run (s19 glm53 or s20 dsv41, method con) is found through the step-0 harness
(research_harness.py); its config.yaml gives the configuration, with run.trials 1, run.cache_dir and run.prompts_dir
absolute, and run.use_cache off in mock mode only. Its whitelisted scenes are loaded through the harness's guards,
checked with the package's scene check, and every episode is run once at trial 0 by the package's own
sgi.run.run_episode. The one hook replaced is sgi.run._given_inferrer: where the package gives
pipeline.constraint_inferrer(OraclePeople(truth, th), trial=trial) under run.people: oracle, the driver gives
pipeline.constraint_inferrer(DetectorPeople(...), trial=trial) for the scene being run. The code is the package at
commit 202c165 (src_202c165/, extracted with git archive), imported from there and asserted; its prompts must equal
the recorded run's prompts/ byte for byte.

Run folders: DIR/<time>_<prefix><arm>_<model>_con/ with config.yaml (plus the blocks people_source and i09_live),
prompts/, results.jsonl, summary.csv and viz/, as python -m sgi.run writes them. Every record's config carries
people_source. Per invocation DIR/check_<model>_<mode>_<time>.json holds the cache counts by stage, the people calls
(0), the detector people per scene, and (live) the comparison of each record's plan and context answers with the
recorded t0 record.

Exit codes: 0 complete; 1 some record ended in vlm_error (the same command run again repeats only those calls);
2 refused (the lock, FROZEN.md, the prompts, an upstream miss, an unknown scene), or stopped by a configuration,
scene, key or provider error, or any other error; 3 stopped by the call cap.
"""
from __future__ import annotations

import argparse
import collections
import csv
import fcntl
import hashlib
import json
import logging
import math
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent                      # .../phase2/i09/live
SRC = HERE / "src_202c165" / "src"
PROMPTS = HERE / "src_202c165" / "prompts"
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths                                                # noqa: E402
RD = paths.STILL                                            # results/still-scenes
WORKTREE = paths.REPO                                       # the repository root
HARNESS_DIR = paths.HARNESS                                 # results/still-scenes/analysis-scripts/harness
EVAL_PEOPLE = RD / "people-finder" / "early-detector" / "phase1c" / "a1" / "eval_people.json"
EVAL_PEOPLE_SHA256 = "ffa595453c94b444c8dd030d053d37ac103441a7e74f07e3f172836b3d0c24b3"
FROZEN_SETTING = {"detector": "rtdetr_v2_r50vd", "tau": 0.25, "band": "wide", "flat": False, "d_dup": 0.5,
                  "d_push": 0.2}
CODE_COMMIT = "202c165"

sys.path.insert(0, str(SRC))
sys.path.insert(1, str(HARNESS_DIR))

import yaml                                                 # noqa: E402

import sgi                                                  # noqa: E402
import sgi.run as sgi_run                                   # noqa: E402
import sgi.vlm.env as sgi_env                               # noqa: E402
import sgi.vlm.factory as sgi_factory                       # noqa: E402
from sgi.config import Config                               # noqa: E402
from sgi.geometry import project                            # noqa: E402
from sgi.gt.derive import prepare                           # noqa: E402
from sgi.pipeline import Pipeline                           # noqa: E402
from sgi.prompts import prompt_files, stages_of             # noqa: E402
from sgi.scene import load_scene                            # noqa: E402
from sgi.stages.people import FoundPerson, PeopleFound     # noqa: E402
from sgi.types import GoalResult                            # noqa: E402
from sgi.evaluate import Score                              # noqa: E402
from sgi.viz import episode_figure                          # noqa: E402
from sgi.vlm.base import BackendError, QuotaExceededError, VLMClient, VLMResponse   # noqa: E402
from sgi.vlm.cache import Cache, cache_key                  # noqa: E402
from sgi.vlm.mock import MockBackend, centre_responder      # noqa: E402

import research_harness as H                                # noqa: E402

assert Path(sgi.__file__).resolve().is_relative_to(SRC), f"sgi was not imported from {SRC}: {sgi.__file__}"

log = logging.getLogger("i09_live")

MODELS = {"glm53": ("s19", "glm53"), "dsv41": ("s20", "dsv41")}     # --model -> (stage, model) of the recorded runs
ARMS = ("real", "staffed", "plain")
CAP_TOTAL, CAP_MODEL = 120, 60                              # the approval (18 Oct); never raised
PEOPLE_SOURCE = "detector_i09_a1_frozen"
UNUSED_STAGES = ("people", "goal")                          # asked by no D-live goal step: a call there is a defect


class CapReached(Exception):
    """The next request would be the (cap + 1)-th: refused before it is sent."""


class UnexpectedCall(Exception):
    """A stage that D-live never asks was asked (the people stage above all)."""


class UpstreamMiss(UnexpectedCall):
    """Live mode: a plan, locate or context answer is not in the cache. D-live would redraw it and no longer be the
    recorded C with one change, so it is refused before it is sent."""


class NotSent(Exception):
    """A back end of count-only mode was reached; it never sends."""


# ----------------------------------------------------------------------------------------------------------------
# The call ledger and the cap

class Ledger:
    """Every request sent to a provider, one JSON line each in `path`, written before it is sent; counted over every
    invocation that shares the file. `reserve` refuses the request that would pass a cap. Each reservation re-counts
    the file under an exclusive lock on it, so the caps hold even if two invocations share the file at once."""

    def __init__(self, path: Path, model: str, max_total: int, max_model: int):
        if not (1 <= max_total <= CAP_TOTAL and 1 <= max_model <= CAP_MODEL):
            raise SystemExit(f"error: the caps are at most {CAP_TOTAL} in total and {CAP_MODEL} per model, "
                             f"not {max_total} and {max_model}")
        self.path, self.model, self.max_total, self.max_model = Path(path), model, max_total, max_model
        self.total, self.per_model = 0, collections.Counter()
        self._locked(lambda out: None)
        self.at_start = (self.total, self.per_model[model])

    def _locked(self, act) -> None:
        """Re-count the sent rows of the file and then `act(file)`, all under one exclusive lock on the file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a+", encoding="utf-8") as out:
            fcntl.flock(out.fileno(), fcntl.LOCK_EX)
            try:
                out.seek(0)
                self.total, self.per_model = 0, collections.Counter()
                for line in out.read().splitlines():
                    if line.strip():
                        row = json.loads(line)
                        if row.get("event") == "sent":
                            self.total += 1
                            self.per_model[row["model"]] += 1
                act(out)
                out.flush()
                os.fsync(out.fileno())
            finally:
                fcntl.flock(out.fileno(), fcntl.LOCK_UN)

    def _write(self, row: dict) -> None:
        self._locked(lambda out: out.write(json.dumps(row) + "\n"))

    def reserve(self, info: dict) -> None:
        refused = []

        def act(out):
            if self.total >= self.max_total or self.per_model[self.model] >= self.max_model:
                out.write(json.dumps({"time": _now(), "event": "refused", "model": self.model, **info,
                                      "total": self.total, "model_total": self.per_model[self.model]}) + "\n")
                refused.append(True)
                return
            self.total += 1
            self.per_model[self.model] += 1
            out.write(json.dumps({"time": _now(), "event": "sent", "model": self.model, **info,
                                  "total": self.total, "model_total": self.per_model[self.model]}) + "\n")

        self._locked(act)
        if refused:
            raise CapReached(f"call cap reached: {self.total} sent in total (cap {self.max_total}), "
                             f"{self.per_model[self.model]} for {self.model} (cap {self.max_model}); "
                             f"the next request was not sent")

    def note(self, event: str, info: dict) -> None:
        self._write({"time": _now(), "event": event, "model": self.model, **info})


class Lock:
    """One live invocation at a time per output folder. The ledger's caps hold without it; it keeps two runs from
    interleaving their run folders. Only the invocation that took the lock removes it."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.held = False

    def __enter__(self):
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise SystemExit(f"error: {self.path} exists: another live driver may be running on this folder. "
                             "If none is, remove the file and start again.") from None
        os.write(fd, f"{os.getpid()} {_now()}\n".encode())
        os.close(fd)
        self.held = True
        return self

    def __exit__(self, *exc):
        if self.held:
            self.path.unlink(missing_ok=True)
            self.held = False


@dataclass
class Where:
    """The record being run, for the ledger and the tallies."""
    arm: str = ""
    scene: str = ""
    episode: str = ""
    trial: int = 0

    def info(self) -> dict:
        return {"arm": self.arm, "scene": self.scene, "episode": self.episode, "trial": self.trial}


class GuardedBackend:
    """A real back end behind the ledger: each request is reserved (and refused at the cap) before it is sent."""

    def __init__(self, inner, ledger: Ledger, stage: str, where: Where):
        self.inner, self.ledger, self.stage, self.where = inner, ledger, stage, where
        self.provider = inner.provider

    def call(self, entry, system, parts, json_schema):
        info = {**self.where.info(), "stage": self.stage}
        self.ledger.reserve(info)
        try:
            result = self.inner.call(entry, system, parts, json_schema)
        except Exception as exc:                            # the package's client decides what to do with it
            self.ledger.note("error", {**info, "error": type(exc).__name__})
            raise
        self.ledger.note("answered", {**info, "finish": result.finish})
        return result


class NeverSend:
    """The back end of count-only mode, and of the stages D-live never asks: it carries the provider of its entry
    (so the cache key is the live one) and refuses to be called."""

    def __init__(self, provider: str, why: str):
        self.provider, self.why = provider, why

    def call(self, entry, system, parts, json_schema):
        raise NotSent(self.why)


@dataclass
class Tally:
    """What the clients were asked, by stage: cache hits, misses, requests handed to a back end."""
    rows: list = field(default_factory=list)
    sent_requests: list = field(default_factory=list)       # (stage, where, system, parts, json_schema), mock only

    def add(self, **row) -> None:
        self.rows.append(row)

    def counts(self, rows=None) -> dict:
        out = collections.defaultdict(collections.Counter)
        for row in self.rows if rows is None else rows:
            out[row["stage"]][row["outcome"]] += 1
        return {stage: dict(c) for stage, c in sorted(out.items())}


class DriverClient(VLMClient):
    """The package's VLMClient for one stage, with three changes only: it tallies cache hits and misses; in
    count-only mode a miss is answered locally as vlm_error and never sent; and a stage D-live never asks fails
    loudly. A hit is returned exactly as VLMClient.generate returns it, and a live miss goes through
    VLMClient.generate itself (validation retry, transient retries, pacing, cache write)."""

    def __init__(self, entry, backend, cache, *, stage: str, mode: str, where: Where, tally: Tally, pacer=None,
                 capture: bool = False):
        super().__init__(entry, backend, cache, pacer=pacer)
        self.stage, self.mode, self.where, self.tally, self.capture = stage, mode, where, tally, capture

    def generate(self, parts, schema, system=None, trial=0):
        if self.stage in UNUSED_STAGES:
            raise UnexpectedCall(f"the {self.stage} stage was asked, which D-live never does ({self.where.info()})")
        json_schema = schema.model_json_schema()
        key = cache_key(self.entry.model_copy(update={"provider": self._backend.provider}), system, parts,
                        json_schema, trial)
        base = {**self.where.info(), "stage": self.stage, "key": key, "trial_in_key": trial}
        hit = None if self._cache is None else self._cache.get(key)
        if hit is not None:
            self.tally.add(**base, outcome="hit")
            return hit.model_copy(update={"cached": True})
        if self.mode == "count-only":
            self.tally.add(**base, outcome="miss")
            images = [p for p in parts if hasattr(p, "png")]
            return VLMResponse(status="vlm_error", error="count-only: a cache miss, not sent",
                               provider=self._backend.provider, model=self.entry.model,
                               reasoning_effort=self.entry.reasoning_effort, point_format=self.entry.point_format,
                               image_size=(images[-1].width, images[-1].height) if images else None)
        if self.mode == "live" and self.stage != "constraints":
            self.tally.add(**base, outcome="refused")
            raise UpstreamMiss(f"the {self.stage} answer of {self.where.info()} is not in the cache: D-live sends only "
                               "constraints calls (FROZEN.md section 7, the go-live rule); nothing was sent")
        if self.capture:
            self.tally.sent_requests.append((self.stage, self.where.info(), system, list(parts), json_schema))
        response = super().generate(parts, schema, system=system, trial=trial)
        self.tally.add(**base, outcome="sent", status=response.status,
                       validation_retries=response.validation_retries, transient_retries=response.transient_retries)
        return response


# ----------------------------------------------------------------------------------------------------------------
# The people source

class DetectorPeople:
    """The frozen A1 detector people of one scene as a people source with OraclePeople's interface: in descending
    score, each a FoundPerson at the detector's (x, y) and yaw, its pixel the projection of the point
    person_mark_height_m above that position (as OraclePeople builds it), no label. A person whose point does not
    project (behind the camera) is left out, as in OraclePeople. No model is asked."""

    def __init__(self, scene_id: str, fingerprint: str, observation_sha256: str, people: list[dict], th):
        self.scene_id, self.fingerprint, self.observation_sha256 = scene_id, fingerprint, observation_sha256
        self.people = sorted(people, key=lambda p: -p["score"])        # stable: equal scores keep the file's order
        if [p["score"] for p in self.people] != [p["score"] for p in people]:
            raise SystemExit(f"error: {scene_id}: the frozen people are not in descending score")
        self.th = th
        self.asked = 0

    def __call__(self, observation) -> PeopleFound:
        started = time.perf_counter()
        if observation.sha256() != self.observation_sha256:
            raise SystemExit(f"error: DetectorPeople of {self.scene_id} was given another observation")
        found = []
        for person in self.people:
            marked = project(observation.camera, (person["x"], person["y"], self.th.person_mark_height_m))
            if marked is not None:
                found.append(FoundPerson(xy=(float(person["x"]), float(person["y"])),
                                         yaw=None if person["yaw"] is None else float(person["yaw"]),
                                         pixel=marked[:2], label=None))
        self.asked += 1
        return PeopleFound(found=found, calls=[], call_time_s=0.0, geometry_time_s=time.perf_counter() - started)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


_FINGERPRINTS: dict[Path, str] = {}


def fingerprint(folder: Path) -> str:
    """The harness fingerprint of a scene folder (guarded first), once per resolved folder."""
    real = Path(folder).resolve()
    if real not in _FINGERPRINTS:
        _FINGERPRINTS[real] = H.fingerprint(folder)
    return _FINGERPRINTS[real]


def detector_table() -> dict[str, dict]:
    """The frozen E-real detector people of eval_people.json, keyed by the harness fingerprint of each scene folder;
    the file's sha256 and setting are checked."""
    data = EVAL_PEOPLE.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != EVAL_PEOPLE_SHA256:
        raise SystemExit(f"error: {EVAL_PEOPLE} has sha256 {got}, not the frozen {EVAL_PEOPLE_SHA256}")
    doc = json.loads(data)
    if doc["setting"] != FROZEN_SETTING or doc["facing_variant"] != "F5" or doc["person_mark_height_m"] != 1.1:
        raise SystemExit(f"error: {EVAL_PEOPLE} does not hold A1's frozen setting with facing F5")
    table = {}
    for folder, entry in doc["scenes"].items():
        if entry["domain"] != "E-real":
            continue
        fp = fingerprint(Path(folder))
        if fp in table:
            raise SystemExit(f"error: two entries of {EVAL_PEOPLE.name} are one scene by content: {folder}")
        table[fp] = {"folder": folder, **entry}
    return table


# ----------------------------------------------------------------------------------------------------------------
# One arm

def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def recorded_run(model: str, arm: str):
    stage, mkey = MODELS[model]
    found = H.runs(stage=stage, arm=arm, model=mkey, method="con")
    if len(found) != 1:
        raise SystemExit(f"error: {len(found)} listed runs for {stage} {arm} {mkey} con, not 1")
    return found[0]


def check_prompts(ref) -> dict:
    """The recorded run's prompts/ against src_202c165/prompts: every file equal, byte for byte, or refused."""
    folder = H.guard_file(ref.path / "prompts")
    theirs = {name: _sha256(H.guard_file(folder / name)) for name in sorted(os.listdir(folder))}
    mine = {name: _sha256(PROMPTS / name) for name in theirs}
    differ = [name for name in theirs if theirs[name] != mine[name]]
    if differ:
        raise SystemExit(f"error: {', '.join(differ)} of {PROMPTS} differ from {folder}")
    return theirs


def arm_config(ref, mode: str, cache_dir: Path | None) -> Config:
    """The recorded configuration with trials 1, absolute cache and prompts folders, and (mock) no cache."""
    data = H.recorded_config(ref)[1].model_dump(mode="json")
    data["run"].update(trials=1, prompts_dir=str(PROMPTS), use_cache=mode != "mock",
                       cache_dir=str(cache_dir) if cache_dir is not None else data["run"]["cache_dir"])
    cfg = Config.model_validate(data)
    assert cfg.run.goal == "constraints" and cfg.run.people == "vlm" and cfg.run.constraints == "vlm"
    assert cfg.run.grounding == "vlm" and cfg.run.context_check == "vlm" and cfg.run.grounding_source == "view"
    return cfg


def build_clients(cfg: Config, mode: str, cache_dir: Path | None, ledger: Ledger | None, where: Where, tally: Tally,
                  capture: bool, backend_factory=None) -> dict:
    """One DriverClient per stage of the run. live: the package's real back end behind the ledger, and its pacer;
    count-only: NeverSend; mock: the package's MockBackend. The people and goal stages always get NeverSend.
    `backend_factory(entry) -> back end` replaces the real one (dry runs only)."""
    clients, real = {}, {}
    cache = Cache(cache_dir) if cfg.run.use_cache else None
    for stage in stages_of(cfg.run):
        name = cfg.stages[stage].model
        entry = cfg.models[name]
        pacer = None
        if stage in UNUSED_STAGES:
            backend = NeverSend(entry.provider, f"the {stage} stage is never asked in D-live")
        elif mode == "count-only":
            backend = NeverSend(entry.provider, "count-only mode sends nothing")
        elif mode == "mock":
            backend = MockBackend(centre_responder)
        else:
            if name not in real:
                real[name] = backend_factory(entry) if backend_factory else sgi_factory._backend(entry)
                if real[name].provider != entry.provider:
                    raise SystemExit(f"error: the back end of {name} is {real[name].provider}, not {entry.provider}")
            backend = GuardedBackend(real[name], ledger, stage, where)
            pacer = None if backend_factory else sgi_factory._pacer(entry)
        clients[stage] = DriverClient(entry, backend, cache, stage=stage, mode=mode, where=where, tally=tally,
                                      pacer=pacer, capture=capture)
    return clients


class _Current:
    source: DetectorPeople | None = None


CURRENT = _Current()


def _given_inferrer_detector(run, pipeline, truth, ep, chosen, th, trial):
    """sgi.run._given_inferrer for D-live: as the package's `people: oracle` branch, with DetectorPeople."""
    if run.goal != "constraints" or CURRENT.source is None:
        raise SystemExit("error: D-live runs the constraint method with a detector people source only")
    return pipeline.constraint_inferrer(CURRENT.source, trial=trial)


def compare_upstream(record: dict, recorded) -> dict:
    """The plan and context answers of a D-live record against the recorded t0 record: equal or not."""
    def plan_of(r):
        return None if r is None else {k: v for k, v in r.items() if k != "image_sha256"}
    mine, theirs = record["result"], recorded.result
    stages = lambda calls: [c["stage"] for c in calls or [] if c["stage"] != "people"]       # noqa: E731
    out = {
        "plan_equal": plan_of(record["plan"]) == plan_of(recorded.raw.get("plan")),
        "image_equal": (record["plan"] or {}).get("image_sha256") == (recorded.raw.get("plan") or {}).get("image_sha256"),
        "context_equal": mine.get("context_check") == theirs.get("context_check"),
        "location_equal": mine.get("location") == theirs.get("location"),
        "target_equal": mine.get("target") == theirs.get("target"),
        "calls_equal": stages(mine.get("calls")) == stages(theirs.get("calls")),
        "goal_step_live": any(c["stage"] == "constraints" for c in mine.get("calls") or []),
        "goal_step_recorded": any(c["stage"] == "people" for c in theirs.get("calls") or []),
    }
    out["all_equal"] = all(out[k] for k in ("plan_equal", "image_equal", "context_equal", "location_equal",
                                            "target_equal", "calls_equal"))
    return out


def run_arm(model: str, arm: str, mode: str, out_root: Path, cache_dir: Path | None, ledger: Ledger | None,
            table: dict, tally: Tally, *, prefix: str = "s24_i09_", only=None, capture: bool = False,
            backend_factory=None) -> dict:
    """Run one arm at t0; returns its check entry. Writes the run folder unless in count-only mode."""
    ref = recorded_run(model, arm)
    prompt_hashes = check_prompts(ref)
    cfg = arm_config(ref, mode, cache_dir)
    th = cfg.thresholds
    recorded = {(r.scene_id, r.episode_id): r for r in H.records(ref) if r.trial == 0}
    listed = sorted(sid for d, sid in H.whitelist().scenes if d == ref.scenes_dir)
    scene_ids = sorted({sid for sid, _ in recorded})
    if listed != scene_ids:
        raise SystemExit(f"error: {ref.folder}: the listed scenes and the recorded t0 scenes differ")
    if only:
        scene_ids = [sid for sid in scene_ids if sid in set(only)]
    where = Where(arm=arm)
    clients = build_clients(cfg, mode, cache_dir, ledger, where, tally, capture, backend_factory)
    pipeline = Pipeline(cfg, clients, [])
    name = f"{prefix}{arm}_{model}_con" + ("_mock" if mode == "mock" else "")
    entry = {"arm": arm, "baseline_run": ref.folder, "name": name, "mode": mode, "prompts": prompt_hashes,
             "scenes": {}, "records": 0, "upstream": [], "people_asked": 0, "stopped_by": None}
    first_row = len(tally.rows)

    out = lines = None
    if mode != "count-only":
        while True:                                             # never two runs in one folder: a new second if taken
            out = Path(out_root) / f"{time.strftime('%Y%m%d-%H%M%S')}_{name}"
            try:
                out.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                time.sleep(1.0)
        entry["run_folder"] = str(out)
    doc = {"name": name, "scenes_dir": str(ref.scenes_dir), "config": cfg.model_dump(mode="json"),
           "example_ids": [], "examples_missing": [], "skipped_held_out": 0, "vlm_errors": 0, "incomplete": True,
           "people_source": {"kind": "detector", "tag": PEOPLE_SOURCE, "file": str(EVAL_PEOPLE),
                             "sha256": EVAL_PEOPLE_SHA256, "setting": FROZEN_SETTING, "facing_variant": "F5",
                             "note": "run.people reads vlm only because RunConfig.people has no detector value; "
                                     "no people call is made (the people stage refuses to be asked)"},
           "i09_live": {"mode": mode, "baseline_run": ref.folder, "trial": 0, "code_commit": CODE_COMMIT,
                        "code": str(SRC), "driver_sha256": _sha256(Path(__file__)),
                        "frozen_sha256": _sha256(HERE / "FROZEN.md"), "started": _now()}}
    if out is not None:
        (out / "config.yaml").write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        (out / "prompts").mkdir()
        for path in prompt_files(pipeline.prompts_dir, stages_of(cfg.run)):
            shutil.copy(path, out / "prompts" / path.name)
        lines = open(out / "results.jsonl", "w", encoding="utf-8")

    records, stop = [], None
    try:
        for scene_id in scene_ids:
            folder = H.check_scene(ref, scene_id)                       # guarded, on the whitelist
            inputs, truth = load_scene(folder)
            if inputs.scene_id != scene_id:
                raise SystemExit(f"error: {folder} holds {inputs.scene_id}")
            sgi_run._refuse_failed_check(folder, scene_id, cfg)
            ets = prepare(inputs, truth, cfg)
            held = [e.episode_id for e in inputs.episodes if ets[e.episode_id].gt.situation in cfg.held_out_situations]
            if held:
                raise SystemExit(f"error: {scene_id} holds held-out episodes {held}; D-live runs none")
            fp = fingerprint(folder)
            if fp not in table:
                raise SystemExit(f"error: {scene_id}: no frozen detector people for this scene (fingerprint {fp})")
            frozen = table[fp]
            if frozen["scene_id"] != scene_id:
                raise SystemExit(f"error: {scene_id}: the frozen entry of this fingerprint is {frozen['scene_id']}")
            source = DetectorPeople(scene_id, fp, inputs.observation.sha256(), frozen["people"], th)
            pixel_gap = max((math.dist(project(inputs.observation.camera, (p["x"], p["y"], th.person_mark_height_m))[:2],
                                       p["mark_pixel"]) for p in frozen["people"] if p["mark_pixel"] is not None),
                            default=0.0)
            entry["scenes"][scene_id] = {"fingerprint": fp, "detector_people": len(frozen["people"]),
                                         "max_gap_to_a1_mark_pixel_px": pixel_gap}
            CURRENT.source = source
            for episode in inputs.episodes:
                if (scene_id, episode.episode_id) not in recorded:
                    raise SystemExit(f"error: {scene_id} {episode.episode_id} has no recorded t0 record")
                where.scene, where.episode, where.trial = scene_id, episode.episode_id, 0
                record = sgi_run.run_episode(pipeline, inputs, truth, ets[episode.episode_id], episode, cfg, 0)
                record["example_ids"] = []
                record["config"]["people_source"] = PEOPLE_SOURCE
                people_calls = [c for c in record["result"].get("calls") or [] if c["stage"] == "people"]
                if people_calls:
                    raise UnexpectedCall(f"{scene_id} {episode.episode_id}: a people call is in the record")
                records.append(record)
                entry["records"] += 1
                if mode != "mock":
                    cmp = compare_upstream(record, recorded[(scene_id, episode.episode_id)])
                    entry["upstream"].append({"scene": scene_id, "episode": episode.episode_id,
                                              "status": record["result"]["status"], **cmp})
                if lines is not None:
                    lines.write(json.dumps(record) + "\n")
                    lines.flush()
                    episode_figure(out / "viz" / f"{scene_id}_{episode.episode_id}.png", inputs.observation, truth,
                                   ets[episode.episode_id].gt, GoalResult.model_validate(record["result"]),
                                   Score.model_validate(record["score"]), title=f"{scene_id}/{episode.episode_id}")
            entry["people_asked"] += source.asked
    except (CapReached, QuotaExceededError, BackendError) as error:
        stop = error
        entry["stopped_by"] = f"{type(error).__name__}: {' '.join(str(error).split())}"
    finally:
        CURRENT.source = None
        if lines is not None:
            lines.close()

    rows_here = tally.rows[first_row:]
    entry["cache"] = tally.counts(rows_here)
    entry["vlm_errors"] = sum(r["result"]["status"] == "vlm_error" for r in records)
    entry["successes"] = sum(bool(r["score"]["success"]) for r in records)
    if out is not None:
        if records:
            rows = sgi_run.summarise(records)
            with open(out / "summary.csv", "w", newline="", encoding="utf-8") as table_file:
                writer = csv.DictWriter(table_file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        doc.update(vlm_errors=entry["vlm_errors"], incomplete=entry["vlm_errors"] > 0 or stop is not None)
        doc["i09_live"]["finished"] = _now()
        doc["i09_live"]["cache"] = entry["cache"]
        if stop is not None:
            doc["stopped_by"] = entry["stopped_by"]
        (out / "config.yaml").write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    if stop is not None:
        stop.i09_entry = entry                                  # the stopped arm's counts reach the check file
        raise stop
    return entry


# ----------------------------------------------------------------------------------------------------------------
# The command

def _forbid_keys(why: str) -> None:
    """In count-only and mock modes no key may be read and no real back end built."""
    def refuse(*args, **kwargs):
        raise SystemExit(f"error: {why}: no key is read and no back end is built")
    sgi_env.load_env = sgi_env.require_key = refuse
    sgi_factory.load_env = sgi_factory.require_key = refuse
    sgi_factory._backend = refuse


def count_bounds(rows: list[dict]) -> dict:
    """Count-only: misses by stage, and the lower and upper bounds of the requests a live run would send (an upstream
    miss stops its record, hiding up to 3 calls after a plan miss, 2 after a locate miss, 1 after a context miss)."""
    misses = collections.Counter(r["stage"] for r in rows if r["outcome"] == "miss")
    lower = sum(misses.values())
    return {"misses_by_stage": dict(misses), "lower": lower,
            "upper": lower + 3 * misses["plan"] + 2 * misses["locate"] + misses["context"],
            "upstream_misses": misses["plan"] + misses["locate"] + misses["context"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="driver.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--trial", type=int, default=0, help="only 0 (FROZEN.md: t0 only)")
    parser.add_argument("--out", type=Path, required=True, help="folder for the run folders, ledger and reports")
    parser.add_argument("--cache-dir", type=Path, help="the VLM disk cache (absolute); not used by --mock")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--count-only", action="store_true", help="count cache misses; send nothing")
    mode.add_argument("--mock", action="store_true", help="the package's mock back end, no cache (dry run)")
    parser.add_argument("--max-calls", type=int, default=CAP_TOTAL)
    parser.add_argument("--max-calls-per-model", type=int, default=CAP_MODEL)
    parser.add_argument("--env-file", type=Path, help="live only: the key file sgi.vlm.env reads (never opened here)")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--only", nargs="*", help="run only these scene ids (dry runs)")
    parser.add_argument("--prefix", default="s24_i09_")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    mode = "count-only" if args.count_only else "mock" if args.mock else "live"
    if args.trial != 0:
        parser.error("--trial: only 0 (FROZEN.md runs t0 only)")
    arms = [a for a in args.arms.split(",") if a]
    if not arms or any(a not in ARMS for a in arms):
        parser.error(f"--arms: of {', '.join(ARMS)}")
    if not (1 <= args.max_calls <= CAP_TOTAL and 1 <= args.max_calls_per_model <= CAP_MODEL):
        parser.error(f"the caps are at most {CAP_TOTAL} in total and {CAP_MODEL} per model")
    cache_dir = None
    if mode != "mock":
        if args.cache_dir is None or not args.cache_dir.is_absolute() or not args.cache_dir.is_dir():
            parser.error("--cache-dir: an absolute path to an existing folder is needed (not with --mock)")
        cache_dir = args.cache_dir
    if mode == "live":
        if args.env_file is None or not args.env_file.is_absolute():
            parser.error("live mode needs --env-file, an absolute path; sgi.vlm.env reads it, this script never does")
        sgi_env.load_env.__defaults__ = (args.env_file,)            # as docs/model_notes/tools/exp_run.py does
        sgi_env.require_key.__defaults__ = (args.env_file,)
    else:
        if args.env_file is not None:
            parser.error("--env-file is for live mode only")
        _forbid_keys(f"{mode} mode")
    out_root = args.out.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    sgi_run._given_inferrer = _given_inferrer_detector             # the one hook replaced (FROZEN.md section 3)

    frozen = (HERE / "FROZEN.sha256").read_text().split()[0]
    if _sha256(HERE / "FROZEN.md") != frozen:
        raise SystemExit("error: FROZEN.md does not match FROZEN.sha256: the protocol changed after it was frozen")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    table = detector_table()
    tally = Tally()
    report = {"model": args.model, "mode": mode, "started": _now(), "out": str(out_root),
              "cache_dir": None if cache_dir is None else str(cache_dir), "code": str(SRC), "sgi": sgi.__file__,
              "frozen_sha256": _sha256(HERE / "FROZEN.md"), "driver_sha256": _sha256(Path(__file__)), "arms": []}
    ledger, code, stopped = None, 0, None
    lock = Lock(out_root / "ledger.lock") if mode == "live" else None
    if lock is not None:
        lock.__enter__()                                            # a refusal here leaves the other run's lock alone
    try:
        if lock is not None:
            ledger = Ledger(out_root / "ledger.jsonl", args.model, args.max_calls, args.max_calls_per_model)
            report["ledger_at_start"] = {"total": ledger.at_start[0], args.model: ledger.at_start[1]}
            log.info("ledger: %d sent before this invocation (%d for %s); caps %d and %d", ledger.at_start[0],
                     ledger.at_start[1], args.model, args.max_calls, args.max_calls_per_model)
        for arm in arms:
            try:
                entry = run_arm(args.model, arm, mode, out_root, cache_dir, ledger, table, tally,
                                prefix=args.prefix, only=args.only)
            except (CapReached, QuotaExceededError, BackendError) as error:
                stopped = f"{type(error).__name__}: {' '.join(str(error).split())}"
                code = 3 if isinstance(error, CapReached) else 2
                if getattr(error, "i09_entry", None) is not None:
                    report["arms"].append(error.i09_entry)
                log.error("stopped: %s", stopped)
                break
            report["arms"].append(entry)
            if entry["vlm_errors"] and code == 0 and mode == "live":
                code = 1
            up = entry["upstream"]
            log.info("%s %s: %d records, %d successes, %d vlm_error; cache %s; upstream equal %d of %d",
                     arm, args.model, entry["records"], entry["successes"], entry["vlm_errors"], entry["cache"],
                     sum(u["all_equal"] for u in up), len(up))
    finally:
        if lock is not None:
            lock.__exit__(None, None, None)
    report["stopped_by"] = stopped
    report["finished"] = _now()
    report["cache_total"] = tally.counts()
    if ledger is not None:
        report["ledger_at_end"] = {"total": ledger.total, args.model: ledger.per_model[args.model]}
    if mode == "count-only":
        report["bounds"] = count_bounds(tally.rows)
        path = out_root / f"count_only_{args.model}_{stamp}"
        with open(path.with_suffix(".jsonl"), "w", encoding="utf-8") as rows:
            for row in tally.rows:
                rows.write(json.dumps(row) + "\n")
        path.with_suffix(".json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        b = report["bounds"]
        print(f"count-only {args.model}: {b['lower']} cache misses would be sent (upper bound {b['upper']}); "
              f"misses by stage {b['misses_by_stage']}; upstream misses {b['upstream_misses']}; "
              f"hits {sum(c.get('hit', 0) for c in report['cache_total'].values())}; report {path}.json")
    else:
        path = out_root / f"check_{args.model}_{mode}_{stamp}.json"
        path.write_text(json.dumps(report, indent=1), encoding="utf-8")
        for entry in report["arms"]:
            print(f"{entry['run_folder']}: {entry['successes']} of {entry['records']} records succeed, "
                  f"{entry['vlm_errors']} vlm_error; cache {entry['cache']}")
        if ledger is not None:
            print(f"ledger: {ledger.total} requests sent in all ({ledger.per_model[args.model]} for {args.model}); "
                  f"report {path}")
    if stopped:
        print(f"stopped: {stopped}", file=sys.stderr)
    return code


def _command() -> int:
    """main() with the documented exit codes: a refusal (SystemExit with a message) or any error that escapes main()
    exits 2, so exit 1 means only that some record ended in vlm_error."""
    try:
        return main()
    except SystemExit as stop:
        if isinstance(stop.code, str):
            print(stop.code, file=sys.stderr)
            return 2
        raise
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(_command())
