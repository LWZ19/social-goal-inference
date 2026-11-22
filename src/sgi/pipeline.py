"""The pipeline: plan in the far phase, then locate and infer the goal pose on an observation."""
from __future__ import annotations

import time
from typing import Callable, Sequence

from pydantic import BaseModel

from sgi.config import REPO_ROOT, Config, ConfigError, StageName
from sgi.geometry import lift, point_to_pixel
from sgi.imaging import draw_target_marker, encode_png
from sgi.inputs import ObservationData, TourFrameData
from sgi.prompts import Example, check_prompts_dir, stages_of
from sgi.stages.constraints import ConstraintInferrer
from sgi.stages.context import run_context
from sgi.stages.goal import DirectMarkingInferrer, GoalInferrer, standard_goal
from sgi.stages.locate import run_locate
from sgi.stages.people import PeopleSource, VlmPeople
from sgi.stages.plan import run_plan
from sgi.types import (Call, ContextCheck, GoalResult, GroundedTarget, Location, Plan, PlanOutcome, Point, Status,
                       Subtask)
from sgi.vlm.base import Part, VLMClient, VLMResponse, to_call
from sgi.vlm.factory import make_client

_CONSTRAINTS = "constraints"                       # the run.goal under which the constraint method places the goal
# Ground truth is the harness's to hold: a pipeline cannot find the people of `run.people: oracle` by itself.
_NO_PEOPLE_SOURCE = ("run.people: oracle: the people of ground truth are given by the run harness, python -m sgi.run; "
                     "a pipeline used by itself has to be given a people source, or run.people: vlm")


def _tour_positions(n: int, k: int) -> list[int]:
    """Positions of the frames kept from a tour of `n` when at most `k` are shown: evenly spread, both ends kept."""
    if n <= k:
        return list(range(n))
    if k == 1:
        return [0]
    return [i * (n - 1) // (k - 1) for i in range(k)]


class _TimedClient:
    """What a stage uses of a client, adding up the time spent inside `generate`."""

    def __init__(self, client: VLMClient):
        self.entry = client.entry
        self.inside_s = 0.0
        self._client = client

    def generate(self, parts: Sequence[Part], schema: type[BaseModel], system: str | None = None,
                 trial: int = 0) -> VLMResponse:
        started = time.perf_counter()
        try:
            return self._client.generate(parts, schema, system=system, trial=trial)
        finally:
            self.inside_s += time.perf_counter() - started


class Pipeline:
    """Each stage of the pipeline on its own client and strategy from the config.

    The stage prompts are read from the config's `run.prompts_dir`, a folder under the repository root unless it
    is absolute. A folder without the file of a stage of the run (sgi.prompts.stages_of), or with a file that
    lacks a block, is a ConfigError here, before any call, whichever stages `clients` holds."""

    def __init__(self, cfg: Config, clients: dict[StageName, VLMClient], examples: Sequence[Example] = ()):
        self.cfg = cfg
        self.clients = clients
        self.examples = examples
        self.prompts_dir = REPO_ROOT / cfg.run.prompts_dir     # an absolute folder is kept
        check_prompts_dir(self.prompts_dir, stages_of(cfg.run))

    def _timed[T](self, stage: StageName,
                  work: Callable[[_TimedClient], tuple[T, VLMResponse]]) -> tuple[T, VLMResponse, float]:
        """Run one stage on its client: what `work` returns, and the stage's time.

        The time is the wall-clock time of `work` with the response's own `latency_total_s` in place of the
        time `generate` took, so a call answered from the cache counts the same as the call that was stored.
        """
        client = _TimedClient(self.clients[stage])
        started = time.perf_counter()
        output, response = work(client)
        elapsed = time.perf_counter() - started
        return output, response, elapsed - client.inside_s + response.latency_total_s

    def plan(self, request: str, context: str | None, observation: ObservationData | None = None,
             tour: Sequence[TourFrameData] | None = None, trial: int = 0) -> PlanOutcome:
        """Stage 1 on one view or on a tour, exactly one of which is given.

        A plan made on a view carries that view's hash. One made on a tour has none: its navigation subtasks
        name a tour frame, by the index the frame has in the whole tour. The outcome of a tour-mode call lists
        those indices for the frames shown, whatever the call's status.
        """
        if (observation is None) == (tour is None):
            raise ValueError("plan takes exactly one of observation and tour")
        run = self.cfg.run
        kept = None if tour is None else [tour[i] for i in _tour_positions(len(tour), run.max_tour_frames)]

        def work(client):
            if kept is None:
                image, frames = encode_png(observation.rgb, run.max_image_side), None
            else:
                image, frames = None, [(frame.index, encode_png(frame.rgb, run.max_image_side)) for frame in kept]
            return run_plan(client, self.cfg.stages["plan"].strategy, request, context, image, frames, trial=trial,
                            prompts_dir=self.prompts_dir)

        plan, response, time_s = self._timed("plan", work)
        if plan is not None and observation is not None:
            plan = plan.model_copy(update={"image_sha256": observation.sha256()})     # of the array as loaded
        shown = None if kept is None else [frame.index for frame in kept]             # the labels on the images
        return PlanOutcome(status=response.status, plan=plan, call=to_call("plan", response), time_s=time_s,
                           tour_indices=shown)

    def locate(self, observation: ObservationData, subtask: Subtask, request: str, context: str | None,
               trial: int = 0) -> tuple[Location | None, Call, float]:
        """Mark the subtask's target in this observation: the location, the call made and the stage's time.

        The call carries the outcome in its `status`; the location is None unless that is `ok`.
        """
        def work(client):
            image = encode_png(observation.rgb, self.cfg.run.max_image_side)
            return run_locate(client, self.cfg.stages["locate"].strategy, request, context, subtask, image,
                              trial=trial, prompts_dir=self.prompts_dir)

        location, response, time_s = self._timed("locate", work)
        return location, to_call("locate", response), time_s

    def constraint_inferrer(self, people_source: PeopleSource | None = None, trial: int = 0) -> ConstraintInferrer:
        """The inferrer of the constraint method, `run.goal: constraints`, for one trial.

        `people_source` finds the people. Without one a model does, in a call of stage `people`, as
        `run.people: vlm` says; under `run.people: oracle` that is a ConfigError, because the people of ground
        truth are the harness's to give. The constraints are stated by the stage `constraints`, or by nobody
        under `run.constraints: none`. The trial goes into both calls, and so into their cache keys."""
        cfg, th = self.cfg, self.cfg.thresholds
        if people_source is None:
            if cfg.run.people != "vlm":
                raise ConfigError(_NO_PEOPLE_SOURCE)
            people_source = VlmPeople(self.clients["people"], cfg.stages["people"].strategy, th,
                                      max_image_side=cfg.run.max_image_side, trial=trial, prompts_dir=self.prompts_dir)
        return ConstraintInferrer(people_source, self.clients["constraints"], cfg.stages["constraints"].strategy, th,
                                  max_image_side=cfg.run.max_image_side, trial=trial,
                                  constraints_mode=cfg.run.constraints, prompts_dir=self.prompts_dir)

    def infer_goal(self, observation: ObservationData, request: str, context: str | None, plan: Plan,
                   subtask_index: int, *, target_point: Point | None = None,
                   context_check: ContextCheck | None = None, inferrer: GoalInferrer | None = None,
                   trial: int = 0) -> GoalResult:
        """The goal pose for one subtask of the plan on this observation.

        `target_point` stands in for the plan's point and for locate, `context_check` for the context stage, and
        `inferrer` for the method that places the goal where people matter. Without an inferrer the one
        `run.goal` names is built, with this trial: the constraint method under `constraints`, else the
        baseline in which a model marks the goal. A result that stops early carries the status, the calls made
        and whatever was worked out before it stopped. Under `run.goal: constraints` with `run.people: oracle`,
        a call without an inferrer is a ConfigError, raised before any model call.
        """
        cfg, th = self.cfg, self.cfg.thresholds
        if inferrer is None and cfg.run.goal == _CONSTRAINTS:      # here: what it refuses is refused before any call
            inferrer = self.constraint_inferrer(trial=trial)
        subtask = plan.subtasks[subtask_index]
        calls: list[Call] = []
        timings = {"geometry": 0.0}
        location: Location | None = None
        target: GroundedTarget | None = None
        check = context_check                      # the check in effect, whoever made it

        def stopped(status: Status) -> GoalResult:
            return GoalResult(status=status, calls=calls, timings_s=timings, location=location, target=target,
                              context_check=check)

        if subtask.type != "navigate":
            return stopped("no_target")

        point = target_point
        if point is None and subtask.target_point is not None and plan.image_sha256 == observation.sha256():
            point = subtask.target_point           # marked on this very image
        if point is None:
            location, call, timings["locate"] = self.locate(observation, subtask, request, context, trial=trial)
            calls.append(call)
            if call.status != "ok":
                return stopped(call.status)
            point = location.target_point
            if point is None:
                return stopped("no_target")

        started = time.perf_counter()
        camera = observation.camera                # the point is scaled to the full frame, not to the image sent
        u, v = point_to_pixel(point.x, point.y, camera.width, camera.height)
        world = lift(observation.depth, camera, u, v, window=th.lift_window)
        timings["geometry"] = time.perf_counter() - started
        if world is None:
            return stopped("lift_failed")
        target = GroundedTarget(point=point, pixel=(u, v), world=(float(world[0]), float(world[1]), float(world[2])))

        said: ContextCheck | None = None           # the model's own check: the only one the inferrer is given
        if context_check is None:
            def work(client):
                marked = encode_png(draw_target_marker(observation.rgb, u, v), cfg.run.max_image_side)
                return run_context(client, cfg.stages["context"].strategy, request, subtask, marked, th.standoff_m,
                                   trial=trial, prompts_dir=self.prompts_dir)

            said, response, timings["context"] = self._timed("context", work)
            calls.append(to_call("context", response))
            if response.status != "ok":
                return stopped(response.status)
            check = said

        if not check.people_matter:
            result = standard_goal(observation, target, th)
        else:
            if inferrer is None:
                inferrer = DirectMarkingInferrer(self.clients["goal"], cfg.stages["goal"].strategy,
                                                 cfg.run.heading_mode, self.examples, th,
                                                 max_image_side=cfg.run.max_image_side, trial=trial,
                                                 prompts_dir=self.prompts_dir)
            result = inferrer.infer(observation, request, subtask, target, said)
        for stage, seconds in result.timings_s.items():
            timings[stage] = timings.get(stage, 0.0) + seconds
        return result.model_copy(update={"calls": calls + result.calls, "timings_s": timings, "target": target,
                                         "location": location, "context_check": check})


def build_pipeline(cfg: Config, examples: Sequence[Example] = ()) -> Pipeline:
    """The pipeline the config describes, with one client per stage of the run (sgi.prompts.stages_of), built
    from that stage's model entry. No client is built, and no key read, for a stage the run does not have."""
    clients = {stage: make_client(cfg.stages[stage].model, cfg) for stage in stages_of(cfg.run)}
    return Pipeline(cfg, clients, examples)
