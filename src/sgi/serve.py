"""The pipeline as a local HTTP service, for a simulator that runs under another Python (docs/habitat.md).

    python -m sgi.serve [--model ENTRY] [--stage-model STAGE=ENTRY ...] [--strategy STAGE=NAME ...]
                        [--goal constraints|direct_marking] [--host HOST] [--port PORT]

One request is one live observation and one request of a person; the answer is the goal pose of the plan's
first navigation subtask, in the world frame of the observation and in the robot frame. Nothing is scored and
no ground truth is read: the service is given what a robot has, and nothing else.

    GET  /health   {"ok": true, "goal": ..., "models": {stage: entry}}
    POST /goal     the body below; the answer below

The body is JSON. `rgb_png` is a PNG, `depth_npy` an .npy file of an (H, W) float32 array of camera Z in metres,
each in base64. `camera` holds fx, fy, cx, cy, width, height and T_world_camera, a 4 x 4 list that maps optical
camera coordinates (x right, y down, z forward) into the world frame, in which z is up and the floor is z = 0.
`robot` holds x, y and yaw in that frame. `context` may be null.

The answer holds `status` (ok, or why there is no goal), `plan_status`, the `subtasks` of the plan as
[type, description] pairs, `subtask_index` of the one the goal is for, `source` (standard, direct_marking or
constraints), `pose_world` and `pose_robot` as [x, y, theta] or null, `target_world` as [x, y, z] or null, the
`calls` made as [stage, status] pairs and `timings_s`. Under `inference` is what the stages worked out, for a
viewer to show: the `plan` (its reasoning and subtasks, each with the target it names and its point), and, once
the goal stage ran, the `target` (its 0..1000 point, its pixel and its world point), the `context_check` (the
people it names, whether they matter, and why), the `goal_mark` and `goal_pixel` of the baseline, the `people`
found and the `constraints` stated with what the solver made of each, the runner-up `candidates`, the `solver`'s
note and the `flags` that are set; and each call's `stage`, `status`, `model`, `latency_s`, `cached` and `error`.
The models' raw text is left out. A body that cannot be read is answered with status 400 and {"error": ...}.

The goal method is `--goal`: constraints by default, the method that measured best. Under direct_marking the goal
stage is put on cot unless --strategy names its strategy: a service has no example scenes to show.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Sequence

import numpy as np
from PIL import Image

from sgi.config import Config, ConfigError, load_config, with_model_options
from sgi.geometry import Camera, RobotPose
from sgi.inputs import ObservationData
from sgi.pipeline import Pipeline, build_pipeline
from sgi.vlm.base import QuotaExceededError
from sgi.vlm.env import MissingKeyError

log = logging.getLogger("sgi.serve")

_CONSTRAINTS, _BASELINE = "constraints", "direct_marking"
_WITH_EXAMPLES = ("few_shot", "few_shot_cot")      # goal strategies that show example scenes


class RequestError(ValueError):
    """A body the service cannot read: answered with status 400."""


def make_config(model: str | None, stage_models: Sequence[str] = (), strategies: Sequence[str] = (),
                goal: str = _CONSTRAINTS) -> Config:
    """The default configuration under the model options, with `run.goal` set to `goal`. Under direct_marking
    a goal strategy that shows examples is replaced by cot, unless `strategies` names the goal's strategy."""
    cfg = with_model_options(load_config(overrides={"run": {"goal": goal}}), model, stage_models, strategies)
    named = any(value.partition("=")[0].strip() == "goal" for value in strategies)
    if goal == _BASELINE and not named and cfg.stages["goal"].strategy in _WITH_EXAMPLES:
        cfg = with_model_options(cfg, None, (), ["goal=cot"])
    return cfg


def _field(body: dict, key: str) -> Any:
    if key not in body:
        raise RequestError(f"the body has no '{key}'")
    return body[key]


def observation_from(body: dict) -> ObservationData:
    """The observation a /goal body describes. A missing field or one of the wrong shape is a RequestError."""
    try:
        rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(_field(body, "rgb_png")))).convert("RGB"))
        depth = np.load(io.BytesIO(base64.b64decode(_field(body, "depth_npy"))), allow_pickle=False)
        cam, robot = _field(body, "camera"), _field(body, "robot")
        T = np.asarray(cam["T_world_camera"], dtype=np.float64)
        camera = Camera(fx=float(cam["fx"]), fy=float(cam["fy"]), cx=float(cam["cx"]), cy=float(cam["cy"]),
                        width=int(cam["width"]), height=int(cam["height"]), T_world_camera=T)
        pose = RobotPose(x=float(robot["x"]), y=float(robot["y"]), yaw=float(robot["yaw"]))
    except RequestError:
        raise
    except (KeyError, TypeError, ValueError, OSError) as error:
        raise RequestError(f"the observation cannot be read: {error}") from error
    depth = depth.astype(np.float32)
    if T.shape != (4, 4):
        raise RequestError(f"T_world_camera is {T.shape}, not 4 x 4")
    if depth.shape != rgb.shape[:2] or depth.shape != (camera.height, camera.width):
        raise RequestError(f"rgb {rgb.shape[:2]}, depth {depth.shape} and camera {(camera.height, camera.width)} "
                           "differ in size")
    return ObservationData(rgb=rgb, depth=depth, camera=camera, robot=pose)


def _triple(values) -> list[float] | None:
    return None if values is None else [float(v) for v in values]


_CALL_FIELDS = {"stage", "status", "model", "latency_s", "cached", "error"}


def _inference(plan, calls, result=None) -> dict:
    """What the stages worked out, as JSON: the `inference` of a /goal answer."""
    shown = {"plan": None if plan is None else plan.model_dump(mode="json", exclude={"image_sha256"}),
             "calls": [call.model_dump(mode="json", include=_CALL_FIELDS) for call in calls]}
    if result is not None:
        shown.update(result.model_dump(mode="json", exclude={"calls", "status", "source", "pose_world", "pose_robot",
                                                             "timings_s", "flags", "location"}))
        shown["flags"] = [name for name, value in result.flags.model_dump().items() if value]
    return shown


def infer(pipeline: Pipeline, body: dict) -> dict:
    """Plan on the observation and place the goal of the plan's first navigation subtask: the /goal answer."""
    request = _field(body, "request")
    if not isinstance(request, str) or not request.strip():
        raise RequestError("'request' has to be a non-empty string")
    context = body.get("context")
    observation = observation_from(body)
    started = time.perf_counter()
    outcome = pipeline.plan(request, context, observation=observation)
    subtasks = [] if outcome.plan is None else outcome.plan.subtasks
    answer = {"status": outcome.status, "plan_status": outcome.status,
              "subtasks": [[subtask.type, subtask.description] for subtask in subtasks], "subtask_index": None,
              "source": None, "pose_world": None, "pose_robot": None, "target_world": None,
              "calls": [] if outcome.call is None else [[outcome.call.stage, outcome.call.status]],
              "timings_s": {"plan": outcome.time_s},
              "inference": _inference(outcome.plan, [] if outcome.call is None else [outcome.call])}
    index = next((i for i, subtask in enumerate(subtasks) if subtask.type == "navigate"), None)
    if outcome.plan is None or index is None:
        if outcome.plan is not None:
            answer["status"] = "no_target"
        return answer
    result = pipeline.infer_goal(observation, request, context, outcome.plan, index)
    answer.update({"status": result.status, "subtask_index": index, "source": result.source,
                   "pose_world": _triple(result.pose_world), "pose_robot": _triple(result.pose_robot),
                   "target_world": None if result.target is None else _triple(result.target.world),
                   "calls": answer["calls"] + [[call.stage, call.status] for call in result.calls],
                   "timings_s": {**answer["timings_s"], **result.timings_s,
                                 "total": time.perf_counter() - started},
                   "inference": _inference(outcome.plan, [outcome.call, *result.calls], result)})
    return answer


def make_handler(pipeline: Pipeline) -> type[BaseHTTPRequestHandler]:
    cfg = pipeline.cfg
    health = {"ok": True, "goal": cfg.run.goal, "models": {stage: cfg.stages[stage].model for stage in pipeline.clients}}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, health)
            else:
                self._send(404, {"error": f"no such path: {self.path}"})

        def do_POST(self):
            if self.path != "/goal":
                self._send(404, {"error": f"no such path: {self.path}"})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                if not isinstance(body, dict):
                    raise RequestError("the body is not a JSON object")
                answer = infer(pipeline, body)
            except (RequestError, json.JSONDecodeError) as error:
                self._send(400, {"error": str(error)})
                return
            except QuotaExceededError as error:
                self._send(503, {"error": f"the provider's allowance is used up: {error}"})
                return
            log.info("%s: %s, pose_world %s", body.get("request"), answer["status"], answer["pose_world"])
            self._send(200, answer)

        def log_message(self, format, *args):      # the answers are logged above, one line each
            pass

    return Handler


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sgi.serve",
                                     description="Serve the pipeline over HTTP for a live simulator.")
    parser.add_argument("--model", help="entry of the model table in configs/default.yaml, used for every stage")
    parser.add_argument("--stage-model", action="append", default=[], metavar="STAGE=ENTRY",
                        help="entry to use for one stage, applied after --model; may be given several times")
    parser.add_argument("--strategy", action="append", default=[], metavar="STAGE=NAME",
                        help="prompting strategy for one stage; may be given several times")
    parser.add_argument("--goal", choices=(_CONSTRAINTS, _BASELINE), default=_CONSTRAINTS,
                        help="the method that places the goal where people matter (default: constraints)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    try:
        pipeline = build_pipeline(make_config(args.model, args.stage_model, args.strategy, args.goal))
    except (MissingKeyError, ConfigError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    server = HTTPServer((args.host, args.port), make_handler(pipeline))
    models = ", ".join(f"{stage}={pipeline.cfg.stages[stage].model}" for stage in pipeline.clients)
    print(f"sgi.serve: listening on http://{args.host}:{args.port} (goal {args.goal}; {models})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
