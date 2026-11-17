"""Where the scripted people are, at 2 Hz from t = 0, with the robot standing at its start and no request sent.

Zero model calls: the episode gets a client that raises on any use. The world is built by the same class as the
bystander runs (experiment_oracle.OracleEpisode, which also plays scenario events) with the same seed, so the people
follow the same random draws. The people never react to the robot except that a walker waits up to 2 s for anyone
ahead of it, the robot included; so a dump is exact for any arm while its robot has not moved (until its first
answer is delivered, for every arm but the move-while-thinking ones), and may differ near the robot after that.

Run with the sim's Python, from the repo root:
    PYTHONPATH=sim ~/miniforge3/envs/sgi-habitat/bin/python <this file> --scenario sim/scenarios/X.yaml --seed N \
        --out DIR/X/seedN.json [--seconds 92]
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import yaml

from sgi_habitat import experiment_oracle as O
from sgi_habitat import run as R


class _NoClient:
    """A model client that refuses: a dump must never ask anything."""

    def __getattr__(self, name):
        raise RuntimeError(f"a dump makes no model calls (client.{name} was used)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scenario", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=92.0)
    ap.add_argument("--every", type=float, default=0.5)
    args = ap.parse_args(argv)

    scenario = yaml.safe_load(args.scenario.read_text())
    started = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        episode = O.OracleEpisode(scenario, _NoClient(), Path(tmp), "CT2", args.seed, 10.0, 0.5,
                                  scenario["requests"][0])
        bystanders = set(episode.log["bystanders"])
        rx, ry, ryaw = episode.robot.world_pose
        frames, next_t = [], 0.0
        try:
            while episode.t < args.seconds:
                if episode.t >= next_t - 1e-9:
                    members = [h for h in episode.queue.members if not isinstance(h, O.X.E.RobotInLine)]
                    frames.append({"t": round(episode.t, 3), "people": [
                        {"id": i, "xy": [round(float(h.world_xy[0]), 3), round(float(h.world_xy[1]), 3)],
                         "yaw": round(float(h.yaw), 3), "in_queue": h in members,
                         "order": members.index(h) if h in members else None,
                         "walking": bool(h.path), "aside": h in episode.queue.aside}
                        for i, h in enumerate(episode.humans)]})
                    next_t += args.every
                episode.step(1.0 / R.FPS)
        finally:
            episode.sim.close()
    doc = {"scenario": args.scenario.stem, "seed": args.seed, "every_s": args.every, "fps": R.FPS,
           "robot_start": [round(rx, 3), round(ry, 3), round(ryaw, 3)], "bystanders": sorted(bystanders),
           "note": "robot idle at its start, no request sent; exact for an arm until its robot first moves",
           "made_by": "dump_humans.py", "wall_s": round(time.time() - started, 1), "frames": frames}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc))
    print(f"[dump] {args.scenario.stem} seed {args.seed}: {len(frames)} frames, {len(episode.humans)} people, "
          f"{time.time() - started:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
