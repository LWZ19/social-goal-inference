"""The robot's picture at the request time (t = 2 s) for candidate layouts, for a visual check. Zero model calls.

    PYTHONPATH=sim ~/miniforge3/envs/sgi-habitat/bin/python <this file> --out DIR LAYOUT.yaml [...]
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml

from sgi_habitat import experiment_oracle as O
from sgi_habitat import run as R


class _NoClient:
    def __getattr__(self, name):
        raise RuntimeError(f"no model calls here (client.{name})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("layouts", nargs="+", type=Path)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for path in args.layouts:
        scenario = yaml.safe_load(path.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            ep = O.OracleEpisode(scenario, _NoClient(), Path(tmp), "CT2", args.seed, 10.0, 0.5, scenario["requests"][0])
            try:
                while ep.t < 2.0:
                    ep.step(1.0 / R.FPS)
                frames = ep.render()
                rgb = np.ascontiguousarray(frames["rgb"][..., :3])
                view = np.ascontiguousarray(frames["view"][..., :3])
                h = rgb.shape[0]
                view = cv2.resize(view, (int(view.shape[1] * h / view.shape[0]), h))
                cv2.imwrite(str(args.out / f"{path.stem}.jpg"), cv2.cvtColor(np.hstack([rgb, view]), cv2.COLOR_RGB2BGR))
            finally:
                ep.sim.close()
        print(f"[render] {path.stem}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
