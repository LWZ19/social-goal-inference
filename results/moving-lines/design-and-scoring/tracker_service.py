"""People tracks from the robot's own camera, as a local HTTP service for the sim (research arm CTd, E1D_FROZEN.md).

    .venv/bin/python tracker_service.py [--port 8897] [--detector http://127.0.0.1:8899/detect]

POST /track {"key": episode id, "rgb_png", "depth_npy", "camera", "robot", "t"} runs the E1d detector service on the
frame, A1's deployed people on it (own-camera/detector-on-sim-frames/e1d/tracker.py: people_from_raw), and that episode's Tracker.update; it answers
{"tracks": [{"id", "x", "y", "yaw", "stale"}], "detected": n, "ms": round trip}. One Tracker per key, kept until the
service stops. Uses only what the robot has: the frame's rgb, depth and camera.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402
E1D = paths.MOVING / "own-camera" / "detector-on-sim-frames" / "e1d"
sys.path.insert(0, str(E1D))
import tracker as T  # noqa: E402  (E1d's frozen tracker, pinned in E1D.sha256)

from sgi.serve import observation_from  # noqa: E402

TRACKERS: dict[str, T.Tracker] = {}
LOCKS: dict[str, threading.Lock] = {}
GUARD = threading.Lock()
DETECTOR_URL = "http://127.0.0.1:8899/detect"


def track(body: dict) -> dict:
    started = time.perf_counter()
    key = str(body["key"])
    observation = observation_from(body)
    raw = T.detect(base64.b64decode(body["rgb_png"]), url=DETECTOR_URL)
    people = T.people_from_raw(raw, observation.depth, observation.camera)
    with GUARD:
        TRACKERS.setdefault(key, T.Tracker())
        lock = LOCKS.setdefault(key, threading.Lock())
    with lock:
        TRACKERS[key].update(people, observation.camera, observation.depth, float(body["t"]))
        tracks = TRACKERS[key].as_people()
    return {"tracks": tracks, "detected": len(people), "ms": round((time.perf_counter() - started) * 1000, 1)}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send(200, {"ok": True, "episodes": len(TRACKERS)} if self.path == "/health" else {"error": "no such path"})

    def do_POST(self):
        if self.path != "/track":
            self._send(404, {"error": "no such path"})
            return
        try:
            self._send(200, track(json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))))
        except Exception as error:                        # the sim treats a failed frame as no update
            self._send(500, {"error": f"{type(error).__name__}: {error}"})

    def log_message(self, format, *args):
        pass


def main() -> int:
    global DETECTOR_URL
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8897)
    ap.add_argument("--detector", default=DETECTOR_URL)
    args = ap.parse_args()
    DETECTOR_URL = args.detector
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"tracker service on {args.port}, detector {DETECTOR_URL}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
