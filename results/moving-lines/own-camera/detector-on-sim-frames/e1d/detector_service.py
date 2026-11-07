"""E1d detector service: A1's frozen RT-DETRv2 + ViTPose behind a local HTTP endpoint, for detector tracking in the sim.
Run in the torch sidecar, read-only, never importing sgi:
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -I detector_service.py --port 8899
POST /detect  {"rgb_png": base64}  ->  {"boxes": [[x0,y0,x1,y1,score],...], "pose": [{"box_index": i, "kp": [...]}],
                                        "W": w, "H": h, "t_det_s": s, "t_pose_s": s}
The answer has the fields of A1's raw pass line, so A1's pipeline.run_image turns it into people (in the sgi env, with
the frame's depth and camera). One model instance; requests are served one at a time. GET /health -> {"ok": true}.
"""
import sys

sys.dont_write_bytecode = True

import base64  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from http.server import BaseHTTPRequestHandler, HTTPServer  # noqa: E402

RESEARCH = "results/2026-10-17-beyond-vlm"
INFER = RESEARCH + "/phase1c/i09/a1/sidecar/infer.py"
INFER_SHA = "bee1322fd692bf0e85e820871413d486fda04d5770f72b5d3d75e18266709484"
DETECTOR = "rtdetr_v2_r50vd"


def load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8899
    assert hashlib.sha256(open(INFER, "rb").read()).hexdigest() == INFER_SHA, "infer.py sha256 mismatch"
    infer = load_by_path("infer_a1", INFER)
    assert infer.BOX_MIN == 0.05 and infer.POSE_MIN == 0.10
    torch = infer.setup()
    dev = torch.device("cuda")
    det = infer.load_detector(torch, DETECTOR, dev)
    pproc, pmodel = infer.load_pose(torch, dev)
    from PIL import Image

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send(200 if self.path == "/health" else 404, {"ok": self.path == "/health"})

        def do_POST(self):
            if self.path != "/detect":
                return self._send(404, {"error": "unknown path"})
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                img = Image.open(io.BytesIO(base64.b64decode(body["rgb_png"]))).convert("RGB")
                boxes, (t_det, _) = infer.detect(torch, *det, img, dev, infer.BOX_MIN)
                pb = [b for b in boxes if b[4] >= infer.POSE_MIN]
                kps, t_pose = infer.pose(torch, pproc, pmodel, img, [b[:4] for b in pb], dev)
                self._send(200, {"boxes": boxes, "pose": [{"box_index": i, "kp": kps[i]} for i in range(len(pb))],
                                 "W": img.size[0], "H": img.size[1], "detector": DETECTOR,
                                 "t_det_s": t_det, "t_pose_s": t_pose})
            except Exception as e:                         # a bad request never takes the service down
                self._send(400, {"error": repr(e)})

        def log_message(self, fmt, *args):
            print(time.strftime("%H:%M:%S"), fmt % args, flush=True)

    print(f"detector service on 127.0.0.1:{port} ({DETECTOR} + ViTPose, A1 frozen)", flush=True)
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
