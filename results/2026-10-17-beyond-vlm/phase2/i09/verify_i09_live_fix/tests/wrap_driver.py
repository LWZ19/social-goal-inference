"""Test verifier's wrapper around the NEW driver.py (adapted from verify_i09_live_build/wrap_driver.py).

    wrap_driver.py SPEC.json -- <driver args...>

The driver runs as its own __main__ via runpy.run_path(LIVE/driver.py, run_name="__main__"), so the driver's own
`if __name__ == "__main__": raise SystemExit(_command())` runs: the real command path and its exit codes.

SPEC: {"tag": str, "fake_gateway": bool, "invalid_at": [int], "transient_at": [int], "raise_at": [int],
       "sticky_transient": bool, "sleep_s": float, "scene_error_on": str, "capture_mock": bool,
       "block_paths": [str]}

Added in this process before the driver runs:
- audit.install(tag): blocks and counts .env opens, the main cache, writes into the main checkout, network.
- fake_gateway: sgi.vlm.factory._backend is replaced by a fake with the entry's own provider (so cache keys are the
  live ones), answering as the package's centre_responder does; options: an invalid answer at given call numbers
  (validation retry), a TransientError at given call numbers, a RuntimeError at given call numbers, a TransientError
  on EVERY attempt of the first distinct request it sees (sticky_transient: that record ends in vlm_error), a sleep
  per call. Every call is appended at once to <V>/fake/fake_<tag>_<pid>.jsonl (survives a SIGINT).
- scene_error_on: sgi.run._refuse_failed_check raises the package's SceneError for that scene id.
- capture_mock: sgi.vlm.mock.MockBackend keeps every request (texts, image hashes, schema title).
At exit it writes <V>/wrap/wrap_<tag>.json with the exit code, the fake's calls and the audit counts.
"""
import hashlib
import json
import os
import runpy
import sys
import time
from pathlib import Path

V = Path(__file__).resolve().parent
sys.path.insert(0, str(V))
import audit  # noqa: E402

LIVE = Path("results/2026-10-17-beyond-vlm/phase2/i09/live")
spec = json.loads(Path(sys.argv[1]).read_text())
assert sys.argv[2] == "--"
driver_args = sys.argv[3:]
TAG = spec["tag"]
AUD = audit.install(TAG, spec.get("block_paths", []))

sys.path.insert(0, str(LIVE / "src_202c165" / "src"))
import sgi  # noqa: E402
import sgi.run as sgi_run  # noqa: E402
import sgi.vlm.factory as factory  # noqa: E402
import sgi.vlm.mock as mockmod  # noqa: E402
from sgi.scene import SceneError  # noqa: E402
from sgi.types import Usage  # noqa: E402
from sgi.vlm.base import BackendResult, ImagePart, TextPart, TransientError  # noqa: E402

assert Path(sgi.__file__).resolve().is_relative_to(LIVE / "src_202c165" / "src")
FAKE_CALLS = []
CAPTURED = []
(V / "fake").mkdir(exist_ok=True)
(V / "wrap").mkdir(exist_ok=True)
FAKE_LOG = V / "fake" / f"fake_{TAG}_{os.getpid()}.jsonl"
STICKY = {"digest": None}


def _digest(system, parts):
    h = hashlib.sha256((system or "").encode())
    for p in parts:
        h.update(p.text.encode() if isinstance(p, TextPart) else hashlib.sha256(p.png).digest())
    return h.hexdigest()[:16]


class FakeGateway:
    def __init__(self, entry):
        self.provider = entry.provider
        self.entry = entry

    def call(self, entry, system, parts, json_schema):
        n = len(FAKE_CALLS) + 1
        dig = _digest(system, parts)
        row = {"n": n, "pid": os.getpid(), "t": time.time(), "model": entry.model, "provider": entry.provider,
               "base_url": entry.base_url, "schema_title": json_schema.get("title"), "n_parts": len(parts),
               "digest": dig, "person_lines": sum(p.text.count("Person ") for p in parts if isinstance(p, TextPart))}
        outcome = "ok"
        if spec.get("sticky_transient") and STICKY["digest"] is None:
            STICKY["digest"] = dig
        if spec.get("sticky_transient") and dig == STICKY["digest"]:
            outcome = "transient"
        elif n in spec.get("transient_at", []):
            outcome = "transient"
        elif n in spec.get("raise_at", []):
            outcome = "runtime_error"
        elif n in spec.get("invalid_at", []):
            outcome = "invalid"
        row["outcome"] = outcome
        FAKE_CALLS.append(row)
        with open(FAKE_LOG, "a") as f:
            f.write(json.dumps(row) + "\n")
        if spec.get("sleep_s"):
            time.sleep(spec["sleep_s"])
        if outcome == "transient":
            raise TransientError("verifier fake: transient", retry_after=0.01)
        if outcome == "runtime_error":
            raise RuntimeError("verifier fake: an unexpected error mid-run")
        if outcome == "invalid":
            return BackendResult("this is not json", "ok", Usage())
        return BackendResult(json.dumps(mockmod.centre_responder(parts, json_schema)), "ok", Usage())


def _fake_backend(entry):
    return FakeGateway(entry)


def _real_backend_forbidden(entry):
    raise PermissionError("verifier: a real back end was requested and refused")


factory._backend = _fake_backend if spec.get("fake_gateway") else _real_backend_forbidden

if spec.get("restore_sigint"):
    # A background job of a non-interactive shell inherits SIGINT = SIG_IGN, so Python never installs its handler;
    # restore it so a SIGINT acts like Ctrl-C in a terminal (KeyboardInterrupt in the main thread).
    import signal
    signal.signal(signal.SIGINT, signal.default_int_handler)

if spec.get("scene_error_on"):
    _orig_refuse = sgi_run._refuse_failed_check

    def _refuse(folder, scene_id, cfg):
        if scene_id == spec["scene_error_on"]:
            raise SceneError(f"{scene_id}: verifier-injected scene check failure")
        return _orig_refuse(folder, scene_id, cfg)

    sgi_run._refuse_failed_check = _refuse

if spec.get("capture_mock"):
    Orig = mockmod.MockBackend

    class CapturingMock(Orig):
        def call(self, entry, system, parts, json_schema):
            CAPTURED.append({"title": json_schema.get("title"),
                             "texts": [p.text for p in parts if isinstance(p, TextPart)],
                             "images": [hashlib.sha256(p.png).hexdigest() for p in parts if isinstance(p, ImagePart)]})
            return super().call(entry, system, parts, json_schema)

    mockmod.MockBackend = CapturingMock

code = None
sys.argv = [str(LIVE / "driver.py"), *driver_args]
try:
    runpy.run_path(str(LIVE / "driver.py"), run_name="__main__")
    code = 0
except SystemExit as exc:
    code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else repr(exc.code))
finally:
    out = {"tag": TAG, "args": driver_args, "exit": code, "pid": os.getpid(), "audit": AUD,
           "fake_calls": FAKE_CALLS, "captured": CAPTURED}
    (V / "wrap" / f"wrap_{TAG}.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"WRAP {TAG}: exit {code}; fake calls {len(FAKE_CALLS)}; captured {len(CAPTURED)}; {audit.summary()}",
          flush=True)
sys.exit(code if isinstance(code, int) else 9)
