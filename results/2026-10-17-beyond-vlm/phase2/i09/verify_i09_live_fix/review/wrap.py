"""Reviewer's wrapper: runs live/driver.py as __main__ (runpy, so _command() and its exit codes are what is tested),
with the audit hook of audit.py and sgi.vlm.factory._backend replaced by an in-process fake gateway. Zero calls.

    wrap.py SPEC.json -- <driver args...>

SPEC: {"tag": str, "sleep_s": float, "transient_constraints": int (the first N constraints requests raise
TransientError with retry_after 0.01), "kbint_at": int (the N-th request raises KeyboardInterrupt),
"backend_error_at": int (the N-th request raises BackendError)}
Writes review/wrap_<tag>.json with the exit status as seen in-process (None when a BaseException escapes) and the
fake's requests, and also exits with the driver's status.
"""
import json
import os
import runpy
import sys
import time
from pathlib import Path

REVIEW = Path(__file__).resolve().parent
sys.path.insert(0, str(REVIEW))
import audit  # noqa: E402

LIVE = REVIEW.parents[1] / "live"
spec = json.loads(Path(sys.argv[1]).read_text())
assert sys.argv[2] == "--"
driver_args = sys.argv[3:]
TAG = spec["tag"]

sys.path.insert(0, str(LIVE / "src_202c165" / "src"))
import sgi.vlm.factory as factory  # noqa: E402
from sgi.types import Usage  # noqa: E402
from sgi.vlm.base import BackendError, BackendResult, TransientError  # noqa: E402
from sgi.vlm.mock import centre_responder  # noqa: E402

CALLS = []


class FakeGateway:
    def __init__(self, entry):
        self.provider = entry.provider

    def call(self, entry, system, parts, json_schema):
        n = len(CALLS) + 1
        title = json_schema.get("title")
        CALLS.append({"n": n, "pid": os.getpid(), "model": entry.model, "title": title, "t": time.time()})
        if spec.get("sleep_s"):
            time.sleep(spec["sleep_s"])
        if n == spec.get("kbint_at"):
            raise KeyboardInterrupt("reviewer fake: interrupt")
        if n == spec.get("backend_error_at"):
            raise BackendError("reviewer fake: a non-transient provider error")
        if title == "ConstraintsWire":
            k = sum(c["title"] == "ConstraintsWire" for c in CALLS)
            if k <= spec.get("transient_constraints", 0):
                raise TransientError("reviewer fake: transient", retry_after=0.01)
        return BackendResult(json.dumps(centre_responder(parts, json_schema)), "ok", Usage())


factory._backend = lambda entry: FakeGateway(entry)

code = "not set"
sys.argv = [str(LIVE / "driver.py"), *driver_args]
try:
    runpy.run_path(str(LIVE / "driver.py"), run_name="__main__")
    code = 0
except SystemExit as exc:
    code = exc.code
finally:
    out = {"tag": TAG, "args": driver_args, "exit_in_process": code, "fake_calls": CALLS,
           "env_hits": len(audit.HITS["env"]), "main_cache_hits": len(audit.HITS["main_cache"])}
    (REVIEW / f"wrap_{TAG}.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"WRAP {TAG}: exit {code!r}; fake calls {len(CALLS)} "
          f"({', '.join(sorted({c['title'] for c in CALLS}))}); env hits {len(audit.HITS['env'])}", file=sys.stderr)
sys.exit(code if isinstance(code, int) else 9)
