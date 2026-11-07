"""Verifier's wrapper around the builder's driver.py (the code under test, run as its own __main__ via runpy).

    wrap_driver.py SPEC.json -- <driver args...>

SPEC: {"tag": str, "fake_gateway": bool, "capture_mock": bool, "invalid_at": [int], "transient_at": [int]}

What it adds, all in this process, before the driver runs:
- an audit hook that BLOCKS (raises) and logs: any open of a file named like a .env file; any access under the main
  checkout's .cache; any write-mode open, mkdir, rename, remove or copy under the main checkout; any socket connect or
  address lookup (no network); any subprocess. Write-mode opens elsewhere are logged with their path.
- fake_gateway: sgi.vlm.factory._backend is replaced by a counting fake with the entry's own provider, answering as the
  package's centre_responder does (optionally an invalid answer or a TransientError at given call numbers). The real
  back end and the key reader are never reached.
- capture_mock: sgi.vlm.mock.MockBackend is replaced by a subclass that keeps every request (system, parts, schema).
At exit it writes <V>/wrap_<tag>.json with the exit code, the audit log, the fake's calls and the captured requests
(texts and image hashes; constraints images saved as PNG).
"""
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

V = Path(__file__).resolve().parent
LIVE = V.parent / "live"
MAIN = "."
spec = json.loads(Path(sys.argv[1]).read_text())
assert sys.argv[2] == "--"
driver_args = sys.argv[3:]
TAG = spec["tag"]
AUDIT = {"blocked": [], "writes": [], "fs_ops": [], "subprocess": []}


def _is_env_name(p: str) -> bool:
    name = os.path.basename(p.rstrip("/")).lower()
    return name == ".env" or name.startswith(".env.") or name.endswith(".env")


def _hook(event, args):
    if event == "open":
        path, mode, flags = (list(args) + [None, None, None])[:3]
        if isinstance(path, int) or path is None:
            return
        p = os.path.abspath(os.fsdecode(path))
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
        if any(b in p for b in spec.get("block_paths", [])):
            AUDIT["blocked"].append(["key_file", p])
            raise PermissionError(f"verifier: blocked open of the named key file {p}")
        if _is_env_name(p):
            AUDIT["blocked"].append(["env", p])
            raise PermissionError(f"verifier: blocked open of an env file {p}")
        if p.startswith(MAIN + "/.cache"):
            AUDIT["blocked"].append(["main_cache", p])
            raise PermissionError(f"verifier: blocked access to the main checkout's cache {p}")
        if writing:
            if p.startswith(MAIN + "/"):
                AUDIT["blocked"].append(["main_write", p])
                raise PermissionError(f"verifier: blocked write into the main checkout {p}")
            AUDIT["writes"].append(p)
    elif event in ("os.mkdir", "os.rename", "os.remove", "os.rmdir", "shutil.copyfile", "shutil.rmtree"):
        paths = [os.path.abspath(os.fsdecode(a)) for a in args if isinstance(a, (str, bytes, os.PathLike))]
        for p in paths:
            if p.startswith(MAIN + "/"):
                AUDIT["blocked"].append([event, p])
                raise PermissionError(f"verifier: blocked {event} in the main checkout {p}")
        AUDIT["fs_ops"].append([event, *paths])
    elif event in ("socket.connect", "socket.getaddrinfo", "socket.gethostbyname", "urllib.Request"):
        AUDIT["blocked"].append([event, repr(args)[:200]])
        raise PermissionError(f"verifier: blocked network event {event}")
    elif event in ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn"):
        AUDIT["subprocess"].append([event, repr(args)[:300]])


sys.addaudithook(_hook)
sys.path.insert(0, str(LIVE / "src_202c165" / "src"))
import sgi  # noqa: E402
import sgi.vlm.factory as factory  # noqa: E402
import sgi.vlm.mock as mockmod  # noqa: E402
from sgi.vlm.base import BackendResult, ImagePart, TextPart, TransientError  # noqa: E402
from sgi.types import Usage  # noqa: E402

assert Path(sgi.__file__).resolve().is_relative_to(LIVE / "src_202c165" / "src")
FAKE_CALLS = []
CAPTURED = []


def _summ(system, parts, schema):
    return {"system": system, "texts": [p.text for p in parts if isinstance(p, TextPart)],
            "images": [hashlib.sha256(p.png).hexdigest() for p in parts if isinstance(p, ImagePart)],
            "schema": schema}


class FakeGateway:
    def __init__(self, entry):
        self.provider = entry.provider
        self.entry = entry

    def call(self, entry, system, parts, json_schema):
        n = len(FAKE_CALLS) + 1
        FAKE_CALLS.append({"n": n, "model": entry.model, "provider": entry.provider, "base_url": entry.base_url,
                           "schema_title": json_schema.get("title"), "n_parts": len(parts)})
        if n in spec.get("transient_at", []):
            raise TransientError("verifier fake: transient")
        if n in spec.get("invalid_at", []):
            return BackendResult("this is not json", "ok", Usage())
        return BackendResult(json.dumps(mockmod.centre_responder(parts, json_schema)), "ok", Usage())


def _fake_backend(entry):
    return FakeGateway(entry)


if spec.get("fake_gateway"):
    factory._backend = _fake_backend

if spec.get("capture_mock"):
    Orig = mockmod.MockBackend

    class CapturingMock(Orig):
        def call(self, entry, system, parts, json_schema):
            rec = _summ(system, parts, json_schema)
            rec["entry_model"] = entry.model
            CAPTURED.append(rec)
            if "Person 1" in "\n".join(rec["texts"]) or json_schema.get("title", "").lower().startswith("constraint"):
                imgs = [p for p in parts if isinstance(p, ImagePart)]
                if imgs:
                    d = V / f"wrap_{TAG}_images"
                    d.mkdir(exist_ok=True)
                    (d / f"req{len(CAPTURED):03d}.png").write_bytes(imgs[-1].png)
                    rec["saved_image"] = str(d / f"req{len(CAPTURED):03d}.png")
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
    out = {"tag": TAG, "args": driver_args, "exit": code, "audit": AUDIT, "fake_calls": FAKE_CALLS,
           "captured": CAPTURED}
    (V / f"wrap_{TAG}.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"WRAP {TAG}: exit {code}; fake calls {len(FAKE_CALLS)}; captured {len(CAPTURED)}; blocked "
          f"{AUDIT['blocked']}; subprocess {len(AUDIT['subprocess'])}")
sys.exit(code if isinstance(code, int) else 9)
