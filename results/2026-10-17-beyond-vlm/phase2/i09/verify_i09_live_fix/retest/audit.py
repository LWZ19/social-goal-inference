"""Test verifier's audit hook (adapted from verify_i09_live_build/wrap_driver.py), shared by every script here.

install(tag) adds a sys.audit hook that BLOCKS (raises PermissionError) and counts:
- any open of a path whose name is a .env file (".env", ".env.*", "*.env");
- any access under the main checkout's .cache (.cache);
- any write-mode open, mkdir, rename, remove or copy under the main checkout;
- any socket connect or address lookup, any urllib request (no network);
- any open of a path containing one of `block_paths` (the non-existent --env-file of a test).
Write-mode opens elsewhere are logged. At exit (atexit) it writes <V>/audit/audit_<tag>_<pid>.json.
"""
import atexit
import json
import os
import sys
from pathlib import Path

V = Path(__file__).resolve().parent
MAIN = "."
AUDIT = {"tag": None, "pid": None, "env_hits": 0, "main_cache_hits": 0, "network_hits": 0, "main_write_hits": 0,
         "key_file_hits": 0, "blocked": [], "writes_outside_V": [], "n_writes": 0, "subprocess": []}
_BLOCK = []


def _is_env_name(p: str) -> bool:
    name = os.path.basename(p.rstrip("/")).lower()
    return name == ".env" or name.startswith(".env.") or name.endswith(".env")


def _allowed_write(p: str) -> bool:
    home = str(Path.home())
    return (p.startswith(str(V)) or p.startswith(home + "/.cache/") or p.startswith(home + "/.config/matplotlib")
            or p.startswith("/tmp/") or p.startswith("/dev/") or "/__pycache__/" in p or p.startswith("/proc/"))


def _hook(event, args):
    if event == "open":
        path, mode, flags = (list(args) + [None, None, None])[:3]
        if isinstance(path, int) or path is None:
            return
        p = os.path.abspath(os.fsdecode(path))
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
        if any(b in p for b in _BLOCK):
            AUDIT["key_file_hits"] += 1
            AUDIT["blocked"].append(["key_file", p])
            raise PermissionError(f"verifier: blocked open of the named key file {p}")
        if _is_env_name(p):
            AUDIT["env_hits"] += 1
            AUDIT["blocked"].append(["env", p])
            raise PermissionError(f"verifier: blocked open of an env file {p}")
        if p.startswith(MAIN + "/.cache"):
            AUDIT["main_cache_hits"] += 1
            AUDIT["blocked"].append(["main_cache", p])
            raise PermissionError(f"verifier: blocked access to the main checkout's cache {p}")
        if writing:
            if p.startswith(MAIN + "/"):
                AUDIT["main_write_hits"] += 1
                AUDIT["blocked"].append(["main_write", p])
                raise PermissionError(f"verifier: blocked write into the main checkout {p}")
            AUDIT["n_writes"] += 1
            if not _allowed_write(p) and len(AUDIT["writes_outside_V"]) < 200:
                AUDIT["writes_outside_V"].append(p)
    elif event in ("os.mkdir", "os.rename", "os.remove", "os.rmdir", "shutil.copyfile", "shutil.rmtree",
                   "os.replace", "os.unlink", "os.symlink", "os.link", "os.truncate"):
        paths = [os.path.abspath(os.fsdecode(a)) for a in args if isinstance(a, (str, bytes, os.PathLike))]
        for p in paths:
            if p.startswith(MAIN + "/.cache"):
                AUDIT["main_cache_hits"] += 1
                AUDIT["blocked"].append([event, p])
                raise PermissionError(f"verifier: blocked {event} in the main cache {p}")
            if p.startswith(MAIN + "/"):
                AUDIT["main_write_hits"] += 1
                AUDIT["blocked"].append([event, p])
                raise PermissionError(f"verifier: blocked {event} in the main checkout {p}")
    elif event in ("os.listdir", "os.scandir"):
        paths = [os.path.abspath(os.fsdecode(a)) for a in args if isinstance(a, (str, bytes, os.PathLike))]
        for p in paths:
            if p.startswith(MAIN + "/.cache"):
                AUDIT["main_cache_hits"] += 1
                AUDIT["blocked"].append([event, p])
                raise PermissionError(f"verifier: blocked listing of the main cache {p}")
    elif event in ("socket.connect", "socket.getaddrinfo", "socket.gethostbyname", "urllib.Request",
                   "socket.sendto", "socket.bind"):
        AUDIT["network_hits"] += 1
        AUDIT["blocked"].append([event, repr(args)[:200]])
        raise PermissionError(f"verifier: blocked network event {event}")
    elif event in ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn"):
        AUDIT["subprocess"].append([event, repr(args)[:300]])


def _dump():
    try:
        d = V / "audit"
        d.mkdir(exist_ok=True)
        (d / f"audit_{AUDIT['tag']}_{AUDIT['pid']}.json").write_text(json.dumps(AUDIT, indent=1, default=str))
    except Exception as exc:                        # never mask the process's own exit
        print(f"audit dump failed: {exc}", file=sys.stderr)


def install(tag: str, block_paths=()):
    AUDIT["tag"], AUDIT["pid"] = tag, os.getpid()
    _BLOCK.extend(block_paths)
    sys.addaudithook(_hook)
    atexit.register(_dump)
    return AUDIT


def summary() -> str:
    return (f"env_hits {AUDIT['env_hits']} main_cache_hits {AUDIT['main_cache_hits']} network_hits "
            f"{AUDIT['network_hits']} main_write_hits {AUDIT['main_write_hits']} key_file_hits {AUDIT['key_file_hits']} "
            f"writes_outside_V {len(AUDIT['writes_outside_V'])}")
