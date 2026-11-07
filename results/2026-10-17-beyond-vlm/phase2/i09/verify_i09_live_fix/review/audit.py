"""Reviewer's audit hook: imported first by every script in this folder.

Blocks (raises PermissionError) and counts: any open of a path whose name is .env, starts with .env. or ends with .env;
any access under the main checkout's .cache; any socket connect or address lookup. At exit the counts are appended to
review/audit_log.jsonl with the script name, so the report can say how often the hook fired.
"""
import atexit
import json
import os
import sys
import time

REVIEW = os.path.dirname(os.path.abspath(__file__))
MAIN = "."
HITS = {"env": [], "main_cache": [], "network": []}


def _is_env_name(p: str) -> bool:
    name = os.path.basename(p.rstrip("/")).lower()
    return name == ".env" or name.startswith(".env.") or name.endswith(".env")


def _hook(event, args):
    if event == "open":
        path = args[0] if args else None
        if path is None or isinstance(path, int):
            return
        p = os.path.abspath(os.fsdecode(path))
        if _is_env_name(p):
            HITS["env"].append(p)
            raise PermissionError(f"reviewer audit: blocked open of an env file {p}")
        if p.startswith(MAIN + "/.cache"):
            HITS["main_cache"].append(p)
            raise PermissionError(f"reviewer audit: blocked access to the main checkout's cache {p}")
    elif event in ("os.listdir", "os.scandir"):
        path = args[0] if args else None
        if isinstance(path, (str, bytes, os.PathLike)):
            p = os.path.abspath(os.fsdecode(path))
            if p.startswith(MAIN + "/.cache"):
                HITS["main_cache"].append(p)
                raise PermissionError(f"reviewer audit: blocked listing of the main checkout's cache {p}")
    elif event in ("socket.connect", "socket.getaddrinfo", "socket.gethostbyname"):
        HITS["network"].append(event)
        raise PermissionError(f"reviewer audit: blocked network event {event}")


sys.addaudithook(_hook)


@atexit.register
def _report():
    row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "pid": os.getpid(), "argv": sys.argv[:3],
           "env_hits": len(HITS["env"]), "main_cache_hits": len(HITS["main_cache"]),
           "network_hits": len(HITS["network"]), "detail": {k: v[:5] for k, v in HITS.items() if v}}
    with open(os.path.join(REVIEW, "audit_log.jsonl"), "a", encoding="utf-8") as out:
        out.write(json.dumps(row) + "\n")
