"""T6: the dryrun.py checks against the NEW driver, without editing LIVE and without the main cache.

    run_dryrun.py OUT_DIR part [part ...]

dryrun.py's source is read and exec'd in a fresh module whose __file__ is LIVE/dryrun.py (so `import driver as D`
imports LIVE/driver.py, the new build), under __name__ "dryrun_exec" (its __main__ block does not run). Its OUT
(LIVE/dryrun) is then replaced by OUT_DIR before any part runs, so every write goes to my folder. Each part runs in
a try: an exception is recorded with its type and message. The audit hook of audit.py is on.
"""
import json
import sys
import traceback
import types
from pathlib import Path

V = Path(__file__).resolve().parent
sys.path.insert(0, str(V))
import audit  # noqa: E402

AUD = audit.install("dryrun")
LIVE = Path("results/2026-10-17-beyond-vlm/phase2/i09/live")
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
parts = sys.argv[2:]

M = types.ModuleType("dryrun_exec")
M.__file__ = str(LIVE / "dryrun.py")
sys.modules["dryrun_exec"] = M
exec(compile((LIVE / "dryrun.py").read_text(encoding="utf-8"), str(LIVE / "dryrun.py"), "exec"), M.__dict__)
assert Path(M.D.__file__).resolve() == LIVE / "driver.py"
M.OUT = out
report = {"driver_sha256": M.D._sha256(LIVE / "driver.py"), "parts": {}}
for name in parts:
    M.RESULTS.clear()
    fn = {"a": M.part_a, "b": M.part_b, "b2": M.part_b2, "c": M.part_c}[name]
    err = None
    try:
        fn()
    except BaseException as exc:                    # noqa: BLE001 - record and go on
        err = {"type": type(exc).__name__, "message": " ".join(str(exc).split())[:600],
               "traceback_tail": traceback.format_exc().splitlines()[-6:]}
        print(f"PART {name} RAISED {err['type']}: {err['message']}", flush=True)
    report["parts"][name] = {"checks": [{"check": n, "pass": ok, "evidence": e} for n, ok, e in M.RESULTS],
                             "raised": err}
report["audit"] = audit.summary()
(out / "run_dryrun_result.json").write_text(json.dumps(report, indent=1, default=str))
for name, p in report["parts"].items():
    print(name, [(c["check"].split(" ")[0], c["pass"]) for c in p["checks"]], "raised:",
          None if p["raised"] is None else p["raised"]["type"])
print(audit.summary())
