"""After the driver change at ~05:54 (BUILD.sha256 driver line 84bab2e4 -> 41940c3c), the builder's synthetic
sets carry the old driver sha256, so 8 of the builder's 25 tests now read 'not applicable'. This script regenerates
the synthetic sets with the builder's own make_synthetic.py (unchanged, hashed) into verify/synthetic_now, then reruns
the builder's TESTS list with the synthetic paths pointed there (outputs verify/builder_tests_now_out). Writes
verify/BUILDER_TESTS_NOW.json. Zero calls; nothing outside verify/ is written.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
LA = HERE.parent
PY = ".venv/bin/python"
ENV = {**os.environ, "PYTHONPATH": "", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1"}
NEW = HERE / "synthetic_now"


def main():
    proc = subprocess.run(["nice", "-n", "10", PY, str(LA / "make_synthetic.py"), "--root", str(NEW)],
                          capture_output=True, text=True, env=ENV)
    print("make_synthetic exit", proc.returncode, proc.stderr[-500:])
    spec = importlib.util.spec_from_file_location("run_tests_now", LA / "run_tests.py")
    rt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rt)
    rt.OUT = HERE / "builder_tests_now_out"
    rt.OUT.mkdir(exist_ok=True)
    results = []
    for test in rt.TESTS:
        name, runs, *rest = test
        runs = Path(runs)
        if runs.is_relative_to(rt.SYN):
            runs = NEW / runs.relative_to(rt.SYN)
        got = rt.run(name, runs, *rest)
        results.append(got)
        print(f"{'PASS' if got['pass'] else 'FAIL'} {name}: exit {got['exit']} reading {got.get('reading', '-')} "
              f"successes {got.get('successes', '-')}" + (f" problems {got['problems']}" if got["problems"] else ""))
    (HERE / "BUILDER_TESTS_NOW.json").write_text(json.dumps(results, indent=1, default=str))
    print(f"{sum(r['pass'] for r in results)} of {len(results)} pass")


if __name__ == "__main__":
    main()
