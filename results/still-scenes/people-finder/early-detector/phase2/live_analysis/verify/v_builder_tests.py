"""Re-run the builder's 25 tests (run_tests.TESTS, unchanged) with outputs in verify/builder_tests_out, so the builder's
own test_out/ is left as evidence. Also checks the guards that the tests cannot reach from the command line, by
importing analyse.py as a module and pointing it at modified COPIES (never at the originals):
FROZEN.md changed, a baseline constant changed, an A2 row changed, an I01 row changed. Writes verify/BUILDER_TESTS.json.
"""
from __future__ import annotations

import importlib.util
import io
import json
import shutil
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402
HERE = Path(__file__).resolve().parent
LA = HERE.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def builder_tests():
    rt = load("run_tests_v", LA / "run_tests.py")
    rt.OUT = HERE / "builder_tests_out"
    rt.OUT.mkdir(exist_ok=True)
    out = []
    for test in rt.TESTS:
        flags = list(test[2])
        if "--out" in flags:            # the out-inside-live test: keep it (it must refuse before writing anything)
            pass
        got = rt.run(*test)
        out.append(got)
        print(f"{'PASS' if got['pass'] else 'FAIL'} {got['test']}: exit {got['exit']} reading {got.get('reading', '-')} "
              f"successes {got.get('successes', '-')}" + (f" problems {got['problems']}" if got["problems"] else ""))
    return out


def guard_tests():
    """Each case: patch module globals of a fresh import of analyse.py, run main() with c_copy, expect a refusal."""
    work = HERE / "guard_copies"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    RD = paths.STILL                    # results/still-scenes
    LIVE = LA.parent / "live"
    cases = []
    # FROZEN.md copy with one threshold changed, FROZEN.sha256 copy listing its new hash
    fz = work / "FROZEN.md"
    fz.write_text((LIVE / "FROZEN.md").read_text().replace("= **66**", "= **65**"))
    import hashlib
    (work / "FROZEN.sha256").write_text(hashlib.sha256(fz.read_bytes()).hexdigest() + "  FROZEN.md\n")
    cases.append(("frozen_changed_copy", {"FROZEN_MD": fz, "FROZEN_SHA_FILE": work / "FROZEN.sha256"}))
    cases.append(("baseline_constant_changed", {"BASELINES": {"glm53": {"C": (65, 14), "D": (69, 19), "P": (69, 19)},
                                                              "dsv41": {"C": (65, 25), "D": (71, 28), "P": (70, 27)}}}))
    a2 = work / "a2"
    shutil.copytree(RD / "people-finder" / "early-detector" / "phase1c" / "a2", a2, ignore=shutil.ignore_patterns("base_*", "rows_new_*", "rows_ph_*", "*.log"))
    rows = (a2 / "rows_old_glm53.jsonl").read_text().splitlines()
    r = json.loads(rows[0]); r["D"] = not r["D"]; rows[0] = json.dumps(r)
    (a2 / "rows_old_glm53.jsonl").write_text("\n".join(rows) + "\n")
    cases.append(("a2_row_changed_copy", {"A2_DIR": a2}))
    i01 = work / "i01"
    i01.mkdir()
    for m in ("glm53", "dsv41"):
        shutil.copy(RD / "replay-harness" / "repairs" / f"rows_old_{m}.jsonl", i01)
    rows = (i01 / "rows_old_dsv41.jsonl").read_text().splitlines()
    r = json.loads(rows[3]); r["P"] = not r["P"]; rows[3] = json.dumps(r)
    (i01 / "rows_old_dsv41.jsonl").write_text("\n".join(rows) + "\n")
    cases.append(("i01_row_changed_copy", {"I01_DIR": i01}))
    # A2 row changed AND its A2.sha256 copy updated: then only the frozen-number assertion can catch a D change
    a2b = work / "a2b"
    shutil.copytree(a2, a2b)
    lines = []
    for line in (a2b / "A2.sha256").read_text().splitlines():
        h, name = line.split()
        if name == "rows_old_glm53.jsonl":
            h = hashlib.sha256((a2b / name).read_bytes()).hexdigest()
        lines.append(f"{h}  {name}")
    (a2b / "A2.sha256").write_text("\n".join(lines) + "\n")
    cases.append(("a2_row_changed_rehashed_copy", {"A2_DIR": a2b}))
    out = []
    for name, patch in cases:
        mod = load(f"analyse_{name}", LA / "analyse.py")
        for k, v in patch.items():
            setattr(mod, k, v)
        buf_o, buf_e = io.StringIO(), io.StringIO()
        with redirect_stdout(buf_o), redirect_stderr(buf_e):
            try:
                code = mod.main(["--runs", str(LA / "synthetic" / "c_copy" / "runs"), "--out", str(HERE / "guard_out" / name)])
            except mod.Refused as stop:
                code, msg = 2, str(stop.code)
            else:
                msg = ""
        first = msg.splitlines()[:3]
        ok = code == 2
        out.append({"case": name, "exit": code, "refusal": first, "pass": ok})
        print(f"{'PASS' if ok else 'FAIL'} guard {name}: exit {code} {first}")
    return out


if __name__ == "__main__":
    b = builder_tests()
    g = guard_tests()
    (HERE / "BUILDER_TESTS.json").write_text(json.dumps({"builder_tests": b, "guards": g}, indent=1, default=str))
    print(f"builder tests {sum(x['pass'] for x in b)} of {len(b)}; guards {sum(x['pass'] for x in g)} of {len(g)}")
