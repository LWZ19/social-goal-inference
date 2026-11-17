"""Zero-call tests of analyse.py: the driver's mock and fake-gateway dry-run folders (parsing, refusals, validity),
and the synthetic variants of make_synthetic.py (the reading and its boundaries). Run make_synthetic.py first.

    run_tests.py        writes test_out/<test>/{tables.txt,summary.json} and test_out/TESTS.json; exit 1 on a failure
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
I09 = HERE.parent
LIVE = I09 / "live"
SYN = HERE / "synthetic"
OUT = HERE / "test_out"
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402
PY = paths.REPO / ".venv" / "bin" / "python"

ENV = {**os.environ, "PYTHONPATH": "", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
       "PYTHONDONTWRITEBYTECODE": "1"}

# name, runs folder, flags, expected exit, expected reading (None: not checked), expected successes, extra checks
TESTS = [
    # (i) the driver's own dry-run folders: parsing only, never a result
    ("dryrun_a_mock_strict", LIVE / "dryrun" / "a_mock", [], 2, None, None, []),
    ("dryrun_a_mock", LIVE / "dryrun" / "a_mock", ["--accept-mock", "--allow-incomplete"], 3, "not applicable",
     {"glm53": 2, "dsv41": None}, [("n", "glm53", 10), ("fails", "mode live"), ("fails", "driver sha256")]),
    ("dryrun_pythonpath_mock", LIVE / "dryrun" / "a_pythonpath_mock", ["--accept-mock", "--allow-incomplete"], 3,
     "not applicable", None, [("n", "dsv41", 2), ("n", "glm53", None)]),
    ("verify_build_a_mock", I09 / "verify_i09_live_build" / "a_mock", ["--accept-mock", "--allow-incomplete"], 3,
     "not applicable", None, [("n", "glm53", 16), ("check_entries", 3)]),
    ("dryrun_c6_main_capstop", LIVE / "dryrun" / "c_cap" / "c6_main", ["--allow-incomplete"], 3, "not applicable",
     None, [("n", "glm53", 1), ("n", "dsv41", None), ("fails", "no refused line")]),
    ("dryrun_c6_main_full_strict", LIVE / "dryrun" / "c_cap" / "c6_main_full", [], 2, None, None, []),
    ("dryrun_c6_main_full", LIVE / "dryrun" / "c_cap" / "c6_main_full", ["--allow-incomplete"], 3, "not applicable",
     None, [("n", "glm53", 4), ("fails", "upstream")]),
    ("dryrun_b2_three_folders", LIVE / "dryrun" / "b2_count_only_filled", ["--allow-incomplete"], 3,
     "not applicable", None, [("taken", "20261018-013304_s24_i09_real_glm53_con")]),
    # (ii) synthetic copies of recorded data
    ("syn_c_copy", SYN / "c_copy" / "runs", [], 0, "mixed", {"glm53": 64, "dsv41": 65},
     [("flips_vs_C", 0), ("breaks_vs_C", 0)]),
    ("syn_d_replay", SYN / "d_replay" / "runs", [], 0, "consistent with A2", {"glm53": 69, "dsv41": 71},
     [("flips_vs_D", 0), ("breaks_vs_D", 0)]),
    ("syn_contradicts", SYN / "contradicts" / "runs", [], 0, "contradicts A2", {"glm53": 63, "dsv41": 64}, []),
    ("syn_edge_consistent", SYN / "edge_consistent" / "runs", [], 0, "consistent with A2",
     {"glm53": 66, "dsv41": 68}, []),
    ("syn_edge_mixed_ds67", SYN / "edge_mixed_ds67" / "runs", [], 0, "mixed", {"glm53": 66, "dsv41": 67}, []),
    ("syn_edge_mixed_glm63", SYN / "edge_mixed_glm63" / "runs", [], 0, "mixed", {"glm53": 63, "dsv41": 65}, []),
    ("syn_missing_arm_strict", SYN / "neg_missing_arm" / "runs", [], 2, None, None, []),
    ("syn_missing_arm", SYN / "neg_missing_arm" / "runs", ["--allow-incomplete"], 3, "not applicable",
     {"glm53": 64, "dsv41": 49}, [("n", "dsv41", 56)]),
    ("syn_cap_stop_strict", SYN / "neg_cap_stop" / "runs", [], 2, None, None, []),
    ("syn_cap_stop", SYN / "neg_cap_stop" / "runs", ["--allow-incomplete"], 3, "not applicable", None,
     [("fails", "no refused line"), ("fails", "folder complete")]),
    ("syn_upstream", SYN / "neg_upstream" / "runs", [], 3, "not applicable", None,
     [("fails", "check file upstream"), ("fails", "compared again here")]),
    ("syn_driver_sha", SYN / "neg_driver_sha" / "runs", [], 3, "not applicable", None, [("fails", "driver sha256")]),
    ("syn_people_call", SYN / "neg_people_call" / "runs", [], 3, "not applicable", None,
     [("fails", "no people call")]),
    ("syn_vlm_error_rerun_strict", SYN / "vlm_error_rerun" / "runs", [], 2, None, None, []),
    ("syn_vlm_error_rerun", SYN / "vlm_error_rerun" / "runs", ["--count-vlm-errors-as-failures"], 0, "mixed",
     {"glm53": 63, "dsv41": 65}, [("note", "FROZEN 5.1")]),
    ("syn_later_incomplete", SYN / "later_incomplete" / "runs", [], 0, "mixed", {"glm53": 64, "dsv41": 65},
     [("folder_note", "later incomplete")]),
    ("refuse_out_inside_live", SYN / "c_copy" / "runs", ["--out", str(LIVE / "analysis_must_not_exist")], 2, None,
     None, [("absent", str(LIVE / "analysis_must_not_exist"))]),
]


def run(name, runs, flags, exit_expected, reading_expected, successes_expected, extra) -> dict:
    out = OUT / name
    argv = [str(PY), str(HERE / "analyse.py"), "--runs", str(runs)]
    if "--out" not in flags:
        argv += ["--out", str(out)]
    argv += flags
    proc = subprocess.run(["nice", "-n", "10", *argv], capture_output=True, text=True, env=ENV, cwd=str(HERE))
    got = {"test": name, "argv": " ".join(argv[1:]), "exit": proc.returncode, "problems": []}
    if proc.returncode != exit_expected:
        got["problems"].append(f"exit {proc.returncode}, expected {exit_expected}")
    if proc.returncode == 2:
        got["refusal"] = proc.stderr.strip().splitlines()[:3]
    summary = None
    if proc.returncode in (0, 3) and (out / "summary.json").exists():
        summary = json.loads((out / "summary.json").read_text())
        got["reading"] = summary["reading"]["reading"]
        got["successes"] = summary["reading"]["successes"]
        got["n"] = summary["reading"]["n"]
        got["failed_gating_checks"] = [r["check"] for r in summary["validity"] if not r["pass"] and r["gates_reading"]]
        if reading_expected is not None and got["reading"] != reading_expected:
            got["problems"].append(f"reading {got['reading']!r}, expected {reading_expected!r}")
        if successes_expected is not None and got["successes"] != successes_expected:
            got["problems"].append(f"successes {got['successes']}, expected {successes_expected}")
    for check in extra:
        kind = check[0]
        if kind == "absent":
            if Path(check[1]).exists():
                got["problems"].append(f"{check[1]} exists")
            continue
        if summary is None:
            got["problems"].append(f"no summary for {check}")
            continue
        if kind == "n" and summary["reading"]["n"].get(check[1]) != check[2]:
            got["problems"].append(f"n {check[1]} = {summary['reading']['n'].get(check[1])}, expected {check[2]}")
        elif kind == "fails" and not any(check[1] in r["check"] for r in summary["validity"] if not r["pass"]):
            got["problems"].append(f"no failed check containing {check[1]!r}")
        elif kind == "check_entries" and sum(len(v) == 1 for v in summary["check_entries"].values()) != check[1]:
            got["problems"].append(f"check entries {summary['check_entries']}")
        elif kind == "taken" and not any(check[1] in line for line in summary["folders"]["taken"]):
            got["problems"].append(f"{check[1]} not taken: {summary['folders']['taken']}")
        elif kind == "note" and not any(check[1] in n for n in summary["reading"]["notes"]):
            got["problems"].append(f"no reading note with {check[1]!r}")
        elif kind == "folder_note" and not any(check[1] in n for n in summary["folders"]["notes"]):
            got["problems"].append(f"no folder note with {check[1]!r}")
        elif kind in ("flips_vs_C", "breaks_vs_C", "flips_vs_D", "breaks_vs_D"):
            what, base = kind.split("_vs_")
            for model, m in summary["measures"].items():
                value = m["m3"][base][what]
                if value != check[1]:
                    got["problems"].append(f"{model} {what} vs {base} = {value}, expected {check[1]}")
    got["pass"] = not got["problems"]
    return got


def main() -> int:
    OUT.mkdir(exist_ok=True)
    results = [run(*test) for test in TESTS]
    (OUT / "TESTS.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    for r in results:
        print(f"{'PASS' if r['pass'] else 'FAIL'} {r['test']}: exit {r['exit']}, reading {r.get('reading', '-')}, "
              f"successes {r.get('successes', '-')}" + (f"; problems {r['problems']}" if r["problems"] else ""))
    print(f"{sum(r['pass'] for r in results)} of {len(results)} pass")
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
