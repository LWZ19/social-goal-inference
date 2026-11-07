"""Verifier's own planted-defect variants of the builder's synthetic c_copy folder, and analyse.py run on each.

    v_defects.py      writes verify/defects/<variant>/runs, verify/defects_out/<variant>/, verify/DEFECTS.json

Zero calls. Each variant copies synthetic/c_copy/runs (recorded t0 records in the driver's format, built by the
builder's make_synthetic.py through the harness) and plants one defect. Nothing is written outside verify/.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent                    # .../live_analysis/verify
LA = HERE.parent
LIVE = LA.parent / "live"
# optional argv: SRC_RUNS TAG (defaults: the builder's synthetic/c_copy/runs, outputs defects/, defects_out/, DEFECTS.json)
SRC = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else LA / "synthetic" / "c_copy" / "runs"
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
DEF = HERE / f"defects{TAG}"
OUT = HERE / f"defects{TAG}_out"
PY = ".venv/bin/python"
ENV = {**os.environ, "PYTHONPATH": "", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
       "PYTHONDONTWRITEBYTECODE": "1"}
STALE = "e6e73181d47cfeb1d22ec9fcc91bcdbb6ec625e32dc002be4a877e75d2cf9470"      # the first build
PREVIOUS = "84bab2e469e793ce9351aca1eb3322627ea4e94f3c91b527fc2f5a1f245497cf"   # the second build (BUILD before 05:55)

import yaml  # noqa: E402

G = {"real": "20261019-001000_s24_i09_real_glm53_con", "staffed": "20261019-001001_s24_i09_staffed_glm53_con",
     "plain": "20261019-001002_s24_i09_plain_glm53_con"}
D = {"real": "20261019-011000_s24_i09_real_dsv41_con", "staffed": "20261019-011001_s24_i09_staffed_dsv41_con",
     "plain": "20261019-011002_s24_i09_plain_dsv41_con"}
GCHK, DCHK = "check_glm53_live_20261019-001000.json", "check_dsv41_live_20261019-011000.json"


def fresh(name: str) -> Path:
    runs = DEF / name / "runs"
    if runs.parent.exists():
        shutil.rmtree(runs.parent)
    shutil.copytree(SRC, runs)
    return runs


def rj(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def wj(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


def ry(path):
    return yaml.safe_load(Path(path).read_text())


def wy(path, doc):
    Path(path).write_text(yaml.safe_dump(doc, sort_keys=False))


def goal_idx(rows):
    return [i for i, r in enumerate(rows) if any(c["stage"] == "constraints" for c in r["result"]["calls"])]


VARIANTS = []


def variant(name, flags, expect_exit, expect_fail=(), expect_reading=None, note=""):
    def deco(fn):
        VARIANTS.append((name, fn, flags, expect_exit, expect_fail, expect_reading, note))
        return fn
    return deco


@variant("d01_missing_record", [], 3, ["28 records", "84 records"], "not applicable",
         "one GLM real record removed from results.jsonl and from the check file's upstream list")
def _(runs):
    p = runs / G["real"] / "results.jsonl"
    rows = rj(p)
    gone = rows.pop(5)
    wj(p, rows)
    chk = json.loads((runs / GCHK).read_text())
    arm = next(a for a in chk["arms"] if a["arm"] == "real")
    arm["upstream"] = [u for u in arm["upstream"] if (u["scene"], u["episode"]) != (gone["scene_id"], gone["episode_id"])]
    (runs / GCHK).write_text(json.dumps(chk, indent=1))


@variant("d02_upstream_plan_silent", [], 3, ["compared again here"], "not applicable",
         "one GLM staffed record's plan reasoning changed; the check file still says all_equal (a driver that missed it)")
def _(runs):
    p = runs / G["staffed"] / "results.jsonl"
    rows = rj(p)
    rows[3]["plan"]["reasoning"] = rows[3]["plan"]["reasoning"] + " (changed)"
    wj(p, rows)


@variant("d03_upstream_target_silent", [], 3, ["compared again here"], "not applicable",
         "one DeepSeek plain record's grounded target pixel changed; check file says all_equal")
def _(runs):
    p = runs / D["plain"] / "results.jsonl"
    rows = rj(p)
    i = next(i for i, r in enumerate(rows) if r["result"].get("target"))
    rows[i]["result"]["target"]["pixel"] = [1.0, 1.0]
    wj(p, rows)


@variant("d04_upstream_image_sha_silent", [], 3, ["compared again here"], "not applicable",
         "one GLM plain record's plan.image_sha256 changed; check file says all_equal")
def _(runs):
    p = runs / G["plain"] / "results.jsonl"
    rows = rj(p)
    rows[0]["plan"]["image_sha256"] = "0" * 64
    wj(p, rows)


@variant("d05_upstream_extra_call_silent", [], 3, ["compared again here"], "not applicable",
         "one DeepSeek real record gets an extra locate call in result.calls; check file says all_equal")
def _(runs):
    p = runs / D["real"] / "results.jsonl"
    rows = rj(p)
    c = copy.deepcopy(rows[2]["result"]["calls"][0])
    c["stage"] = "locate"
    rows[2]["result"]["calls"].insert(1, c)
    wj(p, rows)


@variant("d06_upstream_context_flagged", [], 3, ["check file upstream", "compared again here"], "not applicable",
         "one GLM real context answer changed and the check file says so")
def _(runs):
    p = runs / G["real"] / "results.jsonl"
    rows = rj(p)
    i = next(i for i, r in enumerate(rows) if r["result"].get("context_check"))
    rows[i]["result"]["context_check"]["reasoning"] = "x"
    wj(p, rows)
    chk = json.loads((runs / GCHK).read_text())
    arm = next(a for a in chk["arms"] if a["arm"] == "real")
    for u in arm["upstream"]:
        if (u["scene"], u["episode"]) == (rows[i]["scene_id"], rows[i]["episode_id"]):
            u["context_equal"] = u["all_equal"] = False
    (runs / GCHK).write_text(json.dumps(chk, indent=1))


@variant("d07_people_call", [], 3, ["no people call"], "not applicable",
         "a people call inserted into one DeepSeek staffed goal-step record")
def _(runs):
    p = runs / D["staffed"] / "results.jsonl"
    rows = rj(p)
    i = goal_idx(rows)[0]
    c = copy.deepcopy(next(c for c in rows[i]["result"]["calls"] if c["stage"] == "constraints"))
    c["stage"] = "people"
    calls = rows[i]["result"]["calls"]
    calls.insert(len(calls) - 1, c)
    wj(p, rows)


@variant("d08_incomplete_crash_strict", [], 2, [], None,
         "GLM plain config.yaml incomplete: true, vlm_errors 0, no stopped_by (what a crashed invocation leaves)")
def _(runs):
    p = runs / G["plain"] / "config.yaml"
    doc = ry(p)
    doc["incomplete"] = True
    wy(p, doc)


@variant("d09_incomplete_crash_vlmflag", ["--count-vlm-errors-as-failures"], 2, [], None,
         "as d08, with --count-vlm-errors-as-failures: a crashed folder must not be taken as vlm_error-only")
def _(runs):
    _crash(runs)


def _crash(runs):
    p = runs / G["plain"] / "config.yaml"
    doc = ry(p)
    doc["incomplete"] = True
    wy(p, doc)


@variant("d10_incomplete_crash_allow", ["--allow-incomplete"], 3, ["folder complete"], "not applicable",
         "as d08 with --allow-incomplete")
def _(runs):
    _crash(runs)


@variant("d11_config_no_incomplete_key", [], 2, [], None, "DeepSeek real config.yaml has no 'incomplete' key")
def _(runs):
    p = runs / D["real"] / "config.yaml"
    doc = ry(p)
    doc.pop("incomplete")
    wy(p, doc)


@variant("d12_config_unparsable", [], 2, [], None, "DeepSeek staffed config.yaml is not YAML")
def _(runs):
    (runs / D["staffed"] / "config.yaml").write_text("incomplete: false\n  : : [\n")


@variant("d13_stale_driver_sha_config", [], 3, ["driver sha256 in config.yaml"], "not applicable",
         "DeepSeek staffed config.yaml carries the first build's driver sha256 e6e73181")
def _(runs):
    p = runs / D["staffed"] / "config.yaml"
    doc = ry(p)
    doc["i09_live"]["driver_sha256"] = STALE
    wy(p, doc)


@variant("d13b_previous_build_sha_config", [], 3, ["driver sha256 in config.yaml"], "not applicable",
         "GLM plain config.yaml carries the second build's driver sha256 84bab2e4 (stale once BUILD says 41940c3c)")
def _(runs):
    p = runs / G["plain"] / "config.yaml"
    doc = ry(p)
    if doc["i09_live"]["driver_sha256"] == PREVIOUS:
        doc["i09_live"]["driver_sha256"] = STALE         # the source set already carries 84bab2e4: plant e6e73181
    else:
        doc["i09_live"]["driver_sha256"] = PREVIOUS
    wy(p, doc)


@variant("d14_stale_driver_sha_checkfile", [], 3, ["check file driver sha256"], "not applicable",
         "the GLM check file carries the first build's driver sha256")
def _(runs):
    chk = json.loads((runs / GCHK).read_text())
    chk["driver_sha256"] = STALE
    (runs / GCHK).write_text(json.dumps(chk, indent=1))


@variant("d15_frozen_sha_config", [], 3, ["FROZEN sha256 in config.yaml"], "not applicable",
         "GLM real config.yaml records another FROZEN sha256")
def _(runs):
    p = runs / G["real"] / "config.yaml"
    doc = ry(p)
    doc["i09_live"]["frozen_sha256"] = "f" * 64
    wy(p, doc)


@variant("d16_trial1_record", [], 3, ["every record at trial 0"], "not applicable", "one GLM real record at trial 1")
def _(runs):
    p = runs / G["real"] / "results.jsonl"
    rows = rj(p)
    rows[0]["trial"] = 1
    wj(p, rows)


@variant("d17_duplicate_record", [], 3, ["28 records", "84 records"], "not applicable",
         "one DeepSeek plain record duplicated (29 lines)")
def _(runs):
    p = runs / D["plain"] / "results.jsonl"
    rows = rj(p)
    rows.append(copy.deepcopy(rows[4]))
    wj(p, rows)


@variant("d18_people_source_missing", [], 3, ["config.people_source"], "not applicable",
         "one GLM staffed record without config.people_source")
def _(runs):
    p = runs / G["staffed"] / "results.jsonl"
    rows = rj(p)
    rows[7]["config"].pop("people_source")
    wj(p, rows)


@variant("d19_config_people_source_block", [], 3, ["people_source is the frozen detector file"], "not applicable",
         "GLM plain config.yaml people_source block with another file sha256")
def _(runs):
    p = runs / G["plain"] / "config.yaml"
    doc = ry(p)
    doc["people_source"]["sha256"] = "a" * 64
    wy(p, doc)


@variant("d20_ledger_context_send", [], 3, ["only constraints requests sent"], "not applicable",
         "one ledger 'sent' line of stage context")
def _(runs):
    p = runs / "ledger.jsonl"
    rows = rj(p)
    rows[0]["stage"] = "context"
    wj(p, rows)


@variant("d21_ledger_over_model_cap", [], 3, ["sent within the caps"], "not applicable",
         "7 extra DeepSeek sent lines in the ledger (61 for DeepSeek)")
def _(runs):
    p = runs / "ledger.jsonl"
    rows = rj(p)
    extra = [dict(r) for r in rows if r["event"] == "sent" and r["model"] == "dsv41"][:7]
    wj(p, rows + extra)


@variant("d22_ledger_refused_line", [], 3, ["no refused line"], "not applicable", "one refused line in the ledger")
def _(runs):
    p = runs / "ledger.jsonl"
    rows = rj(p)
    rows.append({"time": "x", "event": "refused", "model": "glm53", "arm": "plain", "stage": "constraints"})
    wj(p, rows)


@variant("d23_no_ledger", [], 3, ["ledger.jsonl present"], "not applicable", "ledger.jsonl removed")
def _(runs):
    (runs / "ledger.jsonl").unlink()


@variant("d24_three_invocations", [], 3, ["at most one re-run"], "not applicable",
         "two extra GLM live check files (copies; two re-runs)")
def _(runs):
    for stamp in ("20261019-002000", "20261019-003000"):
        shutil.copy(runs / GCHK, runs / f"check_glm53_live_{stamp}.json")


@variant("d25_vlm_error_no_rerun_flag", ["--count-vlm-errors-as-failures"], 3, ["after one re-run"], "not applicable",
         "a vlm_error left in GLM real after ONE invocation only (no re-run), with --count-vlm-errors-as-failures")
def _(runs):
    p = runs / G["real"] / "results.jsonl"
    rows = rj(p)
    i = goal_idx(rows)[0]
    rows[i]["result"]["status"] = "vlm_error"
    rows[i]["result"]["calls"][-1]["status"] = "vlm_error"
    rows[i]["score"]["success"] = False
    wj(p, rows)
    c = runs / G["real"] / "config.yaml"
    doc = ry(c)
    doc["incomplete"], doc["vlm_errors"] = True, 1
    wy(c, doc)


@variant("d26_config_lies_vlm_error", [], 3, ["no vlm_error record"], "not applicable",
         "config.yaml says complete but one DeepSeek real record has status vlm_error")
def _(runs):
    p = runs / D["real"] / "results.jsonl"
    rows = rj(p)
    i = goal_idx(rows)[0]
    rows[i]["result"]["status"] = "vlm_error"
    rows[i]["score"]["success"] = False
    wj(p, rows)


@variant("d27_situation_changed", [], 3, ["situation is the recorded one"], "not applicable",
         "one DeepSeek staffed record's labels.situation changed to another dev situation")
def _(runs):
    p = runs / D["staffed"] / "results.jsonl"
    rows = rj(p)
    i = next(i for i, r in enumerate(rows) if r["labels"]["situation"] == "no_one")
    rows[i]["labels"]["situation"] = "waiting_line"
    wj(p, rows)


@variant("d28_baseline_run_wrong", [], 3, ["baseline run is the recorded C run"], "not applicable",
         "GLM real config.yaml names the staffed recorded run as its baseline")
def _(runs):
    p = runs / G["real"] / "config.yaml"
    doc = ry(p)
    doc["i09_live"]["baseline_run"] = "20261011-160352_s19_staffed_glm53_con"
    wy(p, doc)


@variant("d29_mode_mock_in_config", [], 3, ["mode live"], "not applicable",
         "DeepSeek plain config.yaml i09_live.mode = mock (folder name without _mock)")
def _(runs):
    p = runs / D["plain"] / "config.yaml"
    doc = ry(p)
    doc["i09_live"]["mode"] = "mock"
    wy(p, doc)


@variant("d30_scene_from_other_arm", [], 3, ["the recorded t0 keys"], "not applicable",
         "one GLM real record's scene_id replaced by its staffed twin (a key not recorded for this arm)")
def _(runs):
    p = runs / G["real"] / "results.jsonl"
    rows = rj(p)
    rows[1]["scene_id"] = "staffed_" + rows[1]["scene_id"][len("real_"):]
    wj(p, rows)


@variant("d31_check_entry_missing", [], 3, ["one check-file entry"], "not applicable",
         "the DeepSeek check file loses its plain entry")
def _(runs):
    chk = json.loads((runs / DCHK).read_text())
    chk["arms"] = [a for a in chk["arms"] if a["arm"] != "plain"]
    (runs / DCHK).write_text(json.dumps(chk, indent=1))


@variant("d32_check_people_stage", [], 3, ["no people stage"], "not applicable",
         "the GLM check file's staffed cache counts include a people stage")
def _(runs):
    chk = json.loads((runs / GCHK).read_text())
    arm = next(a for a in chk["arms"] if a["arm"] == "staffed")
    arm["cache"]["people"] = {"sent": 1}
    (runs / GCHK).write_text(json.dumps(chk, indent=1))


# --- probes of what is NOT gated (expected exit 0 means the analysis lets it through) -------------------------------

@variant("p01_crash_then_rerun", [], None, [], None,
         "PROBE: an EARLIER crashed GLM real folder (incomplete: true, vlm_errors 0, no stopped_by, 5 records, "
         "no check file: a crash writes none) plus its 5 constraints sends in the ledger; the complete folders are a "
         "second invocation, against REQUEST ('exit 2: do not re-run'). Does the analysis flag it?")
def _(runs):
    early = runs / "20261019-000500_s24_i09_real_glm53_con"
    shutil.copytree(runs / G["real"], early)
    rows = rj(early / "results.jsonl")[:5]
    wj(early / "results.jsonl", rows)
    doc = ry(early / "config.yaml")
    doc["incomplete"], doc["vlm_errors"] = True, 0
    wy(early / "config.yaml", doc)
    led = rj(runs / "ledger.jsonl")
    extra = [{"time": "x", "event": "sent", "model": "glm53", "arm": "real", "scene": r["scene_id"],
              "episode": r["episode_id"], "trial": 0, "stage": "constraints"} for r in rows]
    wj(runs / "ledger.jsonl", extra + led)


@variant("p02_backend_stop_then_rerun", [], None, [], None,
         "PROBE: a first GLM invocation stopped by BackendError (exit 2; its check file says stopped_by) and the "
         "controller re-ran anyway; the complete folders are the re-run's")
def _(runs):
    chk = json.loads((runs / GCHK).read_text())
    first = copy.deepcopy(chk)
    first["stopped_by"] = "BackendError: synthetic"
    first["arms"] = [dict(first["arms"][0], stopped_by="BackendError: synthetic",
                          run_folder=str(runs / "20261019-000500_s24_i09_real_glm53_con"))]
    (runs / "check_glm53_live_20261019-000500.json").write_text(json.dumps(first, indent=1))
    early = runs / "20261019-000500_s24_i09_real_glm53_con"
    shutil.copytree(runs / G["real"], early)
    doc = ry(early / "config.yaml")
    doc["incomplete"], doc["stopped_by"] = True, "BackendError: synthetic"
    wy(early / "config.yaml", doc)


@variant("p03_parse_failed_goal_step", [], None, [], None,
         "PROBE: one GLM goal step's constraints call ended parse_failed (status parse_failed, no constraints); the "
         "folder is complete. Expect it counted as a failure, reading applies, M4/M5 n drops by one")
def _(runs):
    p = runs / G["staffed"] / "results.jsonl"
    rows = rj(p)
    i = next(i for i in goal_idx(rows) if rows[i]["score"]["success"])
    r = rows[i]["result"]
    r["status"] = "parse_failed"
    r["calls"][-1]["status"] = "parse_failed"
    r["constraints"] = r["solver"] = r["candidates"] = r["pose_world"] = r["pose_robot"] = None
    rows[i]["score"]["success"] = False
    wj(p, rows)


@variant("p04_lock_file_stays", [], None, [], None,
         "PROBE: ledger.lock present, as driver 41940c3c leaves it after every live run by design (flock, never "
         "removed). Expect: reading applies; how is the lock reported?")
def _(runs):
    (runs / "ledger.lock").write_text("12345 2026-10-19T00:00:00+0800\n")


def run(name, flags) -> dict:
    out = OUT / name
    if out.exists():
        shutil.rmtree(out)
    argv = [PY, str(LA / "analyse.py"), "--runs", str(DEF / name / "runs"), "--out", str(out), *flags]
    proc = subprocess.run(["nice", "-n", "10", *argv], capture_output=True, text=True, env=ENV, cwd=str(HERE))
    got = {"exit": proc.returncode, "stderr_head": proc.stderr.strip().splitlines()[:4]}
    s = out / "summary.json"
    if proc.returncode in (0, 3) and s.exists():
        summary = json.loads(s.read_text())
        got["reading"] = summary["reading"]["reading"]
        got["rule_on_these_numbers"] = summary["reading"]["rule_on_these_numbers"]
        got["successes"] = summary["reading"]["successes"]
        got["failed_gating"] = [r["check"] for r in summary["validity"] if not r["pass"] and r["gates_reading"]]
        got["failed_reported"] = [r["check"] for r in summary["validity"] if not r["pass"] and not r["gates_reading"]]
        got["folder_notes"] = summary["folders"]["notes"]
        got["ledger"] = (summary.get("ledger") or {}).get("sent_per_model")
        got["m4_live_steps"] = {m: (v or {}).get("m4", {}).get("live", {}).get("goal_steps_with_answer")
                                for m, v in summary["measures"].items()}
    return got


def main() -> int:
    assert not DEF.resolve().is_relative_to(LIVE.resolve())
    results = []
    for name, fn, flags, exp_exit, exp_fail, exp_reading, note in VARIANTS:
        runs = fresh(name)
        fn(runs)
        got = run(name, flags)
        problems = []
        if exp_exit is not None and got["exit"] != exp_exit:
            problems.append(f"exit {got['exit']}, expected {exp_exit}")
        for frag in exp_fail:
            if not any(frag in c for c in got.get("failed_gating", [])):
                problems.append(f"no failed gating check containing {frag!r}")
        if exp_reading is not None and got.get("reading") != exp_reading:
            problems.append(f"reading {got.get('reading')!r}, expected {exp_reading!r}")
        row = {"variant": name, "flags": flags, "planted": note, "expected_exit": exp_exit, **got,
               "verdict": "PROBE" if exp_exit is None else ("PASS" if not problems else "FAIL"),
               "problems": problems}
        results.append(row)
        print(f"{row['verdict']:5} {name}: exit {got['exit']} reading {got.get('reading', '-')} "
              f"gating fails {got.get('failed_gating', '-')}" + (f" PROBLEMS {problems}" if problems else "")
              + (f" | reported {got.get('failed_reported')} notes {got.get('folder_notes')} ledger {got.get('ledger')}"
                 f" m4 {got.get('m4_live_steps')} successes {got.get('successes')}" if exp_exit is None else "")
              + (f" | stderr {got['stderr_head'][:2]}" if got["exit"] == 2 else ""))
    (HERE / f"DEFECTS{TAG}.json").write_text(json.dumps(results, indent=1))
    n = [r for r in results if r["verdict"] != "PROBE"]
    print(f"{sum(r['verdict'] == 'PASS' for r in n)} of {len(n)} planted defects refused or flagged as expected")
    return 0 if all(r["verdict"] == "PASS" for r in n) else 1


if __name__ == "__main__":
    raise SystemExit(main())
