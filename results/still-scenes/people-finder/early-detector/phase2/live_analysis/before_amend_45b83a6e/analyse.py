"""I09 live check (phase 2): the offline analysis of FROZEN.md sections 5-6, run after the live run of arm D-live.

    analyse.py [--runs DIR] [--out DIR] [--allow-incomplete] [--count-vlm-errors-as-failures] [--accept-mock]

No model is called, no network is used and no .env file is read. The recorded runs are read only through the step-0
harness (results/still-scenes/analysis-scripts/harness/research_harness.py); the live run folders, check files, ledger and count-only
reports are read as files from --runs; the saved baseline rows of A2 (phase1c/i09/a2/rows_old_*.jsonl) and I01
(phase1/i01_repairs/rows_old_*.jsonl) are read as files. Nothing is rescored: every success is the package's own
`score.success` as written in a record or a row (FROZEN.md section 5.1). It writes only tables.txt and summary.json
into --out, which may not lie inside phase2/i09/live/.

Steps:
  1. FROZEN.md must match FROZEN.sha256 and the hash in this file; its baselines (section 4), thresholds (section 6),
     trial spread (section 6), recorded goal steps (section 2) and detector time (section 5.6) are parsed from the
     note and must equal the constants below.
  2. The baselines are derived again: recorded C at t0 from the six recorded runs (through the harness), A2's D
     replay and the P column from the A2 rows, I01's P from the I01 rows; they must equal the frozen numbers, or
     the analysis stops (exit 2).
  3. Per (arm, model) the latest complete run folder `<time>_s24_i09_<arm>_<model>_con/` (config.yaml
     `incomplete: false`) is taken (REQUEST.md section 2). A missing or incomplete one is refused (exit 2) unless
     --allow-incomplete (then the latest folder is taken and the reading is "not applicable"), or, for a folder
     incomplete only through vlm_error records after one re-run, --count-vlm-errors-as-failures (FROZEN.md 5.1).
  4. Validity (FROZEN.md sections 2, 6, 7): complete folders, the check files' upstream comparison (all_equal for
     every record), the same comparison done again here from the records, no people call anywhere, 84 records per
     model with the recorded t0 keys, every record's config.people_source, the driver sha256 in config.yaml against
     BUILD.sha256's driver line now, the FROZEN.md sha256, the ledger (sent in all and per model, refused lines).
  5. The pre-registered reading (FROZEN.md section 6), and the measures of section 5, each labelled.

Exit codes: 0 the analysis is written and the pre-registered reading applies; 3 the analysis is written but the
reading is not applicable (incomplete or invalid input); 2 refused (FROZEN.md changed, a baseline does not
reproduce, a missing or incomplete folder without --allow-incomplete, --out inside live/, or any other error).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True                      # nothing is written next to any module imported here

HERE = Path(__file__).resolve().parent              # .../phase2/i09/live_analysis
I09 = HERE.parent                                   # .../phase2/i09
LIVE = I09 / "live"
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths                                        # noqa: E402
RD = paths.STILL                                    # results/still-scenes
WORKTREE = paths.REPO                               # the repository root
HARNESS_DIR = paths.HARNESS                         # results/still-scenes/analysis-scripts/harness

FROZEN_MD = LIVE / "FROZEN.md"
FROZEN_SHA_FILE = LIVE / "FROZEN.sha256"
BUILD_SHA_FILE = LIVE / "BUILD.sha256"
DRIVER = LIVE / "driver.py"
A2_DIR = RD / "people-finder" / "early-detector" / "phase1c" / "a2"
I01_DIR = RD / "replay-harness" / "repairs"
DEFAULT_RUNS = LIVE / "runs"
DEFAULT_OUT = HERE / "result"
ANALYSE_SHA_FILE = HERE / "ANALYSE.sha256"

sys.path.insert(0, str(HARNESS_DIR))
import yaml                                         # noqa: E402

import research_harness as H                        # noqa: E402

# ---------------------------------------------------------------------------------------------------------------
# What FROZEN.md fixes (checked against the note itself at every run)

FROZEN_SHA256 = "7e1d321237f323dcb6747288ac9680138bfeeadff305de3990f6743f06898b84"
EVAL_PEOPLE_SHA256 = "ffa595453c94b444c8dd030d053d37ac103441a7e74f07e3f172836b3d0c24b3"
PEOPLE_SOURCE = "detector_i09_a1_frozen"
# The I01 rows have no hash file of their own; these are their sha256 when this script was written (18 Oct). A2's
# rows are checked against A2.sha256.
I01_ROWS_SHA256 = {"glm53": "16db3dd0bf0f8ed5635745f776ca4df7249c8bf5c5ab551955380965ae816d2f",
                   "dsv41": "0df90e6d7d62fdf0467a9ebe794ee84b7d1c30453523ab2926a93761f698bfa8"}

MODELS = {"glm53": {"stage": "s19", "label": "GLM"}, "dsv41": {"stage": "s20", "label": "DeepSeek"}}
LABEL_TO_MODEL = {spec["label"]: model for model, spec in MODELS.items()}
ARMS = ("real", "staffed", "plain")
SITUATIONS = {"no_one": 36, "waiting_line": 30, "conversation_group": 18}      # t0 records per model
RECORDS_PER_MODEL, RECORDS_PER_ARM = 84, 28
CAP_TOTAL, CAP_MODEL = 120, 60

# Section 4: successes of 84 at t0 (waiting_line successes of 30).
BASELINES = {"glm53": {"C": (64, 14), "D": (69, 19), "P": (69, 19)},
             "dsv41": {"C": (65, 25), "D": (71, 28), "P": (70, 27)}}
# Section 6: the per-trial spread (t0, t1, t2) of recorded C and of the D replay.
TRIALS = {"glm53": {"C": (64, 65, 57), "D": (69, 68, 65)}, "dsv41": {"C": (65, 60, 57), "D": (71, 65, 63)}}
# Section 2: recorded t0 goal steps (records with a people call) per arm.
GOAL_STEPS = {"glm53": {"real": 17, "staffed": 16, "plain": 15}, "dsv41": {"real": 18, "staffed": 20, "plain": 16}}
# Section 6: the thresholds of the reading.
CONSISTENT_AT = {"glm53": 66, "dsv41": 68}
CONTRADICTS_AT = {"glm53": 63, "dsv41": 64}
# Section 5.6: A1's warm detector-plus-pose runtime, a constant (seconds); the verifier's value as a second line.
DETECTOR_MS, DETECTOR_MS_VERIFIER = 55.6, 61.9
DETECTOR_S, DETECTOR_S_VERIFIER = DETECTOR_MS / 1000.0, DETECTOR_MS_VERIFIER / 1000.0

FOLDER_RE = re.compile(r"^(\d{8}-\d{6})_s24_i09_(real|staffed|plain)_(glm53|dsv41)_con(_mock)?$")
CHECK_RE = re.compile(r"^check_(glm53|dsv41)_(live|mock)_(\d{8}-\d{6})\.json$")
COUNT_RE = re.compile(r"^count_only_(glm53|dsv41)_(\d{8}-\d{6})\.json$")

LABELS = {
    "reading": "PRE-REGISTERED (FROZEN 6): the reading",
    "m1": "PRE-REGISTERED (FROZEN 5.1): successes; the reading's input",
    "m2": "PRE-REGISTERED measure (FROZEN 5.2); no role in the reading",
    "m3": "PRE-REGISTERED measure (FROZEN 5.3); no role in the reading",
    "m4": "PRE-REGISTERED measure (FROZEN 5.4); no role in the reading",
    "m5": "PRE-REGISTERED measure (FROZEN 5.5); no role in the reading",
    "m6": "DESCRIPTIVE (FROZEN 5.6 calls the latency descriptive)",
    "m7": "DESCRIPTIVE extras (FROZEN 5.7)",
    "added": "DESCRIPTIVE, added by this analysis (not in FROZEN)",
}


class Refused(SystemExit):
    """The analysis cannot go on: the message says why."""


def refuse(message: str):
    raise Refused(f"error: {message}")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_text(path: Path) -> str:
    return Path(H.guard_file(path)).read_text(encoding="utf-8")


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in read_text(path).splitlines() if line.strip()]


def layout_of(arm: str, scene_id: str) -> str:
    """The layout of a scene: its id without the arm it was rendered in (real_eval_08_x -> eval_08_x)."""
    if not scene_id.startswith(arm + "_"):
        refuse(f"the scene {scene_id} does not start with its arm {arm}")
    return scene_id[len(arm) + 1:]


# ---------------------------------------------------------------------------------------------------------------
# Step 1: FROZEN.md

def check_frozen() -> dict:
    """FROZEN.md's hash, and the numbers this analysis uses, parsed from the note and compared with the constants."""
    got = sha256_file(FROZEN_MD)
    listed = read_text(FROZEN_SHA_FILE).split()[0]
    if not (got == listed == FROZEN_SHA256):
        refuse(f"FROZEN.md has sha256 {got}; FROZEN.sha256 says {listed}; this analysis was written for "
               f"{FROZEN_SHA256}")
    text = read_text(FROZEN_MD)
    flat = " ".join(text.split())
    parsed: dict = {"baselines": {}, "trials": {"glm53": {}, "dsv41": {}}}
    for label, c, cl, d, dl, p, pl in re.findall(
            r"^\| (GLM|DeepSeek) \| (\d+) \((\d+)\) \| (\d+) \((\d+)\) \| (\d+) \((\d+)\) \|$", text, re.M):
        parsed["baselines"][LABEL_TO_MODEL[label]] = {"C": (int(c), int(cl)), "D": (int(d), int(dl)),
                                                      "P": (int(p), int(pl))}
    m = re.search(r"GLM >= max\((\d+), (\d+) − 3\) = \*\*(\d+)\*\* and DeepSeek >= max\((\d+), (\d+) − 3\) = "
                  r"\*\*(\d+)\*\*", flat)
    n = re.search(r"GLM <= (\d+) and DeepSeek <= (\d+)", flat)
    t = re.search(r"recorded C per trial is GLM (\d+) / (\d+) / (\d+) and DeepSeek (\d+) / (\d+) / (\d+) of 84", flat)
    u = re.search(r"D replay GLM (\d+) / (\d+) / (\d+) and DeepSeek (\d+) / (\d+) / (\d+)", flat)
    g = re.search(r"\*\*GLM (\d+)\*\* \(real (\d+), staffed (\d+), plain (\d+)\), \*\*DeepSeek (\d+)\*\* "
                  r"\((\d+), (\d+), (\d+)\)", flat)
    k = re.search(r"median ([\d.]+) ms; the verifier's ([\d.]+) ms", flat)
    if not all((m, n, t, u, g, k)) or set(parsed["baselines"]) != set(MODELS):
        refuse("FROZEN.md: a number this analysis needs was not found in the note")
    parsed["consistent_at"] = {"glm53": int(m.group(3)), "dsv41": int(m.group(6))}
    parsed["consistent_terms"] = {"glm53": (int(m.group(1)), int(m.group(2))),
                                  "dsv41": (int(m.group(4)), int(m.group(5)))}
    parsed["contradicts_at"] = {"glm53": int(n.group(1)), "dsv41": int(n.group(2))}
    parsed["trials"]["glm53"]["C"] = tuple(int(x) for x in t.groups()[:3])
    parsed["trials"]["dsv41"]["C"] = tuple(int(x) for x in t.groups()[3:])
    parsed["trials"]["glm53"]["D"] = tuple(int(x) for x in u.groups()[:3])
    parsed["trials"]["dsv41"]["D"] = tuple(int(x) for x in u.groups()[3:])
    parsed["goal_steps"] = {"glm53": dict(zip(ARMS, (int(x) for x in g.groups()[1:4]))),
                            "dsv41": dict(zip(ARMS, (int(x) for x in g.groups()[5:8])))}
    if int(g.group(1)) != sum(parsed["goal_steps"]["glm53"].values()) or \
            int(g.group(5)) != sum(parsed["goal_steps"]["dsv41"].values()):
        refuse("FROZEN.md: the recorded goal steps do not add up")
    parsed["detector_ms"] = (float(k.group(1)), float(k.group(2)))
    expected = {"baselines": BASELINES, "trials": TRIALS, "goal_steps": GOAL_STEPS, "consistent_at": CONSISTENT_AT,
                "contradicts_at": CONTRADICTS_AT, "detector_ms": (DETECTOR_MS, DETECTOR_MS_VERIFIER)}
    for key, value in expected.items():
        if parsed[key] != value:
            refuse(f"FROZEN.md says {key} = {parsed[key]}, this analysis has {value}")
    for model in MODELS:                            # the thresholds follow from the baselines as section 6 says
        c, d = BASELINES[model]["C"][0], BASELINES[model]["D"][0]
        if parsed["consistent_terms"][model] != (c, d) or CONSISTENT_AT[model] != max(c, d - 3) \
                or CONTRADICTS_AT[model] != c - 1:
            refuse(f"FROZEN.md section 6: the thresholds of {model} do not follow from its baselines")
    return {"sha256": got, "parsed": {key: parsed[key] for key in expected}}


def build_sha() -> dict:
    """BUILD.sha256's driver line now, and driver.py's sha256 now."""
    lines = [line.split() for line in read_text(BUILD_SHA_FILE).splitlines() if line.strip()]
    driver = [words[0] for words in lines if len(words) == 2 and words[1] == "driver.py"]
    if len(driver) != 1:
        refuse(f"{BUILD_SHA_FILE}: one driver.py line is expected")
    return {"build_driver_sha256": driver[0], "driver_py_sha256_now": sha256_file(DRIVER),
            "build_matches_driver_now": driver[0] == sha256_file(DRIVER)}


# ---------------------------------------------------------------------------------------------------------------
# Step 2: the baselines

def load_baselines() -> dict:
    """Recorded C (through the harness) and the A2 and I01 rows, per model; every frozen number asserted."""
    a2_hashes = dict(reversed(line.split()) for line in read_text(A2_DIR / "A2.sha256").splitlines() if line.strip())
    out, problems = {}, []
    for model, spec in MODELS.items():
        refs, success, t0, tallies = {}, {}, {}, {}
        for arm in ARMS:
            found = H.runs(stage=spec["stage"], arm=arm, model=model, method="con")
            if len(found) != 1:
                refuse(f"{len(found)} listed runs for {spec['stage']} {arm} {model} con, not 1")
            ref = refs[arm] = found[0]
            tally = collections.Counter()
            for rec in H.records(ref, tally=tally):
                key = (arm, rec.scene_id, rec.episode_id, rec.trial)
                if key in success:
                    refuse(f"{ref.folder}: two records of {key}")
                success[key] = bool(rec.score.get("success"))
                if rec.trial == 0:
                    t0[key[:3]] = rec
            tallies[ref.folder] = dict(tally)
        a2_path, i01_path = A2_DIR / f"rows_old_{model}.jsonl", I01_DIR / f"rows_old_{model}.jsonl"
        if sha256_file(a2_path) != a2_hashes.get(a2_path.name):
            refuse(f"{a2_path} does not match A2.sha256")
        if sha256_file(i01_path) != I01_ROWS_SHA256[model]:
            refuse(f"{i01_path} has changed since this analysis was written ({I01_ROWS_SHA256[model]})")
        a2 = rows_by_key(jsonl(a2_path), model, refs, a2_path)
        i01 = rows_by_key(jsonl(i01_path), model, refs, i01_path)
        if set(a2) != set(success) or set(i01) != set(success):
            refuse(f"{model}: the A2 or I01 rows do not cover the recorded records one to one")
        for key, row in a2.items():
            if row["C_rec"] != success[key]:
                problems.append(f"{model} {key}: A2 C_rec {row['C_rec']} against the record's {success[key]}")
            if row["P"] != i01[key]["P"]:
                problems.append(f"{model} {key}: A2 P {row['P']} against I01 P {i01[key]['P']}")
            if row["layout"] != layout_of(key[0], key[1]):
                problems.append(f"{model} {key}: layout {row['layout']}")
        for key, rec in t0.items():
            if a2[(*key, 0)]["situation"] != rec.situation:
                problems.append(f"{model} {key}: situation differs between the record and the A2 row")
        c0 = {key: success[(*key, 0)] for key in t0}
        d0 = {key: bool(a2[(*key, 0)]["D"]) for key in t0}
        p0 = {key: bool(i01[(*key, 0)]["P"]) for key in t0}
        situation = {key: rec.situation for key, rec in t0.items()}
        derived = {name: (sum(col.values()), sum(v for key, v in col.items() if situation[key] == "waiting_line"))
                   for name, col in (("C", c0), ("D", d0), ("P", p0))}
        trials = {"C": tuple(sum(v for key, v in success.items() if key[3] == t) for t in (0, 1, 2)),
                  "D": tuple(sum(bool(row["D"]) for key, row in a2.items() if key[3] == t) for t in (0, 1, 2))}
        goal = {arm: sum(has_stage(rec.result, "people") for key, rec in t0.items() if key[0] == arm) for arm in ARMS}
        counts = collections.Counter(situation.values())
        if len(t0) != RECORDS_PER_MODEL or dict(counts) != SITUATIONS:
            problems.append(f"{model}: {len(t0)} t0 records, situations {dict(counts)}")
        if derived != BASELINES[model]:
            problems.append(f"{model}: baselines derived {derived}, frozen {BASELINES[model]}")
        if trials != TRIALS[model]:
            problems.append(f"{model}: per-trial successes derived {trials}, frozen {TRIALS[model]}")
        if goal != GOAL_STEPS[model]:
            problems.append(f"{model}: recorded t0 goal steps derived {goal}, frozen {GOAL_STEPS[model]}")
        by_situation = {name: {s: sum(v for key, v in col.items() if situation[key] == s) for s in SITUATIONS}
                        for name, col in (("C", c0), ("D", d0), ("P", p0))}
        out[model] = {"refs": refs, "t0": t0, "C": c0, "D": d0, "P": p0, "situation": situation,
                      "derived": derived, "trials": trials, "goal_steps": goal, "by_situation": by_situation,
                      "tallies": tallies, "a2_rows": str(a2_path), "i01_rows": str(i01_path)}
    if problems:
        refuse("the baselines do not reproduce the frozen numbers:\n  " + "\n  ".join(problems))
    return out


def rows_by_key(rows: list[dict], model: str, refs: dict, path: Path) -> dict:
    out = {}
    for row in rows:
        if row.get("wording") != "old" or row.get("model") != model or row.get("arm") not in refs:
            refuse(f"{path}: a row of another wording, model or arm: {row.get('run')}")
        if row["run"] != refs[row["arm"]].folder:
            refuse(f"{path}: a row of {row['run']}, not {refs[row['arm']].folder}")
        key = (row["arm"], row["scene"], row["episode"], row["trial"])
        if key in out:
            refuse(f"{path}: two rows of {key}")
        out[key] = row
    return out


def has_stage(result: dict, stage: str) -> bool:
    return any(call.get("stage") == stage for call in result.get("calls") or [])


# ---------------------------------------------------------------------------------------------------------------
# Step 3: the live folders

def read_config(folder: Path) -> dict:
    """config.yaml of a run folder; a folder without a readable one is taken as incomplete."""
    try:
        return yaml.safe_load(read_text(folder / "config.yaml")) or {}
    except (OSError, yaml.YAMLError) as error:
        return {"incomplete": True, "stopped_by": f"config.yaml unreadable: {type(error).__name__}"}


def discover(runs: Path, accept_mock: bool) -> dict:
    """Every run folder, check file and count-only report under `runs`, and its ledger."""
    folders, ignored = collections.defaultdict(list), []
    checks, counts = [], []
    for path in sorted(Path(H.guard_file(runs)).iterdir()):
        H.guard_file(path)
        m = FOLDER_RE.match(path.name)
        if m and path.is_dir():
            if m.group(4) and not accept_mock:
                ignored.append(f"{path.name} (a mock folder; --accept-mock reads it)")
                continue
            doc = read_config(path)
            folders[(m.group(2), m.group(3))].append({
                "stamp": m.group(1), "path": path, "name": path.name, "mock": bool(m.group(4)), "doc": doc,
                "complete": doc.get("incomplete") is False,
                "vlm_error_only": doc.get("incomplete") is True and not doc.get("stopped_by")
                                  and (doc.get("vlm_errors") or 0) > 0})
            continue
        c = CHECK_RE.match(path.name)
        if c:
            checks.append({"model": c.group(1), "mode": c.group(2), "stamp": c.group(3), "path": path,
                           "doc": json.loads(read_text(path))})
            continue
        n = COUNT_RE.match(path.name)
        if n:
            counts.append({"model": n.group(1), "stamp": n.group(2), "path": path, "doc": json.loads(read_text(path))})
    ledger = runs / "ledger.jsonl"
    return {"folders": folders, "ignored": ignored, "checks": checks, "counts": counts,
            "ledger": jsonl(ledger) if ledger.exists() else None, "ledger_path": str(ledger),
            "lock_left": (runs / "ledger.lock").exists()}


def choose(found: dict, args) -> tuple[dict, list[str], list[str]]:
    """Per (arm, model) the folder taken, the problems that refuse the analysis, and notes."""
    chosen, problems, notes = {}, [], []
    for model in MODELS:
        for arm in ARMS:
            cands = sorted(found["folders"].get((arm, model), []), key=lambda f: f["stamp"])
            complete = [f for f in cands if f["complete"]]
            what = f"s24_i09_{arm}_{model}_con"
            if complete:
                pick = complete[-1]
                pick["taken_as"] = "latest complete"
                later = [f["name"] for f in cands if f["stamp"] > pick["stamp"]]
                if later:
                    notes.append(f"{what}: later incomplete folder(s) not taken: {', '.join(later)}")
            elif cands and args.count_vlm_errors_as_failures and cands[-1]["vlm_error_only"]:
                pick = cands[-1]
                pick["taken_as"] = "latest, incomplete only through vlm_error records (FROZEN 5.1: counted as failures)"
            elif cands and args.allow_incomplete:
                pick = cands[-1]
                pick["taken_as"] = "latest, INCOMPLETE (--allow-incomplete)"
                problems.append(f"{what}: no complete folder; took {pick['name']} "
                                f"(incomplete: {pick['doc'].get('incomplete')}, stopped_by: "
                                f"{pick['doc'].get('stopped_by')}, vlm_errors: {pick['doc'].get('vlm_errors')})")
            else:
                pick = None
                problems.append(f"{what}: " + ("missing" if not cands else
                                               "no complete folder (" + ", ".join(
                                                   f"{f['name']}: incomplete {f['doc'].get('incomplete')}, "
                                                   f"stopped_by {f['doc'].get('stopped_by')}, "
                                                   f"vlm_errors {f['doc'].get('vlm_errors')}" for f in cands) + ")"))
            if len(cands) > 1:
                notes.append(f"{what}: {len(cands)} folders found: {', '.join(f['name'] for f in cands)}")
            if pick is not None:
                chosen[(arm, model)] = pick
    return chosen, problems, notes


def load_live_records(pick: dict, arm: str, model: str, base: dict) -> list[dict]:
    """The records of a chosen folder, each guarded as the harness guards a record before its other fields."""
    ref = base[model]["refs"][arm]
    held = H.held_out(ref)
    out = []
    path = pick["path"] / "results.jsonl"
    for raw in jsonl(path) if path.exists() else []:
        situation = (raw.get("labels") or {}).get("situation")
        scene_id = raw.get("scene_id")
        if situation is None or situation in held or not isinstance(scene_id, str) or H.forbidden(scene_id):
            refuse(f"{pick['name']}: a record of a held-out or unknown situation or scene; nothing of it is read")
        out.append(raw)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Step 4: validity

def compare_upstream(record: dict, recorded) -> dict:
    """The plan and context answers of a D-live record against the recorded t0 record, as driver.py compares them
    (compare_upstream, same rule), done again here from the records."""
    def plan_of(r):
        return None if r is None else {k: v for k, v in r.items() if k != "image_sha256"}
    mine, theirs = record["result"], recorded.result

    def stages(calls):
        return [c["stage"] for c in calls or [] if c["stage"] != "people"]
    out = {
        "plan_equal": plan_of(record.get("plan")) == plan_of(recorded.raw.get("plan")),
        "image_equal": (record.get("plan") or {}).get("image_sha256")
                       == (recorded.raw.get("plan") or {}).get("image_sha256"),
        "context_equal": mine.get("context_check") == theirs.get("context_check"),
        "location_equal": mine.get("location") == theirs.get("location"),
        "target_equal": mine.get("target") == theirs.get("target"),
        "calls_equal": stages(mine.get("calls")) == stages(theirs.get("calls")),
    }
    out["all_equal"] = all(out.values())
    return out


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, name: str, ok: bool, detail: str = "", gates: bool = True):
        self.rows.append({"check": name, "pass": bool(ok), "detail": detail, "gates_reading": gates})

    def failed(self, gating_only: bool = True) -> list[dict]:
        return [r for r in self.rows if not r["pass"] and (r["gates_reading"] or not gating_only)]


def validate(found: dict, chosen: dict, live: dict, base: dict, frozen: dict, build: dict, args) -> tuple:
    chk = Checks()
    info: dict = {"check_entries": {}, "ledger": None, "count_only": {}, "invocations": {}}
    checks_by_folder = {}
    for check in found["checks"]:
        for entry in check["doc"].get("arms", []):
            folder = Path(entry.get("run_folder", "")).name
            if folder:
                checks_by_folder.setdefault(folder, []).append((check, entry))
    for model in MODELS:
        live_checks = [c for c in found["checks"] if c["model"] == model and c["mode"] == "live"]
        info["invocations"][model] = [c["path"].name for c in live_checks]
        chk.add(f"{model}: at most one re-run (FROZEN 7: two live check files at most)", len(live_checks) <= 2,
                f"{len(live_checks)} live check file(s): {', '.join(c['path'].name for c in live_checks) or 'none'}")
        stopped = [f"{c['path'].name}: {c['doc'].get('stopped_by')}" for c in live_checks if c["doc"].get("stopped_by")]
        chk.add(f"{model}: no live invocation stopped (cap, quota, back end; REQUEST: no re-run after exit 2 or 3)",
                not stopped, "; ".join(stopped), gates=False)
        recs = [r for arm in ARMS for r in live.get((arm, model), [])]
        chk.add(f"{model}: {RECORDS_PER_MODEL} records", len(recs) == RECORDS_PER_MODEL, f"{len(recs)} records")
        for arm in ARMS:
            what = f"{model} {arm}"
            pick = chosen.get((arm, model))
            if pick is None:
                chk.add(f"{what}: folder present", False, "no folder taken")
                continue
            doc, i09 = pick["doc"], pick["doc"].get("i09_live") or {}
            ref = base[model]["refs"][arm]
            chk.add(f"{what}: folder complete (incomplete: false, no stopped_by)",
                    pick["complete"] and not doc.get("stopped_by"),
                    f"{pick['name']}: incomplete {doc.get('incomplete')}, stopped_by {doc.get('stopped_by')}, "
                    f"vlm_errors {doc.get('vlm_errors')}", gates=not pick["vlm_error_only"]
                    or not args.count_vlm_errors_as_failures)
            if pick["vlm_error_only"] and args.count_vlm_errors_as_failures:
                chk.add(f"{what}: incomplete only through vlm_error after one re-run (FROZEN 5.1)",
                        len(info["invocations"][model]) == 2 and not doc.get("stopped_by"),
                        f"{doc.get('vlm_errors')} vlm_error record(s), counted as failures; live invocations of "
                        f"{model}: {len(info['invocations'][model])}")
            chk.add(f"{what}: mode live", i09.get("mode") == "live" and not pick["mock"], f"mode {i09.get('mode')}")
            chk.add(f"{what}: baseline run is the recorded C run", i09.get("baseline_run") == ref.folder,
                    f"{i09.get('baseline_run')} (expected {ref.folder})")
            chk.add(f"{what}: trial 0", i09.get("trial") == 0, f"trial {i09.get('trial')}")
            chk.add(f"{what}: driver sha256 in config.yaml = BUILD.sha256 driver line now",
                    i09.get("driver_sha256") == build["build_driver_sha256"],
                    f"config {i09.get('driver_sha256')}, BUILD {build['build_driver_sha256']}")
            chk.add(f"{what}: FROZEN sha256 in config.yaml = FROZEN.md now", i09.get("frozen_sha256") == frozen["sha256"],
                    f"config {i09.get('frozen_sha256')}")
            ps = doc.get("people_source") or {}
            chk.add(f"{what}: config.yaml people_source is the frozen detector file",
                    ps.get("tag") == PEOPLE_SOURCE and ps.get("sha256") == EVAL_PEOPLE_SHA256
                    and ps.get("kind") == "detector", f"tag {ps.get('tag')}, sha256 {ps.get('sha256')}")
            recs = live.get((arm, model), [])
            keys = [(arm, r["scene_id"], r["episode_id"]) for r in recs]
            expected = {key for key in base[model]["t0"] if key[0] == arm}
            chk.add(f"{what}: {RECORDS_PER_ARM} records, the recorded t0 keys once each",
                    len(keys) == len(set(keys)) == RECORDS_PER_ARM and set(keys) == expected,
                    f"{len(keys)} records, {len(set(keys))} distinct, {len(set(keys) - expected)} not recorded, "
                    f"{len(expected - set(keys))} recorded but missing")
            chk.add(f"{what}: every record at trial 0", all(r.get("trial") == 0 for r in recs))
            sit_bad = sum(r["labels"]["situation"] != base[model]["situation"].get(key)
                          for r, key in zip(recs, keys))
            chk.add(f"{what}: every record's situation is the recorded one", sit_bad == 0, f"{sit_bad} differ")
            bad_ps = sum((r.get("config") or {}).get("people_source") != PEOPLE_SOURCE for r in recs)
            chk.add(f"{what}: every record's config.people_source = {PEOPLE_SOURCE}", bad_ps == 0,
                    f"{bad_ps} record(s) without it")
            people_calls = sum(has_stage(r["result"], "people") for r in recs)
            chk.add(f"{what}: no people call in any record", people_calls == 0, f"{people_calls} record(s) with one")
            errors = sum(r["result"].get("status") == "vlm_error" for r in recs)
            chk.add(f"{what}: no vlm_error record", errors == 0, f"{errors} vlm_error record(s)",
                    gates=not args.count_vlm_errors_as_failures)
            own = [(key, compare_upstream(r, base[model]["t0"][key])) for r, key in zip(recs, keys)
                   if key in base[model]["t0"]]
            unequal = [f"{key[1]} {key[2]}" for key, cmp in own if not cmp["all_equal"]]
            chk.add(f"{what}: upstream answers equal the recorded t0 ones (compared again here)",
                    len(own) == len(recs) and not unequal,
                    f"{len(own) - len(unequal)} of {len(recs)} equal" + (f"; differ: {', '.join(unequal)}" if unequal
                                                                          else ""))
            entries = checks_by_folder.get(pick["name"], [])
            info["check_entries"][f"{arm} {model}"] = [str(c["path"].name) for c, _ in entries]
            if len(entries) != 1:
                chk.add(f"{what}: one check-file entry for the folder", False, f"{len(entries)} found")
                continue
            check, entry = entries[0]
            up = entry.get("upstream") or []
            up_keys = {(arm, u["scene"], u["episode"]) for u in up}
            false = [f"{u['scene']} {u['episode']}" for u in up if not u.get("all_equal")]
            chk.add(f"{what}: check file upstream all_equal for every record",
                    len(up) == len(recs) and up_keys == set(keys) and not false,
                    f"{check['path'].name}: {len(up)} entries, {len(up) - len(false)} all_equal"
                    + (f"; false: {', '.join(false)}" if false else ""))
            cache = entry.get("cache") or {}
            refused_up = sum((cache.get(s) or {}).get("refused", 0) for s in cache)
            chk.add(f"{what}: check file: no people stage, no refused upstream request, not stopped",
                    "people" not in cache and refused_up == 0 and not entry.get("stopped_by")
                    and check["doc"].get("mode") == "live",
                    f"cache {cache}; stopped_by {entry.get('stopped_by')}")
            chk.add(f"{what}: check file driver sha256 = BUILD.sha256 driver line now",
                    check["doc"].get("driver_sha256") == build["build_driver_sha256"],
                    f"{check['doc'].get('driver_sha256')}")
    # The ledger: every request sent, over every invocation that used the folder.
    led = found["ledger"]
    if led is None:
        chk.add("ledger.jsonl present", False, f"{found['ledger_path']} not found")
    else:
        events = collections.Counter(row.get("event") for row in led)
        sent = [row for row in led if row.get("event") == "sent"]
        per_model = collections.Counter(row.get("model") for row in sent)
        per_stage = collections.Counter((row.get("model"), row.get("stage")) for row in sent)
        refused = [row for row in led if row.get("event") == "refused"]
        info["ledger"] = {"events": dict(events), "sent_total": len(sent), "sent_per_model": dict(per_model),
                          "sent_per_model_stage": {f"{m} {s}": n for (m, s), n in sorted(per_stage.items())},
                          "refused_lines": refused, "error_lines": [row for row in led if row.get("event") == "error"]}
        chk.add("ledger: no refused line (no cap stop)", not refused, f"{len(refused)} refused line(s)")
        chk.add(f"ledger: sent within the caps ({CAP_TOTAL} in all, {CAP_MODEL} per model)",
                len(sent) <= CAP_TOTAL and all(n <= CAP_MODEL for n in per_model.values()),
                f"{len(sent)} in all; {dict(per_model)}")
        chk.add("ledger: only constraints requests sent (no people, plan, locate or context send)",
                all(row.get("stage") == "constraints" for row in sent),
                f"by model and stage {info['ledger']['sent_per_model_stage']}")
    chk.add("no ledger.lock left behind", not found["lock_left"], "", gates=False)
    # Count-only reports (FROZEN 2(a) and 7, the go-live rule): reported; the decisive check is the one after the run.
    for model in MODELS:
        reports = sorted((c for c in found["counts"] if c["model"] == model), key=lambda c: c["stamp"])
        if not reports:
            chk.add(f"{model}: count-only report present", False, "none found", gates=False)
            continue
        last = reports[-1]["doc"]
        b = last.get("bounds") or {}
        info["count_only"][model] = {"file": reports[-1]["path"].name, "bounds": b,
                                     "cache_total": last.get("cache_total")}
        chk.add(f"{model}: count-only found 0 upstream misses and at most {CAP_MODEL} (upper bound)",
                b.get("upstream_misses") == 0 and (b.get("upper") or 0) <= CAP_MODEL,
                f"{reports[-1]['path'].name}: {b}", gates=False)
    chk.add("BUILD.sha256's driver line = driver.py now", build["build_matches_driver_now"],
            f"BUILD {build['build_driver_sha256']}, driver.py {build['driver_py_sha256_now']}", gates=False)
    return chk, info


# ---------------------------------------------------------------------------------------------------------------
# Step 5: the reading and the measures

def reading(successes: dict) -> str:
    """FROZEN.md section 6, on successes of the 84 t0 records per model."""
    if all(successes[m] >= CONSISTENT_AT[m] for m in MODELS):
        return "consistent with A2"
    if all(successes[m] <= CONTRADICTS_AT[m] for m in MODELS):
        return "contradicts A2"
    return "mixed"


def quantile(values, q: float):
    """numpy's default (linear) quantile of a list; None for an empty one."""
    xs = sorted(values)
    if not xs:
        return None
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def latency_of(result: dict, stages: tuple[str, ...], geometry: bool = True) -> float:
    calls = sum(c.get("latency_total_s") or 0.0 for c in result.get("calls") or [] if c.get("stage") in stages)
    return calls + ((result.get("timings_s") or {}).get("geometry", 0.0) if geometry else 0.0)


def summary_stats(values) -> dict:
    return {"n": len(values), "median": quantile(values, 0.5), "p90": quantile(values, 0.9)}


def answered(result: dict) -> bool:
    """A goal step with a constraints answer: the solver ran on a valid answer."""
    return result.get("constraints") is not None and result.get("solver") is not None


def usability(steps: list[dict]) -> dict:
    status, per_step, no_usable, cu, eb = collections.Counter(), [], 0, [], []
    for raw in steps:
        r = raw["result"]
        for use in r["constraints"]:
            status[use["status"]] += 1
        per_step.append(len(r["constraints"]))
        no_usable += bool((r.get("flags") or {}).get("no_usable_constraint"))
        cu.append((raw.get("measures") or {}).get("constraints_unusable"))
        eb.append((raw.get("measures") or {}).get("exclusions_broken"))
    return {"goal_steps_with_answer": len(steps),
            "status": {s: status.get(s, 0) for s in ("used", "unusable", "superseded", "cut")},
            "constraints": sum(per_step), "per_goal_step_mean": mean(per_step),
            "per_goal_step_median": quantile(per_step, 0.5),
            "no_usable_constraint_steps": no_usable,
            "constraints_unusable_sum": sum(v for v in cu if v is not None), "constraints_unusable_mean": mean(cu),
            "exclusions_broken_sum": sum(v for v in eb if v is not None), "exclusions_broken_mean": mean(eb),
            "steps_with_exclusions_broken": sum(1 for v in eb if v)}


def numbers_named(steps: list[dict]) -> dict:
    named = in_list = steps_outside = steps_naming = list_not_1_to_n = 0
    distinct_named = distinct_in = 0
    for raw in steps:
        r = raw["result"]
        shown = {p["number"] for p in r.get("people") or []}
        if shown != set(range(1, len(r.get("people") or []) + 1)):
            list_not_1_to_n += 1
        numbers = [n for use in r["constraints"] for n in use["constraint"].get("people") or []]
        named += len(numbers)
        in_list += sum(n in shown for n in numbers)
        steps_naming += bool(numbers)
        steps_outside += any(n not in shown for n in numbers)
        distinct_named += len(set(numbers))
        distinct_in += len(set(numbers) & shown)
    return {"goal_steps_with_answer": len(steps), "numbers_named": named, "in_list": in_list,
            "share_in_list": in_list / named if named else None,
            "steps_naming_any": steps_naming, "steps_naming_one_not_in_list": steps_outside,
            "share_steps_naming_one_not_in_list": steps_outside / len(steps) if steps else None,
            "distinct_per_step_named": distinct_named, "distinct_per_step_in_list": distinct_in,
            "lists_not_numbered_1_to_n": list_not_1_to_n}


def flips_breaks(live: dict, baseline: dict, situation: dict) -> dict:
    """Record level: flip = baseline failure and D-live success; break = baseline success and D-live failure."""
    keys = sorted(set(live) & set(baseline))
    flips = [k for k in keys if not baseline[k] and live[k]]
    breaks = [k for k in keys if baseline[k] and not live[k]]
    net = collections.Counter()
    for k in flips:
        net[layout_of(k[0], k[1])] += 1
    for k in breaks:
        net[layout_of(k[0], k[1])] -= 1

    def where(ks):
        return {"records": [f"{k[1]} {k[2]}" for k in ks],
                "scenes": sorted({k[1] for k in ks}), "layouts": sorted({layout_of(k[0], k[1]) for k in ks}),
                "by_situation": dict(collections.Counter(situation[k] for k in ks))}
    return {"n": len(keys), "baseline_successes": sum(baseline[k] for k in keys),
            "live_successes": sum(live[k] for k in keys), "flips": len(flips), "breaks": len(breaks),
            "net": len(flips) - len(breaks), "flip": where(flips), "break": where(breaks),
            "net_by_layout": {lay: n for lay, n in sorted(net.items()) if n}}


def measures(live: dict, base: dict, chosen: dict, found: dict) -> dict:
    out = {}
    for model in MODELS:
        b = base[model]
        recs = {(arm, r["scene_id"], r["episode_id"]): r for arm in ARMS for r in live.get((arm, model), [])}
        recs = {k: r for k, r in recs.items() if k in b["t0"]}          # an unknown key fails validity above
        if not recs:
            out[model] = None
            continue
        situation = {key: b["situation"][key] for key in recs}
        success = {key: bool((r.get("score") or {}).get("success")) for key, r in recs.items()}
        status = collections.Counter(r["result"].get("status") for r in recs.values())
        m: dict = {"n_records": len(recs)}
        # M1 and the per-arm addition
        m["m1"] = {"successes": sum(success.values()), "n": len(recs), "vlm_error": status.get("vlm_error", 0),
                   "status": dict(status)}
        m["per_arm"] = {arm: {"n": sum(k[0] == arm for k in recs), "live": sum(v for k, v in success.items() if k[0] == arm),
                              "C": sum(b["C"][k] for k in recs if k[0] == arm),
                              "D": sum(b["D"][k] for k in recs if k[0] == arm),
                              "P": sum(b["P"][k] for k in recs if k[0] == arm)} for arm in ARMS}
        # M2
        m["m2"] = {s: {"n": sum(situation[k] == s for k in recs),
                       "live": sum(v for k, v in success.items() if situation[k] == s),
                       **{name: sum(b[name][k] for k in recs if situation[k] == s) for name in ("C", "D", "P")}}
                   for s in SITUATIONS}
        # M3
        m["m3"] = {name: flips_breaks(success, b[name], situation) for name in ("C", "D", "P")}
        # Goal steps
        live_steps = {k: r for k, r in recs.items() if has_stage(r["result"], "constraints")}
        c_steps = {k: rec.raw for k, rec in b["t0"].items() if k in recs and has_stage(rec.result, "people")}
        live_ans = [r for r in live_steps.values() if answered(r["result"])]
        c_ans = [r for r in c_steps.values() if answered(r["result"])]
        m["goal_steps"] = {"live": len(live_steps), "live_with_answer": len(live_ans),
                           "C": len(c_steps), "C_with_answer": len(c_ans),
                           "in_both": len(set(live_steps) & set(c_steps)),
                           "live_only": [f"{k[1]} {k[2]}" for k in sorted(set(live_steps) - set(c_steps))],
                           "C_only": [f"{k[1]} {k[2]}" for k in sorted(set(c_steps) - set(live_steps))]}
        # M4, M5
        m["m4"] = {"live": usability(live_ans), "C": usability(c_ans)}
        m["m5"] = {"live": numbers_named(live_ans), "C": numbers_named(c_ans)}
        # M6
        live_lat = {k: latency_of(r["result"], ("constraints",)) for k, r in live_steps.items()}
        c_lat = {k: latency_of(r["result"], ("people", "constraints")) for k, r in c_steps.items()}
        live_call = {k: latency_of(r["result"], ("constraints",), geometry=False) for k, r in live_steps.items()}
        c_call = {k: latency_of(r["result"], ("people", "constraints"), geometry=False) for k, r in c_steps.items()}
        both = sorted(set(live_lat) & set(c_lat))
        m["m6"] = {
            "all": {"live": summary_stats(list(live_lat.values())), "C": summary_stats(list(c_lat.values())),
                    "live_plus_detector_55_6ms": summary_stats([v + DETECTOR_S for v in live_lat.values()]),
                    "live_plus_detector_61_9ms": summary_stats([v + DETECTOR_S_VERIFIER for v in live_lat.values()])},
            "paired": {"n": len(both), "live": summary_stats([live_lat[k] for k in both]),
                       "C": summary_stats([c_lat[k] for k in both]),
                       "live_minus_C": summary_stats([live_lat[k] - c_lat[k] for k in both]),
                       "live_plus_detector_55_6ms_minus_C": summary_stats(
                           [live_lat[k] + DETECTOR_S - c_lat[k] for k in both])},
            "calls_alone": {"live": summary_stats(list(live_call.values())), "C": summary_stats(list(c_call.values())),
                            "paired_live_minus_C": summary_stats([live_call[k] - c_call[k] for k in both])},
        }
        # M7
        live_people = [len(r["result"].get("people") or []) for r in live_steps.values()
                       if r["result"].get("people") is not None]
        c_people = [len(r["result"].get("people") or []) for r in c_steps.values()
                    if r["result"].get("people") is not None]

        def measure_stats(raws, name):
            values = [(r.get("measures") or {}).get(name) for r in raws]
            present = [v for v in values if v is not None]
            return {"n": len(present), "mean": mean(present), "sum": sum(present) if present else 0}
        method = ("people_found_share", "people_unmatched", "members_precision", "members_recall")
        retries = collections.defaultdict(collections.Counter)
        for r in recs.values():
            for call in r["result"].get("calls") or []:
                row = retries[call["stage"]]
                row["calls"] += 1
                vr, tr = call.get("validation_retries") or 0, call.get("transient_retries") or 0
                if call.get("cached"):                  # a cached call carries the retries of the call stored
                    row["cached"] += 1
                    row["retries_carried_by_cached"] += vr + tr
                else:
                    row["fresh"] += 1
                    row["attempts"] += 1 + vr + tr
                    row["validation_retries"] += vr
                    row["transient_retries"] += tr
                row[f"status_{call.get('status', 'ok')}"] += 1
        c_retries = collections.Counter()
        for r in c_steps.values():
            for call in r["result"].get("calls") or []:
                if call["stage"] in ("people", "constraints"):
                    c_retries[f"{call['stage']}_validation_retries"] += call.get("validation_retries") or 0
                    c_retries[f"{call['stage']}_transient_retries"] += call.get("transient_retries") or 0
        detector_per_scene = {}
        for arm in ARMS:
            pick = chosen.get((arm, model))
            if pick is None:
                continue
            for check in found["checks"]:
                for entry in check["doc"].get("arms", []):
                    if Path(entry.get("run_folder", "")).name == pick["name"]:
                        for scene, s in (entry.get("scenes") or {}).items():
                            detector_per_scene[scene] = s.get("detector_people")
        people_asked = sum(entry.get("people_asked", 0) for check in found["checks"]
                           for entry in check["doc"].get("arms", [])
                           if any(Path(entry.get("run_folder", "")).name == p["name"]
                                  for (a, mm), p in chosen.items() if mm == model))
        m["m7"] = {
            "people_per_goal_step": {"live_detector": {"n": len(live_people), "mean": mean(live_people),
                                                       "median": quantile(live_people, 0.5),
                                                       "min": min(live_people, default=None),
                                                       "max": max(live_people, default=None)},
                                     "C_vlm": {"n": len(c_people), "mean": mean(c_people),
                                               "median": quantile(c_people, 0.5),
                                               "min": min(c_people, default=None), "max": max(c_people, default=None)}},
            "frozen_detector_people_per_scene": {"scenes": len(detector_per_scene),
                                                 "total": sum(v or 0 for v in detector_per_scene.values()),
                                                 "mean": mean(list(detector_per_scene.values()))},
            "detector_source_asked": people_asked,
            "method_measures": {"live": {n: measure_stats(live_steps.values(), n) for n in method},
                                "C": {n: measure_stats(c_steps.values(), n) for n in method}},
            "calls_by_stage_live": {stage: dict(c) for stage, c in sorted(retries.items())},
            "recorded_C_goal_step_retries": dict(c_retries),
        }
        out[model] = m
    return out


# ---------------------------------------------------------------------------------------------------------------
# Output

def fmt(v, digits: int = 3) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def table(headers: list[str], rows: list[list]) -> list[str]:
    cells = [[str(h) for h in headers]] + [[fmt(c) for c in row] for row in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    lines = ["  " + "  ".join(c.ljust(w) for c, w in zip(cells[0], widths)),
             "  " + "  ".join("-" * w for w in widths)]
    lines += ["  " + "  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in cells[1:]]
    return lines


def ms(v):
    return None if v is None else round(v * 1000.0, 1)


def render(s: dict) -> str:
    L = []
    add = L.append
    add("I09 live check: offline analysis of FROZEN.md sections 5-6 (arm D-live)")
    add(f"generated {s['generated']}; analyse.py sha256 {s['analyse_sha256']}"
        f" (ANALYSE.sha256: {s['analyse_sha256_file'] or 'not written yet'}; match: {s['analyse_matches_hash_file']})")
    add(f"runs folder: {s['runs']}")
    add(f"code: harness {s['harness']}; sgi {s['sgi']}")
    if s["synthetic"]:
        add("")
        add("!" * 100)
        add("TEST INPUT: the run folders are SYNTHETIC (built by make_synthetic.py from recorded data): NOT A RESULT")
        for note in s["synthetic"]:
            add(f"  {note}")
        add("!" * 100)
    add("")
    add("== Reading [" + LABELS["reading"] + "]")
    r = s["reading"]
    add("  successes at t0: " + ", ".join(f"{MODELS[m]['label']} {fmt(r['successes'].get(m))} of "
                                          f"{fmt(r['n'].get(m))}" for m in MODELS) + " (the rule is on 84 per model)")
    add(f"  rule: consistent with A2 if GLM >= {CONSISTENT_AT['glm53']} and DeepSeek >= {CONSISTENT_AT['dsv41']}; "
        f"contradicts A2 if GLM <= {CONTRADICTS_AT['glm53']} and DeepSeek <= {CONTRADICTS_AT['dsv41']}; mixed otherwise")
    add(f"  READING: {r['reading']}")
    if r["not_applicable_because"]:
        add("  not applicable because:")
        for why in r["not_applicable_because"]:
            add(f"    - {why}")
        if r["rule_on_these_numbers"]:
            add(f"  (for information only, NOT the reading: the rule on these numbers would give "
                f"'{r['rule_on_these_numbers']}')")
    for note in r["notes"]:
        add(f"  note: {note}")
    add("")
    add("== 0. Inputs: the folders taken")
    for line in s["folders"]["taken"]:
        add(f"  {line}")
    for line in s["folders"]["notes"]:
        add(f"  note: {line}")
    for line in s["folders"]["ignored"]:
        add(f"  ignored: {line}")
    add(f"  check files: {', '.join(s['folders']['check_files']) or 'none'}")
    add(f"  count-only reports: {', '.join(s['folders']['count_files']) or 'none'}")
    add("")
    add("== 1. Validity (FROZEN 2, 6, 7; a failed gating check makes the reading not applicable)")
    for row in s["validity"]:
        add(f"  [{'PASS' if row['pass'] else 'FAIL'}]{'' if row['gates_reading'] else ' (reported, not gating)'} "
            f"{row['check']}" + (f": {row['detail']}" if row["detail"] else ""))
    led = s["ledger"]
    if led:
        add(f"  ledger: events {led['events']}; sent {led['sent_total']} in all; per model {led['sent_per_model']}; "
            f"by model and stage {led['sent_per_model_stage']}; refused lines {len(led['refused_lines'])}; "
            f"error lines {len(led['error_lines'])}")
    for model, c in s["count_only"].items():
        add(f"  count-only {model}: {c['file']}: bounds {c['bounds']}; cache {c['cache_total']}")
    add(f"  BUILD.sha256 driver line {s['build']['build_driver_sha256']}; driver.py now "
        f"{s['build']['driver_py_sha256_now']}")
    add("")
    add("== 2. Baselines derived again (asserted equal to FROZEN 2, 4 and 6; the analysis stops otherwise)")
    rows = []
    for model in MODELS:
        b = s["baselines"][model]
        rows.append([MODELS[model]["label"], f"{b['derived']['C'][0]} ({b['derived']['C'][1]})",
                     f"{b['derived']['D'][0]} ({b['derived']['D'][1]})", f"{b['derived']['P'][0]} ({b['derived']['P'][1]})",
                     " / ".join(map(str, b["trials"]["C"])), " / ".join(map(str, b["trials"]["D"])),
                     f"{sum(b['goal_steps'].values())} ({', '.join(str(b['goal_steps'][a]) for a in ARMS)})",
                     "equal" if b["equal_frozen"] else "DIFFER"])
    L += table(["model", "C t0 (lines)", "D replay t0", "I01 P t0", "C t0/t1/t2", "D t0/t1/t2",
                "C goal steps (real, staffed, plain)", "vs FROZEN"], rows)
    rows = [[MODELS[m]["label"], name] + [s["baselines"][m]["by_situation"][name][sit] for sit in SITUATIONS]
            for m in MODELS for name in ("C", "D", "P")]
    L += table(["model", "baseline", *[f"{sit} (of {n})" for sit, n in SITUATIONS.items()]], rows)
    add(f"  sources: recorded C = score.success of the records (harness); D = column D of {A2_DIR.name}/rows_old_*; "
        f"P = column P of {I01_DIR.name}/rows_old_* (equal to A2's P column record by record, asserted)")
    for model in MODELS:
        m = s["measures"].get(model)
        add("")
        add("#" * 100)
        add(f"# {MODELS[model]['label']} ({model})" + ("" if m else ": NO LIVE RECORDS"))
        add("#" * 100)
        if not m:
            continue
        add("")
        add(f"== M1. Successes over the t0 records [{LABELS['m1']}]")
        add(f"  D-live {m['m1']['successes']} of {m['m1']['n']}; vlm_error records {m['m1']['vlm_error']} "
            f"(counted as failures); status {m['m1']['status']}")
        add(f"  baselines on the same records: C {m['m3']['C']['baseline_successes']}, D replay "
            f"{m['m3']['D']['baseline_successes']}, I01 P {m['m3']['P']['baseline_successes']} (n {m['m3']['C']['n']})")
        add(f"  per arm [{LABELS['added']}]:")
        L += table(["arm", "n", "D-live", "C", "D replay", "I01 P"],
                   [[a, v["n"], v["live"], v["C"], v["D"], v["P"]] for a, v in m["per_arm"].items()])
        add("")
        add(f"== M2. Line successes and by situation [{LABELS['m2']}]")
        L += table(["situation", "n", "D-live", "C", "D replay", "I01 P"],
                   [[sit, v["n"], v["live"], v["C"], v["D"], v["P"]] for sit, v in m["m2"].items()])
        add("")
        add(f"== M3. Flips and breaks, record level [{LABELS['m3']}; P is secondary]")
        L += table(["against", "n", "flips", "breaks", "net", "flip scenes", "flip layouts", "break scenes",
                    "break layouts"],
                   [[name, v["n"], v["flips"], v["breaks"], v["net"], len(v["flip"]["scenes"]),
                     len(v["flip"]["layouts"]), len(v["break"]["scenes"]), len(v["break"]["layouts"])]
                    for name, v in m["m3"].items()])
        for name, v in m["m3"].items():
            add(f"  against {name}: flips by situation {v['flip']['by_situation']}, breaks by situation "
                f"{v['break']['by_situation']}; net by layout {v['net_by_layout'] or '{}'}")
            if v["flip"]["records"]:
                add(f"    flips: {', '.join(v['flip']['records'])}")
            if v["break"]["records"]:
                add(f"    breaks: {', '.join(v['break']['records'])}")
        g = m["goal_steps"]
        add("")
        add(f"  goal steps: D-live {g['live']} ({g['live_with_answer']} with a constraints answer); recorded C "
            f"{g['C']} ({g['C_with_answer']}); in both {g['in_both']}"
            + (f"; D-live only {g['live_only']}" if g["live_only"] else "")
            + (f"; C only {g['C_only']}" if g["C_only"] else ""))
        add("")
        add(f"== M4. Constraint usability, goal steps with a constraints answer [{LABELS['m4']}]")
        rows = []
        for arm_name in ("live", "C"):
            u = m["m4"][arm_name]
            rows.append(["D-live" if arm_name == "live" else "recorded C", u["goal_steps_with_answer"],
                         u["constraints"], u["status"]["used"], u["status"]["unusable"], u["status"]["superseded"],
                         u["status"]["cut"], u["no_usable_constraint_steps"], u["constraints_unusable_sum"],
                         u["constraints_unusable_mean"], u["exclusions_broken_sum"], u["exclusions_broken_mean"],
                         u["per_goal_step_mean"], u["per_goal_step_median"]])
        L += table(["arm", "steps", "constraints", "used", "unusable", "superseded", "cut", "no_usable steps",
                    "c_unusable sum", "mean", "excl_broken sum", "mean", "per step mean", "median"], rows)
        add("")
        add(f"== M5. Numbers named that exist in the list shown [{LABELS['m5']}]")
        rows = []
        for arm_name in ("live", "C"):
            u = m["m5"][arm_name]
            rows.append(["D-live (detector list)" if arm_name == "live" else "recorded C (VLM's list)",
                         u["goal_steps_with_answer"], u["numbers_named"], u["in_list"], u["share_in_list"],
                         u["steps_naming_one_not_in_list"], u["share_steps_naming_one_not_in_list"],
                         u["lists_not_numbered_1_to_n"]])
        L += table(["arm", "steps", "numbers named", "in list", "share", "steps naming one not in list", "share",
                    "lists not 1..N"], rows)
        add(f"  distinct per goal step [{LABELS['added']}]: D-live {m['m5']['live']['distinct_per_step_in_list']} of "
            f"{m['m5']['live']['distinct_per_step_named']}; C {m['m5']['C']['distinct_per_step_in_list']} of "
            f"{m['m5']['C']['distinct_per_step_named']}")
        add("")
        add(f"== M6. Latency of the goal step, ms [{LABELS['m6']}]")
        add("  D-live = latency_total_s of its constraints call + timings_s.geometry; recorded C = latency_total_s of "
            "its people call + its constraints call + timings_s.geometry")
        lat = m["m6"]
        rows = [["all goal steps", "D-live", lat["all"]["live"]["n"], ms(lat["all"]["live"]["median"]),
                 ms(lat["all"]["live"]["p90"])],
                ["all goal steps", "recorded C", lat["all"]["C"]["n"], ms(lat["all"]["C"]["median"]),
                 ms(lat["all"]["C"]["p90"])],
                ["all goal steps", "D-live + 55.6 ms detector (A1)", lat["all"]["live_plus_detector_55_6ms"]["n"],
                 ms(lat["all"]["live_plus_detector_55_6ms"]["median"]), ms(lat["all"]["live_plus_detector_55_6ms"]["p90"])],
                ["all goal steps", "D-live + 61.9 ms detector (verifier)", lat["all"]["live_plus_detector_61_9ms"]["n"],
                 ms(lat["all"]["live_plus_detector_61_9ms"]["median"]), ms(lat["all"]["live_plus_detector_61_9ms"]["p90"])],
                ["paired", "D-live", lat["paired"]["live"]["n"], ms(lat["paired"]["live"]["median"]),
                 ms(lat["paired"]["live"]["p90"])],
                ["paired", "recorded C", lat["paired"]["C"]["n"], ms(lat["paired"]["C"]["median"]),
                 ms(lat["paired"]["C"]["p90"])],
                ["paired", "D-live - C (per step)", lat["paired"]["live_minus_C"]["n"],
                 ms(lat["paired"]["live_minus_C"]["median"]), ms(lat["paired"]["live_minus_C"]["p90"])],
                ["paired", "D-live + 55.6 ms - C", lat["paired"]["live_plus_detector_55_6ms_minus_C"]["n"],
                 ms(lat["paired"]["live_plus_detector_55_6ms_minus_C"]["median"]),
                 ms(lat["paired"]["live_plus_detector_55_6ms_minus_C"]["p90"])],
                ["calls alone", "D-live constraints call", lat["calls_alone"]["live"]["n"],
                 ms(lat["calls_alone"]["live"]["median"]), ms(lat["calls_alone"]["live"]["p90"])],
                ["calls alone", "C people + constraints calls", lat["calls_alone"]["C"]["n"],
                 ms(lat["calls_alone"]["C"]["median"]), ms(lat["calls_alone"]["C"]["p90"])],
                ["calls alone", "paired D-live - C", lat["calls_alone"]["paired_live_minus_C"]["n"],
                 ms(lat["calls_alone"]["paired_live_minus_C"]["median"]),
                 ms(lat["calls_alone"]["paired_live_minus_C"]["p90"])]]
        L += table(["set", "what", "n", "median", "p90"], rows)
        add("  Recorded latencies are those of the calls first made on 11-13 Oct (a cached call carries the latency of "
            "the call stored); D-live's are new; the gateway's load differs; the detector time is a constant, not "
            "measured live.")
        add("")
        add(f"== M7. Extras [{LABELS['m7']}]")
        x = m["m7"]
        pp = x["people_per_goal_step"]
        add(f"  people in the numbered list per goal step: D-live (detector) n {pp['live_detector']['n']}, mean "
            f"{fmt(pp['live_detector']['mean'])}, median {fmt(pp['live_detector']['median'])}, range "
            f"{pp['live_detector']['min']}-{pp['live_detector']['max']}; recorded C (VLM) n {pp['C_vlm']['n']}, mean "
            f"{fmt(pp['C_vlm']['mean'])}, median {fmt(pp['C_vlm']['median'])}, range {pp['C_vlm']['min']}-"
            f"{pp['C_vlm']['max']}")
        fd = x["frozen_detector_people_per_scene"]
        add(f"  frozen detector people per scene (check files): {fd['scenes']} scenes, {fd['total']} people, mean "
            f"{fmt(fd['mean'])}; DetectorPeople asked {x['detector_source_asked']} time(s) (one per goal step)")
        rows = []
        for name in ("people_found_share", "people_unmatched", "members_precision", "members_recall"):
            lv, cv = x["method_measures"]["live"][name], x["method_measures"]["C"][name]
            rows.append([name, lv["n"], lv["mean"], lv["sum"], cv["n"], cv["mean"], cv["sum"]])
        L += table(["record measure (goal steps)", "D-live n", "mean", "sum", "C n", "mean", "sum"], rows)
        rows = [[stage, c.get("calls", 0), c.get("cached", 0), c.get("fresh", 0), c.get("attempts", 0),
                 c.get("validation_retries", 0), c.get("transient_retries", 0), c.get("retries_carried_by_cached", 0),
                 ", ".join(f"{k[7:]} {v}" for k, v in sorted(c.items()) if k.startswith("status_"))]
                for stage, c in x["calls_by_stage_live"].items()]
        L += table(["D-live calls in the records", "calls", "cached", "fresh", "attempts (fresh)",
                    "validation retries (fresh)", "transient retries (fresh)", "retries carried by cached", "status"],
                   rows)
        attempts = sum(c.get("attempts", 0) for c in x["calls_by_stage_live"].values())
        sent = (s["ledger"] or {}).get("sent_per_model", {}).get(model)
        add(f"  calls made: ledger sent {fmt(sent)} for {model}; attempts of the fresh calls in the records taken "
            f"{attempts} (equal when one invocation made every folder taken)")
        add(f"  recorded C goal steps, retries as recorded: {x['recorded_C_goal_step_retries']}")
    add("")
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------------------------

def jsonable(obj):
    if isinstance(obj, dict):
        return {(" ".join(map(str, k)) if isinstance(k, tuple) else str(k)): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="analyse.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS, help="the driver's --out folder (default live/runs)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="folder for tables.txt and summary.json (default live_analysis/result); not inside live/")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="take a missing or incomplete (arm, model) as it is; the reading is then not applicable")
    parser.add_argument("--count-vlm-errors-as-failures", action="store_true",
                        help="FROZEN 5.1: after one re-run, take a folder incomplete only through vlm_error records "
                             "and count those records as failures")
    parser.add_argument("--accept-mock", action="store_true",
                        help="test only: also read the driver's --mock folders (..._con_mock); never a result")
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if out == LIVE.resolve() or out.is_relative_to(LIVE.resolve()):
        refuse(f"--out {out} lies inside {LIVE}; this analysis never writes there")
    runs = args.runs.resolve()
    if not runs.is_dir():
        refuse(f"--runs {runs} is not a folder (has the live run happened?)")
    t_start = time.monotonic()
    me = sha256_file(Path(__file__))
    recorded_hash = next((words[0] for words in (line.split() for line in read_text(ANALYSE_SHA_FILE).splitlines())
                          if words[1:] == ["analyse.py"]), None) if ANALYSE_SHA_FILE.exists() else None

    frozen = check_frozen()
    build = build_sha()
    base = load_baselines()
    found = discover(runs, args.accept_mock)
    chosen, problems, notes = choose(found, args)
    if problems and not args.allow_incomplete:
        refuse("the six run folders are not all there and complete (REQUEST.md section 2: the latest complete "
               "folder per name):\n  " + "\n  ".join(problems)
               + "\n  --allow-incomplete analyses what is there (reading: not applicable); for a folder incomplete "
                 "only through vlm_error records after one re-run, --count-vlm-errors-as-failures applies FROZEN 5.1")
    live = {key: load_live_records(pick, key[0], key[1], base) for key, pick in chosen.items()}
    chk, info = validate(found, chosen, live, base, frozen, build, args)
    meas = measures(live, base, chosen, found)

    successes = {m: (meas[m]["m1"]["successes"] if meas.get(m) else None) for m in MODELS}
    why = [f"{r['check']}: {r['detail']}" if r["detail"] else r["check"] for r in chk.failed()]
    why = problems + why
    read_notes = []
    if any(p.get("vlm_error_only") and args.count_vlm_errors_as_failures for p in chosen.values()):
        read_notes.append("vlm_error records left after one re-run are counted as failures (FROZEN 5.1; FROZEN 6 "
                          "defines complete as no cap stop). The dispatch's summary said the reading needs no "
                          "vlm_error left; decided separately.")
    rule = reading(successes) if all(v is not None for v in successes.values()) else None
    synthetic = sorted({str((p["doc"].get("i09_live") or {}).get("synthetic")) for p in chosen.values()
                        if (p["doc"].get("i09_live") or {}).get("synthetic")})
    summary = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "analyse_sha256": me,
        "analyse_sha256_file": recorded_hash, "analyse_matches_hash_file": None if recorded_hash is None
        else recorded_hash == me, "runs": str(runs), "out": str(out), "harness": H.__file__,
        "sgi": sys.modules["sgi"].__file__ if "sgi" in sys.modules else None,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "synthetic": synthetic, "frozen": frozen, "build": build,
        "reading": {"label": LABELS["reading"], "successes": successes,
                    "n": {m: (meas[m]["m1"]["n"] if meas.get(m) else None) for m in MODELS},
                    "thresholds": {"consistent_at": CONSISTENT_AT, "contradicts_at": CONTRADICTS_AT},
                    "reading": rule if not why else "not applicable",
                    "not_applicable_because": why, "rule_on_these_numbers": rule if why else None,
                    "notes": read_notes},
        "folders": {"taken": [f"{arm} {model}: {p['name']} [{p['taken_as']}]" for (arm, model), p in sorted(chosen.items())],
                    "notes": notes, "ignored": found["ignored"],
                    "check_files": [c["path"].name for c in found["checks"]],
                    "count_files": [c["path"].name for c in found["counts"]]},
        "validity": chk.rows, "ledger": info["ledger"], "count_only": info["count_only"],
        "invocations": info["invocations"], "check_entries": info["check_entries"],
        "baselines": {m: {"derived": b["derived"], "trials": b["trials"], "goal_steps": b["goal_steps"],
                          "by_situation": b["by_situation"], "runs": {a: r.folder for a, r in b["refs"].items()},
                          "harness_tallies": b["tallies"], "a2_rows": b["a2_rows"], "i01_rows": b["i01_rows"],
                          "equal_frozen": b["derived"] == BASELINES[m] and b["trials"] == TRIALS[m]
                          and b["goal_steps"] == GOAL_STEPS[m], "frozen": BASELINES[m]} for m, b in base.items()},
        "labels": LABELS, "measures": meas, "seconds": None,
    }
    summary["seconds"] = round(time.monotonic() - t_start, 1)
    text = render(summary)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables.txt").write_text(text, encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(jsonable(summary), indent=1), encoding="utf-8")
    print(text)
    print(f"wrote {out / 'tables.txt'} and {out / 'summary.json'} ({summary['seconds']} s)")
    return 0 if not why else 3


def _command() -> int:
    try:
        return main()
    except Refused as stop:
        print(stop.code, file=sys.stderr)
        return 2
    except SystemExit as stop:
        if isinstance(stop.code, str):              # the harness refuses with a message
            print(stop.code, file=sys.stderr)
            return 2
        raise
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(_command())
