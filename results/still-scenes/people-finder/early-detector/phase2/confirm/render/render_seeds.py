"""Render driver for the I09 unseen-layout confirmation (phase2/i09/confirm/FROZEN.md, sections 1.2-1.6).

Runs the frozen generate command once per seed, one seed at a time, in ascending order:

    cd <worktree> && PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 \
      .venv/bin/python -m sgi.synth.realistic.generate --variant real --seed k \
      --out results/still-scenes/scene-seeds/phase2/realistic/seed<k> --only <the 14 ids>

One addition to the environment, which changes no output: TMPDIR points at render/tmp, so that the generator's
temporary folders (the placeholder layouts it replays, Blender's job folders) stay inside this agent's folder.

Modes:
    --trial            seed 200, real_eval_00_no_one_place only (timing trial; its time counts on the clock)
    --seeds A-B        the seeds A..B in order, stopping once the summed render wall time reaches 3 h
    --manifest         write render/manifest.json from the files and the logged state

Rules kept here (the task's hard rules and FROZEN.md 1.4, 10):
- before each command: nvidia-smi free >= 3000 MiB and free -m available >= 3000 MiB, else wait 5 min and log it;
  waits are not counted on the clock;
- the clock is the summed wall time of the generate commands, start to end; no new seed starts once it has
  reached 3 h 00 min, a running one finishes; no process is ever killed;
- a technical failure is run again (the generator skips completed scenes); a generator stop (no dressing passes
  in 5 attempts) is logged and the next seed follows (FROZEN 1.5 contingency);
- this script reads logs, file lists, render_job.json, and from scene.json only its scene id and the episodes'
  situation fields (asked for by the task); it opens no picture for viewing (sha256 of the bytes only) and no
  detector output.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402

WT = paths.REPO                                    # the repository root
RESEARCH = paths.STILL                             # results/still-scenes
CONFIRM = RESEARCH / "people-finder/early-detector/phase2/confirm"
RENDER = CONFIRM / "render"
LOGS = RENDER / "logs"
STATE = RENDER / "state.json"
EVENTS = RENDER / "render_events.jsonl"
SEEDS_ROOT = RESEARCH / "scene-seeds/phase2/realistic"
SEEDS_REL = SEEDS_ROOT.relative_to(WT)
FROZEN = CONFIRM / "FROZEN.md"
FROZEN_SHA = "f55341c03cd5e1d38c91b7e0142ef10f0870ccb2b43c5bf4edfac07d1996e751"
CONFIG_SHA = "c651f4c484fff15d33290e13f28bc53c9221c8edd79c0b7a24f24a128098b4fc"

IDS = ["real_eval_00_no_one_place", "real_eval_01_no_one_place", "real_eval_02_no_one_place",
       "real_eval_03_no_one_deep_place", "real_eval_04_no_one_object", "real_eval_05_no_one_person",
       "real_eval_06_waiting_line_1", "real_eval_07_waiting_line_2", "real_eval_08_waiting_line_4",
       "real_eval_09_waiting_line_6", "real_eval_10_conversation_group_2", "real_eval_11_conversation_group_3",
       "real_eval_12_conversation_group_4", "real_eval_13_waiting_line_3"]
SITUATIONS = {"no_one", "waiting_line", "conversation_group"}
BANNED = re.compile(r"group_photo|lift_wait|heldout|held_out|eval_1[6-9]", re.IGNORECASE)
SEED_MIN, SEED_MAX = 200, 219
BUDGET_S = 3 * 3600.0
MIN_GPU_FREE_MIB = 3000
MIN_RAM_AVAIL_MIB = 3000
POLL_S = 300
MAX_RUNS_PER_SEED = 4          # one run plus up to three re-runs after technical failures
SCENE_LINE = re.compile(r"(real_eval_\d\d_[a-z_0-9]+): (rendered|skipped) in ([0-9.]+) s, after (\d+) dressings given up")
SUMMARY_LINE = re.compile(r": (\d+) scenes rendered, (\d+) skipped as unchanged, (\d+) dressed again")
GEN_STOP = "no dressing passed the scene check"


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def event(kind: str, **fields) -> None:
    line = {"time": now(), "event": kind, **fields}
    with EVENTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
    print(json.dumps(line), flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_state() -> dict:
    if STATE.is_file():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"render_wall_s": 0.0, "wait_s": 0.0, "commands": [], "seeds": {}}


def save_state(state: dict) -> None:
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    os.replace(tmp, STATE)


def gpu_status() -> dict:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total,utilization.gpu",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60)
    used, free, total, util = (int(x.strip()) for x in out.stdout.strip().splitlines()[0].split(","))
    return {"used_mib": used, "free_mib": free, "total_mib": total, "util_pct": util}


def ram_status() -> dict:
    out = subprocess.run(["free", "-m"], capture_output=True, text=True, timeout=60)
    mem = next(line.split() for line in out.stdout.splitlines() if line.startswith("Mem:"))
    return {"total_mib": int(mem[1]), "used_mib": int(mem[2]), "available_mib": int(mem[6])}


def wait_for_resources(label: str, state: dict) -> float:
    """Block until the GPU and RAM rules hold; returns the seconds waited (logged, not counted on the clock)."""
    begun, polls = time.monotonic(), 0
    while True:
        gpu, ram = gpu_status(), ram_status()
        ok = gpu["free_mib"] >= MIN_GPU_FREE_MIB and ram["available_mib"] >= MIN_RAM_AVAIL_MIB
        event("resources", label=label, gpu=gpu, ram=ram, ok=ok)
        if ok:
            waited = time.monotonic() - begun
            if polls:
                event("wait_end", label=label, waited_s=round(waited, 1), polls=polls)
                state["wait_s"] = state.get("wait_s", 0.0) + waited
                return waited
            return 0.0
        event("wait", label=label, poll_s=POLL_S,
              reason=f"gpu free {gpu['free_mib']} MiB (need {MIN_GPU_FREE_MIB}), "
                     f"ram available {ram['available_mib']} MiB (need {MIN_RAM_AVAIL_MIB})")
        polls += 1
        time.sleep(POLL_S)


def run_generate(seed: int, ids: list[str], label: str, state: dict) -> dict:
    """One generate command, timed; its stdout and stderr go to logs/<label>.log."""
    LOGS.mkdir(parents=True, exist_ok=True)
    (RENDER / "tmp").mkdir(parents=True, exist_ok=True)
    out_rel = SEEDS_REL / f"seed{seed}"
    cmd = ["nice", "-n", "10", ".venv/bin/python", "-m", "sgi.synth.realistic.generate", "--variant", "real",
           "--seed", str(seed), "--out", str(out_rel), "--only", *ids]
    env = dict(os.environ)
    env.update({"PYTHONPATH": "", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                "TMPDIR": str(RENDER / "tmp")})
    log_path = LOGS / f"{label}.log"
    event("command_start", label=label, seed=seed, cmd=cmd, cwd=str(WT), log=str(log_path.relative_to(RENDER)))
    started_at, begun = now(), time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# {started_at} cwd={WT}\n# PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "
                  f"MKL_NUM_THREADS=2 TMPDIR={RENDER / 'tmp'} {' '.join(cmd)}\n")
        log.flush()
        done = subprocess.run(cmd, cwd=WT, env=env, stdout=log, stderr=subprocess.STDOUT)
    wall = time.monotonic() - begun
    text = log_path.read_text(encoding="utf-8", errors="replace")
    scenes = {m.group(1): {"outcome": m.group(2), "seconds": float(m.group(3)), "given_up": int(m.group(4))}
              for m in SCENE_LINE.finditer(text)}
    summary = SUMMARY_LINE.search(text)
    kind = "ok" if done.returncode == 0 else ("generator_stop" if GEN_STOP in text else "technical_failure")
    record = {"label": label, "seed": seed, "ids": ids, "start": started_at, "end": now(),
              "wall_s": round(wall, 2), "returncode": done.returncode, "outcome": kind, "scenes": scenes,
              "summary": None if summary is None else {"rendered": int(summary.group(1)),
                                                        "skipped": int(summary.group(2)),
                                                        "dressed_again": int(summary.group(3))},
              "log": str(log_path.relative_to(RENDER))}
    if kind != "ok":
        record["log_tail"] = text.splitlines()[-25:]
    state["render_wall_s"] = state.get("render_wall_s", 0.0) + wall
    state["commands"].append(record)
    save_state(state)
    event("command_end", label=label, seed=seed, wall_s=round(wall, 1), returncode=done.returncode, outcome=kind,
          render_wall_total_s=round(state["render_wall_s"], 1),
          scenes_logged=len(scenes), summary=record["summary"])
    return record


def check_names() -> list[str]:
    bad = []
    if SEEDS_ROOT.exists():
        for path in SEEDS_ROOT.rglob("*"):
            if BANNED.search(path.name):
                bad.append(str(path))
    return bad


def check_seed(seed: int, expected: list[str]) -> dict:
    """The per-seed assertions. Returns {"complete": bool, "problems": [...], "scenes": {...}}."""
    sys.path.insert(0, str(WT / "results/still-scenes/analysis-scripts/harness"))
    import research_harness as H                     # noqa: E402  guard_scene refuses held-out or excluded names
    from sgi.config import load_config, DEFAULT_CONFIG
    from sgi.scene import load_scene

    problems: list[str] = []
    seed_dir = SEEDS_ROOT / f"seed{seed}"
    eval_dir = seed_dir / "eval"
    th = load_config(DEFAULT_CONFIG).thresholds.model_dump(mode="json")
    found = sorted(p.name for p in eval_dir.iterdir()) if eval_dir.is_dir() else []
    if sorted(found) != sorted(expected):
        problems.append(f"eval/ holds {found}, expected {sorted(expected)}")
    others = sorted(p.name for p in seed_dir.iterdir() if p.name != "eval") if seed_dir.is_dir() else []
    examples = seed_dir / "examples"
    if examples.exists() and any(examples.iterdir()):
        problems.append("examples/ is not empty")
    if [o for o in others if o != "examples"]:
        problems.append(f"seed folder holds more than eval/: {others}")
    bad_names = check_names()
    if bad_names:
        problems.append(f"banned names under {SEEDS_ROOT}: {bad_names}")
    scenes = {}
    for sid in expected:
        folder = eval_dir / sid
        info: dict = {"folder": str(folder.relative_to(WT))}
        try:
            needed = ["scene.json", "rgb.png", "depth.npy", "render_job.json",
                      *(f"tour/{i:03d}.png" for i in range(8))]
            missing = [n for n in needed if not (folder / n).is_file()]
            if missing:
                raise AssertionError(f"missing files {missing}")
            H.guard_scene(folder)
            load_scene(folder)
            scene = json.loads((folder / "scene.json").read_text(encoding="utf-8"))
            if scene["scene_id"] != sid:
                raise AssertionError(f"scene_id {scene['scene_id']} != folder {sid}")
            situations = sorted({ep["ground_truth"]["situation"] for ep in scene["episodes"]})
            if not set(situations) <= SITUATIONS:
                raise AssertionError(f"situations {situations} outside {sorted(SITUATIONS)}")
            job = json.loads((folder / "render_job.json").read_text(encoding="utf-8"))
            if job.get("variant") != "real" or job.get("seed") != seed or job.get("renderer", {}).get("name") != "blender":
                raise AssertionError(f"render job says variant={job.get('variant')} seed={job.get('seed')} "
                                     f"renderer={job.get('renderer')}")
            if job.get("thresholds") != th:
                raise AssertionError("render job thresholds differ from configs/default.yaml")
            tour_files = sorted(p.name for p in (folder / "tour").iterdir())
            info.update({"situations": situations, "n_episodes": len(scene["episodes"]),
                         "attempt": job["attempt"], "renderer_version": job["renderer"]["version"],
                         "tour_files": len(tour_files), "ok": True})
        except (Exception, SystemExit) as error:     # noqa: BLE001  H.Forbidden is a SystemExit; recorded
            info.update({"ok": False, "error": f"{type(error).__name__}: {error}"})
            problems.append(f"{sid}: {type(error).__name__}: {error}")
        scenes[sid] = info
    return {"complete": not problems, "problems": problems, "scenes": scenes}


def preflight() -> None:
    assert sha256(FROZEN) == FROZEN_SHA, "FROZEN.md sha256 mismatch"
    assert sha256(WT / "configs/default.yaml") == CONFIG_SHA, "configs/default.yaml sha256 mismatch"


def do_seed(seed: int, ids: list[str], label_base: str, state: dict) -> dict:
    runs = []
    for run in range(1, MAX_RUNS_PER_SEED + 1):
        waited = wait_for_resources(f"{label_base}_run{run}", state)
        save_state(state)
        rec = run_generate(seed, ids, f"{label_base}_run{run}", state)
        rec["waited_before_s"] = round(waited, 1)
        runs.append(rec["label"])
        if rec["outcome"] in ("ok", "generator_stop"):
            return {"outcome": rec["outcome"], "runs": runs}
        event("rerun", seed=seed, after=rec["label"], reason="technical failure; the same command is run again")
    return {"outcome": "technical_failure_exhausted", "runs": runs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", action="store_true")
    ap.add_argument("--seeds", default=None, help="A-B")
    ap.add_argument("--manifest", action="store_true")
    args = ap.parse_args(argv)
    os.chdir(WT)
    preflight()
    RENDER.mkdir(parents=True, exist_ok=True)
    state = load_state()

    if args.trial:
        event("trial_begin", seed=SEED_MIN, ids=IDS[:1])
        res = do_seed(SEED_MIN, IDS[:1], f"seed{SEED_MIN}_trial", state)
        state.setdefault("trial", {}).update(res)
        save_state(state)
        event("trial_end", **res, render_wall_total_s=round(state["render_wall_s"], 1))
        return 0 if res["outcome"] == "ok" else 1

    if args.seeds:
        lo, hi = (int(x) for x in args.seeds.split("-"))
        assert SEED_MIN <= lo <= hi <= SEED_MAX, "seeds outside 200-219"
        for seed in range(lo, hi + 1):
            if state["seeds"].get(str(seed), {}).get("complete"):
                event("seed_skip", seed=seed, reason="already complete in state")
                continue
            if state["render_wall_s"] >= BUDGET_S:
                event("stop", reason="render wall time reached 3 h", render_wall_total_s=round(state["render_wall_s"], 1),
                      next_seed_not_started=seed)
                state["stopped"] = {"reason": "budget", "at": now(), "next_seed_not_started": seed}
                save_state(state)
                return 0
            event("seed_begin", seed=seed, render_wall_total_s=round(state["render_wall_s"], 1))
            res = do_seed(seed, IDS, f"seed{seed}", state)
            check = check_seed(seed, IDS)
            state["seeds"][str(seed)] = {**res, "complete": res["outcome"] == "ok" and check["complete"],
                                         "problems": check["problems"], "checked_at": now()}
            save_state(state)
            event("seed_end", seed=seed, outcome=res["outcome"], complete=state["seeds"][str(seed)]["complete"],
                  problems=check["problems"], render_wall_total_s=round(state["render_wall_s"], 1))
            if res["outcome"] == "technical_failure_exhausted":
                event("halt", seed=seed, reason="technical failures repeated; stopping for a human look")
                state["stopped"] = {"reason": "technical_failure_exhausted", "at": now(), "seed": seed}
                save_state(state)
                return 2
        if hi == SEED_MAX:
            state["stopped"] = {"reason": "after seed 219", "at": now()}
            save_state(state)
            event("stop", reason="after seed 219", render_wall_total_s=round(state["render_wall_s"], 1))
        return 0

    if args.manifest:
        write_manifest(state)
        return 0
    ap.error("name a mode")
    return 2


def write_manifest(state: dict) -> None:
    # seconds per scene: the last command that rendered (or skipped) it in its own seed's runs
    per_scene: dict[tuple[int, str], dict] = {}
    for rec in state["commands"]:
        for sid, s in rec["scenes"].items():
            key = (rec["seed"], sid)
            if s["outcome"] == "rendered" or key not in per_scene:
                per_scene[key] = {**s, "command": rec["label"]}
    entries, seeds = [], []
    for seed in range(SEED_MIN, SEED_MAX + 1):
        st = state["seeds"].get(str(seed))
        if st is None:
            continue
        check = check_seed(seed, IDS)
        complete = st.get("outcome") == "ok" and check["complete"]
        seed_cmds = [r for r in state["commands"] if r["seed"] == seed]
        seeds.append({"seed": seed, "complete": complete, "outcome": st.get("outcome"), "runs": st.get("runs"),
                      "problems": check["problems"],
                      "command_wall_s": round(sum(r["wall_s"] for r in seed_cmds), 2),
                      "dressed_again": sum((r.get("summary") or {}).get("dressed_again", 0) for r in seed_cmds)})
        for row, sid in enumerate(IDS):
            folder = SEEDS_ROOT / f"seed{seed}" / "eval" / sid
            info = check["scenes"][sid]
            timing = per_scene.get((seed, sid), {})
            entry = {"seed": seed, "row": row, "scene_id": sid, "folder": str(folder.relative_to(WT)),
                     "situations": info.get("situations"), "ok": info.get("ok"),
                     "seconds": timing.get("seconds"), "outcome": timing.get("outcome"),
                     "rendered_by": timing.get("command"),
                     "dressings_given_up": timing.get("given_up"),
                     "attempt": info.get("attempt"), "renderer_version": info.get("renderer_version")}
            if folder.is_dir():
                for name in ("rgb.png", "scene.json", "depth.npy", "render_job.json"):
                    if (folder / name).is_file():
                        entry[f"sha256_{name.replace('.', '_')}"] = sha256(folder / name)
            entries.append(entry)
    rendered_secs = [e["seconds"] for e in entries if e["outcome"] == "rendered" and e["seconds"] is not None]
    rendered_secs.sort()
    median = None
    if rendered_secs:
        n = len(rendered_secs)
        median = rendered_secs[n // 2] if n % 2 else 0.5 * (rendered_secs[n // 2 - 1] + rendered_secs[n // 2])
    manifest = {
        "written": now(),
        "note": "Render manifest for phase2/i09/confirm (FROZEN.md sha256 " + FROZEN_SHA + "). "
                "Not the manifest.json of FROZEN section 2, which manifest.py writes.",
        "frozen_sha256": FROZEN_SHA,
        "command": "cd <worktree> && PYTHONPATH= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 "
                   "TMPDIR=<render>/tmp nice -n 10 .venv/bin/python -m sgi.synth.realistic.generate --variant real "
                   "--seed k --out " + str(SEEDS_REL) + "/seed<k> --only <14 ids>",
        "render_wall_s_total": round(state["render_wall_s"], 2),
        "wait_s_total": round(state.get("wait_s", 0.0), 2),
        "budget_s": BUDGET_S,
        "stopped": state.get("stopped"),
        "trial": state.get("trial"),
        "seeds_complete": [s["seed"] for s in seeds if s["complete"]],
        "seeds_left_out": [{"seed": s["seed"], "reason": s["outcome"], "problems": s["problems"]}
                           for s in seeds if not s["complete"]]
                          + [{"seed": k, "reason": "not started (stop)"} for k in range(SEED_MIN, SEED_MAX + 1)
                             if str(k) not in state["seeds"]],
        "scene_seconds": {"n_rendered": len(rendered_secs), "median": median,
                          "min": rendered_secs[0] if rendered_secs else None,
                          "max": rendered_secs[-1] if rendered_secs else None,
                          "mean": round(sum(rendered_secs) / len(rendered_secs), 2) if rendered_secs else None,
                          "definition": "the generator's own per-scene time (its log line), rendered scenes only"},
        "seeds": seeds,
        "scenes": entries,
        "commands": [{k: v for k, v in r.items() if k != "scenes"} for r in state["commands"]],
    }
    out = RENDER / "manifest.json"
    out.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (RENDER / "manifest.sha256").write_text(f"{sha256(out)}  manifest.json\n", encoding="utf-8")
    event("manifest", path=str(out.relative_to(WT)), scenes=len(entries),
          seeds_complete=manifest["seeds_complete"])


if __name__ == "__main__":
    raise SystemExit(main())
