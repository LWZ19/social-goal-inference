"""I09 confirmation, step 4 (FROZEN.md sections 1.5, 2 and 8 C4): the test manifest, written once.

Runs after FROZEN.sha256 (and DEVIATIONS_1.sha256) verify and after the render step's log says it has stopped.
Completeness is determined from the files, not from a status file. Every scene folder is guarded (H.guard_scene and
the section 3 path guard) before its scene.json is read, then loaded with sgi.scene.load_scene. Writes manifest.json
and MANIFEST.sha256. Project venv.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import confirm_lib as CL  # noqa: E402
import pathguard  # noqa: E402
from sgi.config import load_config  # noqa: E402
from sgi.scene import load_scene, read_scene_file  # noqa: E402

H = CL.H
NAME_BAN = re.compile(r"group_photo|lift_wait|heldout|held_out|eval_1[6-9]", re.I)
ALLOWED_SITUATIONS = {"no_one", "waiting_line", "conversation_group"}
REQUIRED_FILES = ("scene.json", "rgb.png", "depth.npy", "render_job.json")


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def render_stopped() -> dict:
    ev = [json.loads(x) for x in (CL.HERE / "render/render_events.jsonl").read_text().splitlines() if x.strip()]
    stops = [e for e in ev if e.get("event") == "stop"]
    assert stops, "the render step's log has no stop event"
    running = []
    for pid in os.listdir("/proc"):
        if pid.isdigit():
            try:
                cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            except OSError:
                continue
            if "sgi.synth.realistic.generate" in cmd or ("blender" in cmd and str(CL.WORKTREE) in cmd):
                running.append(cmd[:200])
    assert not running, f"render processes still running: {running}"
    return stops[-1]


def check_names() -> int:
    """No file or folder name anywhere under phase2/seeds/realistic/ holds a banned name (case ignored)."""
    n = 0
    for top, dirs, files in os.walk(CL.SEEDS, followlinks=False):
        for name in dirs + files:
            n += 1
            assert not NAME_BAN.search(name), f"banned name under seeds: {Path(top) / name}"
            assert not (Path(top) / name).is_symlink(), f"link under seeds: {Path(top) / name}"
    return n


def check_seed(k: int, thresholds_default: dict):
    """(complete, problems, entries) for seed k, from the files."""
    seed_dir = CL.SEEDS / f"seed{k}"
    problems, entries = [], []
    top = sorted(os.listdir(seed_dir))
    extra_top = [x for x in top if x not in ("eval", "examples")]
    if extra_top:
        problems.append(f"unexpected entries in seed{k}/: {extra_top}")
    ex = seed_dir / "examples"
    if ex.exists() and any(ex.iterdir()):
        problems.append("examples/ is not empty")
    ev = seed_dir / "eval"
    if not ev.is_dir():
        return False, problems + ["no eval/ folder"], []
    names = sorted(os.listdir(ev))
    if names != sorted(CL.ROW_IDS):
        problems.append(f"eval/ holds {names}, not exactly the 14 row ids")
    for row, sid in enumerate(CL.ROW_IDS):
        folder = ev / sid
        if not folder.is_dir():
            problems.append(f"missing scene {sid}")
            continue
        pathguard.check(folder)
        missing = [f for f in REQUIRED_FILES if not (folder / f).is_file()]
        if missing:
            problems.append(f"{sid}: missing {missing}")
            continue
        H.guard_scene(folder)                                  # guarded before scene.json is read
        sf = read_scene_file(folder)
        tour = [folder / fr.rgb for fr in sf.tour]
        if not tour or not all(t.is_file() for t in tour):
            problems.append(f"{sid}: tour frames missing ({len(tour)} named)")
            continue
        rgb = folder / sf.observation.rgb
        depth = folder / sf.observation.depth
        assert rgb == folder / "rgb.png" and depth == folder / "depth.npy", (rgb, depth)
        pathguard.check(rgb)
        H.guard_file(rgb)
        if sf.scene_id != sid:
            problems.append(f"{sid}: scene.json scene_id {sf.scene_id}")
        sits = {ep.ground_truth.situation for ep in sf.episodes}
        assert sits <= ALLOWED_SITUATIONS, f"{sid}: situation outside the first pass: {sits}"
        job = json.loads((folder / "render_job.json").read_text())
        if job.get("variant") != "real" or job.get("seed") != k or job.get("renderer", {}).get("name") != "blender":
            problems.append(f"{sid}: render job variant {job.get('variant')}, seed {job.get('seed')}, renderer "
                            f"{job.get('renderer')}")
        if job.get("thresholds") != thresholds_default:
            diff = sorted(set(job.get("thresholds", {}).items()) ^ set(thresholds_default.items()))
            problems.append(f"{sid}: render-job thresholds differ from configs/default.yaml: {diff[:6]}")
        try:
            load_scene(folder)
        except Exception as exc:                               # noqa: BLE001
            problems.append(f"{sid}: load_scene failed: {exc!r}"[:300])
            continue
        entries.append({"seed": k, "row": row, "row_id": sid.removeprefix("real_"), "scene_id": sid,
                        "folder": str(folder), "rgb": str(rgb), "sha256_rgb_png": sha(rgb),
                        "sha256_depth_npy": sha(depth), "sha256_scene_json": sha(folder / "scene.json"),
                        "sha256_render_job_json": sha(folder / "render_job.json"),
                        "attempt": job.get("attempt"), "renderer_version": job.get("renderer", {}).get("version"),
                        "situations": sorted(sits), "n_tour_frames": len(tour)})
    complete = not problems and len(entries) == 14
    return complete, problems, entries


def main():
    out_path = CL.HERE / "manifest.json"
    assert not out_path.exists() and not (CL.HERE / "MANIFEST.sha256").exists(), "manifest.json exists already"
    stop = render_stopped()
    th = load_config().thresholds.model_dump()
    n_names = check_names()
    seed_dirs = sorted(os.listdir(CL.SEEDS))
    for d in seed_dirs:
        m = re.fullmatch(r"seed(\d+)", d)
        assert m and int(m.group(1)) in CL.SEED_RANGE, f"unexpected folder under seeds/realistic: {d}"
    entries, left_out, seeds_complete = [], [], []
    for k in CL.SEED_RANGE:
        if f"seed{k}" not in seed_dirs:
            left_out.append({"seed": k, "reason": "not rendered: the render step stopped on the 3 h clock before "
                                                  "this seed (render log 'stop' event)"})
            continue
        complete, problems, ents = check_seed(k, th)
        if complete:
            seeds_complete.append(k)
            entries += ents
        else:
            left_out.append({"seed": k, "reason": "incomplete", "problems": problems})
        print(f"seed{k}: complete {complete}; {len(ents)} scenes; problems {problems}", flush=True)
    required_ok = all(k in seeds_complete for k in CL.REQUIRED_SEEDS)
    # cross-check against the render step's own per-scene hashes (DEVIATIONS_1.md R4); reported, not used
    rm = json.loads((CL.HERE / "render/manifest.json").read_text())
    rm_by = {(s["seed"], s["scene_id"]): s for s in rm["scenes"]}
    xdiff = []
    for e in entries:
        r = rm_by.get((e["seed"], e["scene_id"]))
        if r is None:
            xdiff.append([e["seed"], e["scene_id"], "absent from the render step's manifest"])
            continue
        for key in ("sha256_rgb_png", "sha256_depth_npy", "sha256_scene_json", "sha256_render_job_json", "attempt",
                    "renderer_version"):
            if r.get(key) != e[key]:
                xdiff.append([e["seed"], e["scene_id"], key])
    sits = {}
    for e in entries:
        for s in e["situations"]:
            sits[s] = sits.get(s, 0) + 1
    images = [{"scene_id": e["scene_id"], "domain": "test", "set": "real", "rgb": e["rgb"],
               "sha256": e["sha256_rgb_png"], "seed": e["seed"], "row": e["row"]} for e in entries]
    assert len({i["rgb"] for i in images}) == len(images)
    out = {"purpose": "I09 unseen-layout confirmation test manifest (FROZEN.md section 2); written once",
           "written": time.strftime('%Y-%m-%dT%H:%M:%S%z'), "protocol_hashes": CL.PROTOCOL_HASHES,
           "render_stop_event": stop, "names_checked_under_seeds": n_names,
           "seeds_complete": seeds_complete, "seeds_left_out": left_out,
           "required_seeds_200_209_complete": required_ok,
           "outcome_if_incomplete": None if required_ok else "incomplete (FROZEN.md section 1.5)",
           "counts": {"seeds": len(seeds_complete), "layouts": len(entries), "situations": sits},
           "render_manifest_crosscheck": {"render_manifest_sha256": CL.sha(CL.HERE / "render/manifest.json"),
                                          "differences": xdiff},
           "order": "seed ascending, then row ascending",
           "entries": entries, "images": images}
    out_path.write_text(json.dumps(out, indent=1))
    (CL.HERE / "MANIFEST.sha256").write_text(f"{CL.sha(out_path)}  manifest.json\n")
    os.chmod(out_path, 0o444)
    os.chmod(CL.HERE / "MANIFEST.sha256", 0o444)
    print(f"manifest.json: {len(entries)} layouts in seeds {seeds_complete}; left out {[x['seed'] for x in left_out]}; "
          f"required 200-209 complete: {required_ok}; situations {sits}; render-manifest differences {len(xdiff)}; "
          f"{out['written']}")


if __name__ == "__main__":
    main()
