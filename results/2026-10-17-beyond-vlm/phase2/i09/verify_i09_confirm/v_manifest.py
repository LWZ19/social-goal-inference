"""Verifier: independent check of the render set and of the compute agent's manifest.json (project venv).

Reads only: FROZEN.md/sha, configs/default.yaml, src/sgi/synth/generate.py's _EVAL table (row designs), the scene
folders under phase2/seeds/realistic (each guarded by v_guard and H.guard_scene before scene.json is read), the
render step's events log, the compute agent's manifest.json. Writes v_images.json (image list for the verifier's
own sidecar) and v_manifest_check.txt in this folder. No model, no network.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
RESEARCH = Path("results/2026-10-17-beyond-vlm")
WT = Path(".-scenes")
CONFIRM = RESEARCH / "phase2/i09/confirm"
SEEDS = RESEARCH / "phase2/seeds/realistic"
sys.path.insert(0, str(WT / "docs/model_notes/tools"))
import research_harness as H  # noqa: E402
from sgi.config import load_config  # noqa: E402
from sgi.scene import load_scene, read_scene_file  # noqa: E402
import ast  # noqa: E402

spec = importlib.util.spec_from_file_location("v_guard", HERE / "v_guard.py")
VG = importlib.util.module_from_spec(spec)
spec.loader.exec_module(VG)

FROZEN_SHA = "f55341c03cd5e1d38c91b7e0142ef10f0870ccb2b43c5bf4edfac07d1996e751"
DEFAULT_YAML_SHA = "c651f4c484fff15d33290e13f28bc53c9221c8edd79c0b7a24f24a128098b4fc"
BAN = re.compile(r"group_photo|lift_wait|heldout|held_out|eval_1[6-9]", re.I)
FIRST_PASS = {"no_one", "waiting_line", "conversation_group"}
L = []


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def out(*a):
    s = " ".join(str(x) for x in a)
    L.append(s)
    print(s, flush=True)


def row_ids():
    ids = []
    tree = ast.parse((WT / "src/sgi/synth/generate.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_EVAL")
    for i, (recipe, n, kind, _by, _g2) in enumerate(ast.literal_eval(node.value)):
        name = f"{recipe}_{kind}" if recipe == "no_one" else f"{recipe}_{n}"
        ids.append(f"real_eval_{i:02d}_{name}")
    return ids


def main():
    assert sha(CONFIRM / "FROZEN.md") == FROZEN_SHA
    assert (CONFIRM / "FROZEN.sha256").read_text().split()[0] == FROZEN_SHA
    assert sha(WT / "configs/default.yaml") == DEFAULT_YAML_SHA
    out(f"v_manifest {time.strftime('%Y-%m-%dT%H:%M:%S%z')}; FROZEN.md sha256 verified {FROZEN_SHA}")
    ids = row_ids()
    frozen_ids = re.findall(r"`(real_eval_\d\d_[a-z_0-9]+)`", (CONFIRM / "FROZEN.md").read_text())
    assert sorted(set(frozen_ids)) == sorted(ids), (frozen_ids, ids)
    out(f"14 row ids from G._EVAL equal FROZEN 1.2's table: {len(ids)}")

    # every name under seeds/realistic
    n_names = 0
    for top, dirs, files in os.walk(SEEDS):
        for nm in dirs + files:
            n_names += 1
            assert not BAN.search(nm), Path(top) / nm
            assert not (Path(top) / nm).is_symlink(), Path(top) / nm
    out(f"names under seeds/realistic checked: {n_names}; none banned, no links")
    seed_dirs = sorted(os.listdir(SEEDS))
    seeds = []
    for d in seed_dirs:
        m = re.fullmatch(r"seed(\d+)", d)
        assert m, d
        k = int(m.group(1))
        assert 200 <= k <= 219, d
        seeds.append(k)
    seeds.sort()
    out(f"seed folders: {seeds}")

    th_default = load_config().thresholds.model_dump()
    entries = []
    sits_count = {}
    complete = []
    for k in seeds:
        sd = SEEDS / f"seed{k}"
        top = sorted(os.listdir(sd))
        problems = []
        if [x for x in top if x not in ("eval", "examples")]:
            problems.append(f"extra top entries {top}")
        if (sd / "examples").exists() and any((sd / "examples").iterdir()):
            problems.append("examples not empty")
        names = sorted(os.listdir(sd / "eval"))
        if names != sorted(ids):
            problems.append(f"eval holds {names}")
        for row, sid in enumerate(ids):
            f = sd / "eval" / sid
            assert VG.check(str(f / "scene.json")) == k
            H.guard_scene(f)
            sfile = read_scene_file(f)
            req = ["scene.json", "rgb.png", "depth.npy", "render_job.json"]
            miss = [x for x in req if not (f / x).is_file()]
            tour = [f / fr.rgb for fr in sfile.tour]
            if miss or not tour or not all(t.is_file() for t in tour):
                problems.append(f"{sid}: missing {miss} or tour")
            if sfile.scene_id != sid:
                problems.append(f"{sid}: scene_id {sfile.scene_id}")
            sits = sorted({ep.ground_truth.situation for ep in sfile.episodes})
            if not set(sits) <= FIRST_PASS:
                problems.append(f"{sid}: situations {sits}")
            for s_ in sits:
                sits_count[s_] = sits_count.get(s_, 0) + 1
            job = json.loads((f / "render_job.json").read_text())
            if job["variant"] != "real" or job["seed"] != k or job["renderer"]["name"] != "blender":
                problems.append(f"{sid}: job {job['variant']} {job['seed']} {job['renderer']}")
            if job["thresholds"] != th_default:
                problems.append(f"{sid}: thresholds differ")
            load_scene(f)
            assert sfile.observation.rgb == "rgb.png" and sfile.observation.depth == "depth.npy"
            entries.append({"seed": k, "row": row, "scene_id": sid, "folder": str(f), "rgb": str(f / "rgb.png"),
                            "sha256_rgb_png": sha(f / "rgb.png"), "sha256_depth_npy": sha(f / "depth.npy"),
                            "sha256_scene_json": sha(f / "scene.json"),
                            "sha256_render_job_json": sha(f / "render_job.json"), "attempt": job["attempt"],
                            "renderer_version": job["renderer"]["version"], "situations": sits})
        ok = not problems
        if ok:
            complete.append(k)
        out(f"seed{k}: complete {ok}; problems {problems}")
    out(f"complete seeds {complete}; required 200-209 complete: {all(k in complete for k in range(200, 210))}; "
        f"layouts {len(entries)}; situations per scene {sits_count}")
    out(f"attempt > 0: {[(e['seed'], e['scene_id'], e['attempt']) for e in entries if e['attempt']]}")

    # compare with the compute agent's manifest.json
    cm = json.loads((CONFIRM / "manifest.json").read_text())
    assert (CONFIRM / "MANIFEST.sha256").read_text().split()[0] == sha(CONFIRM / "manifest.json")
    keys = ("seed", "row", "scene_id", "folder", "rgb", "sha256_rgb_png", "sha256_depth_npy", "sha256_scene_json",
            "sha256_render_job_json", "attempt", "renderer_version", "situations")
    diffs = []
    if len(cm["entries"]) != len(entries):
        diffs.append(("count", len(cm["entries"]), len(entries)))
    for a, b in zip(cm["entries"], entries):
        for kk in keys:
            if a.get(kk) != b.get(kk):
                diffs.append((b["seed"], b["scene_id"], kk, a.get(kk), b.get(kk)))
    img_ok = [(i["rgb"], i["sha256"]) for i in cm["images"]] == [(e["rgb"], e["sha256_rgb_png"]) for e in entries]
    out(f"compute manifest.json: {len(cm['entries'])} entries, seeds_complete {cm['seeds_complete']}, "
        f"required flag {cm['required_seeds_200_209_complete']}; field differences vs verifier: {len(diffs)} "
        f"{diffs[:5]}; images list identical: {img_ok}; left out {[x['seed'] for x in cm['seeds_left_out']]}")

    # render log: clock and stop rule
    ev = [json.loads(x) for x in (CONFIRM / "render/render_events.jsonl").read_text().splitlines() if x.strip()]
    ends = [e for e in ev if e["event"] == "command_end"]
    starts = [e for e in ev if e["event"] == "command_start"]
    total = 0.0
    viol = []
    for s_, e_ in zip(starts, ends):
        assert s_["label"] == e_["label"]
        if total >= 3 * 3600:
            viol.append(s_["label"])
        total += e_["wall_s"]
        assert e_["returncode"] == 0
    cmd_seeds = sorted({e["seed"] for e in starts})
    only_ok = all(("--only" in s_["cmd"]) and set(s_["cmd"][s_["cmd"].index("--only") + 1:]) <= set(ids)
                  and s_["cmd"][s_["cmd"].index("--variant") + 1] == "real" for s_ in starts)
    stop = [e for e in ev if e["event"] == "stop"]
    out(f"render commands {len(starts)} ({[s_['label'] for s_ in starts]}); seeds commanded {cmd_seeds}; sum of "
        f"wall_s {total:.1f} s; commands started at or after 3 h: {viol}; every command --variant real with --only "
        f"ids among the 14: {only_ok}; stop event {stop[-1] if stop else None}")

    t_frozen = os.stat(CONFIRM / "FROZEN.md").st_mtime
    newer = []
    for top, dirs, files in os.walk(WT / "data"):
        for nm in dirs + files:
            p = Path(top) / nm
            try:
                st = os.lstat(p)
            except OSError:
                continue
            if max(st.st_mtime, st.st_ctime) >= t_frozen - 3600:
                newer.append(str(p))
    out(f"worktree data/: entries with mtime or ctime after (FROZEN.md time - 1 h): {len(newer)} {newer[:5]}")
    stray = []
    for top, dirs, files in os.walk(WT):
        if "/.git" in top or "/.venv" in top or "/.tools" in top or "/assets" in top:
            dirs[:] = []
            continue
        for d in dirs:
            if re.fullmatch(r"seed2[01]\d", d) and not top.startswith(str(SEEDS)):
                stray.append(str(Path(top) / d))
    out(f"seed200-219 folders outside phase2/seeds/realistic in the worktree: {stray}")
    ph = sorted(int(m.group(1)) for d in os.listdir(RESEARCH / "phase2/seeds/placeholder")
                if (m := re.fullmatch(r"seed(\d+)", d)))
    out(f"phase2/seeds/placeholder seeds (another step): {ph[0]}-{ph[-1]} ({len(ph)}); any in 3-99: "
        f"{[k for k in ph if 3 <= k <= 99]}; any in 200-219: {[k for k in ph if 200 <= k <= 219]}")

    imgs = [{"seed": e["seed"], "row": e["row"], "scene_id": e["scene_id"], "rgb": e["rgb"],
             "sha256": e["sha256_rgb_png"]} for e in entries if e["seed"] in complete]
    (HERE / "v_images.json").write_text(json.dumps({"written": time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                                                    "images": imgs}, indent=1))
    (HERE / "v_manifest_check.txt").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
