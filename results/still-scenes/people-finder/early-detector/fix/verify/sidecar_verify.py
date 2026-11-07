"""Verifier task (2): re-run A1's frozen detector + pose pass on ONE fit seed (seed 116, 14 images) and compare the
raw outputs with phase3/i09_fix/raw_fit/rtdetr_v2_r50vd.jsonl bit for bit (boxes, pose keypoints, image fields).

Run in the torch sidecar, read-only, never importing sgi:
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 nice -n 10 python -I -B \
      sidecar_verify.py [--compare-only]
Imports A1's phase1c/i09/a1/sidecar/infer.py by path (sha256 asserted) and calls its run_pass("eval", ...) unchanged;
only infer.guard (my own path guard: seed116 rgb.png files of my manifest) and infer.wait_for_gpu (>= 3,000 MiB GPU
free and >= 3,000 MiB RAM available, else stop) are replaced. Weights: infer.SNAP (pinned snapshots, asserted).
pip freeze before and after must match. Writes only into verify/ (manifest_seed116.json, raw_seed116/, logs).
"""
import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

RESEARCH = "results/2026-10-17-beyond-vlm"
FIX = RESEARCH + "/phase3/i09_fix"
VER = FIX + "/verify"
SEEDS = RESEARCH + "/phase3/seeds/realistic_train"
INFER = RESEARCH + "/phase1c/i09/a1/sidecar/infer.py"
INFER_SHA = "bee1322fd692bf0e85e820871413d486fda04d5770f72b5d3d75e18266709484"
SEED = 116
ROWS = ("real_eval_00_no_one_place", "real_eval_01_no_one_place", "real_eval_02_no_one_place",
        "real_eval_03_no_one_deep_place", "real_eval_04_no_one_object", "real_eval_05_no_one_person",
        "real_eval_06_waiting_line_1", "real_eval_07_waiting_line_2", "real_eval_08_waiting_line_4",
        "real_eval_09_waiting_line_6", "real_eval_10_conversation_group_2", "real_eval_11_conversation_group_3",
        "real_eval_12_conversation_group_4", "real_eval_13_waiting_line_3")
OUTDIR = VER + f"/raw_seed{SEED}"
MAN = VER + f"/manifest_seed{SEED}.json"
DET = "rtdetr_v2_r50vd"
BAD = re.compile(r"phase2/seeds|held_?out|(^|[_/])s18|s20_held|eval_1[6-9]|group_photo|lift_wait", re.I)


def log(*a):
    print(time.strftime('%Y-%m-%dT%H:%M:%S%z'), *a, flush=True)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def my_guard(p):
    for q in (p, os.path.realpath(p)):
        assert not BAD.search(q), q
        m = re.fullmatch(re.escape(SEEDS) + r"/seed(\d+)/eval/([a-z0-9_]+)/rgb\.png", q)
        assert m and int(m.group(1)) == SEED and m.group(2) in ROWS, q
    return p


def load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gpu():
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=True).stdout.split(",")
    return int(o[0]), int(o[1])


def ram():
    o = subprocess.run(["free", "-m"], capture_output=True, text=True, check=True).stdout.splitlines()
    return int(o[1].split()[1 + o[0].split().index("available")])


def freeze():
    return subprocess.run([sys.executable, "-I", "-m", "pip", "freeze", "--disable-pip-version-check",
                           "--no-cache-dir"], capture_output=True, text=True, check=True).stdout


def build_manifest():
    ref = {e["rgb"]: e for e in json.load(open(FIX + "/manifest_fit.json"))["images"]}
    imgs = []
    for row in ROWS:
        p = my_guard(f"{SEEDS}/seed{SEED}/eval/{row}/rgb.png")
        h = sha(p)
        e = ref[p]
        assert e["sha256"] == h and e["seed"] == SEED and e["row_id"] == row, p
        imgs.append({"scene_id": e["scene_id"], "domain": e["domain"], "set": e["set"], "rgb": p, "sha256": h})
    json.dump({"purpose": f"verifier: one-seed sidecar re-run (seed {SEED})", "images": imgs}, open(MAN, "w"), indent=1)
    return imgs


def compare():
    old = {x["rgb"]: x for x in map(json.loads, open(FIX + f"/raw_fit/{DET}.jsonl").read().splitlines())}
    new = [json.loads(x) for x in open(OUTDIR + f"/{DET}.jsonl").read().splitlines()]
    assert len(new) == 14
    keys = ("scene_id", "domain", "set", "rgb", "sha256", "W", "H", "detector", "boxes", "pose")
    diffs, nb, nk = [], 0, 0
    for x in new:
        o = old[x["rgb"]]
        for k in keys:
            if x[k] != o[k]:
                diffs.append((x["scene_id"], k))
        # also compare the exact float bits (JSON round trip of float64 is exact)
        for b1, b2 in zip(x["boxes"], o["boxes"]):
            nb += 1
            if [float.hex(v) for v in b1] != [float.hex(v) for v in b2]:
                diffs.append((x["scene_id"], "box bits"))
        for p1, p2 in zip(x["pose"], o["pose"]):
            for k1, k2 in zip(p1["kp"], p2["kp"]):
                nk += 1
                if [float.hex(v) for v in k1] != [float.hex(v) for v in k2]:
                    diffs.append((x["scene_id"], "kp bits"))
        if len(x["boxes"]) != len(o["boxes"]) or len(x["pose"]) != len(o["pose"]):
            diffs.append((x["scene_id"], "counts", len(x["boxes"]), len(o["boxes"]), len(x["pose"]), len(o["pose"])))
    old_meta = json.load(open(FIX + "/raw_fit/meta.json"))
    new_meta = json.load(open(OUTDIR + "/meta.json"))
    env_keys = ("detectors", "box_min", "pose_min", "gpu", "driver", "torch", "cuda", "cudnn", "transformers",
                "cudnn_deterministic", "cudnn_benchmark", "tf32_matmul", "tf32_cudnn", "dtype", "python")
    env_diff = {k: (new_meta.get(k), old_meta.get(k)) for k in env_keys if new_meta.get(k) != old_meta.get(k)}
    log(f"compared 14 images, {nb} boxes, {nk} keypoints; differences: {len(diffs)} {diffs[:20]}; "
        f"environment differences: {env_diff}")
    return diffs, env_diff


def main():
    if "--compare-only" in sys.argv:
        compare()
        return
    assert sha(INFER) == INFER_SHA, "infer.py sha256"
    assert not os.path.exists(OUTDIR + f"/{DET}.jsonl"), "re-run exists already"
    imgs = build_manifest()
    infer = load_by_path("infer_a1_verify", INFER)
    assert infer.BOX_MIN == 0.05 and infer.POSE_MIN == 0.10
    assert infer.SNAP[DET].endswith("282494075698cab9faa1096ae26856890030c817")
    assert infer.SNAP["vitpose_base_simple"].endswith("a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45")
    allowed = {e["rgb"] for e in imgs}

    def guard(p):
        my_guard(p)
        assert p in allowed, p
        return p

    def wait_for_gpu():
        used, free = gpu()
        avail = ram()
        apps = infer.gpu_apps()
        log(f"before start: GPU used {used} MiB free {free} MiB; RAM available {avail} MiB; compute apps: "
            f"{apps or 'none'}")
        assert free >= 3000 and avail >= 3000 and not apps, "resources not free: stop"
        return used

    infer.guard = guard
    infer.wait_for_gpu = wait_for_gpu
    f0 = freeze()
    open(VER + "/pip_freeze_before.txt", "w").write(f0)
    log(f"pip freeze before: {len(f0.splitlines())} lines, sha256 {hashlib.sha256(f0.encode()).hexdigest()[:8]}")
    infer.run_pass("eval", MAN, OUTDIR, (DET,))
    f1 = freeze()
    open(VER + "/pip_freeze_after.txt", "w").write(f1)
    assert f1 == f0, "pip freeze changed"
    log(f"pip freeze unchanged ({len(f1.splitlines())} lines)")
    compare()


if __name__ == "__main__":
    main()
