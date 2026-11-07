"""I09 confirmation, step 5 (FROZEN.md sections 3 and 9; DEVIATIONS_1.md C2, C3): the one detector + pose pass.

Run in the torch sidecar, never importing sgi:
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -I infer_confirm.py

It imports A1's phase1c/i09/a1/sidecar/infer.py by path (sha256 asserted; bytecode writing off) and calls its
run_pass("eval", manifest.json, raw/, ("rtdetr_v2_r50vd",)) unchanged: RT-DETRv2 boxes >= 0.05, ViTPose on boxes
>= 0.10 as one batch per image, fp32, eval mode, seed 0, cudnn deterministic, benchmark off, TF32 off, no warm-up.
Only two functions are replaced: infer.guard (the test-set path guard plus manifest membership) and
infer.wait_for_gpu (the shared-machine rule: >= 3,000 MiB GPU free and >= 3,000 MiB RAM available, else wait
300 s, logged). pip freeze is recorded before and after and must not change. Writes raw/, pip_freeze_*.txt and
RAW.sha256.
"""
import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

CONFIRM = "results/2026-10-17-beyond-vlm/phase2/i09/confirm"
RESEARCH = "results/2026-10-17-beyond-vlm"
INFER = RESEARCH + "/phase1c/i09/a1/sidecar/infer.py"
INFER_SHA = "bee1322fd692bf0e85e820871413d486fda04d5770f72b5d3d75e18266709484"
FROZEN_SHA = "f55341c03cd5e1d38c91b7e0142ef10f0870ccb2b43c5bf4edfac07d1996e751"
MANIFEST = CONFIRM + "/manifest.json"
RAW = CONFIRM + "/raw"
DETECTOR = "rtdetr_v2_r50vd"
GPU_FREE_MIN = 3000
RAM_AVAIL_MIN = 3000
POLL_S = 300


def log(*a):
    print(time.strftime('%Y-%m-%dT%H:%M:%S%z'), *a, flush=True)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def verify_sha_file(sha_file):
    folder = os.path.dirname(sha_file)
    out = {}
    for line in open(sha_file).read().splitlines():
        if line.strip():
            want, name = line.split()
            got = sha(os.path.join(folder, name))
            assert got == want, f"{name}: {got} != {want} ({sha_file})"
            out[name] = want
    return out


def load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pip_freeze():
    out = subprocess.run([sys.executable, "-I", "-m", "pip", "freeze", "--disable-pip-version-check", "--no-cache-dir"],
                         capture_output=True, text=True, check=True)
    return out.stdout


def ram_available_mib():
    out = subprocess.run(["free", "-m"], capture_output=True, text=True, check=True).stdout.splitlines()
    head = out[0].split()
    mem = out[1].split()
    return int(mem[1 + head.index("available")])


def gpu_query():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout.split(",")
    return int(out[0]), int(out[1])


def main():
    hashes = verify_sha_file(CONFIRM + "/FROZEN.sha256")
    assert hashes["FROZEN.md"] == FROZEN_SHA
    for f in sorted(os.listdir(CONFIRM)):
        if f.startswith("DEVIATIONS_") and f.endswith(".md"):
            hashes |= verify_sha_file(CONFIRM + "/" + f[:-3] + ".sha256")
    hashes |= verify_sha_file(CONFIRM + "/MANIFEST.sha256")
    assert sha(INFER) == INFER_SHA, "infer.py sha256 mismatch (C1)"
    hashes["infer.py"] = INFER_SHA
    man = json.load(open(MANIFEST))
    assert man["required_seeds_200_209_complete"] is True
    allowed = {e["rgb"]: e["sha256"] for e in man["images"]}
    assert len(allowed) == len(man["images"]) == man["counts"]["layouts"]
    assert os.path.exists(RAW + "/" + DETECTOR + ".jsonl") is False, "raw pass exists already (FROZEN section 9)"

    pg = load_by_path("pathguard_confirm", CONFIRM + "/pathguard.py")
    infer = load_by_path("infer_a1", INFER)
    assert infer.BOX_MIN == 0.05 and infer.POSE_MIN == 0.10
    assert infer.SNAP[DETECTOR].endswith("282494075698cab9faa1096ae26856890030c817")
    assert infer.SNAP["vitpose_base_simple"].endswith("a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45")
    for p in allowed:                                       # every path guarded before the GPU is touched
        pg.check(p)

    def guard(path):
        pg.check(path)
        assert path in allowed, f"refused: not in manifest.json: {path}"
        return path

    def wait_for_gpu():
        while True:
            used, free = gpu_query()
            avail = ram_available_mib()
            apps = infer.gpu_apps()
            log(f"before start: nvidia-smi memory.used {used} MiB, memory.free {free} MiB; free -m available "
                f"{avail} MiB; compute apps: {apps or 'none'}")
            if free >= GPU_FREE_MIN and avail >= RAM_AVAIL_MIN:
                return used
            log(f"waiting {POLL_S} s (need >= {GPU_FREE_MIN} MiB GPU free and >= {RAM_AVAIL_MIN} MiB RAM available)")
            time.sleep(POLL_S)

    infer.guard = guard
    infer.wait_for_gpu = wait_for_gpu
    if "--preflight" in sys.argv:                          # no model, no image read, nothing written
        for p in allowed:
            guard(p)
        n = len(pip_freeze().splitlines())
        used, free = gpu_query()
        log(f"preflight ok: hashes {json.dumps(hashes)}; {len(allowed)} paths guarded; pip freeze {n} lines; "
            f"GPU used {used} MiB free {free} MiB; RAM available {ram_available_mib()} MiB")
        return
    freeze_before = pip_freeze()
    open(CONFIRM + "/pip_freeze_before.txt", "w").write(freeze_before)
    log(f"hashes verified: {json.dumps(hashes)}")
    log(f"pip freeze before: {len(freeze_before.splitlines())} lines; {len(allowed)} images; detector {DETECTOR}")
    infer.run_pass("eval", MANIFEST, RAW, (DETECTOR,))
    freeze_after = pip_freeze()
    open(CONFIRM + "/pip_freeze_after.txt", "w").write(freeze_after)
    assert freeze_after == freeze_before, "pip freeze changed during the pass"
    lines = [json.loads(x) for x in open(RAW + "/" + DETECTOR + ".jsonl").read().splitlines()]
    assert len(lines) == len(allowed), (len(lines), len(allowed))
    assert [x["rgb"] for x in lines] == [e["rgb"] for e in man["images"]]
    assert all(allowed[x["rgb"]] == x["sha256"] for x in lines)
    meta = json.load(open(RAW + "/meta.json"))
    assert meta["manifest_sha256"] == sha(MANIFEST) and meta["detectors"] == [DETECTOR]
    with open(CONFIRM + "/RAW.sha256", "w") as fh:
        for name in ("raw/" + DETECTOR + ".jsonl", "raw/meta.json", "pip_freeze_before.txt", "pip_freeze_after.txt"):
            fh.write(f"{sha(CONFIRM + '/' + name)}  {name}\n")
    log(f"done: {len(lines)} raw lines; pip freeze unchanged ({len(freeze_after.splitlines())} lines); RAW.sha256 written")


if __name__ == "__main__":
    main()
