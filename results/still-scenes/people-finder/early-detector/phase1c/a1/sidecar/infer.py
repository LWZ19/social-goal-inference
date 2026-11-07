"""I09 A1 torch sidecar (FROZEN.md section 2; DEVIATIONS.md D1, D2, C1, C2, C15). Never imports sgi.

Run with:
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -I infer.py MODE ...

  fit  MANIFEST_FIT OUT_DIR                     both detectors on every fit image: person boxes >= 0.05, ViTPose on
                                                each detector's own boxes >= 0.10 (one batch per image and detector);
                                                detector-only latency per image after 5 warm-ups (C1).
  eval MANIFEST_EVAL OUT_DIR DETECTOR           the chosen detector only (D2), same outputs.
  time MANIFEST_EVAL MANIFEST_FIT OUT_DIR DETECTOR TAU
                                                the deployed path (section 9): decoded image -> detector at TAU ->
                                                ViTPose on boxes >= TAU -> keypoints on the CPU, synchronised; 5
                                                warm-ups on the first 5 F-real images, then each E-real image once.

Image paths come only from manifests a project-venv script wrote through research_harness (and fitdata.py); each is
path-guarded and its sha256 checked. Writes JSONL (one line per image and detector) and a meta JSON.
"""
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import threading
import time

FORBIDDEN_SUBSTR = ("keep/heldout", "s18", "s20_held", "eval_16", "eval_17", "eval_18", "eval_19", "group_photo",
                    "lift_wait")
SNAP = {
    "rtdetr_v2_r50vd": "~/.cache/huggingface/hub/models--PekingU--rtdetr_v2_r50vd/snapshots/282494075698cab9faa1096ae26856890030c817",
    "dfine_large": "~/.cache/huggingface/hub/models--ustc-community--dfine-large-coco/snapshots/aa8f8211545e4cff5adf52ea512f9ac74a9824ec",
    "vitpose_base_simple": "~/.cache/huggingface/hub/models--usyd-community--vitpose-base-simple/snapshots/a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45",
}
DETECTORS = ("rtdetr_v2_r50vd", "dfine_large")
BOX_MIN = 0.05
POSE_MIN = 0.10
N_WARM = 5
GPU_BUSY_MIB = 500
POLL_S = 180


def log(*a):
    print(time.strftime('%Y-%m-%dT%H:%M:%S%z'), *a, flush=True)


def guard(path: str) -> str:
    """FROZEN.md section 2: refuse held-out or reserved paths. A seed<k> component must be seed100-109 directly under
    phase1b/fresh_seeds, or seed0-2 directly under keep/dev_new (the whitelisted E-ph location)."""
    low = path.lower()
    for s in FORBIDDEN_SUBSTR:
        assert s not in low, f"refused: {s} in {path}"
    parts = path.split("/")
    for i, part in enumerate(parts):
        m = re.fullmatch(r"seed(\d+)", part.lower())
        if m:
            k = int(m.group(1))
            parent = "/".join(parts[:i])
            ok = (100 <= k <= 109 and parent.endswith("phase1b/fresh_seeds")) or \
                 (k in (0, 1, 2) and parent.endswith("keep/dev_new"))
            assert ok, f"refused: seed folder {part} in {path}"
    return path


def gpu_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout.split()
    return int(out[0])


def gpu_apps():
    return subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


def wait_for_gpu():
    while True:
        used = gpu_used_mib()
        apps = gpu_apps()
        log(f"nvidia-smi before start: memory.used {used} MiB; compute apps: {apps or 'none'}")
        if used <= GPU_BUSY_MIB:
            return used
        log(f"GPU busy (> {GPU_BUSY_MIB} MiB held by another process); waiting {POLL_S} s")
        time.sleep(POLL_S)


class MemPoll:
    """Polls nvidia-smi memory.used every 0.5 s; keeps the maximum."""

    def __init__(self):
        self.max = 0
        self._stop = False
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop:
            try:
                self.max = max(self.max, gpu_used_mib())
            except Exception:
                pass
            time.sleep(0.5)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop = True
        self._t.join(timeout=2)


def load_images(man):
    """[(entry, PIL image, read+decode seconds)] with path guard and sha256 check."""
    from PIL import Image
    out = []
    for e in man["images"]:
        p = guard(e["rgb"])
        t0 = time.perf_counter()
        data = open(p, "rb").read()
        assert hashlib.sha256(data).hexdigest() == e["sha256"], f"sha256 mismatch: {p}"
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img.load()
        out.append((e, img, time.perf_counter() - t0))
    return out


def setup():
    import torch
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def env_info(torch):
    import transformers
    drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True,
                         text=True).stdout.strip()
    return {"gpu": torch.cuda.get_device_name(0), "driver": drv, "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "transformers": transformers.__version__,
            "cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "dtype": "float32", "python": sys.version.split()[0]}


def load_detector(torch, name, dev):
    from transformers import DFineForObjectDetection, RTDetrImageProcessor, RTDetrV2ForObjectDetection
    cls = RTDetrV2ForObjectDetection if name == "rtdetr_v2_r50vd" else DFineForObjectDetection
    proc = RTDetrImageProcessor.from_pretrained(SNAP[name])
    model = cls.from_pretrained(SNAP[name]).to(dev).eval().float()
    person = [int(k) for k, v in model.config.id2label.items() if v == "person"]
    assert person == [0], person
    return proc, model


def load_pose(torch, dev):
    from transformers import VitPoseForPoseEstimation, VitPoseImageProcessor
    proc = VitPoseImageProcessor.from_pretrained(SNAP["vitpose_base_simple"])
    model = VitPoseForPoseEstimation.from_pretrained(SNAP["vitpose_base_simple"]).to(dev).eval().float()
    return proc, model


def detect(torch, proc, model, img, dev, threshold):
    """Person boxes [x0, y0, x1, y1, score] in descending score, and (total s, forward s); synchronised."""
    W, H = img.size
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        inputs = proc(images=img, return_tensors="pt").to(dev)
        torch.cuda.synchronize()
        tf0 = time.perf_counter()
        res = model(**inputs)
        torch.cuda.synchronize()
        tf1 = time.perf_counter()
        post = proc.post_process_object_detection(res, threshold=threshold, target_sizes=[(H, W)])[0]
        labels = post["labels"].cpu().numpy()
        boxes = post["boxes"].cpu().numpy()
        scores = post["scores"].cpu().numpy()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    keep = [i for i in range(len(labels)) if int(labels[i]) == 0]
    order = sorted(keep, key=lambda i: -float(scores[i]))
    out = [[float(boxes[i][0]), float(boxes[i][1]), float(boxes[i][2]), float(boxes[i][3]), float(scores[i])]
           for i in order]
    return out, (t1 - t0, tf1 - tf0)


def pose(torch, proc, model, img, boxes, dev):
    """ViTPose on the given xyxy boxes as one batch: [[[u, v, s] x 17] per box] and seconds; synchronised."""
    if not boxes:
        return [], 0.0
    coco = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        pin = proc(images=img, boxes=[coco], return_tensors="pt").to(dev)
        pres = model(**pin)
        pp = proc.post_process_pose_estimation(pres, boxes=[coco])[0]
        kps = [(r["keypoints"].cpu().numpy(), r["scores"].cpu().numpy()) for r in pp]
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    out = [[[float(k[j, 0]), float(k[j, 1]), float(s[j])] for j in range(k.shape[0])] for k, s in kps]
    assert len(out) == len(boxes) and all(len(o) == 17 for o in out)
    return out, t1 - t0


def run_pass(mode, man_path, out_dir, detectors):
    torch = setup()
    man = json.load(open(man_path))
    os.makedirs(out_dir, exist_ok=True)
    meta = {"mode": mode, "manifest": man_path, "manifest_sha256": hashlib.sha256(open(man_path, "rb").read()).hexdigest(),
            "detectors": list(detectors), "box_min": BOX_MIN, "pose_min": POSE_MIN, "n_warm": N_WARM,
            "start": time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    meta["gpu_used_before_MiB"] = wait_for_gpu()
    dev = torch.device("cuda")
    meta.update(env_info(torch))
    imgs = load_images(man)
    log(f"{mode}: {len(imgs)} images guarded and sha256-checked")
    torch.cuda.reset_peak_memory_stats()
    with MemPoll() as mp:
        dets = {d: load_detector(torch, d, dev) for d in detectors}
        pproc, pmodel = load_pose(torch, dev)
        warm = [x for x in imgs if x[0]["domain"] == "F-real"][:N_WARM] if mode == "fit" else []   # eval: no timing use
        for d in detectors:
            for e, img, _ in warm:
                b, _ = detect(torch, *dets[d], img, dev, BOX_MIN)
                pose(torch, pproc, pmodel, img, [x[:4] for x in b if x[4] >= POSE_MIN], dev)
        meta["warmup_images"] = [e["scene_id"] for e, _, _ in warm]
        fhs = {d: open(os.path.join(out_dir, f"{d}.jsonl"), "w") for d in detectors}
        for k, (e, img, t_read) in enumerate(imgs):
            for d in detectors:
                boxes, (t_det, t_fwd) = detect(torch, *dets[d], img, dev, BOX_MIN)
                pb = [b for b in boxes if b[4] >= POSE_MIN]
                kps, t_pose = pose(torch, pproc, pmodel, img, [b[:4] for b in pb], dev)
                line = {"scene_id": e["scene_id"], "domain": e["domain"], "set": e["set"], "rgb": e["rgb"],
                        "sha256": e["sha256"], "W": img.size[0], "H": img.size[1], "detector": d, "boxes": boxes,
                        "pose": [{"box_index": i, "kp": kps[i]} for i in range(len(pb))],
                        "t_det_s": t_det, "t_fwd_s": t_fwd, "t_pose_s": t_pose, "t_read_decode_s": t_read}
                fhs[d].write(json.dumps(line) + "\n")
            if k % 50 == 0:
                log(f"{mode}: image {k + 1}/{len(imgs)} {e['scene_id']}")
        for fh in fhs.values():
            fh.close()
        meta["peak_mem_allocated_MiB"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        meta["peak_mem_reserved_MiB"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
    meta["peak_nvidia_smi_MiB"] = mp.max
    meta["end"] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    log(json.dumps(meta))


def run_time(man_eval, man_fit, out_dir, detector, tau):
    torch = setup()
    me, mf = json.load(open(man_eval)), json.load(open(man_fit))
    os.makedirs(out_dir, exist_ok=True)
    meta = {"mode": "time", "detector": detector, "tau": tau, "manifest_eval": man_eval, "manifest_fit": man_fit,
            "manifest_eval_sha256": hashlib.sha256(open(man_eval, "rb").read()).hexdigest(),
            "start": time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    meta["gpu_used_before_MiB"] = wait_for_gpu()
    dev = torch.device("cuda")
    meta.update(env_info(torch))
    warm = load_images({"images": [e for e in mf["images"] if e["domain"] == "F-real"][:N_WARM]})
    real = load_images({"images": [e for e in me["images"] if e["domain"] == "E-real"]})
    assert len(real) == 42, len(real)
    torch.cuda.reset_peak_memory_stats()
    with MemPoll() as mp:
        dproc, dmodel = load_detector(torch, detector, dev)
        pproc, pmodel = load_pose(torch, dev)

        def deployed(img):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            boxes, _ = detect(torch, dproc, dmodel, img, dev, tau)
            kps, _ = pose(torch, pproc, pmodel, img, [b[:4] for b in boxes], dev)
            torch.cuda.synchronize()
            return boxes, kps, time.perf_counter() - t0

        for e, img, _ in warm:
            deployed(img)
        meta["warmup_images"] = [e["scene_id"] for e, _, _ in warm]
        with open(os.path.join(out_dir, "timing.jsonl"), "w") as fh:
            for e, img, t_read in real:
                boxes, kps, t = deployed(img)
                fh.write(json.dumps({"scene_id": e["scene_id"], "rgb": e["rgb"], "sha256": e["sha256"], "t_s": t,
                                     "t_read_decode_s": t_read, "boxes": boxes, "kp": kps}) + "\n")
        meta["peak_mem_allocated_MiB"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        meta["peak_mem_reserved_MiB"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
    meta["peak_nvidia_smi_MiB"] = mp.max
    meta["end"] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    log(json.dumps(meta))


def main():
    mode = sys.argv[1]
    if mode == "fit":
        run_pass("fit", sys.argv[2], sys.argv[3], DETECTORS)
    elif mode == "eval":
        assert sys.argv[4] in DETECTORS
        run_pass("eval", sys.argv[2], sys.argv[3], (sys.argv[4],))
    elif mode == "time":
        assert sys.argv[5] in DETECTORS
        run_time(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], float(sys.argv[6]))
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
