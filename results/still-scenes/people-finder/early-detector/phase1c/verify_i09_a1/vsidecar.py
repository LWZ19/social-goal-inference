"""I09 A1 verifier: torch sidecar (GPU inference only; never imports sgi; run with -I and HF offline).

  raw  MANIFEST OUT_DIR    RT-DETRv2 on every manifest image (person boxes >= 0.05, descending score) and ViTPose on
                           its boxes >= 0.10 as one batch per image; D-FINE the same on the fit (worked-example) images.
  time MANIFEST OUT_DIR    the deployed path at tau 0.25 (decoded image -> detector -> ViTPose on boxes >= tau ->
                           keypoints on the CPU, synchronised): 5 warm-ups on the first 5 F-real images, then every
                           E-real image once (pass A), and the whole loop again (pass B) to show run-to-run spread.

Paths come only from manifest_verify.json, written by manifest.py in the project venv through research_harness.
Each path is guarded (no keep/heldout, s18, s20_held, eval_16..19, group_photo, lift_wait; a seed<k> component must be
seed100..109 under phase1b/fresh_seeds or seed0..2 directly under keep/dev_new, the harness-whitelisted E-ph scenes),
and its sha256 is checked.
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

SNAP = {
    "rtdetr_v2_r50vd": "~/.cache/huggingface/hub/models--PekingU--rtdetr_v2_r50vd/snapshots/282494075698cab9faa1096ae26856890030c817",
    "dfine_large": "~/.cache/huggingface/hub/models--ustc-community--dfine-large-coco/snapshots/aa8f8211545e4cff5adf52ea512f9ac74a9824ec",
    "vitpose": "~/.cache/huggingface/hub/models--usyd-community--vitpose-base-simple/snapshots/a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45",
}
BAD = ("keep/heldout", "s18", "s20_held", "eval_16", "eval_17", "eval_18", "eval_19", "group_photo", "lift_wait")
TAU = 0.25


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def log(fh, msg):
    line = f"{now()} {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def check_path(p):
    low = p.lower()
    for b in BAD:
        assert b not in low, (b, p)
    parts = p.split("/")
    for i, part in enumerate(parts):
        m = re.fullmatch(r"seed(\d+)", part.lower())
        if m:
            k, parent = int(m.group(1)), "/".join(parts[:i])
            assert (100 <= k <= 109 and parent.endswith("phase1b/fresh_seeds")) or \
                   (k <= 2 and parent.endswith("keep/dev_new")), p
    return p


def smi(query):
    return subprocess.run(["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"], capture_output=True,
                          text=True).stdout.strip()


def gpu_wait(fh):
    while True:
        used = int(smi("gpu=memory.used").split()[0])
        apps = smi("compute-apps=pid,process_name,used_memory")
        log(fh, f"nvidia-smi: used {used} MiB; apps: {apps or 'none'}")
        if used <= 500:
            return used
        log(fh, "GPU busy; waiting 180 s")
        time.sleep(180)


class Poll(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.stop = 0, False

    def run(self):
        while not self.stop:
            try:
                self.peak = max(self.peak, int(smi("gpu=memory.used").split()[0]))
            except Exception:
                pass
            time.sleep(0.5)


def images(man, pred):
    from PIL import Image
    out = []
    for e in man["images"]:
        if not pred(e):
            continue
        data = open(check_path(e["rgb"]), "rb").read()
        assert hashlib.sha256(data).hexdigest() == e["sha256"], e["rgb"]
        im = Image.open(io.BytesIO(data)).convert("RGB")
        im.load()
        out.append((e, im))
    return out


def torch_setup():
    import torch
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def env(torch):
    import transformers
    return {"gpu": torch.cuda.get_device_name(0), "driver": smi("gpu=driver_version"), "torch": torch.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "transformers": transformers.__version__, "python": sys.version.split()[0], "dtype": "float32",
            "cudnn_deterministic": torch.backends.cudnn.deterministic, "tf32": torch.backends.cuda.matmul.allow_tf32}


class Det:
    def __init__(self, torch, name, dev):
        from transformers import DFineForObjectDetection, RTDetrImageProcessor, RTDetrV2ForObjectDetection
        self.torch, self.dev = torch, dev
        self.proc = RTDetrImageProcessor.from_pretrained(SNAP[name])
        klass = RTDetrV2ForObjectDetection if name == "rtdetr_v2_r50vd" else DFineForObjectDetection
        self.model = klass.from_pretrained(SNAP[name]).to(dev).float().eval()
        assert self.model.config.id2label[0] == "person"

    def __call__(self, im, thr):
        torch = self.torch
        with torch.no_grad():
            x = self.proc(images=im, return_tensors="pt").to(self.dev)
            y = self.model(**x)
            r = self.proc.post_process_object_detection(y, threshold=thr, target_sizes=[(im.size[1], im.size[0])])[0]
            lab, box, sc = r["labels"].cpu().numpy(), r["boxes"].cpu().numpy(), r["scores"].cpu().numpy()
        rows = [[*map(float, box[i]), float(sc[i])] for i in range(len(lab)) if int(lab[i]) == 0]
        rows.sort(key=lambda b: -b[4])
        return rows


class Pose:
    def __init__(self, torch, dev):
        from transformers import VitPoseForPoseEstimation, VitPoseImageProcessor
        self.torch, self.dev = torch, dev
        self.proc = VitPoseImageProcessor.from_pretrained(SNAP["vitpose"])
        self.model = VitPoseForPoseEstimation.from_pretrained(SNAP["vitpose"]).to(dev).float().eval()

    def __call__(self, im, boxes):
        if not boxes:
            return []
        torch = self.torch
        coco = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes]
        with torch.no_grad():
            x = self.proc(images=im, boxes=[coco], return_tensors="pt").to(self.dev)
            y = self.model(**x)
            res = self.proc.post_process_pose_estimation(y, boxes=[coco])[0]
            out = []
            for r in res:
                k, s = r["keypoints"].cpu().numpy(), r["scores"].cpu().numpy()
                out.append([[float(k[j, 0]), float(k[j, 1]), float(s[j])] for j in range(17)])
        return out


def mode_raw(man_path, out):
    os.makedirs(out, exist_ok=True)
    fh = open(os.path.join(out, "raw.log"), "a")
    torch = torch_setup()
    man = json.load(open(man_path))
    meta = {"mode": "raw", "start": now(), "manifest_sha256": hashlib.sha256(open(man_path, "rb").read()).hexdigest()}
    meta["gpu_used_before"] = gpu_wait(fh)
    meta.update(env(torch))
    imgs = images(man, lambda e: True)
    log(fh, f"{len(imgs)} images guarded and hash-checked")
    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    poll = Poll()
    poll.start()
    pose = Pose(torch, dev)
    for name in ("rtdetr_v2_r50vd", "dfine_large"):
        det = Det(torch, name, dev)
        with open(os.path.join(out, f"{name}.jsonl"), "w") as w:
            for e, im in imgs:
                if name == "dfine_large" and e["part"] != "fit":
                    continue
                boxes = det(im, 0.05)
                kp = pose(im, [b[:4] for b in boxes if b[4] >= 0.10])
                w.write(json.dumps({"scene_id": e["scene_id"], "domain": e["domain"], "set": e["set"],
                                    "rgb": e["rgb"], "boxes": boxes, "kp": kp}) + "\n")
        log(fh, f"{name} done")
        del det
    meta["peak_alloc_MiB"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    poll.stop = True
    poll.join(2)
    meta["peak_smi_MiB"] = poll.peak
    meta["end"] = now()
    json.dump(meta, open(os.path.join(out, "raw_meta.json"), "w"), indent=1)
    log(fh, json.dumps(meta))


def mode_time(man_path, out):
    os.makedirs(out, exist_ok=True)
    fh = open(os.path.join(out, "time.log"), "a")
    torch = torch_setup()
    man = json.load(open(man_path))
    meta = {"mode": "time", "start": now(), "tau": TAU}
    meta["gpu_used_before"] = gpu_wait(fh)
    meta.update(env(torch))
    warm = images(man, lambda e: e["domain"] == "F-real")[:5]
    real = images(man, lambda e: e["domain"] == "E-real")
    assert len(real) == 42
    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    poll = Poll()
    poll.start()
    det, pose = Det(torch, "rtdetr_v2_r50vd", dev), Pose(torch, dev)

    def run(im):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        b = det(im, TAU)
        k = pose(im, [x[:4] for x in b])
        torch.cuda.synchronize()
        return b, k, time.perf_counter() - t0

    for _, im in warm:
        run(im)
    meta["warm"] = [e["scene_id"] for e, _ in warm]
    with open(os.path.join(out, "timing.jsonl"), "w") as w:
        for pas in ("A", "B"):
            for e, im in real:
                b, k, t = run(im)
                w.write(json.dumps({"pass": pas, "scene_id": e["scene_id"], "set": e["set"], "t_s": t, "boxes": b,
                                    "kp": k}) + "\n")
    meta["peak_alloc_MiB"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    poll.stop = True
    poll.join(2)
    meta["peak_smi_MiB"] = poll.peak
    meta["end"] = now()
    json.dump(meta, open(os.path.join(out, "time_meta.json"), "w"), indent=1)
    log(fh, json.dumps(meta))


if __name__ == "__main__":
    {"raw": mode_raw, "time": mode_time}[sys.argv[1]](sys.argv[2], sys.argv[3])
