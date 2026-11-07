"""Verifier's own detector + pose pass (torch sidecar; never imports sgi; installs nothing).

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -I v_infer.py

Same snapshots and settings as FROZEN.md section 3: RT-DETRv2-r50vd person boxes with score >= 0.05 (descending
score), ViTPose-base-simple on every box with score >= 0.10 as one batch per image; fp32, eval mode,
torch.manual_seed(0), cudnn deterministic, benchmark off, TF32 off, no warm-up. Image paths come only from
v_images.json (written by v_manifest.py in the project venv); every path is checked by v_guard and its sha256 checked.
Writes raw/v_rtdetr_v2_r50vd.jsonl, raw/v_meta.json, pip_freeze_v_before.txt, pip_freeze_v_after.txt here.
"""
import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

V = "results/2026-10-17-beyond-vlm/phase2/i09/verify_i09_confirm"
DET_SNAP = ("~/.cache/huggingface/hub/models--PekingU--rtdetr_v2_r50vd/snapshots/"
            "282494075698cab9faa1096ae26856890030c817")
POSE_SNAP = ("~/.cache/huggingface/hub/models--usyd-community--vitpose-base-simple/snapshots/"
             "a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45")
BOX_MIN, POSE_MIN = 0.05, 0.10


def log(*a):
    print(time.strftime('%Y-%m-%dT%H:%M:%S%z'), *a, flush=True)


def freeze():
    return subprocess.run([sys.executable, "-I", "-m", "pip", "freeze", "--disable-pip-version-check",
                           "--no-cache-dir"], capture_output=True, text=True, check=True).stdout


def resources():
    q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=True).stdout.split(",")
    fm = subprocess.run(["free", "-m"], capture_output=True, text=True, check=True).stdout.splitlines()
    avail = int(fm[1].split()[1 + fm[0].split().index("available")])
    return int(q[0]), int(q[1]), avail


def main():
    spec = importlib.util.spec_from_file_location("v_guard", V + "/v_guard.py")
    vg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vg)
    man = json.load(open(V + "/v_images.json"))["images"]
    for e in man:
        vg.check(e["rgb"])
    out_path = V + "/raw/v_rtdetr_v2_r50vd.jsonl"
    assert not os.path.exists(out_path), "raw output exists"
    os.makedirs(V + "/raw", exist_ok=True)
    while True:
        used, free, avail = resources()
        log(f"resources: GPU used {used} MiB free {free} MiB; RAM available {avail} MiB")
        if free >= 3000 and avail >= 3000:
            break
        log("waiting 300 s")
        time.sleep(300)
    f_before = freeze()
    open(V + "/pip_freeze_v_before.txt", "w").write(f_before)

    import torch
    from PIL import Image
    import transformers
    from transformers import (RTDetrImageProcessor, RTDetrV2ForObjectDetection, VitPoseForPoseEstimation,
                              VitPoseImageProcessor)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda")
    dproc = RTDetrImageProcessor.from_pretrained(DET_SNAP)
    dmodel = RTDetrV2ForObjectDetection.from_pretrained(DET_SNAP).to(dev).eval().float()
    assert [int(k) for k, v in dmodel.config.id2label.items() if v == "person"] == [0]
    pproc = VitPoseImageProcessor.from_pretrained(POSE_SNAP)
    pmodel = VitPoseForPoseEstimation.from_pretrained(POSE_SNAP).to(dev).eval().float()
    torch.cuda.reset_peak_memory_stats()
    meta = {"start": time.strftime('%Y-%m-%dT%H:%M:%S%z'), "torch": torch.__version__,
            "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0),
            "det_snapshot": DET_SNAP, "pose_snapshot": POSE_SNAP, "box_min": BOX_MIN, "pose_min": POSE_MIN,
            "gpu_used_before_MiB": used}
    with open(out_path, "w") as fh:
        for k, e in enumerate(man):
            p = e["rgb"]
            vg.check(p)
            data = open(p, "rb").read()
            assert hashlib.sha256(data).hexdigest() == e["sha256"], p
            img = Image.open(io.BytesIO(data)).convert("RGB")
            W, H = img.size
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                x = dproc(images=img, return_tensors="pt").to(dev)
                r = dmodel(**x)
                post = dproc.post_process_object_detection(r, threshold=BOX_MIN, target_sizes=[(H, W)])[0]
            lab = post["labels"].cpu().numpy()
            bx = post["boxes"].cpu().numpy()
            sc = post["scores"].cpu().numpy()
            torch.cuda.synchronize()
            t_det = time.perf_counter() - t0
            idx = sorted([i for i in range(len(lab)) if int(lab[i]) == 0], key=lambda i: -float(sc[i]))
            boxes = [[float(bx[i][0]), float(bx[i][1]), float(bx[i][2]), float(bx[i][3]), float(sc[i])] for i in idx]
            pb = [b for b in boxes if b[4] >= POSE_MIN]
            kps = []
            t_pose = 0.0
            if pb:
                coco = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in pb]
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                with torch.no_grad():
                    pin = pproc(images=img, boxes=[coco], return_tensors="pt").to(dev)
                    pres = pmodel(**pin)
                    pp = pproc.post_process_pose_estimation(pres, boxes=[coco])[0]
                for res in pp:
                    kk, ss = res["keypoints"].cpu().numpy(), res["scores"].cpu().numpy()
                    kps.append([[float(kk[j, 0]), float(kk[j, 1]), float(ss[j])] for j in range(kk.shape[0])])
                torch.cuda.synchronize()
                t_pose = time.perf_counter() - t1
            assert len(kps) == len(pb) and all(len(q) == 17 for q in kps)
            fh.write(json.dumps({"seed": e["seed"], "row": e["row"], "scene_id": e["scene_id"], "rgb": p,
                                 "sha256": e["sha256"], "W": W, "H": H, "boxes": boxes,
                                 "pose": [{"box_index": i, "kp": kps[i]} for i in range(len(pb))],
                                 "t_det_s": t_det, "t_pose_s": t_pose}) + "\n")
            if k % 50 == 0:
                log(f"image {k + 1}/{len(man)}")
    meta["peak_mem_allocated_MiB"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    meta["end"] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    meta["n_images"] = len(man)
    f_after = freeze()
    open(V + "/pip_freeze_v_after.txt", "w").write(f_after)
    meta["pip_freeze_unchanged"] = f_after == f_before
    meta["pip_freeze_lines"] = len(f_after.splitlines())
    json.dump(meta, open(V + "/raw/v_meta.json", "w"), indent=1)
    log(json.dumps(meta))
    assert f_after == f_before, "pip freeze changed"


if __name__ == "__main__":
    main()
