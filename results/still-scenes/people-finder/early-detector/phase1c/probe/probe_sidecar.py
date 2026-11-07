"""I09 pre-freeze API probe, torch sidecar: load RT-DETRv2, D-FINE and ViTPose from their local snapshots ONCE and
run them on the ONE fit image named in probe_manifest.json, to confirm the API, the output format and the
left/right keypoint convention. Never imports sgi. Run with:
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -I probe_sidecar.py MANIFEST OUT
"""
import json
import re
import sys
import time

FORBIDDEN_SUBSTR = ("keep/heldout", "s18", "s20_held", "eval_16", "eval_17", "eval_18", "eval_19", "group_photo",
                    "lift_wait")
SNAP = {
    "rtdetr_v2_r50vd": "~/.cache/huggingface/hub/models--PekingU--rtdetr_v2_r50vd/snapshots/282494075698cab9faa1096ae26856890030c817",
    "dfine_large": "~/.cache/huggingface/hub/models--ustc-community--dfine-large-coco/snapshots/aa8f8211545e4cff5adf52ea512f9ac74a9824ec",
    "vitpose_base_simple": "~/.cache/huggingface/hub/models--usyd-community--vitpose-base-simple/snapshots/a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45",
}


def guard(path: str) -> str:
    low = path.lower()
    for s in FORBIDDEN_SUBSTR:
        assert s not in low, f"refused: {s} in {path}"
    parts = path.split("/")
    for i, part in enumerate(parts):
        m = re.fullmatch(r"seed(\d+)", part)
        if m:
            k = int(m.group(1))
            parent = "/".join(parts[:i])
            ok = (100 <= k <= 109 and parent.endswith("phase1b/fresh_seeds")) or \
                 (k in (0, 1, 2) and parent.endswith("keep/dev_new"))
            assert ok, f"refused: seed folder {part} in {path}"
    return path


def main():
    import numpy as np
    import torch
    import transformers
    from PIL import Image
    from transformers import (DFineForObjectDetection, RTDetrImageProcessor, RTDetrV2ForObjectDetection,
                              VitPoseForPoseEstimation, VitPoseImageProcessor)

    man = json.load(open(sys.argv[1]))
    out_path = sys.argv[2]
    assert len(man["images"]) == 1
    img_path = guard(man["images"][0]["rgb"])
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda")
    info = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
            "transformers": transformers.__version__, "image": img_path}
    image = Image.open(img_path).convert("RGB")
    W, H = image.size
    info["size"] = [W, H]
    torch.cuda.reset_peak_memory_stats()
    out = {"info": info, "detectors": {}}
    boxes_for_pose = None
    for name, cls in (("rtdetr_v2_r50vd", RTDetrV2ForObjectDetection), ("dfine_large", DFineForObjectDetection)):
        proc = RTDetrImageProcessor.from_pretrained(SNAP[name])
        model = cls.from_pretrained(SNAP[name]).to(dev).eval().float()
        id2label = model.config.id2label
        person_ids = [int(k) for k, v in id2label.items() if v == "person"]
        with torch.no_grad():
            inputs = proc(images=image, return_tensors="pt").to(dev)
            t0 = time.perf_counter()
            res = model(**inputs)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            post = proc.post_process_object_detection(res, threshold=0.05, target_sizes=[(H, W)])[0]
        keep = [i for i, lab in enumerate(post["labels"].tolist()) if lab in person_ids]
        boxes = post["boxes"][keep].cpu().numpy().tolist()
        scores = post["scores"][keep].cpu().numpy().tolist()
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        out["detectors"][name] = {
            "processor": type(proc).__name__, "input_shape": list(inputs["pixel_values"].shape),
            "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
            "person_ids": person_ids, "n_labels": len(id2label), "label0": id2label.get(0, id2label.get("0")),
            "post_keys": sorted(post.keys()), "first_forward_s": round(t1 - t0, 4),
            "boxes_xyxy": [[round(v, 2) for v in boxes[i]] for i in order], "scores": [round(scores[i], 4) for i in order]}
        if boxes_for_pose is None:
            boxes_for_pose = [boxes[i] for i in order if scores[i] >= 0.10]
        del model
        torch.cuda.empty_cache()
    pproc = VitPoseImageProcessor.from_pretrained(SNAP["vitpose_base_simple"])
    pmodel = VitPoseForPoseEstimation.from_pretrained(SNAP["vitpose_base_simple"]).to(dev).eval().float()
    coco = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes_for_pose]
    with torch.no_grad():
        pin = pproc(images=image, boxes=[coco], return_tensors="pt").to(dev)
        pres = pmodel(**pin)
        pp = pproc.post_process_pose_estimation(pres, boxes=[coco])[0]
    out["pose"] = {"processor": type(pproc).__name__, "input_shape": list(pin["pixel_values"].shape),
                   "id2label": pmodel.config.id2label, "result_keys": sorted(pp[0].keys()) if pp else [],
                   "per_box": []}
    for b, r in zip(boxes_for_pose, pp):
        kp = r["keypoints"].cpu().numpy()
        sc = r["scores"].cpu().numpy()
        out["pose"]["per_box"].append({"box_xyxy": [round(v, 2) for v in b],
                                       "keypoints": [[round(float(kp[j, 0]), 2), round(float(kp[j, 1]), 2),
                                                      round(float(sc[j]), 3)] for j in range(kp.shape[0])]})
    out["info"]["peak_mem_MiB"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    json.dump(out, open(out_path, "w"), indent=1)
    print(json.dumps(out["info"]))
    for name, d in out["detectors"].items():
        print(name, d["processor"], d["input_shape"], d["params_M"], d["person_ids"], d["label0"], d["post_keys"],
              d["first_forward_s"])
        print("  boxes>=0.10:", [(b, s) for b, s in zip(d["boxes_xyxy"], d["scores"]) if s >= 0.10])
    print("pose", out["pose"]["processor"], out["pose"]["input_shape"], out["pose"]["result_keys"])
    names = out["pose"]["id2label"]
    for pb in out["pose"]["per_box"]:
        print(" box", pb["box_xyxy"])
        for j, k in enumerate(pb["keypoints"]):
            if j <= 6 or j in (11, 12):
                print("   ", j, names.get(j, names.get(str(j))), k)


if __name__ == "__main__":
    main()
