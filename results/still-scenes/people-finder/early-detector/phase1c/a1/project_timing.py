"""I09 A1 runtime, project side (FROZEN.md section 9; DEVIATIONS.md C15). Steps 1-7 plus the frozen facing variant on
each E-real image on the CPU (2 BLAS/OMP threads), with depth, camera, boxes and keypoints in memory (the sidecar's
timing pass output: boxes >= tau* and their keypoints). 5 warm-ups on the first 5 F-real images, then each of the 42
E-real images timed once. Also checks that the deployed path's people equal the evaluation path's (raw pass). Writes
timing/project_timing.json. Project venv."""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import pipeline as P  # noqa: E402
import fitdata as FD  # noqa: E402
import research_harness as H  # noqa: E402


def main():
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        assert os.environ.get(k) == "2", k
    chosen = json.loads((HERE / "chosen.json").read_text())
    fix = json.loads((HERE / "chosen_fix.json").read_text())
    s, variant = chosen["setting"], fix["facing_variant"]
    tau = s["tau"]
    man_e = json.loads((HERE / "manifest_eval.json").read_text())
    man_f = json.loads((HERE / "manifest_fit.json").read_text())
    tim = {x["scene_id"]: x for x in map(json.loads, (HERE / "timing" / "timing.jsonl").read_text().splitlines())}
    raw_e = {x["scene_id"]: x for x in map(json.loads, (HERE / "raw_eval" / f"{s['detector']}.jsonl").read_text().splitlines())}
    raw_f = {(x["set"], x["scene_id"]): x for x in map(json.loads, (HERE / "raw_fit" / f"{s['detector']}.jsonl").read_text().splitlines())}

    def deployed_input(boxes, kps):
        return {"boxes": boxes, "pose": [{"box_index": i, "kp": kps[i]} for i in range(len(boxes))]}

    def from_raw(raw):
        boxes = [b for b in raw["boxes"] if b[4] >= tau]
        kp = {p["box_index"]: p["kp"] for p in raw["pose"]}
        return deployed_input(boxes, [kp[i] for i in range(len(boxes))])

    warm = []
    for e in [e for e in man_f["images"] if e["domain"] == "F-real"][:5]:
        ref = next(r for r in FD.fit_refs() if r[1] == e["set"] and r[2] == e["scene_id"])
        sc, _ = FD.load(ref[3])
        warm.append((e["scene_id"], sc.observation.depth, sc.observation.camera, from_raw(raw_f[(e["set"], e["scene_id"])])))
    by_id = {sref.scene_id: sref for sref in H.distinct_scenes()}
    real = []
    for e in [e for e in man_e["images"] if e["domain"] == "E-real"]:
        sref = by_id[e["scene_id"]]
        sc = H.scene(sref.runs[0], sref.scene_id)
        t = tim[e["scene_id"]]
        real.append((e["scene_id"], sc.observation.depth, sc.observation.camera, deployed_input(t["boxes"], t["kp"])))
    assert len(real) == 42
    for _, depth, cam, inp in warm:
        P.run_image(inp, depth, cam, s, variant)
    times, same = [], 0
    rows = []
    for sid, depth, cam, inp in real:
        t0 = time.perf_counter()
        ppl = P.run_image(inp, depth, cam, s, variant)
        dt = time.perf_counter() - t0
        times.append(dt)
        ref_ppl = P.run_image(from_raw(raw_e[sid]), depth, cam, s, variant)
        eq = len(ppl) == len(ref_ppl) and all(
            math.dist(a["xy"], b["xy"]) < 1e-3 and ((a["yaw"] is None) == (b["yaw"] is None)) and
            (a["yaw"] is None or abs(math.remainder(a["yaw"] - b["yaw"], 2 * math.pi)) < 1e-2)
            for a, b in zip(ppl, ref_ppl))
        same += eq
        rows.append({"scene_id": sid, "t_s": dt, "n_boxes": len(inp["boxes"]), "n_people": len(ppl),
                     "same_as_eval_path": eq})
    out = {"written": time.strftime('%Y-%m-%dT%H:%M:%S%z'), "threads": {k: os.environ.get(k) for k in
           ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}, "setting": s, "variant": variant,
           "warmup": [w[0] for w in warm], "n": len(times), "median_s": statistics.median(times),
           "p90_s": C.p90(times), "max_s": max(times), "deployed_equals_eval_path": [same, len(real)],
           "rows": rows}
    (HERE / "timing" / "project_timing.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
