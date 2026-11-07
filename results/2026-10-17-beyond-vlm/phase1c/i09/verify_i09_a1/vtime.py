"""I09 A1 verifier: runtime (FROZEN.md section 9). Reads the verifier's own timing pass (timing/timing.jsonl, two
passes A and B over the 42 E-real images after 5 F-real warm-ups) and its raw pass; checks keypoint agreement; times
the verifier's own project-side pipeline (2 threads) on the timing pass's inputs; compares with the compute agent's
a1/timing files (read as data). Writes vtime.txt."""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, 'results/2026-10-17-beyond-vlm/phase1b/i03/stageA')
sys.path.insert(0, 'results/2026-10-17-beyond-vlm/phase1b/i03')
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H  # noqa: E402
import fitdata as FD  # noqa: E402
import vpipe as VP  # noqa: E402

SET = dict(tau=0.25, band="wide", flat=False, d_dup=0.50, d_push=0.20)


def p90(x):
    s = sorted(x)
    return s[math.ceil(0.9 * len(s)) - 1]


def main():
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        assert os.environ.get(k) == "2"
    L = []
    tim = [json.loads(l) for l in open(HERE / "timing" / "timing.jsonl")]
    raw = {json.loads(l)["scene_id"] + "|" + json.loads(l)["set"]: json.loads(l) for l in open(HERE / "raw" / "rtdetr_v2_r50vd.jsonl")}
    for pas in ("A", "B"):
        t = [r["t_s"] for r in tim if r["pass"] == pas]
        L.append(f"sidecar pass {pas}: n {len(t)} median {1000 * statistics.median(t):.1f} ms, p90 {1000 * p90(t):.1f} ms, max {1000 * max(t):.1f} ms")
    theirs = [json.loads(l) for l in open(HERE.parent / "a1" / "timing" / "timing.jsonl")]
    tt = [r["t_s"] for r in theirs]
    L.append(f"compute agent's sidecar timing (a1/timing/timing.jsonl): n {len(tt)} median {1000 * statistics.median(tt):.1f} ms, p90 {1000 * p90(tt):.1f} ms")
    # keypoint agreement, my timing pass A vs my raw pass, boxes >= tau paired by index
    maxk = maxb = 0.0
    nimg_ok = 0
    for r in [r for r in tim if r["pass"] == "A"]:
        rw = raw[r["scene_id"] + "|" + r["set"]]
        rb = [b for b in rw["boxes"] if b[4] >= 0.25]
        assert len(rb) == len(r["boxes"])
        db = float(np.abs(np.array(rb).reshape(-1, 5) - np.array(r["boxes"]).reshape(-1, 5)).max()) if rb else 0.0
        dk = float(np.abs(np.array(rw["kp"][:len(rb)]).reshape(-1, 17, 3)[..., :2] - np.array(r["kp"]).reshape(-1, 17, 3)[..., :2]).max()) if rb else 0.0
        maxb, maxk = max(maxb, db), max(maxk, dk)
        nimg_ok += dk <= 1.0
    L.append(f"timing pass vs raw pass: {nimg_ok}/42 images keypoints within 1 px; max kp diff {maxk:.2e} px; max box diff {maxb:.2e} px")
    # project side
    warm = []
    for ref in [r for r in FD.fit_refs() if r[0] == "F-real"][:5]:
        sc, _ = FD.load(ref[3])
        rw = raw[ref[2] + "|" + ref[1]]
        rb = [b for b in rw["boxes"] if b[4] >= 0.25]
        warm.append((sc.observation, rb, rw["kp"][:len(rb)]))
    by_id = {s.scene_id: s for s in H.distinct_scenes()}
    obs_of = {}
    for r in [r for r in tim if r["pass"] == "A"]:
        s = by_id[r["scene_id"]]
        obs_of[r["scene_id"]] = H.scene(s.runs[0], s.scene_id).observation

    def run(obs, boxes, kps):
        bases = [VP.box_base(obs, b, k) for b, k in zip(boxes, kps)]
        return VP.people(bases, obs.camera.T_world_camera[:2, 3], **SET)
    for obs, b, k in warm:
        run(obs, b, k)
    pt, same = [], 0
    for r in [r for r in tim if r["pass"] == "A"]:
        obs = obs_of[r["scene_id"]]
        t0 = time.perf_counter()
        ppl = run(obs, r["boxes"], r["kp"])
        pt.append(time.perf_counter() - t0)
        rw = raw[r["scene_id"] + "|" + r["set"]]
        rb = [b for b in rw["boxes"] if b[4] >= 0.25]
        ref = run(obs, rb, rw["kp"][:len(rb)])
        same += len(ppl) == len(ref) and all(math.dist((a["x"], a["y"]), (b["x"], b["y"])) < 1e-3 and
                                            ((a["yaw"]["F5"] is None) == (b["yaw"]["F5"] is None)) and
                                            (a["yaw"]["F5"] is None or abs(math.remainder(a["yaw"]["F5"] - b["yaw"]["F5"], 2 * math.pi)) < 1e-2)
                                            for a, b in zip(ppl, ref))
    L.append(f"verifier project side (box_base computes the flat share too, so an upper bound; 2 threads): median "
             f"{1000 * statistics.median(pt):.2f} ms, p90 {1000 * p90(pt):.2f} ms; deployed = eval path people in {same}/42")
    pj = json.load(open(HERE.parent / "a1" / "timing" / "project_timing.json"))
    L.append(f"compute agent's project side: median {1000 * pj['median_s']:.2f} ms, p90 {1000 * pj['p90_s']:.2f} ms")
    sA = [r["t_s"] for r in tim if r["pass"] == "A"]
    L.append(f"criterion (verifier): median sidecar {1000 * statistics.median(sA):.1f} + median project "
             f"{1000 * statistics.median(pt):.2f} = {1000 * (statistics.median(sA) + statistics.median(pt)):.1f} ms <= 1000 ms: "
             f"{statistics.median(sA) + statistics.median(pt) <= 1.0}")
    (HERE / "vtime.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
