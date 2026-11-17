"""Verifier: compare the verifier's own raw pass with the compute agent's raw pass (stdlib only)."""
import hashlib
import json
import statistics

V = "results/still-scenes/people-finder/early-detector/phase2/verify_i09_confirm"
C = "results/still-scenes/people-finder/early-detector/phase2/confirm"


def main():
    rawsha = {l.split()[1]: l.split()[0] for l in open(C + "/RAW.sha256") if l.strip()}
    assert hashlib.sha256(open(C + "/raw/rtdetr_v2_r50vd.jsonl", "rb").read()).hexdigest() == rawsha["raw/rtdetr_v2_r50vd.jsonl"]
    mine = [json.loads(x) for x in open(V + "/raw/v_rtdetr_v2_r50vd.jsonl")]
    theirs = [json.loads(x) for x in open(C + "/raw/rtdetr_v2_r50vd.jsonl")]
    assert [m["rgb"] for m in mine] == [t["rgb"] for t in theirs] and len(mine) == 182
    exact_imgs = 0
    nb_diff = []
    max_box = max_score = max_kp = max_kps = 0.0
    nboxes = npose = 0
    above = {0.25: [0, 0], 0.10: [0, 0]}
    for m, t in zip(mine, theirs):
        assert m["sha256"] == t["sha256"] and (m["W"], m["H"]) == (t["W"], t["H"])
        same = m["boxes"] == t["boxes"] and [p["kp"] for p in m["pose"]] == [p["kp"] for p in t["pose"]] and \
            [p["box_index"] for p in m["pose"]] == [p["box_index"] for p in t["pose"]]
        exact_imgs += same
        if len(m["boxes"]) != len(t["boxes"]) or len(m["pose"]) != len(t["pose"]):
            nb_diff.append((m["rgb"][-60:], len(m["boxes"]), len(t["boxes"]), len(m["pose"]), len(t["pose"])))
            continue
        for a, b in zip(m["boxes"], t["boxes"]):
            nboxes += 1
            max_box = max(max_box, max(abs(a[i] - b[i]) for i in range(4)))
            max_score = max(max_score, abs(a[4] - b[4]))
        for a, b in zip(m["pose"], t["pose"]):
            npose += 1
            for ka, kb in zip(a["kp"], b["kp"]):
                max_kp = max(max_kp, abs(ka[0] - kb[0]), abs(ka[1] - kb[1]))
                max_kps = max(max_kps, abs(ka[2] - kb[2]))
    tm = [m["t_det_s"] + m["t_pose_s"] for m in mine]
    tt = [t["t_det_s"] + t["t_pose_s"] for t in theirs]
    p90 = lambda xs: sorted(xs)[max(0, -(-9 * len(xs) // 10) - 1)]
    L = [f"raw comparison: {len(mine)} images; bit-identical boxes, scores and keypoints: {exact_imgs}/{len(mine)}",
         f"images with a different box or pose count: {len(nb_diff)} {nb_diff[:5]}",
         f"over {nboxes} boxes and {npose} pose sets: max |box coord diff| {max_box:.3g} px, max |score diff| "
         f"{max_score:.3g}, max |keypoint coord diff| {max_kp:.3g} px, max |keypoint score diff| {max_kps:.3g}",
         f"total boxes >= 0.05: verifier {sum(len(m['boxes']) for m in mine)}, compute {sum(len(t['boxes']) for t in theirs)}; "
         f">= 0.25: verifier {sum(b[4] >= 0.25 for m in mine for b in m['boxes'])}, compute {sum(b[4] >= 0.25 for t in theirs for b in t['boxes'])}",
         f"runtime t_det+t_pose (s), no warm-up: verifier median {statistics.median(tm)*1000:.1f} ms p90 {p90(tm)*1000:.1f} ms; "
         f"compute median {statistics.median(tt)*1000:.1f} ms p90 {p90(tt)*1000:.1f} ms (n {len(tt)})"]
    open(V + "/v_compare_raw.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


main()
