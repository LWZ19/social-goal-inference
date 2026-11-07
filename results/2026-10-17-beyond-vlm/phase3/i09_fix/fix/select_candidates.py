"""I09 fix 2, step 3 of FIX_PROTOCOL.md (the protocol's select.py; renamed so it cannot shadow the stdlib
module select): score every candidate on fit-train (seeds 110-115, 84 images), apply the
selection rule fixed in section 6, and write the choice (chosen_fix2.json, chosen_fix2_precheck.json, CHOSEN2.sha256)
before fit-validate or the development scenes are scored by any candidate. Also writes fix_select.txt,
select_summary.json and select_rows.jsonl. Fit-validate is not read. Project venv.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixlib as X  # noqa: E402

F, P, PS, C = X.F, X.P, X.PS, X.C
OUT = F.HERE
PROTOCOL_SHA = "59654e7bf67d90827ba2fa1e74f23fbe96ce64497d5a2633a5cea9c63c4fa600"
RULE = ("Eligible: fit-train recall >= 0.97, line-member recall >= 0.97, non-profile facing within 45 deg >= 0.90. "
        "Choose the eligible candidate with the fewest false people per image on fit-train; ties: fewer parameters, "
        "then the lower candidate number; none eligible: no fix chosen.")


def verify_inputs():
    assert F.sha(OUT / "FIX_PROTOCOL.md") == PROTOCOL_SHA
    assert (OUT / "FIX_PROTOCOL.sha256").read_text().split()[0] == PROTOCOL_SHA
    for sha_file in ("MANIFEST_FIT.sha256", "RAW_FIT.sha256"):
        for line in (OUT / sha_file).read_text().splitlines():
            want, name = line.split()
            assert F.sha(OUT / name) == want, name
    for name in ("chosen_fix2.json", "chosen_fix2_precheck.json", "CHOSEN2.sha256"):
        assert not (OUT / name).exists(), f"{name} exists: select.py runs once"


def eligible(sm):
    return (sm["recall"] is not None and sm["recall"] >= 0.97 and sm["line_recall"] is not None
            and sm["line_recall"] >= 0.97 and sm["np_w45_share"] is not None and sm["np_w45_share"] >= 0.90)


def choose(summ):
    el = [cid for cid in X.CANDIDATES if eligible(summ[cid])]
    if not el:
        return None, el
    order = list(X.CANDIDATES)
    best = min(el, key=lambda cid: (summ[cid]["fb_per_image"], X.n_params(X.CANDIDATES[cid]), order.index(cid)))
    return best, el


def main():
    t0 = time.time()
    verify_inputs()
    s, variant = F.frozen_setting()
    raw = F.raw_lines(OUT / "raw_fit/rtdetr_v2_r50vd.jsonl")
    th = PS.default_thresholds()
    PS.check_thresholds(th)
    accs = {cid: X.Acc() for cid in X.CANDIDATES}
    out_rows = []
    n = 0
    for e in F.fit_manifest()["images"]:
        if e["domain"] != "fit-train":
            continue
        assert e["seed"] in F.PG.FIT_TRAIN
        F.PG.check(e["rgb"])
        x = raw[e["rgb"]]
        assert x["sha256"] == e["sha256"]
        sc = F.load_fit_scene(e["folder"])
        assert F.sha(Path(e["folder"]) / "depth.npy") == e["sha256_depth"]
        obs = sc.observation
        people = X.run_all(x, obs.depth, obs.camera, s, variant)
        # ---- people fixed above for every candidate; truth read only from here on
        res = X.score_image(accs, people, sc, th, variant, with_classes=True)
        n += 1
        for cid, (r, prow, sec, ppl) in res.items():
            out_rows.append({
                "candidate": cid, "seed": e["seed"], "row_id": e["row_id"], "scene_id": e["scene_id"],
                "visible": r["n_visible"], "matched": r["matched"], "false": r["false"], "exempt": r["exempt"],
                "secondary_false": sec["false"], "secondary_exempt": sec["exempt"],
                "misses": [{"pid": m["id"], "role_class": m["role_class"]} for m in r["misses"]],
                "dets": [{"number": k + 1, "index": p["index"], "score": p["score"], "box": p["box"], "xy": p["xy"],
                          "m1": p["m1"], "matched_to": r["matches"].get(k + 1),
                          "false": (k + 1) in r["false_numbers"], "exempt": (k + 1) in r["exempt_numbers"]}
                         for k, p in enumerate(ppl)]})
    assert n == 84
    summ = {cid: X.summary(a) for cid, a in accs.items()}
    a1 = summ["C00"]
    assert (a1["matched"], a1["visible"], a1["false"], a1["exempt"]) == (226, 238, 17, 5), a1
    best, el = choose(summ)
    code = X.code_sha()
    lines = ["I09 fix 2, fit-train (seeds 110-115, 84 images): every candidate of FIX_PROTOCOL.md (sha256 "
             f"{PROTOCOL_SHA[:16]}...), I03's scorer unchanged (facing full, exempt_points, default thresholds).", "",
             "Id   blocks                        params  eligible  " + "summary"]
    for cid, blocks in X.CANDIDATES.items():
        sm = summ[cid]
        lines.append(f"{cid}  {'+'.join(blocks) or 'A1 frozen':28s}  {X.n_params(blocks):6d}  "
                     f"{'yes' if cid in el else 'no ':8s}  {X.fmt_summary(sm)}")
    lines += ["", "Per candidate: secondary scorer variant (false / exempt / reclassified; false per image) | M3 audit "
              "(A1 detections near hidden-by-rule staff with the mark in the picture: total, lost) | M1 block fired on "
              "kept boxes (of which A1 had matched) | images with a false person | post hoc classes | misses"]
    for cid in X.CANDIDATES:
        sm = summ[cid]
        sec = sm["secondary"]
        lines.append(f"{cid}  secondary {sec['false']} / {sec['exempt']} / {sec['moved']} ({sec['fb_per_image']:.3f}) | "
                     f"M3 {sm['m3_audit']['a1_near_hidden_staff']}, lost {sm['m3_audit']['lost']} | fired "
                     f"{sm['m1_fired_kept']} ({sm['m1_fired_kept_a1_matched']}) | images with false "
                     f"{sm['images_with_false']} | {sm['posthoc_classes']} | misses "
                     f"{[(m['pid'], m['role_class'], m['view']) for m in sm['misses']]}")
    lines += ["", "Selection rule (FIX_PROTOCOL.md section 6): " + RULE,
              f"Eligible: {el}",
              f"Chosen: {best} ({'+'.join(X.CANDIDATES[best]) if best else '-'}; "
              f"{X.n_params(X.CANDIDATES[best]) if best else '-'} parameters)" if best else "Chosen: none"]
    if best:
        ties = [cid for cid in el if summ[cid]["fb_per_image"] == summ[best]["fb_per_image"]]
        lines.append(f"Candidates tied on false per image with the chosen one: {ties} (parameters "
                     f"{[X.n_params(X.CANDIDATES[c]) for c in ties]})")
    lines.append(f"Code sha256: {code}")
    lines.append(f"Run time {time.time() - t0:.1f} s")
    text = "\n".join(lines) + "\n"
    (OUT / "fix_select.txt").write_text(text)
    F.C.jdump(summ, OUT / "select_summary.json")
    (OUT / "select_rows.jsonl").write_text("".join(json.dumps(o, default=float) + "\n" for o in out_rows))
    chosen = {
        "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol": "FIX_PROTOCOL.md", "protocol_sha256": PROTOCOL_SHA,
        "code_sha256": code, "fitlib_sha256": F.sha(F.HERE / "fitlib.py"),
        "pinned_a1_i03": {str(k): v for k, v in F.PINNED.items()},
        "raw_fit_sha256": F.sha(OUT / "raw_fit/rtdetr_v2_r50vd.jsonl"),
        "base_setting": s, "facing_variant": variant,
        "constants": {"GAP_M": X.GAP_M, "D030_M": X.D030_M, "IMG_CONT": X.IMG_CONT, "IMG_CAP_M": X.IMG_CAP_M,
                      "HEAD_KEYPOINTS": list(X.HEAD), "KP_CONF": P.CONF, "KP_WINDOW": P.KP_WINDOW,
                      "BAND_M": P.BAND_M, "MIN_KEPT": P.MIN_KEPT, "FLAT_MIN_Q": P.FLAT_MIN_Q,
                      "STAFF_EXEMPT_M (secondary scorer only)": X.STAFF_EXEMPT_M},
        "rule": RULE,
        "candidates": {cid: {"blocks": list(b), "params": X.n_params(b), "eligible": cid in el,
                             "recall": summ[cid]["recall"], "line_recall": summ[cid]["line_recall"],
                             "np_w45_share": summ[cid]["np_w45_share"], "false": summ[cid]["false"],
                             "fb_per_image": summ[cid]["fb_per_image"],
                             "secondary_false": summ[cid]["secondary"]["false"]}
                       for cid, b in X.CANDIDATES.items()},
        "eligible": el,
        "chosen": best,
        "chosen_blocks": list(X.CANDIDATES[best]) if best else None,
        "chosen_params": X.n_params(X.CANDIDATES[best]) if best else None,
        "fit_train": summ[best] if best else None,
        "a1_fit_train": a1,
        "checks": None,
    }
    body = json.dumps(chosen, indent=1, default=float)
    (OUT / "chosen_fix2_precheck.json").write_text(body)
    (OUT / "chosen_fix2.json").write_text(body)
    (OUT / "CHOSEN2.sha256").write_text(f"{F.sha(OUT / 'chosen_fix2_precheck.json')}  chosen_fix2_precheck.json\n")
    print(text)


if __name__ == "__main__":
    main()
