"""I09 fix 2, step 4 of FIX_PROTOCOL.md (the protocol's confirm.py; renamed alongside select_candidates.py): the
checks of the chosen fix, after the choice is written and hashed. Only the chosen candidate and A1 frozen (C00) are
scored, on
  - fit-validate (seeds 116-119, 56 images): the check of section 7, with Wilson and exact Poisson intervals;
  - A1's 42 development E-real scenes (through the harness, as whatif.py): recall >= 0.95 and false <= 3;
  - A1's F-real (21) and F-ph (268) fit data: information only.
Then completes chosen_fix2.json with the checks (the choice block unchanged) and writes fix_confirm.txt,
confirm_summary.json, confirm_rows.jsonl and fix_tables.txt (fix_select.txt + fix_confirm.txt). Project venv.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixlib as X  # noqa: E402

F, P, PS, C, H = X.F, X.P, X.PS, X.C, X.H
OUT = F.HERE
PROTOCOL_SHA = "59654e7bf67d90827ba2fa1e74f23fbe96ce64497d5a2633a5cea9c63c4fa600"
sys.path.insert(0, str(F.I03 / "stageA"))
import fitdata as FD  # noqa: E402


def verify():
    assert F.sha(OUT / "FIX_PROTOCOL.md") == PROTOCOL_SHA
    want = (OUT / "CHOSEN2.sha256").read_text().split()[0]
    assert F.sha(OUT / "chosen_fix2_precheck.json") == want
    assert (OUT / "chosen_fix2.json").read_bytes() == (OUT / "chosen_fix2_precheck.json").read_bytes(), \
        "chosen_fix2.json already completed: confirm_chosen.py runs once"
    ch = json.loads((OUT / "chosen_fix2_precheck.json").read_text())
    assert ch["protocol_sha256"] == PROTOCOL_SHA
    assert ch["code_sha256"] == X.code_sha(), "fix/ code changed since the choice"
    for sha_file in ("MANIFEST_FIT.sha256", "RAW_FIT.sha256"):
        for line in (OUT / sha_file).read_text().splitlines():
            w, name = line.split()
            assert F.sha(OUT / name) == w, name
    for path, w in F.DEV_RAW.items():
        assert F.sha(path) == w, path
    return ch, want


def interval_block(sm):
    k_f, n_img = sm["false"], sm["images"]
    lo, hi = X.poisson_exact(k_f)
    rl = sm["by_role"]
    return {
        "recall": [sm["matched"], sm["visible"], sm["recall"], X.wilson(sm["matched"], sm["visible"])],
        "line_recall": [rl["line"][0], rl["line"][1], sm["line_recall"], X.wilson(*rl["line"])],
        "staff_recall": [rl["staff"][0], rl["staff"][1], sm["staff_recall"], X.wilson(*rl["staff"])],
        "np_w45": [sm["np_w45"], sm["np_n"], sm["np_w45_share"], X.wilson(sm["np_w45"], sm["np_n"])],
        "images_with_false": [sm["images_with_false"], n_img, sm["images_with_false"] / n_img,
                              X.wilson(sm["images_with_false"], n_img)],
        "false_per_image": [k_f, n_img, k_f / n_img, [lo / n_img, hi / n_img]],
    }


def fmt_iv(name, b):
    k, n, p, (lo, hi) = b
    return f"    {name}: {k}/{n} = {p:.3f} [{lo:.3f}, {hi:.3f}]"


def main():
    t0 = time.time()
    ch, precheck_sha = verify()
    best = ch["chosen"]
    L = []
    if best is None:
        L.append("No candidate was eligible on fit-train; nothing to confirm (FIX_PROTOCOL.md section 6.4).")
        (OUT / "fix_confirm.txt").write_text("\n".join(L) + "\n")
        return
    ids = ["C00", best]
    assert tuple(ch["chosen_blocks"]) == X.CANDIDATES[best]
    s, variant = F.frozen_setting()
    assert ch["base_setting"] == s and ch["facing_variant"] == variant
    sets = {}
    rows_out = []

    # ---- fit-validate (seeds 116-119)
    raw = F.raw_lines(OUT / "raw_fit/rtdetr_v2_r50vd.jsonl")
    th = PS.default_thresholds()
    accs = {i: X.Acc() for i in ids}
    n = 0
    for e in F.fit_manifest()["images"]:
        if e["domain"] != "fit-validate":
            continue
        assert e["seed"] in F.PG.FIT_VALIDATE
        F.PG.check(e["rgb"])
        x = raw[e["rgb"]]
        assert x["sha256"] == e["sha256"]
        sc = F.load_fit_scene(e["folder"])
        assert F.sha(Path(e["folder"]) / "depth.npy") == e["sha256_depth"]
        people = X.run_all(x, sc.observation.depth, sc.observation.camera, s, variant, ids)
        res = X.score_image(accs, people, sc, th, variant, with_classes=True)
        n += 1
        for cid, (r, prow, sec, ppl) in res.items():
            rows_out.append({"set": "fit-validate", "candidate": cid, "seed": e["seed"], "row_id": e["row_id"],
                             "scene_id": e["scene_id"], "visible": r["n_visible"], "matched": r["matched"],
                             "false": r["false"], "exempt": r["exempt"], "secondary_false": sec["false"],
                             "misses": [{"pid": m["id"], "role_class": m["role_class"]} for m in r["misses"]],
                             "false_dets": [{"number": k, "score": ppl[k - 1]["score"], "box": ppl[k - 1]["box"],
                                             "xy": ppl[k - 1]["xy"], "m1": ppl[k - 1]["m1"]}
                                            for k in r["false_numbers"]]})
    assert n == 56
    sets["fit-validate (seeds 116-119, 56)"] = {i: X.summary(a) for i, a in accs.items()}

    # ---- A1 development E-real (42), through the harness
    mane = json.loads((F.A1 / "manifest_eval.json").read_text())
    rawe = {x["rgb"]: x for x in map(json.loads, (F.A1 / "raw_eval/rtdetr_v2_r50vd.jsonl").read_text().splitlines())}
    by_folder = {e["folder"]: e for e in mane["images"]}
    accs = {i: X.Acc() for i in ids}
    n = 0
    for sref in H.distinct_scenes():
        e = by_folder[str(sref.folder)]
        if e["domain"] != "E-real":
            continue
        sc = H.scene(sref.runs[0], sref.scene_id)
        people = X.run_all(rawe[e["rgb"]], sc.observation.depth, sc.observation.camera, s, variant, ids)
        X.score_image(accs, people, sc, C.thresholds_for(sref.runs), variant, with_classes=True)
        n += 1
    assert n == 42
    dev = {i: X.summary(a) for i, a in accs.items()}
    assert (dev["C00"]["matched"], dev["C00"]["visible"], dev["C00"]["false"]) == (116, 118, 3), dev["C00"]
    sets["A1 development E-real (42)"] = dev

    # ---- A1 fit data, information only
    man = json.loads((F.A1 / "manifest_fit.json").read_text())
    meta = json.loads((F.A1 / "raw_fit/meta.json").read_text())
    assert meta["manifest_sha256"] == F.sha(F.A1 / "manifest_fit.json")
    rawf = {(x["set"], x["scene_id"]): x for x in map(json.loads, (F.A1 / "raw_fit/rtdetr_v2_r50vd.jsonl")
                                                       .read_text().splitlines())}
    refs = FD.fit_refs()
    assert [(r[1], r[2]) for r in refs] == [(i["set"], i["scene_id"]) for i in man["images"]]
    accd = {"F-real": {i: X.Acc() for i in ids}, "F-ph": {i: X.Acc() for i in ids}}
    for domain, sset, scene_id, loader in refs:
        sc, thf = FD.load(loader)
        people = X.run_all(rawf[(sset, scene_id)], sc.observation.depth, sc.observation.camera, s, variant, ids)
        X.score_image(accd[domain], people, sc, thf, variant, with_classes=True)
    fr = {i: X.summary(a) for i, a in accd["F-real"].items()}
    fp = {i: X.summary(a) for i, a in accd["F-ph"].items()}
    assert (fr["C00"]["matched"], fr["C00"]["visible"], fr["C00"]["false"]) == (36, 39, 3), fr["C00"]
    assert (fp["C00"]["matched"], fp["C00"]["visible"], fp["C00"]["false"]) == (513, 585, 0), fp["C00"]
    sets["A1 F-real (21; information)"] = fr
    sets["A1 F-ph (268; information)"] = fp

    # ---- checks (FIX_PROTOCOL.md section 7)
    fv = sets["fit-validate (seeds 116-119, 56)"]
    c_fv, a_fv = fv[best], fv["C00"]
    fv_parts = {"recall >= 0.97": c_fv["recall"] >= 0.97, "line recall >= 0.97": c_fv["line_recall"] >= 0.97,
                "non-profile w45 >= 0.90": c_fv["np_w45_share"] >= 0.90,
                "false per image < A1's": c_fv["fb_per_image"] < a_fv["fb_per_image"]}
    holds_fv = all(fv_parts.values())
    c_dev = dev[best]
    dev_parts = {"recall >= 0.95": c_dev["recall"] >= 0.95, "false <= 3": c_dev["false"] <= 3}
    holds_dev = all(dev_parts.values())
    label = ("confirmed" if holds_fv and holds_dev else "fails fit-validate" if holds_dev else
             "fails development" if holds_fv else "fails both")
    iv = {i: interval_block(fv[i]) for i in ids}

    L.append(f"I09 fix 2, checks of the chosen fix {best} ({'+'.join(X.CANDIDATES[best])}) against A1 frozen (C00). "
             f"Choice file sha256 {precheck_sha[:16]}... (CHOSEN2.sha256), written before these were scored.")
    for name, d in sets.items():
        L.append("")
        L.append(f"== {name}")
        for i in ids:
            sm = d[i]
            sec = sm["secondary"]
            L.append(f"  {i} {X.fmt_summary(sm)}")
            L.append(f"      secondary scorer: false {sec['false']}, exempt {sec['exempt']}, reclassified "
                     f"{sec['moved']} ({sec['fb_per_image']:.3f} per image) | M3 audit "
                     f"{sm['m3_audit']['a1_near_hidden_staff']}, lost {sm['m3_audit']['lost']} | M1 fired on kept "
                     f"{sm['m1_fired_kept']} (A1 matched {sm['m1_fired_kept_a1_matched']}) | images with false "
                     f"{sm['images_with_false']} | post hoc {sm['posthoc_classes']}")
            L.append(f"      misses {[(m['pid'], m['role_class'], m['view']) for m in sm['misses']]}")
    L.append("")
    L.append("Fit-validate with 95% intervals (Wilson; exact Poisson for false per image):")
    for i in ids:
        L.append(f"  {i}:")
        for k, b in iv[i].items():
            L.append(fmt_iv(k, b))
    L.append("")
    L.append(f"Check 1, fit-validate holds: {holds_fv} {fv_parts}")
    L.append(f"Check 2, development not worse: {holds_dev} {dev_parts} (A1: {dev['C00']['matched']}/"
             f"{dev['C00']['visible']}, false {dev['C00']['false']})")
    L.append(f"Outcome: {label}")
    L.append(f"Run time {time.time() - t0:.1f} s")
    text = "\n".join(L) + "\n"
    (OUT / "fix_confirm.txt").write_text(text)
    F.C.jdump({"sets": sets, "intervals_fit_validate": iv, "checks": {"fit_validate": fv_parts, "dev": dev_parts},
               "outcome": label}, OUT / "confirm_summary.json")
    (OUT / "confirm_rows.jsonl").write_text("".join(json.dumps(o, default=float) + "\n" for o in rows_out))
    (OUT / "fix_tables.txt").write_text((OUT / "fix_select.txt").read_text() + "\n" + "=" * 100 + "\n\n" + text)
    ch["checks"] = {"written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "precheck_sha256": precheck_sha,
                    "fit_validate": {"holds": holds_fv, "parts": fv_parts, "chosen": fv[best], "a1": fv["C00"],
                                     "intervals": iv},
                    "dev_e_real": {"holds": holds_dev, "parts": dev_parts, "chosen": dev[best], "a1": dev["C00"]},
                    "information": {"F-real": fr, "F-ph": fp},
                    "outcome": label}
    (OUT / "chosen_fix2.json").write_text(json.dumps(ch, indent=1, default=float))
    print(text)


if __name__ == "__main__":
    main()
