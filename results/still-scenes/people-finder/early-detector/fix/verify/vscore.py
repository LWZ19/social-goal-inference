"""Verifier task (1): re-score A1 frozen (C00) and the chosen fix (C07) -- and, to check the selection, every
candidate C00-C10 on fit-train -- from the raw outputs with my own code (vlib.py), then compare every number with
the fix step's outputs. Writes verify/v_rows.jsonl, verify/v_summary.json, verify/v_compare.txt. Project venv."""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402

OUT = Path(__file__).resolve().parent
FX = V.FIXDIR
L = []


def say(s=""):
    L.append(s)
    print(s, flush=True)


def eq(a, b, tol=1e-9):
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a is b
        return abs(a - b) <= tol
    return a == b


DIFFS = []


def cmp(label, mine, theirs, tol=1e-9):
    ok = eq(mine, theirs, tol)
    if not ok:
        DIFFS.append((label, mine, theirs))
    return ok


def main():
    t0 = time.time()
    s, variant = V.frozen_setting()
    say(f"frozen setting {s} variant {variant}")
    assert s == {"detector": "rtdetr_v2_r50vd", "tau": 0.25, "band": "wide", "flat": False, "d_dup": 0.5,
                 "d_push": 0.2} and variant == "F5"
    raw_path = FX / "raw_fit/rtdetr_v2_r50vd.jsonl"
    raw_sha = V.sha(raw_path)
    say(f"raw_fit sha256 {raw_sha}")
    rawl = [json.loads(x) for x in raw_path.read_text().splitlines()]
    raw = {x["rgb"]: x for x in rawl}
    assert len(raw) == len(rawl) == 140
    man = {e["scene_id"]: e for e in json.loads((FX / "manifest_fit.json").read_text())["images"]}
    th = V.PS.default_thresholds()
    V.PS.check_thresholds(th)

    rows_out = []
    recs = {}          # (set, cid) -> list of per-image records
    per_image = {}     # (set, cid, scene_id) -> record

    def do_fit(seeds, setname, ids):
        n = 0
        for k in seeds:
            for row in V.ROWS:
                folder, sc = V.load_fit(k, row)
                sid = f"seed{k}/{row}"
                rgb = folder / "rgb.png"
                x = raw[str(rgb)]
                me = V.sha(rgb)
                assert x["sha256"] == me == man[sid]["sha256"], sid
                assert V.sha(folder / "depth.npy") == man[sid]["sha256_depth"], sid
                assert V.sha(folder / "scene.json") == man[sid]["sha256_scene_json"], sid
                obs = sc.observation
                res = V.run_image(x, obs.depth, obs.camera, s, variant, ids)
                ref = res["C00"]
                r0 = V.score(ref, sc, th)
                ref_idx = {ref[n_ - 1]["index"] for n_ in r0["matches"]}
                for cid in ids:
                    rec = V.score(res[cid], sc, th, ref=ref, ref_matched_idx=ref_idx)
                    recs.setdefault((setname, cid), []).append(rec)
                    per_image[(setname, cid, sid)] = (rec, res[cid])
                    rows_out.append({"set": setname, "candidate": cid, "scene_id": sid, "visible": rec["visible"],
                                     "matched": rec["matched"], "false": rec["false"], "exempt": rec["exempt"],
                                     "sec_false": rec["sec_false"],
                                     "dets": [[p["index"], p["xy"][0], p["xy"][1], p["m1"]] for p in res[cid]]})
                n += 1
        return n

    all_ids = list(V.BLOCKS)
    assert do_fit(V.TRAIN, "fit-train", all_ids) == 84
    summ = {"fit-train": {cid: V.pool(recs[("fit-train", cid)]) for cid in all_ids}}

    # ---- selection rule (FIX_PROTOCOL.md section 6), applied to my numbers
    def elig(p):
        return p["recall"] >= 0.97 and p["line_recall"] >= 0.97 and p["np_share"] >= 0.90
    ft = summ["fit-train"]
    el = [c for c in all_ids if elig(ft[c])]
    order = list(V.BLOCKS)
    nparam = {c: sum(V.NPARAM[b] for b in V.BLOCKS[c]) for c in all_ids}
    best = min(el, key=lambda c: (ft[c]["fpi"], nparam[c], order.index(c))) if el else None
    say(f"my eligible {el}; my chosen {best}; params {nparam}")

    # ---- fit-validate, development E-real, F-real/F-ph: C00 and C07 only
    ids2 = ["C00", "C07"]
    assert do_fit(V.VALID, "fit-validate", ids2) == 56
    summ["fit-validate"] = {cid: V.pool(recs[("fit-validate", cid)]) for cid in ids2}

    mane = json.loads((V.A1 / "manifest_eval.json").read_text())
    rawe = {x["rgb"]: x for x in map(json.loads, (V.A1 / "raw_eval/rtdetr_v2_r50vd.jsonl").read_text().splitlines())}
    byf = {e["folder"]: e for e in mane["images"]}
    n = 0
    for sref in V.H.distinct_scenes():
        e = byf.get(str(sref.folder))
        if e is None or e["domain"] != "E-real":
            continue
        sc = V.H.scene(sref.runs[0], sref.scene_id)
        ths = [V.H.recorded_config(r)[1].thresholds for r in sref.runs]
        for t in ths:
            V.PS.check_thresholds(t)
        keys = ("person_radius_m", "visibility_min_samples", "visibility_depth_tolerance_m", "grid_resolution_m")
        assert all(getattr(t, k) == getattr(ths[0], k) for t in ths for k in keys)
        thd = ths[0]
        res = V.run_image(rawe[e["rgb"]], sc.observation.depth, sc.observation.camera, s, variant, ids2)
        ref = res["C00"]
        r0 = V.score(ref, sc, thd)
        ref_idx = {ref[n_ - 1]["index"] for n_ in r0["matches"]}
        for cid in ids2:
            rec = V.score(res[cid], sc, thd, ref=ref, ref_matched_idx=ref_idx)
            recs.setdefault(("dev", cid), []).append(rec)
        n += 1
    assert n == 42, n
    summ["dev"] = {cid: V.pool(recs[("dev", cid)]) for cid in ids2}

    sys.path.insert(0, str(V.I03 / "stageA"))
    import fitdata as FD
    mf = json.loads((V.A1 / "manifest_fit.json").read_text())
    meta = json.loads((V.A1 / "raw_fit/meta.json").read_text())
    assert meta["manifest_sha256"] == V.sha(V.A1 / "manifest_fit.json")
    rawf = {(x["set"], x["scene_id"]): x for x in map(json.loads, (V.A1 / "raw_fit/rtdetr_v2_r50vd.jsonl")
                                                       .read_text().splitlines())}
    refs = FD.fit_refs()
    assert [(r[1], r[2]) for r in refs] == [(i["set"], i["scene_id"]) for i in mf["images"]]
    for domain, sset, scene_id, loader in refs:
        sc, thf = FD.load(loader)
        res = V.run_image(rawf[(sset, scene_id)], sc.observation.depth, sc.observation.camera, s, variant, ids2)
        ref = res["C00"]
        r0 = V.score(ref, sc, thf)
        ref_idx = {ref[n_ - 1]["index"] for n_ in r0["matches"]}
        for cid in ids2:
            recs.setdefault((domain, cid), []).append(V.score(res[cid], sc, thf, ref=ref, ref_matched_idx=ref_idx))
    summ["F-real"] = {cid: V.pool(recs[("F-real", cid)]) for cid in ids2}
    summ["F-ph"] = {cid: V.pool(recs[("F-ph", cid)]) for cid in ids2}

    # ---- intervals on fit-validate
    iv = {}
    for cid in ids2:
        p = summ["fit-validate"][cid]
        lo, hi = V.garwood95(p["false"])
        npi = p["images"]
        iv[cid] = {"recall": [p["matched"], p["visible"], V.wilson95(p["matched"], p["visible"])],
                   "line": [*p["line"], V.wilson95(*p["line"])], "staff": [*p["staff"], V.wilson95(*p["staff"])],
                   "np_w45": [p["np_w45"], p["np_n"], V.wilson95(p["np_w45"], p["np_n"])],
                   "images_with_false": [p["images_with_false"], npi, V.wilson95(p["images_with_false"], npi)],
                   "fpi": [p["false"], npi, (lo / npi, hi / npi)]}
    fv, dv = summ["fit-validate"], summ["dev"]
    holds_fv = (fv["C07"]["recall"] >= 0.97 and fv["C07"]["line_recall"] >= 0.97 and fv["C07"]["np_share"] >= 0.90
                and fv["C07"]["fpi"] < fv["C00"]["fpi"])
    holds_dev = dv["C07"]["recall"] >= 0.95 and dv["C07"]["false"] <= 3
    label = ("confirmed" if holds_fv and holds_dev else "fails fit-validate" if holds_dev else
             "fails development" if holds_fv else "fails both")

    # ---------------------------------------------------------------------------------- comparisons
    sel = json.loads((FX / "select_summary.json").read_text())
    con = json.loads((FX / "confirm_summary.json").read_text())
    chosen = json.loads((FX / "chosen_fix2.json").read_text())
    pre = json.loads((FX / "chosen_fix2_precheck.json").read_text())

    def cmp_summary(tag, mine, th_):
        cmp(f"{tag} images", mine["images"], th_["images"])
        cmp(f"{tag} visible", mine["visible"], th_["visible"])
        cmp(f"{tag} matched", mine["matched"], th_["matched"])
        for rc in ("line", "staff", "group", "other"):
            cmp(f"{tag} {rc}", list(mine[rc]), list(th_["by_role"][rc]))
        cmp(f"{tag} false", mine["false"], th_["false"])
        cmp(f"{tag} exempt", mine["exempt"], th_["exempt"])
        cmp(f"{tag} near_hidden", mine["near_hidden"], th_["near_hidden"])
        cmp(f"{tag} merged", mine["merged"], th_["merged_pairs"])
        cmp(f"{tag} pairs", mine["pairs"], th_["neighbour_pairs"])
        cmp(f"{tag} xy_median", mine["xy_median"], th_["xy_median"], 1e-12)
        cmp(f"{tag} n_det", mine["n_det"], th_["n_det"])
        cmp(f"{tag} np_w45", mine["np_w45"], th_["np_w45"])
        cmp(f"{tag} np_n", mine["np_n"], th_["np_n"])
        cmp(f"{tag} profile_fb", mine["pf_fb"], th_["profile_fb"])
        cmp(f"{tag} profile_n", mine["pf_n"], th_["profile_n"])
        cmp(f"{tag} images_with_false", mine["images_with_false"], th_["images_with_false"])
        cmp(f"{tag} secondary false", mine["sec_false"], th_["secondary"]["false"])
        cmp(f"{tag} secondary exempt", mine["sec_exempt"], th_["secondary"]["exempt"])
        cmp(f"{tag} secondary moved", mine["sec_moved"], th_["secondary"]["moved"])
        cmp(f"{tag} m3 near", mine["m3_near"], th_["m3_audit"]["a1_near_hidden_staff"])
        cmp(f"{tag} m3 lost (per staff)", mine["m3_lost_staff"], th_["m3_audit"]["lost"])
        cmp(f"{tag} m1 fired kept", mine["fired"], th_["m1_fired_kept"])
        cmp(f"{tag} m1 fired kept A1-matched", mine["fired_a1_matched"], th_["m1_fired_kept_a1_matched"])
        cmp(f"{tag} posthoc classes", mine["classes"], th_["posthoc_classes"])
        cmp(f"{tag} misses", sorted(m for m in mine["misses"]),
            sorted((m["pid"], m["role_class"]) for m in th_["misses"]))

    for cid in all_ids:
        cmp_summary(f"fit-train {cid}", ft[cid], sel[cid])
    names = {"fit-validate": "fit-validate (seeds 116-119, 56)", "dev": "A1 development E-real (42)",
             "F-real": "A1 F-real (21; information)", "F-ph": "A1 F-ph (268; information)"}
    for k, nm in names.items():
        for cid in ids2:
            cmp_summary(f"{k} {cid}", summ[k][cid], con["sets"][nm][cid])
    for cid in ids2:
        t = con["intervals_fit_validate"][cid]
        m = iv[cid]
        for mk, tk in (("recall", "recall"), ("line", "line_recall"), ("staff", "staff_recall"), ("np_w45", "np_w45"),
                       ("images_with_false", "images_with_false"), ("fpi", "false_per_image")):
            cmp(f"interval {cid} {tk} k", m[mk][0], t[tk][0])
            cmp(f"interval {cid} {tk} n", m[mk][1], t[tk][1])
            cmp(f"interval {cid} {tk} lo", m[mk][2][0], t[tk][3][0], 1e-6)
            cmp(f"interval {cid} {tk} hi", m[mk][2][1], t[tk][3][1], 1e-6)
    cmp("eligible", el, pre["eligible"])
    cmp("chosen", best, pre["chosen"])
    cmp("chosen blocks", list(V.BLOCKS[best]), pre["chosen_blocks"])
    cmp("chosen params", nparam[best], pre["chosen_params"])
    for cid in all_ids:
        cmp(f"precheck {cid} params", nparam[cid], pre["candidates"][cid]["params"])
        cmp(f"precheck {cid} eligible", cid in el, pre["candidates"][cid]["eligible"])
        cmp(f"precheck {cid} false", ft[cid]["false"], pre["candidates"][cid]["false"])
        cmp(f"precheck {cid} secondary_false", ft[cid]["sec_false"], pre["candidates"][cid]["secondary_false"])
    cmp("outcome", label, con["outcome"])
    cmp("outcome in chosen_fix2.json", label, chosen["checks"]["outcome"])
    cmp("fit-validate holds", holds_fv, chosen["checks"]["fit_validate"]["holds"])
    cmp("dev holds", holds_dev, chosen["checks"]["dev_e_real"]["holds"])
    cmp("choice block unchanged in chosen_fix2.json", {k: v for k, v in chosen.items() if k != "checks"},
        {k: v for k, v in pre.items() if k != "checks"})

    # per-image rows against select_rows.jsonl / confirm_rows.jsonl
    srows = [json.loads(x) for x in (FX / "select_rows.jsonl").read_text().splitlines()]
    nimg = 0
    for r in srows:
        rec, ppl = per_image[("fit-train", r["candidate"], r["scene_id"])]
        nimg += 1
        for k_m, k_t in (("visible", "visible"), ("matched", "matched"), ("false", "false"), ("exempt", "exempt"),
                         ("sec_false", "secondary_false"), ("sec_exempt", "secondary_exempt")):
            cmp(f"row fit-train {r['candidate']} {r['scene_id']} {k_t}", rec[k_m], r[k_t])
        mine = [(p["index"], round(p["xy"][0], 9), round(p["xy"][1], 9), p["m1"] is not None) for p in ppl]
        theirs = [(d["index"], round(d["xy"][0], 9), round(d["xy"][1], 9), d["m1"] is not None) for d in r["dets"]]
        cmp(f"row fit-train {r['candidate']} {r['scene_id']} dets", mine, theirs)
        fl_m = sorted(rec["false_numbers"])
        fl_t = sorted(d["number"] for d in r["dets"] if d["false"])
        cmp(f"row fit-train {r['candidate']} {r['scene_id']} false numbers", fl_m, fl_t)
    crows = [json.loads(x) for x in (FX / "confirm_rows.jsonl").read_text().splitlines()]
    for r in crows:
        rec, ppl = per_image[("fit-validate", r["candidate"], r["scene_id"])]
        nimg += 1
        for k_m, k_t in (("visible", "visible"), ("matched", "matched"), ("false", "false"), ("exempt", "exempt"),
                         ("sec_false", "secondary_false")):
            cmp(f"row fit-validate {r['candidate']} {r['scene_id']} {k_t}", rec[k_m], r[k_t])
        fx_m = sorted((round(ppl[n_ - 1]["xy"][0], 9), round(ppl[n_ - 1]["xy"][1], 9)) for n_ in rec["false_numbers"])
        fx_t = sorted((round(d["xy"][0], 9), round(d["xy"][1], 9)) for d in r["false_dets"])
        cmp(f"row fit-validate {r['candidate']} {r['scene_id']} false xy", fx_m, fx_t)

    # ---------------------------------------------------------------------------------- report
    def fmt(p):
        return (f"recall {p['matched']}/{p['visible']} = {p['recall']:.3f} (line {p['line'][0]}/{p['line'][1]}, staff "
                f"{p['staff'][0]}/{p['staff'][1]}, group {p['group'][0]}/{p['group'][1]}, other {p['other'][0]}/"
                f"{p['other'][1]}); false {p['false']}/{p['images']} = {p['fpi']:.3f}; exempt {p['exempt']}; merged "
                f"{p['merged']}/{p['pairs']}; xy median {p['xy_median']:.3f}; non-profile w45 {p['np_w45']}/{p['np_n']}"
                f" = {p['np_share']:.3f}; profile f/b {p['pf_fb']}/{p['pf_n']}; images with false "
                f"{p['images_with_false']}; secondary {p['sec_false']}/{p['sec_exempt']}/{p['sec_moved']}; M3 "
                f"{p['m3_near']} lost per-staff {p['m3_lost_staff']} per-box {p['m3_lost_box']}; fired {p['fired']} "
                f"(A1-matched {p['fired_a1_matched']}); classes {p['classes']}")
    say("")
    say("VERIFIER RE-SCORE (own code; objects under test: A1 pipeline.py, I03 people_scorer.py)")
    for setn in ("fit-train", "fit-validate", "dev", "F-real", "F-ph"):
        say(f"== {setn}")
        for cid, p in summ[setn].items():
            say(f"  {cid} {fmt(p)}")
    say("")
    say(f"Selection on my numbers: eligible {el}; chosen {best} "
        f"(tied at {ft[best]['false']}/84: {[c for c in el if ft[c]['fpi'] == ft[best]['fpi']]}; params "
        f"{[nparam[c] for c in el if ft[c]['fpi'] == ft[best]['fpi']]})")
    say("Fit-validate intervals (mine):")
    for cid in ids2:
        say(f"  {cid}: " + "; ".join(f"{k} {v[0]}/{v[1]} [{v[2][0]:.4f}, {v[2][1]:.4f}]" for k, v in iv[cid].items()))
    say(f"Checks on my numbers: fit-validate holds {holds_fv}; dev holds {holds_dev}; outcome {label}")
    say("")
    say(f"Per-image rows compared: {nimg} (select_rows {len(srows)}, confirm_rows {len(crows)})")
    say(f"DIFFERENCES (label | mine | fix step): {len(DIFFS)}")
    for d in DIFFS:
        say(f"  {d[0]} | {d[1]} | {d[2]}")
    say(f"run time {time.time() - t0:.1f} s")
    (OUT / "v_compare.txt").write_text("\n".join(L) + "\n")
    (OUT / "v_summary.json").write_text(json.dumps({"summ": summ, "eligible": el, "chosen": best, "intervals": iv,
                                                    "outcome": label, "diffs": DIFFS, "pins": V.PIN_RESULT,
                                                    "raw_fit_sha256": raw_sha}, indent=1, default=str))
    (OUT / "v_rows.jsonl").write_text("".join(json.dumps(o, default=float) + "\n" for o in rows_out))


if __name__ == "__main__":
    main()
