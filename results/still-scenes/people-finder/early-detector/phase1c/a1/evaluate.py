"""I09 A1 step 7 (FROZEN.md sections 6-10 and 12): score the 68 evaluation scenes once with the frozen setting and
facing variant (chosen.json + chosen_fix.json), the like-for-like table against the blobs and the six VLM lists, the
A0-matched facing per model, the runtime, and the A1 criteria. Evaluation scenes are read only through
research_harness (distinct_scenes, scene, recorded_config); visibility uses the runs' recorded thresholds.

Refuses to run unless every hash verifies (FROZEN, DEVIATIONS, FIT, CHOSEN, ADDENDUM_FIX, FIT_FIX) and the raw
evaluation pass was made from manifest_eval.json. Writes tables.txt, rows.jsonl (per scene), people_rows.jsonl (per
visible true person), eval_people.json (the frozen people of every evaluation scene, for A2) and summary.json.

--dry-fit: the per-scene scoring and the section-6 tables on 6 FIT scenes (raw_fit), writing nothing.
"""
from __future__ import annotations

import collections
import json
import math
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import pipeline as P  # noqa: E402
import people_scorer as PS  # noqa: E402
import research_harness as H  # noqa: E402
from sgi.geometry import in_image, project  # noqa: E402

RESEARCH = Path('results/2026-10-17-beyond-vlm')
I03A = RESEARCH / 'phase1b/i03/stageA'
A0_PAIRS = RESEARCH / 'phase1/a0_facing/a0_pairs.jsonl'
VLM_LABELS = ("Qwen 3.8", "GLM-5.3", "DeepSeek V4.1", "Haiku 5.5", "Sonnet 5.5 low", "Gemini 3.8 low")
A0_LABEL = {"Qwen 3.8 (s19)": "Qwen 3.8", "GLM-5.3 (s19)": "GLM-5.3", "DeepSeek V4.1 (s20)": "DeepSeek V4.1",
            "Haiku 5.5 (s23)": "Haiku 5.5", "Sonnet 5.5 low (s23)": "Sonnet 5.5 low",
            "Gemini 3.8 low (s23)": "Gemini 3.8 low"}
KILL_FACING = 0.806
SCHOOL_FACING = 0.70
L = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    L.append(s)


# ------------------------------------------------------------------------------------------------ frozen inputs
def frozen_final():
    hashes = C.verify_frozen()
    for sha_file in ("FIT.sha256", "FIT_FIX.sha256"):
        for line in (HERE / sha_file).read_text().splitlines():
            want, name = line.split()
            assert C.sha(HERE / name) == want, name
            hashes[name] = want
    assert (HERE / "FIT.sha256").read_text() == (HERE / "CHOSEN.sha256").read_text()
    hashes["ADDENDUM_FIX.md"] = C.verify(HERE / "ADDENDUM_FIX.md", HERE / "ADDENDUM_FIX.sha256")
    chosen = json.loads((HERE / "chosen.json").read_text())
    fix = json.loads((HERE / "chosen_fix.json").read_text())
    assert fix["setting"] == chosen["setting"]
    return chosen["setting"], fix["facing_variant"], hashes, chosen, fix


# ------------------------------------------------------------------------------------------------ one scene
def filter_trace(cands, s):
    lo, hi = P.BANDS[s["band"]]
    out = collections.Counter()
    for c in cands:
        if c["reject"] is not None:
            out["depth: " + c["reject"]] += 1
        elif c["z_top"] > hi:
            out["z_top above band"] += 1
        elif not c["top_touch"] and c["z_top"] < lo:
            out["z_top below band"] += 1
        elif s["flat"] and c["flat"]:
            out["flat"] += 1
        else:
            out["passed filters (kept or duplicate)"] += 1
    return out


def score_one(raw, sc, th, s, variant, meta):
    obs = sc.observation
    cam = obs.camera
    cam_xy = cam.T_world_camera[:2, 3]
    cands = P.candidates(raw, obs.depth, cam, s["tau"])
    ppl = P.people_for(cands, s, cam_xy, variant)
    r, prow = C.score_scene(ppl, sc, th, variant)
    info = PS.truth_info(sc, th)
    lh = info["lower_hidden"]
    vis = info["visible"]
    false_rows = []
    for n in r["false_numbers"] + r["exempt_numbers"]:
        d = ppl[n - 1]
        near_vis = min((math.dist(d["xy"], (p.x, p.y)) for p in vis), default=None)
        false_rows.append({"number": n, "exempt": n in r["exempt_numbers"], "xy": d["xy"], "score": d["score"],
                           "box": d["box"], "nearest_visible_m": near_vis,
                           "near_hidden": any(math.dist(d["xy"], (p.x, p.y)) <= PS.NEAR_M for p in info["hidden"])})
    row = dict(meta) | {k: r[k] for k in PS.POOL_KEYS} | {
        "recall": r["recall"], "xy_err": r["xy_err"], "facing": r["facing"], "misses": r["misses"],
        "false_detail": false_rows, "pairs": r["pairs"], "people": r["people"], "matches": r["matches"],
        "n_candidates": len(cands), "filter_trace": dict(filter_trace(cands, s)), "n_kept": len(ppl)}
    people_rows = []
    for pr in prow:
        people_rows.append(dict(meta) | pr | {"lower_hidden": bool(lh[pr["pid"]]),
                                              "staff_behind": bool(pr["role"] and lh[pr["pid"]])})
    frozen = []
    for p in ppl:
        mk = project(cam, (p["xy"][0], p["xy"][1], 1.1))
        mark = None if mk is None or not in_image(cam, mk[0], mk[1]) else [mk[0], mk[1]]
        frozen.append({"x": p["xy"][0], "y": p["xy"][1], "yaw": p["yaw"], "score": p["score"], "box": p["box"],
                       "torso_pixel": list(p["torso_pixel"]), "mark_pixel": mark,
                       "box_centre": [0.5 * (p["box"][0] + p["box"][2]), 0.5 * (p["box"][1] + p["box"][3])],
                       "yaws_all_variants": p["yaws"]})
    return row, people_rows, frozen


# ------------------------------------------------------------------------------------------------ tables
def fmt_k(k, n):
    if not n:
        return f"{k}/{n} -"
    lo, hi = PS.wilson(k, n)
    return f"{k}/{n} = {k / n:.3f} [{lo:.3f}, {hi:.3f}]"


def f3(x):
    return "-" if x is None else f"{x:.3f}"


def pool_rows(rows):
    Pp = PS.pool(rows)
    Pp["by_role"] = {c: [sum(1 for p in Pp["people"] if p["role_class"] == c and p["matched"]),
                         sum(1 for p in Pp["people"] if p["role_class"] == c)] for c in C.ROLE_CLASSES}
    cls = collections.Counter()
    for m in Pp["misses"]:
        cls["staff behind counters" if m["staff_behind"] else "other staff" if m["role"] else "non-staff"] += 1
    Pp["miss_class"] = cls
    return Pp


def detection_table(groups):
    L.append("group | scenes | visible | recall k/n [Wilson 95%] | false (per image) | raw false (incl. exempt) | "
             "exempt | near-hidden false | merges | merged pairs k/n | median xy err m (n) | misses: staff behind "
             "counters / other staff / non-staff")
    for name, Pp in groups:
        mc = Pp["miss_class"]
        xy = "-" if Pp["xy_err_median"] is None else f"{Pp['xy_err_median']:.3f}"
        L.append(f"{name:12s} | {Pp['scenes']:3d} | {Pp['n_visible']:3d} | {fmt_k(Pp['matched'], Pp['n_visible'])} | "
                 f"{Pp['false']} ({Pp['fb_per_scene']:.3f}) | {Pp['false_raw']} | {Pp['exempt']} | {Pp['near_hidden']} "
                 f"| {Pp['merges']} | {Pp['merged_pairs']}/{Pp['neighbour_pairs']} | {xy} ({len(Pp['xy_err'])}) | "
                 f"{mc['staff behind counters']} / {mc['other staff']} / {mc['non-staff']}")
    L.append("recall by role (line member / group member / staff / other), k/n:")
    for name, Pp in groups:
        L.append(f"  {name:12s} " + " | ".join(f"{c} {Pp['by_role'][c][0]}/{Pp['by_role'][c][1]}" for c in C.ROLE_CLASSES))


def facing_groups(prows, variant_key="err"):
    m = [p for p in prows if p["matched"]]
    out = {vc: C.facing_summary([p[variant_key] if variant_key == "err" else p["errs"][variant_key]
                                 for p in m if p["view"] == vc]) for vc in C.VIEW_CLASSES}
    out["non-profile"] = C.facing_summary([p[variant_key] if variant_key == "err" else p["errs"][variant_key]
                                           for p in m if p["view"] != "profile"])
    out["all matched"] = C.facing_summary([p[variant_key] if variant_key == "err" else p["errs"][variant_key]
                                           for p in m])
    return out


def facing_table(name, fg):
    L.append(f"{name}: view class | matched n | coverage (known) | within 45, unknown wrong | within 45 among known | "
             "median deg, unknown = 180 | median among known | front/back (< 90), unknown wrong | front/back among "
             "known | axis within 45, unknown wrong | axis within 45 among known | axis median, unknown = 180")
    for vc in C.VIEW_CLASSES + ("non-profile", "all matched"):
        f = fg[vc]
        if not f["n"]:
            L.append(f"  {vc:12s} | 0")
            continue
        L.append(f"  {vc:12s} | {f['n']:3d} | {f['known']}/{f['n']} | {f['w45']}/{f['n']} = {f['w45_share']:.3f} | "
                 f"{f3(f['w45_known_share'])} | {f['median_u']:.1f} | {f3(f['median_known'])} | {f['fb']}/{f['n']} = "
                 f"{f['fb_share']:.3f} | {f3(f['fb_known_share'])} | {f['axis_w45']}/{f['n']} = "
                 f"{f['axis_w45_share']:.3f} | {f3(f['axis_w45_known_share'])} | {f['axis_median_u']:.1f}")


# ------------------------------------------------------------------------------------------------ like for like
def like_for_like(vrows, det_by_folder, blob_by_folder):
    L.append("== Like for like (FROZEN.md section 7; phase1b/README.md Table 1.2): one answer per (arm, scene, trial) "
             "with a people list; the detector and the blobs re-weighted to the same keys ==")
    L.append("model | domain | keys (no list) | visible || VLM: recall [Wilson] | false/answer | merged pairs | full "
             "w45 known (unknown) | axis w45/known || detector: recall [Wilson] | false/key | merged pairs | full w45 "
             "unknown-wrong (among known) | axis w45/known || blobs: recall | false/key | merged pairs | axis w45/known")
    out = {}
    for label in VLM_LABELS:
        for dom in C.EVAL_DOMAINS:
            rows = [r for r in vrows if r["model"] == label and r["domain"] == dom]
            if not rows:
                continue
            listed = [r for r in rows if r["has_list"]]
            V = PS.pool(listed)
            D = PS.pool([det_by_folder[r["folder"]] for r in listed])
            B = PS.pool([blob_by_folder[r["folder"]] for r in listed])
            vf, df, bf = V["facing"], D["facing"], B["facing"]
            L.append(f"{label:15s} | {dom:6s} | {len(rows)} ({len(rows) - len(listed)}) | {V['n_visible']} || "
                     f"{fmt_k(V['matched'], V['n_visible'])} | {V['false'] / max(V['scenes'], 1):.3f} | "
                     f"{V['merged_pairs']}/{V['neighbour_pairs']} | {vf['within45_full']}/{vf['known']} "
                     f"({vf['unknown']}) | {vf['within45_axis']}/{vf['known']} || {fmt_k(D['matched'], D['n_visible'])} "
                     f"| {D['false'] / max(D['scenes'], 1):.3f} | {D['merged_pairs']}/{D['neighbour_pairs']} | "
                     f"{df['within45_full']}/{df['matched']} = {df['within45_full'] / max(df['matched'], 1):.3f} "
                     f"({df['within45_full']}/{df['known']}) | {df['within45_axis']}/{df['known']} || "
                     f"{fmt_k(B['matched'], B['n_visible'])} | {B['false'] / max(B['scenes'], 1):.3f} | "
                     f"{B['merged_pairs']}/{B['neighbour_pairs']} | {bf['within45_axis']}/{bf['known']}")
            S = PS.pool(rows)
            SD = PS.pool([det_by_folder[r["folder"]] for r in rows])
            SB = PS.pool([blob_by_folder[r["folder"]] for r in rows])
            if len(rows) != len(listed):
                L.append(f"{'':15s}   sensitivity, keys without a list as empty lists: VLM recall "
                         f"{fmt_k(S['matched'], S['n_visible'])}, false/answer {S['false'] / max(S['scenes'], 1):.3f}; "
                         f"detector over all keys {fmt_k(SD['matched'], SD['n_visible'])}, false/key "
                         f"{SD['false'] / max(SD['scenes'], 1):.3f}; blobs {fmt_k(SB['matched'], SB['n_visible'])}, "
                         f"false/key {SB['false'] / max(SB['scenes'], 1):.3f}")
            out[f"{label}|{dom}"] = {
                "keys": len(rows), "keys_without_list": len(rows) - len(listed),
                "vlm": {"matched": V["matched"], "visible": V["n_visible"], "false": V["false"], "scenes": V["scenes"],
                        "merged_pairs": [V["merged_pairs"], V["neighbour_pairs"]],
                        "facing_full": [vf["within45_full"], vf["known"], vf["unknown"]],
                        "facing_axis": [vf["within45_axis"], vf["known"]]},
                "detector": {"matched": D["matched"], "visible": D["n_visible"], "false": D["false"],
                             "scenes": D["scenes"], "merged_pairs": [D["merged_pairs"], D["neighbour_pairs"]],
                             "facing_full": [df["within45_full"], df["known"], df["matched"]],
                             "facing_axis": [df["within45_axis"], df["known"]]},
                "blobs": {"matched": B["matched"], "visible": B["n_visible"], "false": B["false"], "scenes": B["scenes"],
                          "merged_pairs": [B["merged_pairs"], B["neighbour_pairs"]],
                          "facing_axis": [bf["within45_axis"], bf["known"]]},
                "sensitivity_empty": {"vlm": [S["matched"], S["n_visible"], S["false"], S["scenes"]],
                                      "detector": [SD["matched"], SD["n_visible"], SD["false"], SD["scenes"]],
                                      "blobs": [SB["matched"], SB["n_visible"], SB["false"], SB["scenes"]]}}
    return out


# ------------------------------------------------------------------------------------------------ A0-matched facing
def a0_matched(person_by_key, key="err", label_suffix=""):
    pairs = [json.loads(x) for x in A0_PAIRS.read_text().splitlines()]
    L.append(f"== A0-matched facing (FROZEN.md section 8){label_suffix}: the detector's yaw for the same true person "
             "on each model's A0 pairs (phase1/a0_facing/a0_pairs.jsonl; arm, scene, pid) ==")
    L.append("model | pairs (instances, people, layouts) | detector-matched pairs (unmatched = recall misses) | "
             "coverage (matched, known yaw) | PRIMARY within 45 over detector-matched, unknown wrong [layout bootstrap] "
             "| among known | all pairs, unmatched wrong | distinct instances (det-matched) | model's own on the same "
             "subset, unknown wrong [bootstrap] (among known) | model's own on all its pairs")
    out = {}
    for row_label, label in A0_LABEL.items():
        ps = [p for p in pairs if p["row"] == row_label]
        for p in ps:
            assert (p["scene"], p["pid"]) in person_by_key, (p["scene"], p["pid"])
        dm = [p for p in ps if person_by_key[(p["scene"], p["pid"])]["matched"]]

        def det_err(p):
            pr = person_by_key[(p["scene"], p["pid"])]
            return pr[key] if key == "err" else pr["errs"][key]

        def w(e):
            return e is not None and e <= 45

        k = sum(w(det_err(p)) for p in dm)
        known = [p for p in dm if det_err(p) is not None]
        kk = sum(w(det_err(p)) for p in known)
        inst = {}
        for p in dm:
            inst.setdefault((p["arm"], p["scene"], p["pid"]), w(det_err(p)))
        own_k = sum(w(p["err"]) for p in dm)
        own_known = [p for p in dm if p["err"] is not None]
        own_all = sum(w(p["err"]) for p in ps)
        groups = collections.defaultdict(list)
        for p in dm:
            groups[p["layout"]].append(p)
        ci_det = C.boot_ci(groups, lambda d: sum(w(det_err(p)) for p in d) / len(d) if d else None)
        ci_own = C.boot_ci(groups, lambda d: sum(w(p["err"]) for p in d) / len(d) if d else None)
        n_inst = len({(p["arm"], p["scene"], p["pid"]) for p in ps})
        n_ppl = len({(p["layout"], p["pid"]) for p in ps})
        n_lay = len({p["layout"] for p in ps})
        primary = k / len(dm) if dm else None
        L.append(f"{label:15s} | {len(ps)} ({n_inst}, {n_ppl}, {n_lay}) | {len(dm)} ({len(ps) - len(dm)}) | "
                 f"{len(known)}/{len(ps)} | {k}/{len(dm)} = {f3(primary)} [{f3(ci_det[0])}, {f3(ci_det[1])}] | "
                 f"{kk}/{len(known)} = {f3(kk / len(known) if known else None)} | {k}/{len(ps)} = "
                 f"{f3(k / len(ps) if ps else None)} | {sum(inst.values())}/{len(inst)} = "
                 f"{f3(sum(inst.values()) / len(inst) if inst else None)} | {own_k}/{len(dm)} = "
                 f"{f3(own_k / len(dm) if dm else None)} [{f3(ci_own[0])}, {f3(ci_own[1])}] "
                 f"({sum(w(p['err']) for p in own_known)}/{len(own_known)}) | {own_all}/{len(ps)} = "
                 f"{f3(own_all / len(ps) if ps else None)}")
        byv = {}
        for vc in C.VIEW_CLASSES:
            sub = [p for p in dm if p["view"] == vc]
            byv[vc] = [sum(w(det_err(p)) for p in sub), len(sub), sum(w(p["err"]) for p in sub)]
        L.append(f"{'':15s}   by A0 view class, detector k/n (model's own k): " +
                 "; ".join(f"{vc} {v[0]}/{v[1]} ({v[2]})" for vc, v in byv.items()))
        out[label] = {"pairs": len(ps), "instances": n_inst, "people": n_ppl, "layouts": n_lay,
                      "det_matched": len(dm), "unmatched": len(ps) - len(dm), "coverage_known": len(known),
                      "primary": [k, len(dm)], "primary_value": primary, "primary_ci": ci_det,
                      "among_known": [kk, len(known)], "all_pairs_unmatched_wrong": [k, len(ps)],
                      "distinct_instances": [sum(inst.values()), len(inst)],
                      "own_same_subset": [own_k, len(dm)], "own_ci": ci_own,
                      "own_same_subset_known": [sum(w(p["err"]) for p in own_known), len(own_known)],
                      "own_all": [own_all, len(ps)], "by_view": byv}
    return out


# ------------------------------------------------------------------------------------------------ runtime
def runtime(s):
    tdir = HERE / "timing"
    meta = json.loads((tdir / "meta.json").read_text())
    tim = [json.loads(x) for x in (tdir / "timing.jsonl").read_text().splitlines()]
    proj = json.loads((tdir / "project_timing.json").read_text())
    raw_e = {x["scene_id"]: x for x in map(json.loads, (HERE / "raw_eval" / f"{s['detector']}.jsonl").read_text().splitlines())}
    max_kp, max_box, agree = 0.0, 0.0, 0
    for t in tim:
        r = raw_e[t["scene_id"]]
        rb = [b for b in r["boxes"] if b[4] >= s["tau"]]
        assert len(rb) == len(t["boxes"]), t["scene_id"]
        kp = {p["box_index"]: p["kp"] for p in r["pose"]}
        mb = max([abs(a - b) for x, y in zip(rb, t["boxes"]) for a, b in zip(x[:4], y[:4])], default=0.0)
        mk = max([max(abs(a[0] - b[0]), abs(a[1] - b[1])) for i in range(len(rb)) for a, b in zip(kp[i], t["kp"][i])],
                 default=0.0)
        max_box, max_kp = max(max_box, mb), max(max_kp, mk)
        agree += mb <= 1e-3 and mk <= 1.0
    ts = [t["t_s"] for t in tim]
    out = {"n": len(ts), "sidecar_median_s": statistics.median(ts), "sidecar_p90_s": C.p90(ts),
           "project_median_s": proj["median_s"], "project_p90_s": proj["p90_s"],
           "sum_median_s": statistics.median(ts) + proj["median_s"], "sum_p90_s": C.p90(ts) + proj["p90_s"],
           "read_decode_median_s": statistics.median(t["t_read_decode_s"] for t in tim),
           "peak_mem_allocated_MiB": meta["peak_mem_allocated_MiB"], "peak_mem_reserved_MiB": meta["peak_mem_reserved_MiB"],
           "peak_nvidia_smi_MiB": meta["peak_nvidia_smi_MiB"], "gpu": meta["gpu"], "driver": meta["driver"],
           "torch": meta["torch"], "cuda": meta["cuda"], "kp_agree_images": [agree, len(tim)],
           "kp_max_diff_px": max_kp, "box_max_diff_px": max_box,
           "deployed_equals_eval_path": proj["deployed_equals_eval_path"], "threads": proj["threads"]}
    return out


# ------------------------------------------------------------------------------------------------ main
def bars_of(Pp, fg, rt=None, a0_gemini=None):
    rec = Pp["matched"] / Pp["n_visible"]
    line = Pp["by_role"]["line"]
    fb = Pp["fb_per_scene"]
    med = Pp["xy_err_median"]
    npf, pf = fg["non-profile"], fg["profile"]
    succ = {"recall>=0.95": rec >= 0.95, "line recall>=0.95": (line[0] / line[1] >= 0.95) if line[1] else None,
            "false<=0.10": fb <= 0.10, "median xy<=0.15": med <= 0.15,
            "non-profile median<=25": npf["median_u"] <= 25 if npf["n"] else None,
            "non-profile within45>=0.85": npf["w45_share"] >= 0.85 if npf["n"] else None,
            "profile front/back>=0.90": pf["fb_share"] >= 0.90 if pf["n"] else None,
            "runtime<=1.0 s": None if rt is None else rt["sum_median_s"] <= 1.0}
    kill = {"recall<0.90": rec < 0.90, "false>0.30": fb > 0.30,
            "A0-matched facing (Gemini pairs) < 0.806": None if a0_gemini is None else a0_gemini < KILL_FACING}
    return succ, kill


def main():
    s, variant, hashes, chosen, fix = frozen_final()
    dry = "--dry-fit" in sys.argv
    if dry:
        import fitdata as FD
        raw = {(x["set"], x["scene_id"]): x for x in map(json.loads, (HERE / "raw_fit" / f"{s['detector']}.jsonl").read_text().splitlines())}
        refs = FD.fit_refs()
        picks = refs[:3] + [r for r in refs if r[0] == "F-ph"][:3]
        rows, prows = [], []
        for domain, sset, sid, loader in picks:
            sc, th = FD.load(loader)
            row, pr, fz = score_one(raw[(sset, sid)], sc, th, s, variant, {"domain": domain, "set": sset,
                                                                          "scene_id": sid, "layout": C.layout_of(sid)})
            rows.append(row)
            prows += pr
            print(sid, row["n_kept"], row["matched"], row["n_visible"], row["false"], row["filter_trace"], fz[:1])
        detection_table([("dry", pool_rows(rows))])
        facing_table("dry", facing_groups(prows))
        facing_table("dry F4", facing_groups(prows, "F4"))
        print("\n".join(L))
        return

    t0 = time.monotonic()
    man = json.loads((HERE / "manifest_eval.json").read_text())
    meta_raw = json.loads((HERE / "raw_eval" / "meta.json").read_text())
    assert meta_raw["manifest_sha256"] == C.sha(HERE / "manifest_eval.json")
    assert meta_raw["detectors"] == [s["detector"]]
    raw = {x["rgb"]: x for x in map(json.loads, (HERE / "raw_eval" / f"{s['detector']}.jsonl").read_text().splitlines())}
    by_folder = {e["folder"]: e for e in man["images"]}
    distinct = H.distinct_scenes()
    assert len(distinct) == 68 and {str(d.folder) for d in distinct} == set(by_folder)
    L.append("I09 A1 evaluation (FROZEN.md sections 6-10), scored once with the frozen setting and facing variant.")
    L.append(f"Setting: {C.setting_key(s)}; band {P.BANDS[s['band']]} m; facing variant {variant} "
             f"(ADDENDUM_FIX.md; fix adopted: {fix['fix_adopted']}). Detector run on evaluation images: {s['detector']} "
             f"only. Started {time.strftime('%Y-%m-%dT%H:%M:%S%z')}.")
    L.append(f"Hashes verified: {json.dumps(hashes)}")
    L.append("Definitions: recall = matched / visible (sgi.run._matched_people via phase1b/i03/people_scorer.py, 0.5 m); "
             "false = unmatched detection not within 0.5 m of a non-person target anchor (exempt counted apart); "
             "facing error = |wrap(yaw_det - yaw_true)| on matched people; view class from the angle between the true "
             "yaw and the bearing from the true person to the camera xy (frontal < 45, oblique 45-75, profile 75-105 "
             "inclusive, back > 105); non-profile = frontal + oblique + back; front/back correct = error < 90; unknown "
             "counted as wrong (180 deg in medians) unless 'among known'.")
    L.append("")
    rows, prows, frozen = [], [], {}
    for sref in distinct:
        e = by_folder[str(sref.folder)]
        th = C.thresholds_for(sref.runs)
        sc = H.scene(sref.runs[0], sref.scene_id)
        meta = {"set": e["set"], "domain": e["domain"], "scene_id": sref.scene_id, "layout": C.layout_of(sref.scene_id),
                "folder": str(sref.folder), "run_used": sref.runs[0].folder}
        row, pr, fz = score_one(raw[e["rgb"]], sc, th, s, variant, meta)
        rows.append(row)
        prows += pr
        frozen[str(sref.folder)] = {"scene_id": sref.scene_id, "set": e["set"], "domain": e["domain"],
                                    "listed": e["listed"], "rgb": e["rgb"], "people": fz}
        print(f"{e['set']:12s} {sref.scene_id:38s} kept {row['n_kept']:2d} matched {row['matched']}/{row['n_visible']} "
              f"false {row['false']} exempt {row['exempt']}", flush=True)
    assert collections.Counter(r["domain"] for r in rows) == {"E-real": 42, "E-ph": 26}
    with open(HERE / "rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    with open(HERE / "people_rows.jsonl", "w") as fh:
        for p in prows:
            fh.write(json.dumps(p, default=str) + "\n")
    C.jdump({"setting": s, "facing_variant": variant, "person_mark_height_m": 1.1,
             "note": "people in descending detector score; yaw is the frozen variant's (None allowed); mark_pixel = "
                     "projection of (x, y, 1.1 m), None when behind the camera or outside the image; torso_pixel = "
                     "median (u, v) of the depth-band pixels (DEVIATIONS.md C5, C17)",
             "scenes": frozen}, HERE / "eval_people.json")

    # ---------------------------------------------------------------- section 6
    groups_set = [(ss, pool_rows([r for r in rows if r["set"] == ss])) for ss in C.SETS]
    groups_dom = [(d, pool_rows([r for r in rows if r["domain"] == d])) for d in C.EVAL_DOMAINS]
    pooled = ("pooled 68", pool_rows(rows))
    L.append("== Detection, per set (FROZEN.md section 6) ==")
    detection_table(groups_set)
    L.append("")
    L.append("== Detection, per domain (E-real is the verdict domain) and pooled ==")
    detection_table(groups_dom + [pooled])
    real_rows = [r for r in rows if r["domain"] == "E-real"]
    by_layout = collections.defaultdict(list)
    for r in real_rows:
        by_layout[r["layout"]].append(r)
    ci_rec = C.boot_ci(by_layout, lambda d: sum(r["matched"] for r in d) / sum(r["n_visible"] for r in d))
    real_prows = [p for p in prows if p["domain"] == "E-real"]
    pl = collections.defaultdict(list)
    for p in real_prows:
        pl[p["layout"]].append(p)
    ci_line = C.boot_ci(pl, lambda d: (sum(p["matched"] for p in d if p["role_class"] == "line") /
                                       max(1, sum(1 for p in d if p["role_class"] == "line"))))
    ci_fb = C.boot_ci(by_layout, lambda d: sum(r["false"] for r in d) / len(d))
    L.append(f"E-real layout cluster bootstrap (14 layouts, 2000 draws, seed 0): recall [{ci_rec[0]:.3f}, {ci_rec[1]:.3f}]; "
             f"line recall [{ci_line[0]:.3f}, {ci_line[1]:.3f}]; false per image [{ci_fb[0]:.3f}, {ci_fb[1]:.3f}]")
    L.append("")
    L.append("Missed visible people and false detections, E-real (per scene):")
    for r in real_rows:
        if r["misses"] or r["false"] or r["exempt"]:
            L.append(f"  {r['scene_id']}: misses {[(m['id'], m['role_class'], 'lower hidden' if m['lower_hidden'] else '') for m in r['misses']]}; "
                     f"false/exempt {[(fd['number'], 'exempt' if fd['exempt'] else 'false', round(fd['score'], 3), None if fd['nearest_visible_m'] is None else round(fd['nearest_visible_m'], 2), 'near hidden' if fd['near_hidden'] else '') for fd in r['false_detail']]}")
    L.append("")
    L.append(f"== Facing, all matched people by view class (FROZEN.md section 6), variant {variant} ==")
    fg = {}
    for name in C.SETS:
        fg[name] = facing_groups([p for p in prows if p["set"] == name])
    for name in C.EVAL_DOMAINS:
        fg[name] = facing_groups([p for p in prows if p["domain"] == name])
    fg["pooled 68"] = facing_groups(prows)
    for name in C.EVAL_DOMAINS + ("pooled 68",) + C.SETS:
        facing_table(name, fg[name])
    mreal = collections.defaultdict(list)
    for p in real_prows:
        if p["matched"]:
            mreal[p["layout"]].append(p)

    def st(view_sel, field):
        def f(d):
            sub = [p["err"] for p in d if view_sel(p["view"])]
            return C.facing_summary(sub)[field] if sub else None
        return f

    ci_np_w = C.boot_ci(mreal, st(lambda v: v != "profile", "w45_share"))
    ci_np_m = C.boot_ci(mreal, st(lambda v: v != "profile", "median_u"))
    ci_pf = C.boot_ci(mreal, st(lambda v: v == "profile", "fb_share"))
    ci_all = C.boot_ci(mreal, st(lambda v: True, "w45_share"))
    L.append(f"E-real facing, layout cluster bootstrap (2000 draws, seed 0): non-profile within 45 [{f3(ci_np_w[0])}, "
             f"{f3(ci_np_w[1])}]; non-profile median [{f3(ci_np_m[0])}, {f3(ci_np_m[1])}] deg; profile front/back "
             f"[{f3(ci_pf[0])}, {f3(ci_pf[1])}]; all matched within 45 [{f3(ci_all[0])}, {f3(ci_all[1])}]")
    cov = sum(1 for p in real_prows if p["matched"] and p["yaw"] is not None)
    L.append(f"E-real facing coverage (matched people with a known yaw): {cov}/{sum(p['matched'] for p in real_prows)}")
    L.append("")

    # ---------------------------------------------------------------- section 7
    vrows = [json.loads(x) for x in (I03A / "vlm_rows.jsonl").read_text().splitlines()]
    brows = [json.loads(x) for x in (I03A / "rows.jsonl").read_text().splitlines()]
    blob_by_folder = {r["folder"]: r for r in brows}
    det_by_folder = {r["folder"]: r for r in rows}
    assert set(blob_by_folder) == set(det_by_folder)
    lfl = like_for_like(vrows, det_by_folder, blob_by_folder)
    L.append("")

    # ---------------------------------------------------------------- section 8
    person_by_key = {(p["scene_id"], p["pid"]): p for p in real_prows}
    a0 = a0_matched(person_by_key)
    gem = a0["Gemini 3.8 low"]
    L.append(f"Facing kill population, Gemini's pairs: primary {gem['primary'][0]}/{gem['primary'][1]} = "
             f"{f3(gem['primary_value'])} against 0.806 (fires below); school-panel reading against 0.70: "
             f"{'below' if gem['primary_value'] < SCHOOL_FACING else 'not below'}")
    L.append("")

    # ---------------------------------------------------------------- section 9
    rt = runtime(s)
    L.append("== Runtime (FROZEN.md section 9; DEVIATIONS.md C15) ==")
    L.append(f"GPU {rt['gpu']}, driver {rt['driver']}, torch {rt['torch']} (CUDA {rt['cuda']}); detector "
             f"{s['detector']} at tau {s['tau']}, ViTPose on boxes >= tau, batch 1, fp32; 5 warm-ups on F-real images; "
             f"each of the {rt['n']} E-real images timed once.")
    L.append(f"sidecar (decoded image -> keypoints on CPU, synchronised): median {rt['sidecar_median_s'] * 1000:.1f} ms, "
             f"p90 {rt['sidecar_p90_s'] * 1000:.1f} ms (file read + decode, not counted: median "
             f"{rt['read_decode_median_s'] * 1000:.1f} ms)")
    L.append(f"project side (steps 1-7 + {variant}, CPU, threads {rt['threads']}): median "
             f"{rt['project_median_s'] * 1000:.1f} ms, p90 {rt['project_p90_s'] * 1000:.1f} ms")
    L.append(f"criterion: median sum {rt['sum_median_s'] * 1000:.1f} ms (<= 1000 ms: {rt['sum_median_s'] <= 1.0}); p90 sum "
             f"{rt['sum_p90_s'] * 1000:.1f} ms; peak GPU memory allocated {rt['peak_mem_allocated_MiB']} MiB (reserved "
             f"{rt['peak_mem_reserved_MiB']}; nvidia-smi poll {rt['peak_nvidia_smi_MiB']} MiB total in use)")
    L.append(f"keypoints of the timing pass vs the raw evaluation pass: {rt['kp_agree_images'][0]}/{rt['kp_agree_images'][1]} "
             f"images within 1 px (max diff {rt['kp_max_diff_px']:.4f} px; boxes max diff {rt['box_max_diff_px']:.2e} px); "
             f"deployed-path people equal the evaluation path's in {rt['deployed_equals_eval_path'][0]}/"
             f"{rt['deployed_equals_eval_path'][1]} images")
    L.append("")

    # ---------------------------------------------------------------- section 10
    Pd = dict(groups_dom)
    succ_r, kill_r = bars_of(Pd["E-real"], fg["E-real"], rt, gem["primary_value"])
    succ_p, kill_p = bars_of(Pd["E-ph"], fg["E-ph"], None, None)
    success = all(v is True for v in succ_r.values())
    kill = any(v is True for v in kill_r.values())
    verdict = "success" if success else "kill" if kill else "neither"
    L.append("== A1 criteria (FROZEN.md section 10), judged on E-real; E-ph reported against the same bars ==")
    Pr = Pd["E-real"]
    L.append(f"E-real values: recall {Pr['matched']}/{Pr['n_visible']} = {Pr['matched'] / Pr['n_visible']:.4f}; line "
             f"{Pr['by_role']['line'][0]}/{Pr['by_role']['line'][1]}; false {Pr['false']}/{Pr['scenes']} = "
             f"{Pr['fb_per_scene']:.4f} per image; median xy {Pr['xy_err_median']:.4f} m (n {len(Pr['xy_err'])}); "
             f"non-profile median {fg['E-real']['non-profile']['median_u']:.2f} deg, within 45 "
             f"{fg['E-real']['non-profile']['w45']}/{fg['E-real']['non-profile']['n']}; profile front/back "
             f"{fg['E-real']['profile']['fb']}/{fg['E-real']['profile']['n']}; runtime {rt['sum_median_s'] * 1000:.1f} ms; "
             f"A0-matched facing on Gemini's pairs {f3(gem['primary_value'])}")
    L.append(f"E-real success bars: {succ_r}")
    L.append(f"E-real kill bars: {kill_r}")
    Pp_ = Pd["E-ph"]
    L.append(f"E-ph values: recall {Pp_['matched']}/{Pp_['n_visible']}; line {Pp_['by_role']['line']}; false "
             f"{Pp_['false']}/{Pp_['scenes']}; median xy {Pp_['xy_err_median']:.4f}; non-profile median "
             f"{fg['E-ph']['non-profile']['median_u']}, within 45 {fg['E-ph']['non-profile']['w45']}/"
             f"{fg['E-ph']['non-profile']['n']}; profile front/back {fg['E-ph']['profile']['fb']}/{fg['E-ph']['profile']['n']}")
    L.append(f"E-ph success bars (runtime is E-real only): {succ_p}")
    L.append(f"E-ph kill bars (the facing clause is not judged on E-ph): {kill_p}")
    L.append(f"A1 VERDICT (E-real): {verdict}")
    L.append("")

    # ---------------------------------------------------------------- post hoc (labelled)
    L.append("== POST HOC (not pre-registered; reported for transparency, changes no verdict) ==")
    L.append("F4, the pre-fix facing choice (chosen.json), on the same matched E-real people:")
    facing_table("E-real F4 (post hoc)", facing_groups(real_prows, "F4"))
    a0_f4 = a0_matched(person_by_key, key="F4", label_suffix=" with F4 (POST HOC)")
    L.append(f"{time.monotonic() - t0:.0f} s; finished {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    (HERE / "tables.txt").write_text("\n".join(L) + "\n")

    def jp(Pp):
        return {k: Pp[k] for k in ("scenes", "n_visible", "matched", "recall", "false", "fb_per_scene", "false_raw",
                                   "exempt", "near_hidden", "merges", "neighbour_pairs", "merged_pairs",
                                   "xy_err_median", "staff_behind")} | {
            "n_xy": len(Pp["xy_err"]), "by_role": Pp["by_role"], "miss_class": dict(Pp["miss_class"]),
            "recall_wilson": PS.wilson(Pp["matched"], Pp["n_visible"])}
    summary = {"setting": s, "facing_variant": variant, "hashes": hashes,
               "per_set": {n: jp(Pp) for n, Pp in groups_set}, "per_domain": {n: jp(Pp) for n, Pp in groups_dom},
               "pooled": jp(pooled[1]), "facing": fg,
               "e_real_bootstrap": {"recall": ci_rec, "line_recall": ci_line, "false_per_image": ci_fb,
                                    "non_profile_w45": ci_np_w, "non_profile_median": ci_np_m,
                                    "profile_fb": ci_pf, "all_w45": ci_all},
               "like_for_like": lfl, "a0_matched": a0, "a0_matched_F4_posthoc": a0_f4, "runtime": rt,
               "criteria": {"E-real": {"success": succ_r, "kill": kill_r},
                            "E-ph": {"success": succ_p, "kill": kill_p}}, "verdict": verdict}
    C.jdump(summary, HERE / "summary.json")
    print("\n".join(L[-40:]))


if __name__ == "__main__":
    main()
