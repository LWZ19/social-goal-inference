"""I03 Stage A steps 4-6: score the 68 evaluation scenes once with the frozen setting, score the recorded VLM people
lists on the same scenes with the same scorer, and apply the Stage A criteria (FROZEN.md sections 4-6).

Refuses to run unless ../FIT.sha256 verifies ../fit_selection.json and chosen.json names the same setting. Evaluation
scenes are read only through research_harness (H.distinct_scenes, H.scene, H.records); visibility uses each run's
recorded thresholds (asserted equal on the relevant keys across a scene's runs).

Writes tables.txt, rows.jsonl (one row per evaluation scene, blobs), vlm_rows.jsonl (one row per VLM answer key),
summary.json. --dry-fit runs the blob path on 6 fit scenes only, writing nothing (a code check before the one
evaluation pass).
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
I03 = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(I03))
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H  # noqa: E402
import people_scorer as PS  # noqa: E402
import blobs as B  # noqa: E402

REAL = ("real", "staffed", "plain")
DROP_CLIP = None                                   # set from chosen.json by frozen()
SETS = ("real", "staffed", "plain", "placeholder")
DOMAINS = ("E-real", "E-ph")
ROLE_CLASSES = ("line", "group", "staff", "other")
DIAG = ("no height-map cells within 0.25 m", "rejected: top above the clip (merged with tall structure)",
        "rejected: footprint above foot_hi (merged, too wide)", "rejected: top below top_lo",
        "rejected: footprint below foot_lo", "kept blob matched to another person (merge)",
        "kept blob unmatched, centre beyond 0.5 m")
VLM_PLAN = [("Qwen 3.8", "s19", "qwen38_low", REAL + ("ph", "dev_seed0", "dev_seed1", "dev_seed2")),
            ("GLM-5.3", "s19", "glm53", REAL + ("ph", "dev_seed0", "dev_seed1", "dev_seed2")),
            ("DeepSeek V4.1", "s20", "dsv41", REAL + ("ph", "dev_seed0", "dev_seed1", "dev_seed2")),
            ("Haiku 5.5", "s23", "haiku55", REAL),
            ("Sonnet 5.5 low", "s23", "sonnet55_low", REAL),
            ("Gemini 3.8 low", "s23", "gemini38_low", REAL)]


# ----------------------------------------------------------------------------------------------- frozen inputs
def frozen():
    sel_path = I03 / "fit_selection.json"
    digest = hashlib.sha256(sel_path.read_bytes()).hexdigest()
    recorded = (I03 / "FIT.sha256").read_text().split()[0]
    assert digest == recorded, "fit_selection.json does not match FIT.sha256"
    sel = json.loads(sel_path.read_text())
    chosen = json.loads((HERE / "chosen.json").read_text())
    assert chosen["fit_selection_sha256"] == digest
    assert chosen["setting"] == sel["chosen_setting"]
    assert chosen["facing_variant"] == sel["facing"]["variant"]
    for name, sha in (("ADDENDUM_FIX.md", "ADDENDUM_FIX.sha256"), ("fit_selection_fix.json", "FIT_FIX.sha256")):
        got = hashlib.sha256((I03 / name).read_bytes()).hexdigest()
        assert got == (I03 / sha).read_text().split()[0], name
    fix = json.loads((I03 / "fit_selection_fix.json").read_text())
    assert chosen["fit_selection_fix_sha256"] == hashlib.sha256((I03 / "fit_selection_fix.json").read_bytes()).hexdigest()
    assert chosen["detector"] == fix["union_choice"]["detector"] and chosen["setting"] == fix["union_choice"]["setting"]
    global DROP_CLIP
    DROP_CLIP = bool(chosen["drop_clip"])
    return chosen["setting"], chosen["facing_variant"], digest, sel


def set_of(scenes_dir) -> str:
    d = str(scenes_dir) + "/"
    for name in REAL:
        if f"realistic_final/{name}/" in d:
            return name
    if "dev_new/seed" in d or d.endswith("placeholder/eval/") or d.endswith("/placeholder_first14/"):
        return "placeholder"
    raise SystemExit(f"unknown set: {scenes_dir}")


def domain_of(sset: str) -> str:
    return "E-real" if sset in REAL else "E-ph"


def thresholds_for(refs):
    ths = [H.recorded_config(r)[1].thresholds for r in refs]
    for th in ths:
        PS.check_thresholds(th)
    keys = ("person_radius_m", "visibility_min_samples", "visibility_depth_tolerance_m", "grid_resolution_m")
    for th in ths[1:]:
        assert all(getattr(th, k) == getattr(ths[0], k) for k in keys)
    return ths[0]


# ----------------------------------------------------------------------------------------------- blobs on one scene
def diagnose(person_xy, allb, kept_ids, blob_matched_to, s) -> tuple[str, dict]:
    """Where a missed person's depth went: the blob holding most height-map cells within 0.25 m of them."""
    best, best_n = None, 0
    for b in allb:
        d = B.np.hypot(b["cell_xy"][:, 0] - person_xy[0], b["cell_xy"][:, 1] - person_xy[1])
        n = int((d <= 0.25).sum())
        if n > best_n:
            best, best_n = b, n
    if best is None:
        return DIAG[0], {}
    info = {"top": round(best["top"], 3), "footprint": round(best["footprint"], 3), "cells": best["cells"],
            "cells_near": best_n, "dist": round(math.dist((best["x"], best["y"]), person_xy), 3)}
    top_hi = s["z_hi"] - 0.05
    if best["id"] in kept_ids:
        return (DIAG[5] if best["id"] in blob_matched_to else DIAG[6]), info
    if best["top"] > top_hi + 1e-9:
        return DIAG[1], info
    if best["footprint"] > s["foot_hi"] + 1e-9:
        return DIAG[2], info
    if best["top"] < s["top_lo"]:
        return DIAG[3], info
    return DIAG[4], info


def blob_scene(sc, th, s, variant, sset, scene_id):
    ex = PS.exempt_points(sc.truth)
    kb, allb = B.detect(sc.observation, s, keep_cells=True, drop_clip=DROP_CLIP)
    r = PS.score_people([(b["x"], b["y"], b[f"axis_{variant}"]) for b in kb], sc, th, exempt_xy=ex, facing="axis")
    kept_ids = {b["id"] for b in kb}
    matched_blob_ids = {kb[n - 1]["id"] for n in r["matches"]}
    hid = PS.truth_info(sc, th)["hidden"]
    vis = {p.id: p for p in PS.truth_info(sc, th)["visible"]}
    for m in r["misses"]:
        p = vis[m["id"]]
        m["diag"], m["diag_blob"] = diagnose((p.x, p.y), allb, kept_ids, matched_blob_ids, s)
    false_rows = []
    for n in r["false_numbers"]:
        b = kb[n - 1]
        near_vis = min((math.dist((b["x"], b["y"]), (p.x, p.y)) for p in vis.values()), default=None)
        false_rows.append({"x": round(b["x"], 3), "y": round(b["y"], 3), "top": round(b["top"], 3),
                           "footprint": round(b["footprint"], 3), "cells": b["cells"],
                           "near_hidden": any(math.dist((b["x"], b["y"]), (p.x, p.y)) <= PS.NEAR_M for p in hid),
                           "nearest_visible_m": None if near_vis is None else round(near_vis, 3)})
    row = {"set": sset, "domain": domain_of(sset) if sset in SETS else sset, "scene_id": scene_id,
           "kept": len(kb), **{k: r[k] for k in PS.POOL_KEYS}, "recall": r["recall"],
           "xy_err": [round(e, 4) for e in r["xy_err"]], "facing": r["facing"], "misses": r["misses"],
           "false_blobs": false_rows, "pairs": r["pairs"], "people": r["people"]}
    return row, r


# ----------------------------------------------------------------------------------------------- tables
def fmt_k(k, n):
    if not n:
        return f"{k}/{n} -"
    lo, hi = PS.wilson(k, n)
    return f"{k}/{n} = {k / n:.3f} [{lo:.3f}, {hi:.3f}]"


def pool_rows(rows):
    P = PS.pool(rows)
    P["by_role"] = {c: [sum(1 for p in P["people"] if p["role_class"] == c and p["matched"]),
                        sum(1 for p in P["people"] if p["role_class"] == c)] for c in ROLE_CLASSES}
    cls = collections.Counter()
    diag = collections.Counter()
    for m in P["misses"]:
        cls["staff behind counters" if m["staff_behind"] else "other staff" if m["role"] else "non-staff"] += 1
        if "diag" in m:
            diag[m["diag"]] += 1
    P["miss_class"] = cls
    P["miss_diag"] = diag
    return P


def verdict_of(P):
    has_pairs = P["neighbour_pairs"] > 0
    rec, fb = P["recall"], P["fb_per_scene"]
    mp = P["merged_share"] if has_pairs else None
    succ = rec >= 0.95 and fb <= 0.10 and (not has_pairs or mp <= 0.10)
    kill = rec < 0.85 or fb > 0.30 or (has_pairs and mp > 0.20)
    return {"success": bool(succ), "kill": bool(kill), "recall": rec, "fb": fb, "mp": mp,
            "bars": {"recall>=0.95": rec >= 0.95, "fb<=0.10": fb <= 0.10,
                     "mp<=0.10": None if not has_pairs else mp <= 0.10,
                     "recall<0.85": rec < 0.85, "fb>0.30": fb > 0.30, "mp>0.20": None if not has_pairs else mp > 0.20}}


def blob_table(L, title, groups):
    L.append(title)
    L.append("group | scenes | visible | recall k/n [Wilson 95%] | false (per scene) | exempt | near-hidden false | "
             "merges (detections) | merged pairs k/n | median xy err (n) | axis facing within 45 (known; unknown) | "
             "misses: staff behind counters / other staff / non-staff (share staff-behind)")
    for name, P in groups:
        f = P["facing"]
        mc = P["miss_class"]
        nm = sum(mc.values())
        mp = "" if not P["neighbour_pairs"] else f" = {P['merged_share']:.3f}"
        xy = "-" if P["xy_err_median"] is None else f"{P['xy_err_median']:.3f}"
        fa = "" if not f["known"] else f" = {f['within45_axis'] / f['known']:.3f}"
        sb = "" if not nm else f" ({mc['staff behind counters'] / nm:.2f})"
        L.append(f"{name:12s} | {P['scenes']:3d} | {P['n_visible']:3d} | {fmt_k(P['matched'], P['n_visible'])} | "
                 f"{P['false']} ({P['fb_per_scene']:.3f}) | {P['exempt']} | {P['near_hidden']} | {P['merges']} | "
                 f"{P['merged_pairs']}/{P['neighbour_pairs']}{mp} | {xy} ({len(P['xy_err'])}) | "
                 f"{f['within45_axis']}/{f['known']}{fa}; {f['unknown']} | "
                 f"{mc['staff behind counters']} / {mc['other staff']} / {mc['non-staff']}{sb}")
    L.append("recall by role (line member / group member / staff / other), k/n:")
    for name, P in groups:
        L.append(f"  {name:12s} " + " | ".join(f"{c} {P['by_role'][c][0]}/{P['by_role'][c][1]}" for c in ROLE_CLASSES))
    L.append("where the misses' depth went (descriptive; categories fixed in DEVIATIONS.md before evaluation):")
    for name, P in groups:
        if P["miss_diag"]:
            L.append(f"  {name:12s} " + "; ".join(f"{k}: {v}" for k, v in P["miss_diag"].most_common()))
    L.append("")


# ----------------------------------------------------------------------------------------------- VLM lists
def union_people(recs):
    from sgi.types import DetectedPerson
    union = {}
    for r in recs:
        for p in r.people:
            d = DetectedPerson.model_validate(p)
            k = (round(d.xy[0], 9), round(d.xy[1], 9), round(d.pixel[0], 6), round(d.pixel[1], 6))
            union.setdefault(k, d)
    return [union[k] for k in sorted(union)]


def vlm_rows(canon, blob_by_folder):
    out, checks = [], collections.Counter()
    for label, stage, model, arms in VLM_PLAN:
        for arm in arms:
            refs = H.runs(stage=stage, arm=arm, model=model, method="con")
            assert len(refs) == 1, (stage, arm, model, [r.folder for r in refs])
            ref = refs[0]
            th = thresholds_for([ref])
            by_key = collections.defaultdict(list)
            for rec in H.records(ref):
                by_key[(arm, rec.scene_id, rec.trial)].append(rec)
            for (arm_, scene_id, trial), recs in sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[0][1], str(kv[0][2]))):
                folder = canon[(ref.scenes_dir, scene_id)]
                sc = H.scene(ref, scene_id)
                ex = PS.exempt_points(sc.truth)
                with_list = [r for r in recs if r.people is not None]
                sset = set_of(ref.scenes_dir)
                row = {"model": label, "run": ref.folder, "arm": arm_, "set": sset, "domain": domain_of(sset),
                       "scene_id": scene_id, "folder": str(folder), "trial": trial, "records": len(recs),
                       "records_with_list": len(with_list), "has_list": bool(with_list)}
                ppl = union_people(with_list)
                r = PS.score_people([(d.xy[0], d.xy[1], d.yaw) for d in ppl], sc, th, exempt_xy=ex, facing="full")
                row.update({k: r[k] for k in PS.POOL_KEYS})
                row.update(recall=r["recall"], xy_err=[round(e, 4) for e in r["xy_err"]], facing=r["facing"],
                           misses=r["misses"], people=r["people"])
                checks["keys"] += 1
                checks["keys without a people list"] += not with_list
                out.append(row)
    return out, checks


def vlm_tables(L, vrows, blob_by_folder):
    L.append("== VLM people lists, scored like for like (same scorer, same exemption; one answer per (arm, scene, "
             "trial) = union of its records' people lists) ==")
    L.append("model | domain | answer keys (no list) | visible | recall k/n [Wilson] | false/answer (false; raw "
             "false incl. exempt) | merged pairs k/n | median xy err (n) | full facing within 45 (known; unknown) | "
             "axis facing within 45 (known) || blobs weighted to the same keys: recall | false/key | merged pairs | "
             "axis facing")
    summary = {}
    for label, _, _, _ in VLM_PLAN:
        for dom in DOMAINS + ("pooled",):
            rows = [r for r in vrows if r["model"] == label and (dom == "pooled" or r["domain"] == dom)]
            if not rows:
                continue
            listed = [r for r in rows if r["has_list"]]
            P = PS.pool(listed)
            S = PS.pool(rows)                       # sensitivity: keys without a list as empty lists
            W = PS.pool([blob_by_folder[r["folder"]] for r in listed])
            f, wf = P["facing"], W["facing"]
            L.append(f"{label:15s} | {dom:6s} | {len(rows)} ({len(rows) - len(listed)}) | {P['n_visible']} | "
                     f"{fmt_k(P['matched'], P['n_visible'])} | {P['false'] / max(P['scenes'], 1):.3f} ({P['false']}; "
                     f"{P['false_raw']}) | {P['merged_pairs']}/{P['neighbour_pairs']} | "
                     f"{P['xy_err_median'] if P['xy_err_median'] is None else round(P['xy_err_median'], 3)} "
                     f"({len(P['xy_err'])}) | {f['within45_full']}/{f['known']}; {f['unknown']} | "
                     f"{f['within45_axis']}/{f['known']} || {fmt_k(W['matched'], W['n_visible'])} | "
                     f"{W['false'] / max(W['scenes'], 1):.3f} | {W['merged_pairs']}/{W['neighbour_pairs']} | "
                     f"{wf['within45_axis']}/{wf['known']}")
            if len(rows) != len(listed):
                L.append(f"{'':15s}   sensitivity (keys without a list counted as empty lists): recall "
                         f"{fmt_k(S['matched'], S['n_visible'])}, false/answer {S['false'] / max(S['scenes'], 1):.3f}")
            summary[f"{label}|{dom}"] = {
                "keys": len(rows), "keys_without_list": len(rows) - len(listed), "visible": P["n_visible"],
                "matched": P["matched"], "recall": P["recall"], "false": P["false"], "false_raw": P["false_raw"],
                "false_per_answer": P["false"] / max(P["scenes"], 1),
                "merged_pairs": [P["merged_pairs"], P["neighbour_pairs"]], "xy_err_median": P["xy_err_median"],
                "facing_full": [f["within45_full"], f["known"], f["unknown"]],
                "facing_axis": [f["within45_axis"], f["known"]],
                "sens_recall_empty": S["recall"],
                "blob_weighted": {"matched": W["matched"], "visible": W["n_visible"], "recall": W["recall"],
                                  "false_per_key": W["false"] / max(W["scenes"], 1),
                                  "merged_pairs": [W["merged_pairs"], W["neighbour_pairs"]],
                                  "facing_axis": [wf["within45_axis"], wf["known"]]}}
    L.append("")
    L.append("per model and set (listed keys): recall k/n | false/answer")
    for label, _, _, _ in VLM_PLAN:
        parts = []
        for sset in SETS:
            rows = [r for r in vrows if r["model"] == label and r["set"] == sset and r["has_list"]]
            if rows:
                P = PS.pool(rows)
                parts.append(f"{sset} {P['matched']}/{P['n_visible']} = {P['recall']:.3f}, {P['false'] / P['scenes']:.3f}")
        L.append(f"  {label:15s} " + " | ".join(parts))
    L.append("")
    return summary


# ----------------------------------------------------------------------------------------------- main
def main():
    s, variant, digest, sel = frozen()
    if "--dry-fit" in sys.argv:
        import fitdata as FD
        refs = FD.fit_refs()
        picks = refs[:3] + [r for r in refs if r[0] == "F-ph"][:3]
        for domain, sset, scene_id, loader in picks:
            sc, th = FD.load(loader)
            row, r = blob_scene(sc, th, s, variant, sset, scene_id)
            print(domain, scene_id, row["kept"], row["matched"], row["n_visible"], row["false"],
                  [m["diag"] for m in row["misses"]])
        P = pool_rows([blob_scene(*FD.load(l), s, variant, ss, sid)[0] for _, ss, sid, l in picks])
        L = []
        blob_table(L, "dry", [("fit6", P)])
        print("\n".join(L))
        return
    out_rows = HERE / "rows.jsonl"
    t0 = time.monotonic()
    distinct = H.distinct_scenes()
    assert len(distinct) == 68, len(distinct)
    canon = {}
    for sref in distinct:
        for key in sref.listed:
            canon[key] = sref.folder
    rows, blob_by_folder = [], {}
    # The evaluation is scored once: a first pass wrote rows.jsonl and then stopped in the VLM part on a set-name
    # lookup (keep/placeholder_first14). The blob rows it wrote are read back here, never scored again.
    if out_rows.exists():
        rows = [json.loads(line) for line in out_rows.read_text().splitlines()]
        assert len(rows) == 68 and {r["folder"] for r in rows} == {str(sref.folder) for sref in distinct}
        blob_by_folder = {r["folder"]: r for r in rows}
        distinct_to_score = []
        print("blob rows read back from rows.jsonl (scored once, in the first pass)", flush=True)
    else:
        distinct_to_score = distinct
    for sref in distinct_to_score:
        sset = set_of(sref.scenes_dir)
        th = thresholds_for(sref.runs)
        sc = H.scene(sref.runs[0], sref.scene_id)
        row, r = blob_scene(sc, th, s, variant, sset, sref.scene_id)
        row["folder"] = str(sref.folder)
        row["run_used"] = sref.runs[0].folder
        rows.append(row)
        blob_by_folder[str(sref.folder)] = row
        print(f"{sset:12s} {sref.scene_id:36s} kept {row['kept']:2d} matched {row['matched']}/{row['n_visible']} "
              f"false {row['false']} exempt {row['exempt']}", flush=True)
    n_dom = collections.Counter(r["domain"] for r in rows)
    assert n_dom == {"E-real": 42, "E-ph": 26}, n_dom
    if distinct_to_score:
        with open(out_rows, "w") as fh:
            for r in rows:
                fh.write(json.dumps({k: v for k, v in r.items()}) + "\n")

    L = []
    L.append("I03 Stage A evaluation (FROZEN.md sections 4-5), scored once with the frozen setting.")
    L.append(f"detector: {'with the one general fix (drop clip-reaching cells)' if DROP_CLIP else 'original (the one general fix was tried on fit data and not adopted)'}")
    L.append(f"setting: {B.setting_key(s)}, top_hi {s['z_hi'] - 0.05:.2f}; facing variant {variant}; "
             f"fit_selection.json sha256 {digest}")
    L.append("Definitions: recall = matched / visible (sgi.run._matched_people, 0.5 m); false = unmatched kept blob not "
             "within 0.5 m of a non-person target anchor (exempt ones counted apart); merged pairs = visible "
             "neighbour pairs (< 0.8 m) not both matched with one blob within 0.5 m of both; axis facing = minor "
             "principal axis within 45 deg of the true yaw modulo 180 (chance 0.50).")
    L.append("")
    groups_set = [(sset, pool_rows([r for r in rows if r["set"] == sset])) for sset in SETS]
    groups_dom = [(d, pool_rows([r for r in rows if r["domain"] == d])) for d in DOMAINS]
    pooled = ("pooled 68", pool_rows(rows))
    blob_table(L, "== Blobs, per set ==", groups_set)
    blob_table(L, "== Blobs, per domain (the verdict is per domain) and pooled ==", groups_dom + [pooled])

    verdicts = {d: verdict_of(P) for d, P in groups_dom}
    success = all(v["success"] for v in verdicts.values())
    kill = any(v["kill"] for v in verdicts.values())
    verdict = "success" if success else "kill" if kill else "neither"
    L.append("== Stage A criteria (per domain; success needs all bars in both, kill fires on any bar in either) ==")
    for d, v in verdicts.items():
        L.append(f"{d}: recall {v['recall']:.4f}, false/scene {v['fb']:.4f}, merged share "
                 f"{'n/a' if v['mp'] is None else format(v['mp'], '.4f')}; bars {v['bars']}; "
                 f"success {v['success']}, kill {v['kill']}")
    L.append(f"STAGE A VERDICT: {verdict}")
    L.append("")

    vrows, checks = vlm_rows(canon, blob_by_folder)
    with open(HERE / "vlm_rows.jsonl", "w") as fh:
        for r in vrows:
            fh.write(json.dumps(r) + "\n")
    vsum = vlm_tables(L, vrows, blob_by_folder)
    L.append(f"VLM checks: {dict(checks)}")
    L.append(f"{time.monotonic() - t0:.0f} s")
    (HERE / "tables.txt").write_text("\n".join(L) + "\n")

    def jp(P):
        return {k: P[k] for k in ("scenes", "n_visible", "matched", "recall", "false", "fb_per_scene", "exempt",
                                  "near_hidden", "merges", "neighbour_pairs", "merged_pairs", "merged_share",
                                  "xy_err_median", "staff_behind")} | {
            "n_xy": len(P["xy_err"]), "facing": P["facing"], "miss_class": dict(P["miss_class"]),
            "miss_diag": dict(P["miss_diag"]), "by_role": P["by_role"],
            "recall_wilson": PS.wilson(P["matched"], P["n_visible"])}
    summary = {"setting": s, "facing_variant": variant, "fit_selection_sha256": digest,
               "per_set": {n: jp(P) for n, P in groups_set}, "per_domain": {n: jp(P) for n, P in groups_dom},
               "pooled": jp(pooled[1]), "verdicts": verdicts, "verdict": verdict, "vlm": vsum,
               "vlm_checks": dict(checks)}
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print("\n".join(L))


if __name__ == "__main__":
    main()
