"""I09 confirmation, step 6 (FROZEN.md sections 4-7 and 9): score the test set once.

For each manifest entry (seed ascending, row ascending): the hash checks, the section 3 path guard, H.guard_scene,
PS.FreshScene(load_scene(folder)); then confirm_lib.score_image: P.run_image(raw, depth, cam, setting, "F5") and
C.score_scene (PS.score_people with exempt_xy=PS.exempt_points(truth), facing "full", default thresholds). The criteria
of section 5 are judged once on the pooled test set, on point estimates. Writes tables.txt, rows.jsonl,
people_rows.jsonl and summary.json, hashed into SCORE.sha256. Refuses to run if any of them exists.

--dry-dev: the whole tabulation on the 42 development E-real scenes (A1's recorded raw pass; arms stand in for seeds
200-202), printed to stdout only; no test data is read and nothing is written. A code-path test before the real run.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import collections  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import confirm_lib as CL  # noqa: E402

C, P, PS, H = CL.C, CL.P, CL.PS, CL.H
fmt_k, f3 = CL.fmt_k, CL.f3
OUTPUTS = ("tables.txt", "rows.jsonl", "people_rows.jsonl", "summary.json", "SCORE.sha256")
BATCHES = (("seeds 200-209", range(200, 210)), ("seeds 210-219", range(210, 220)))
L: list[str] = []


# ------------------------------------------------------------------------------------------------ statistics on draws
def _recall(rows):
    n = sum(r["n_visible"] for r in rows)
    return sum(r["matched"] for r in rows) / n if n else None


def _fb(rows):
    return sum(r["false"] for r in rows) / len(rows) if rows else None


def _line(pr):
    n = sum(1 for p in pr if p["role_class"] == "line")
    return sum(1 for p in pr if p["role_class"] == "line" and p["matched"]) / n if n else None


def _xy(pr):
    xs = [p["xy_err"] for p in pr if p["matched"]]
    return statistics.median(xs) if xs else None


def _fsel(sel, field):
    def f(pr):
        errs = [p["err"] for p in pr if p["matched"] and sel(p["view"])]
        return C.facing_summary(errs)[field] if errs else None
    return f


STATS = {   # bar -> (level: 'scene' | 'person', statistic on a draw)
    "S1": ("scene", _recall), "S2": ("person", _line), "S3": ("scene", _fb), "S4": ("person", _xy),
    "S5": ("person", _fsel(lambda v: v != "profile", "median_u")),
    "S6": ("person", _fsel(lambda v: v != "profile", "w45_share")),
    "S7": ("person", _fsel(lambda v: v == "profile", "fb_share")),
    "K1": ("scene", _recall), "K2": ("scene", _fb), "K3": ("person", _fsel(lambda v: True, "w45_share")),
}
BAR = {"S1": (">=", 0.95), "S2": (">=", 0.95), "S3": ("<=", 0.10), "S4": ("<=", 0.15), "S5": ("<=", 25.0),
       "S6": (">=", 0.85), "S7": (">=", 0.90), "K1": ("<", 0.90), "K2": (">", 0.30), "K3": ("<", 0.70)}
LABEL = {"S1": "recall overall >= 0.95 (all visible people)",
         "S2": "recall on line members >= 0.95 (visible line members)",
         "S3": "false people per image <= 0.10 (all test images)",
         "S4": "median position error <= 0.15 m (matched people)",
         "S5": "non-profile facing median <= 25 deg (matched non-profile, unknown = 180)",
         "S6": "non-profile within 45 deg >= 0.85 (matched non-profile, unknown wrong)",
         "S7": "profile front/back correct >= 0.90 (matched profile, unknown wrong)",
         "K1": "KILL recall < 0.90 (all visible people)",
         "K2": "KILL false people per image > 0.30 (all test images)",
         "K3": "KILL all matched within 45 deg < 0.70 (unknown wrong)"}


def cluster_groups(rows, prows, key):
    """{cluster: [items]} for scene rows and for person rows, over the same cluster keys (empty lists kept)."""
    gs, gp = collections.defaultdict(list), collections.defaultdict(list)
    for r in rows:
        gs[key(r)].append(r)
    for p in prows:
        gp[key(p)].append(p)
    for k in gs:
        gp.setdefault(k, [])
    assert set(gp) == set(gs)
    return dict(gs), dict(gp)


def boot_all(gs, gp):
    out = {}
    for b, (level, stat) in STATS.items():
        out[b] = C.boot_ci(gs if level == "scene" else gp, stat)
    return out


def point(m, b):
    return {"S1": m["recall"], "K1": m["recall"],
            "S2": (m["by_role"]["line"][0] / m["by_role"]["line"][1]) if m["by_role"]["line"][1] else None,
            "S3": m["fb_per_image"], "K2": m["fb_per_image"], "S4": m["xy_median"], "S5": m["np_median_u"],
            "S6": (m["np_w45"] / m["np_n"]) if m["np_n"] else None,
            "S7": (m["pf_fb"] / m["pf_n"]) if m["pf_n"] else None,
            "K3": (m["all_w45"] / m["all_n"]) if m["all_n"] else None}[b]


def kn(m, b):
    """(k, n) for proportion bars, else None; the count string for the others."""
    return {"S1": (m["matched"], m["visible"]), "K1": (m["matched"], m["visible"]),
            "S2": tuple(m["by_role"]["line"]), "S6": (m["np_w45"], m["np_n"]), "S7": (m["pf_fb"], m["pf_n"]),
            "K3": (m["all_w45"], m["all_n"])}.get(b)


def value_str(m, b):
    if kn(m, b) is not None:
        k, n = kn(m, b)
        return fmt_k(k, n)
    if b in ("S3", "K2"):
        return f"{m['false']}/{m['images']} = {m['fb_per_image']:.3f} per image"
    if b == "S4":
        return f"{f3(m['xy_median'])} m (n {m['n_xy']})"
    if b == "S5":
        return f"{f3(m['np_median_u'], 1)} deg (n {m['np_n']})"
    raise KeyError(b)


def short_str(m, b):
    if kn(m, b) is not None:
        k, n = kn(m, b)
        return f"{k}/{n}" + (f" = {k / n:.3f}" if n else "")
    if b in ("S3", "K2"):
        return f"{m['false']}/{m['images']} = {m['fb_per_image']:.3f}"
    if b == "S4":
        return f"{f3(m['xy_median'])} (n {m['n_xy']})"
    return f"{f3(m['np_median_u'], 1)} (n {m['np_n']})"


def passing_side(b, ci):
    op, bar = BAR[b]
    lo, hi = ci
    if lo is None:
        return None
    if op == ">=":
        return lo >= bar
    if op == "<=":
        return hi <= bar
    if op == "<":          # kill below: the whole interval clears the kill region when lo >= bar
        return lo >= bar
    return hi <= bar       # kill above


def ci_str(ci, nd=3):
    return "-" if ci[0] is None else f"[{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


# ------------------------------------------------------------------------------------------------ tabulation
def tabulate(rows, prows, raw_lines, meta_raw, dev, header):
    s, variant = CL.frozen_setting()
    m = CL.measures(rows, prows)
    succ, kill = CL.bars(m)
    verdict = CL.verdict_of(succ, kill)
    gs, gp = cluster_groups(rows, prows, lambda r: r["layout"])
    gss, gsp = cluster_groups(rows, prows, lambda r: r["seed"])
    ci_l, ci_s = boot_all(gs, gp), boot_all(gss, gsp)
    dev_m = {name: CL.measures(*rp) for name, rp in dev["rows"].items()}
    L.extend(header)
    seeds = sorted({r["seed"] for r in rows})
    L.append(f"Test set: {len(rows)} images = {len(gs)} layouts (one real-variant image each) in {len(seeds)} seeds "
             f"{seeds[0]}-{seeds[-1]}; visible people {m['visible']} (visible staff {m['staff_visible']}, of whom "
             f"lower body hidden {sum(1 for p in prows if p['staff_behind'])}); hidden people "
             f"{sum(r['n_hidden_people'] for r in rows)}; detections kept {sum(r['n_kept'] for r in rows)}")
    L.append(f"Setting (A1 frozen, chosen.json + chosen_fix.json): {C.setting_key(s)}; band {P.BANDS[s['band']]} m; "
             f"facing {variant}. Scorer: phase1b/i03/people_scorer.py unchanged, facing 'full', exempt_xy = anchors of "
             f"targets without a person, thresholds configs/default.yaml.")
    L.append("Definitions (FROZEN.md section 4): visible = sgi.gt.derive.is_visible; matching = sgi.run._matched_people "
             "(greedy nearest first, one to one, within 0.5 m); recall = matched / visible; false person = unmatched "
             "detection not within 0.5 m of a target anchor with no person (exempt counted apart); position error = xy "
             "distance of a matched pair; view class from |wrap(bearing true person -> camera xy - true yaw)|: frontal "
             "< 45, oblique 45-<75, profile 75-105 inclusive, back > 105; facing error |wrap(yaw_det - yaw_true)|; "
             "within 45 = error <= 45; front/back correct = error < 90; unknown yaw = wrong, 180 deg in medians.")
    L.append("")
    # ---------------------------------------------------------------- main table
    L.append("== Table 1. Criteria (FROZEN.md section 5), pooled test set, judged once on point estimates ==")
    L.append(f"Intervals: Wilson 95% (proportions); layout-cluster bootstrap 95% ({len(gs)} single-image layouts, 2000 "
             "draws, random.Random(0), percentile, via C.boot_ci). Development comparators from A1's recorded rows "
             "(section 7).")
    L.append("bar | test value (n) [Wilson] | layout bootstrap | met? | dev E-real (42 images) | dev real arm (14 images)")
    for b in STATS:
        res = succ if b.startswith("S") else kill
        key = next(k for k in res if k.startswith(b + " "))
        flag = res[key]
        word = ("MET" if flag else "NOT MET") if b.startswith("S") else ("FIRES" if flag else "does not fire")
        L.append(f"{b} {LABEL[b]} | {value_str(m, b)} | {ci_str(ci_l[b], 1 if b == 'S5' else 3)} | {word} | "
                 f"{short_str(dev_m['E-real'], b)} | {short_str(dev_m['real arm'], b)}")
    L.append(f"Development layout bootstrap (A1 summary.json, 14 layouts x 3 arms): recall "
             f"{ci_str(dev['boot']['recall'])}, line {ci_str(dev['boot']['line_recall'])}, false per image "
             f"{ci_str(dev['boot']['false_per_image'])}, non-profile within 45 {ci_str(dev['boot']['non_profile_w45'])}, "
             f"non-profile median {ci_str(dev['boot']['non_profile_median'], 1)}, profile front/back "
             f"{ci_str(dev['boot']['profile_fb'])}, all within 45 {ci_str(dev['boot']['all_w45'])}")
    L.append("")
    L.append(f"VERDICT (FROZEN.md section 5): {verdict.upper()}  -- success bars {sum(v is True for v in succ.values())}"
             f"/7 met; kill clauses fired: {[k for k, v in kill.items() if v] or 'none'}")
    side = {b: passing_side(b, ci_l[b]) for b in STATS}
    L.append("Descriptive (not judged; DEVIATIONS_1.md C10): whole layout-bootstrap interval on the passing side -- " +
             "; ".join(f"{b} {'yes' if v else 'no' if v is not None else '-'}" for b, v in side.items()))
    L.append("")
    # ---------------------------------------------------------------- by role and by row
    L.append("== Table 2. Recall by role class (k/n [Wilson 95%]) ==")
    L.append("role | test | dev E-real | dev real arm")
    for c in C.ROLE_CLASSES:
        L.append(f"{c:6s} | {fmt_k(*m['by_role'][c])} | {'/'.join(map(str, dev_m['E-real']['by_role'][c]))} | "
                 f"{'/'.join(map(str, dev_m['real arm']['by_role'][c]))}")
    L.append("")
    L.append("== Table 3. By row, pooled over seeds: recall k/n [Wilson 95%], false people per image, misses ==")
    L.append("row id | images | visible | recall | line / group / staff / other (matched/visible) | false (per image) | "
             "exempt | misses")
    by_row = {}
    for rid in sorted({r["row_id"] for r in rows}):
        rr = [r for r in rows if r["row_id"] == rid]
        pr = [p for p in prows if p["row_id"] == rid]
        mm = CL.measures(rr, pr)
        by_row[rid] = {"images": mm["images"], "recall": [mm["matched"], mm["visible"]], "false": mm["false"],
                       "fb_per_image": mm["fb_per_image"], "by_role": mm["by_role"], "exempt": mm["exempt"]}
        L.append(f"{rid:32s} | {mm['images']:3d} | {mm['visible']:3d} | {fmt_k(mm['matched'], mm['visible'])} | "
                 + " / ".join(f"{mm['by_role'][c][0]}/{mm['by_role'][c][1]}" for c in C.ROLE_CLASSES)
                 + f" | {mm['false']} ({mm['fb_per_image']:.3f}) | {mm['exempt']} | {mm['visible'] - mm['matched']}")
    L.append("")
    # ---------------------------------------------------------------- facing by view class
    L.append(f"== Table 4. Facing by view class, matched people, variant {variant} (unknown = wrong / 180 deg) ==")
    L.append("view class | n | coverage (known) | within 45 [Wilson] | median deg | front/back < 90 [Wilson] | "
             "dev E-real: n, within 45, median, front/back")
    fg, dfg = m["facing"], dev_m["E-real"]["facing"]
    for vc in C.VIEW_CLASSES + ("non-profile", "all matched"):
        f, d = fg[vc], dfg[vc]
        if not f["n"]:
            L.append(f"{vc:12s} | 0")
            continue
        L.append(f"{vc:12s} | {f['n']:3d} | {f['known']}/{f['n']} | {fmt_k(f['w45'], f['n'])} | {f['median_u']:.1f} | "
                 f"{fmt_k(f['fb'], f['n'])} | {d['n']}, {d['w45']}/{d['n']}, "
                 f"{f3(d['median_u'], 1)}, {d['fb']}/{d['n']}")
    L.append(f"Coverage (matched people with a known yaw): {m['coverage'][0]}/{m['coverage'][1]}. Merged neighbour "
             f"pairs: {m['merged_pairs']}/{m['neighbour_pairs']} (dev E-real {dev_m['E-real']['merged_pairs']}/"
             f"{dev_m['E-real']['neighbour_pairs']}). Merges (detections covering >= 2 visible people, <= 1 matched): "
             f"{m['merges']}. Exempt detections: {m['exempt']}. False people near a hidden person: {m['near_hidden']}.")
    L.append("")
    # ---------------------------------------------------------------- misses
    misses = [p for p in prows if not p["matched"]]
    cls = collections.Counter("staff behind counters" if p["staff_behind"] else "other staff" if p["role"] else
                              "non-staff" for p in misses)
    L.append(f"== Table 5. Misses ({len(misses)} of {m['visible']} visible) ==")
    L.append(f"by class: staff behind counters {cls['staff behind counters']}, other staff {cls['other staff']}, "
             f"non-staff {cls['non-staff']}; by role class: "
             + ", ".join(f"{c} {sum(1 for p in misses if p['role_class'] == c)}" for c in C.ROLE_CLASSES))
    L.append("seed | scene | person | role class (role) | lower body hidden | view (angle) | best raw box score "
             "(box with score >= 0.05 containing the projection of (x, y, 1.1 m)) | nearest kept detection m "
             "(post hoc diagnostic)")
    for p in misses:
        bb = p.get("best_raw_box")
        bb_s = "none" if bb is None else f"{bb['score']:.3f}"
        L.append(f"{p['seed']} | {p['scene_id']} | {p['pid']} | {p['role_class']} ({p['role']}) | {p['lower_hidden']} | "
                 f"{p['view']} ({p['view_angle']:.0f}) | {bb_s} | {f3(p.get('nearest_det_m'), 2)}")
    L.append("")
    # ---------------------------------------------------------------- false people
    fl = [(r, fd) for r in rows for fd in r["false_detail"]]
    L.append(f"== Table 6. False and exempt detections ({m['false']} false, {m['exempt']} exempt) ==")
    L.append("seed | scene | number | false/exempt | detector score | nearest visible person m | near a hidden person")
    for r, fd in fl:
        L.append(f"{r['seed']} | {r['scene_id']} | {fd['number']} | {'exempt' if fd['exempt'] else 'false'} | "
                 f"{fd['score']:.3f} | {f3(fd['nearest_visible_m'], 2)} | {fd['near_hidden']}")
    L.append("")
    # ---------------------------------------------------------------- seed batches
    L.append("== Table 7. Seed batches side by side (descriptive) ==")
    L.append("bar | " + " | ".join(name for name, _ in BATCHES))
    batch_m = {}
    for name, rng in BATCHES:
        rr = [r for r in rows if r["seed"] in rng]
        pr = [p for p in prows if p["seed"] in rng]
        batch_m[name] = CL.measures(rr, pr) if rr else None
    L.append("images | " + " | ".join(str(bm["images"]) if bm else "0" for bm in batch_m.values()))
    for b in STATS:
        L.append(f"{b} | " + " | ".join(short_str(bm, b) if bm else "-" for bm in batch_m.values()))
    L.append("")
    # ---------------------------------------------------------------- seed-cluster bootstrap
    L.append(f"== Table 8. Seed-cluster bootstrap (sensitivity): whole seeds resampled, {len(gss)} clusters, 2000 draws, "
             "seed 0 ==")
    L.append("bar | point | seed bootstrap 95% | layout bootstrap 95% (Table 1)")
    for b in STATS:
        nd = 1 if b == "S5" else 3
        L.append(f"{b} | {f3(point(m, b), nd)} | {ci_str(ci_s[b], nd)} | {ci_str(ci_l[b], nd)}")
    L.append("")
    # ---------------------------------------------------------------- runtime
    tsum = [x["t_det_s"] + x["t_pose_s"] for x in raw_lines]
    tproj = [r["t_project_s"] for r in rows]
    rt = {"n": len(tsum), "sidecar_median_s": statistics.median(tsum), "sidecar_p90_s": C.p90(tsum),
          "project_median_s": statistics.median(tproj), "project_p90_s": C.p90(tproj),
          "gpu": meta_raw.get("gpu"), "gpu_used_before_MiB": meta_raw.get("gpu_used_before_MiB"),
          "peak_mem_allocated_MiB": meta_raw.get("peak_mem_allocated_MiB"),
          "peak_nvidia_smi_MiB": meta_raw.get("peak_nvidia_smi_MiB"), "start": meta_raw.get("start"),
          "end": meta_raw.get("end")}
    L.append("== Runtime (not judged; FROZEN.md section 6) ==")
    L.append(f"Detector pass, t_det_s + t_pose_s per image (detector at score >= 0.05, ViTPose on boxes >= 0.10, no "
             f"warm-up): median {rt['sidecar_median_s'] * 1000:.1f} ms, p90 {rt['sidecar_p90_s'] * 1000:.1f} ms "
             f"(n {rt['n']}); project side (P.run_image, 2 BLAS threads): median {rt['project_median_s'] * 1000:.1f} ms, "
             f"p90 {rt['project_p90_s'] * 1000:.1f} ms. GPU {rt['gpu']}, memory.used before start "
             f"{rt['gpu_used_before_MiB']} MiB, peak allocated {rt['peak_mem_allocated_MiB']} MiB. The GPU is shared and "
             "this is not A1's deployed timing path (that thresholds at tau before ViTPose and warms up 5 times).")
    L.append("")
    # ---------------------------------------------------------------- post hoc
    L.append("== POST HOC (not listed in FROZEN.md section 6; descriptive, changes no verdict) ==")
    L.append("Per seed: images | recall | line | false | non-profile within 45 | profile front/back | all within 45")
    per_seed = {}
    for sd in seeds:
        rr = [r for r in rows if r["seed"] == sd]
        pr = [p for p in prows if p["seed"] == sd]
        mm = CL.measures(rr, pr)
        per_seed[sd] = {b: point(mm, b) for b in STATS}
        L.append(f"seed {sd} | {mm['images']} | {mm['matched']}/{mm['visible']} | {mm['by_role']['line'][0]}/"
                 f"{mm['by_role']['line'][1]} | {mm['false']} | {mm['np_w45']}/{mm['np_n']} | {mm['pf_fb']}/{mm['pf_n']} | "
                 f"{mm['all_w45']}/{mm['all_n']}")
    jm = {k: v for k, v in m.items() if k != "facing"}
    summary = {"verdict": verdict, "success_bars": succ, "kill_clauses": kill, "measures": jm, "facing": m["facing"],
               "layout_bootstrap": ci_l, "seed_bootstrap": ci_s, "passing_side": side,
               "dev_comparators": {n: {b: point(dm, b) for b in STATS} | {"by_role": dm["by_role"],
                                                                          "counts": short_counts(dm)}
                                   for n, dm in dev_m.items()},
               "by_row": by_row, "miss_class": dict(cls),
               "seed_batches": {n: (None if bm is None else {b: point(bm, b) for b in STATS} | short_counts(bm))
                                for n, bm in batch_m.items()},
               "per_seed_posthoc": per_seed, "runtime": rt}
    return summary


def short_counts(m):
    return {"images": m["images"], "visible": m["visible"], "matched": m["matched"], "false": m["false"],
            "n_xy": m["n_xy"], "np": [m["np_w45"], m["np_n"]], "pf_fb": [m["pf_fb"], m["pf_n"]],
            "all_w45": [m["all_w45"], m["all_n"]], "line": m["by_role"]["line"]}


def dev_comparators():
    rec_rows, rec_prows = CL.load_dev_recorded()
    summ = json.loads((CL.A1 / "summary.json").read_text())
    return {"rows": {"E-real": (rec_rows, rec_prows),
                     "real arm": ([r for r in rec_rows if r["set"] == "real"],
                                  [p for p in rec_prows if p["set"] == "real"])},
            "boot": summ["e_real_bootstrap"]}


def add_nearest(rows, prows):
    """Post hoc diagnostic: distance from each missed person to the nearest kept detection of the image."""
    det = {(r["seed"], r["scene_id"]): r["det_people"] for r in rows}
    for p in prows:
        if not p["matched"]:
            d = det[(p["seed"], p["scene_id"])]
            p["nearest_det_m"] = min((math.dist((q["x"], q["y"]), (p["x"], p["y"])) for q in d), default=None)


# ------------------------------------------------------------------------------------------------ main
def dry_dev():
    import c2_repro
    rows, prows, people, raw = c2_repro.dev_rows()
    seed_of = {"real": 200, "staffed": 201, "plain": 202}
    by_folder = {}
    for r in rows:
        r["seed"] = seed_of[r["set"]]
        r["row_id"] = r["layout"]
        r["layout"] = f"seed{r['seed']}/{r['scene_id']}"
        r["det_people"] = people[r["folder"]]
        by_folder[r["folder"]] = r
    for p in prows:
        r = by_folder[p["folder"]]
        p["seed"], p["row_id"], p["layout"] = r["seed"], r["row_id"], r["layout"]
    add_nearest(rows, prows)
    man = json.loads((CL.A1 / "manifest_eval.json").read_text())
    rgb_of = {e["folder"]: e["rgb"] for e in man["images"]}
    raw_lines = [raw[rgb_of[r["folder"]]] for r in rows]
    meta_raw = json.loads((CL.A1 / "raw_eval/meta.json").read_text())
    summary = tabulate(rows, prows, raw_lines, meta_raw, dev_comparators(),
                       ["DRY RUN on development E-real scenes (arms real/staffed/plain stand in for seeds 200/201/202); "
                        "not a result"])
    print("\n".join(L))
    print(json.dumps({k: summary[k] for k in ("verdict", "success_bars", "kill_clauses")}, default=str))


def main():
    if "--dry-dev" in sys.argv:
        return dry_dev()
    t0 = time.monotonic()
    for name in OUTPUTS:
        assert not (CL.HERE / name).exists(), f"{name} exists already: scoring runs once (FROZEN.md section 9)"
    import c3_truth
    import pathguard
    man = c3_truth.load_manifest()                           # MANIFEST.sha256 verified
    raw_hashes = CL.verify_sha_file(CL.HERE / "RAW.sha256")
    meta_raw = json.loads((CL.HERE / "raw/meta.json").read_text())
    assert meta_raw["manifest_sha256"] == CL.sha(CL.HERE / "manifest.json")
    assert meta_raw["detectors"] == ["rtdetr_v2_r50vd"] and meta_raw["mode"] == "eval"
    assert meta_raw["warmup_images"] == []
    raw_lines_all = [json.loads(x) for x in (CL.HERE / "raw/rtdetr_v2_r50vd.jsonl").read_text().splitlines()]
    raw = {x["rgb"]: x for x in raw_lines_all}
    assert len(raw) == len(raw_lines_all) == len(man["entries"])
    s, variant = CL.frozen_setting()
    th = PS.default_thresholds()
    PS.check_thresholds(th)
    rows, prows, raw_lines = [], [], []
    for e in man["entries"]:
        pathguard.check(e["rgb"])
        x = raw[e["rgb"]]
        assert x["sha256"] == e["sha256_rgb_png"] and x["detector"] == "rtdetr_v2_r50vd"
        sc = c3_truth.load_test_scene(e)                     # guards and the rgb/depth/scene.json hashes
        meta = {"seed": e["seed"], "row": e["row"], "row_id": e["row_id"], "scene_id": e["scene_id"],
                "layout": f"seed{e['seed']}/{e['scene_id']}", "folder": e["folder"]}
        row, pr, out = CL.score_image(x, sc, th, s, variant, meta)
        row["det_people"] = out
        rows.append(row)
        prows += pr
        raw_lines.append(x)
    assert len(rows) == man["counts"]["layouts"]
    add_nearest(rows, prows)
    header = [f"I09 unseen-layout confirmation (phase2/i09/confirm): scored once, "
              f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
              f"Hashes verified: {json.dumps(CL.PROTOCOL_HASHES | raw_hashes | {'manifest.json': CL.sha(CL.HERE / 'manifest.json')})}"]
    with open(CL.HERE / "rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    with open(CL.HERE / "people_rows.jsonl", "w") as fh:
        for p in prows:
            fh.write(json.dumps(p, default=str) + "\n")
    summary = tabulate(rows, prows, raw_lines, meta_raw, dev_comparators(), header)
    L.append(f"{time.monotonic() - t0:.0f} s; finished {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    (CL.HERE / "tables.txt").write_text("\n".join(L) + "\n")
    summary["hashes"] = CL.PROTOCOL_HASHES | raw_hashes
    summary["seeds_complete"] = man["seeds_complete"]
    summary["seeds_left_out"] = man["seeds_left_out"]
    CL.jdump(summary, CL.HERE / "summary.json")
    with open(CL.HERE / "SCORE.sha256", "w") as fh:
        for name in OUTPUTS[:-1]:
            fh.write(f"{CL.sha(CL.HERE / name)}  {name}\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
