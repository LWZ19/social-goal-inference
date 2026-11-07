"""I09 A1 step 4 (FROZEN.md sections 4, 5 and 12): the 192-setting grid on FIT data only, the selection rule, the
unit check of the facing sign on fit data, the facing-variant selection, and the fit check of section 10's bars
(the one-general-fix trigger). Then hashes fit_grid.txt, fit_rows.jsonl, fit_people_chosen.jsonl and chosen.json into
FIT.sha256 (copy: CHOSEN.sha256). No evaluation scene, manifest or image is touched. Project venv.

Rule (section 5): feasible = fb_D <= 0.10 in both fit domains; maximise min_D recall_D; ties: lower max_D fb_D, lower
max_D med_D, the faster detector (median detector-only latency over the 289 fit images, DEVIATIONS.md C1), then the
least restrictive setting. None feasible: lowest max fb among min recall >= 0.95; else highest min recall among max
fb <= 0.30; else lowest max fb (each with the remaining keys of the same order as tie-breaks).
"""
from __future__ import annotations

import collections
import json
import math
import statistics
import sys
import time
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import pipeline as P  # noqa: E402
import fitdata as FD  # noqa: E402
import people_scorer as PS  # noqa: E402

RAW = HERE / "raw_fit"
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def load_raw():
    man = json.loads((HERE / "manifest_fit.json").read_text())
    meta = json.loads((RAW / "meta.json").read_text())
    assert meta["manifest_sha256"] == C.sha(HERE / "manifest_fit.json")
    raw = {}
    for d in C.DETECTORS:
        lines = [json.loads(x) for x in (RAW / f"{d}.jsonl").read_text().splitlines()]
        assert [(x["set"], x["scene_id"], x["sha256"]) for x in lines] == \
               [(i["set"], i["scene_id"], i["sha256"]) for i in man["images"]]
        raw[d] = {(x["set"], x["scene_id"]): x for x in lines}
    lat = {d: statistics.median(x["t_det_s"] for x in raw[d].values()) for d in C.DETECTORS}
    return man, meta, raw, lat


def blank():
    return {"scenes": 0, "n_visible": 0, "matched": 0, "false": 0, "exempt": 0, "near_hidden": 0, "merges": 0,
            "neighbour_pairs": 0, "merged_pairs": 0, "n_det": 0, "xy_err": [], "line": [0, 0], "staff": [0, 0]}


def add(acc, r, rows):
    acc["scenes"] += 1
    for k in ("n_visible", "matched", "false", "exempt", "near_hidden", "merges", "neighbour_pairs", "merged_pairs",
              "n_det"):
        acc[k] += r[k]
    acc["xy_err"] += r["xy_err"]
    for row in rows:
        for c in ("line", "staff"):
            if row["role_class"] == c:
                acc[c][1] += 1
                acc[c][0] += row["matched"]


def metrics(acc):
    rec = Fraction(acc["matched"], acc["n_visible"]) if acc["n_visible"] else None
    fb = Fraction(acc["false"], acc["scenes"])
    med = statistics.median(acc["xy_err"]) if acc["xy_err"] else math.inf
    return rec, fb, med


def main():
    t0 = time.monotonic()
    hashes = C.verify_frozen()
    say(f"I09 A1 fit grid, {time.strftime('%Y-%m-%dT%H:%M:%S%z')}; FROZEN.md and DEVIATIONS.md verify: {hashes}")
    man, meta, raw, lat = load_raw()
    say(f"raw fit outputs: {len(man['images'])} images x {len(C.DETECTORS)} detectors; median detector-only latency "
        f"(C1): " + ", ".join(f"{d} {lat[d] * 1000:.2f} ms" for d in C.DETECTORS))
    det_rank = {d: i for i, d in enumerate(sorted(C.DETECTORS, key=lambda d: lat[d]))}
    accs = [{dom: blank() for dom in C.FIT_DOMAINS} for _ in C.GRID]
    refs = FD.fit_refs()
    assert [(r[1], r[2]) for r in refs] == [(i["set"], i["scene_id"]) for i in man["images"]]
    tau_min = min(C.TAUS)
    for k, (domain, sset, scene_id, loader) in enumerate(refs):
        sc, th = FD.load(loader)
        obs = sc.observation
        cam_xy = obs.camera.T_world_camera[:2, 3]
        cands = {d: P.candidates(raw[d][(sset, scene_id)], obs.depth, obs.camera, tau_min) for d in C.DETECTORS}
        for i, s in enumerate(C.GRID):
            ppl = P.people_for(cands[s["detector"]], s, cam_xy, "F1")
            r, rows = C.score_scene(ppl, sc, th, "F1")
            add(accs[i][domain], r, rows)
        if k % 50 == 0:
            say(f"  scene {k + 1}/{len(refs)} {domain} {scene_id} ({time.monotonic() - t0:.0f} s)")

    rows_out = []
    for i, s in enumerate(C.GRID):
        m = {dom: metrics(accs[i][dom]) for dom in C.FIT_DOMAINS}
        min_rec = min(m[d][0] for d in C.FIT_DOMAINS)
        max_fb = max(m[d][1] for d in C.FIT_DOMAINS)
        max_med = max(m[d][2] for d in C.FIT_DOMAINS)
        feasible = all(m[d][1] <= Fraction(1, 10) for d in C.FIT_DOMAINS)
        rows_out.append({"i": i, "setting": s, "key": C.setting_key(s), "feasible": feasible,
                         "min_recall": min_rec, "max_fb": max_fb, "max_med": max_med,
                         "per_domain": {d: {k: v for k, v in accs[i][d].items() if k != "xy_err"} |
                                        {"recall": m[d][0], "fb": m[d][1], "med": m[d][2],
                                         "n_xy": len(accs[i][d]["xy_err"])} for d in C.FIT_DOMAINS}})

    def k_main(r):
        return (-r["min_recall"], r["max_fb"], r["max_med"], det_rank[r["setting"]["detector"]],
                C.restrictive_key(r["setting"]))

    feas = [r for r in rows_out if r["feasible"]]
    if feas:
        chosen, why = min(feas, key=k_main), "step 2-3: feasible (fb <= 0.10 in both fit domains), highest min recall"
    else:
        c1 = [r for r in rows_out if r["min_recall"] >= Fraction(95, 100)]
        c2 = [r for r in rows_out if r["max_fb"] <= Fraction(3, 10)]
        if c1:
            chosen, why = min(c1, key=lambda r: (r["max_fb"], -r["min_recall"], r["max_med"],
                                                 det_rank[r["setting"]["detector"]], C.restrictive_key(r["setting"]))), \
                "fallback 4a: none feasible; lowest max fb among min recall >= 0.95"
        elif c2:
            chosen, why = min(c2, key=k_main), "fallback 4b: none feasible; highest min recall among max fb <= 0.30"
        else:
            chosen, why = min(rows_out, key=lambda r: (r["max_fb"], -r["min_recall"], r["max_med"],
                                                       det_rank[r["setting"]["detector"]],
                                                       C.restrictive_key(r["setting"]))), \
                "fallback 4c: lowest max fb"
    s_star = chosen["setting"]
    ties = [r for r in feas if k_main(r)[:3] == k_main(chosen)[:3]] if feas else []
    say(f"feasible settings: {len(feas)} of {len(rows_out)}; chosen #{chosen['i']}: {chosen['key']} ({why}); "
        f"settings tied with it on recall, fb and median before the detector and restrictiveness tie-breaks: {len(ties)}")

    # ---------------------------------------------------------------- chosen setting: per person, all variants
    people_rows = []
    per_scene = []
    for domain, sset, scene_id, loader in refs:
        sc, th = FD.load(loader)
        obs = sc.observation
        cands = P.candidates(raw[s_star["detector"]][(sset, scene_id)], obs.depth, obs.camera, tau_min)
        ppl = P.people_for(cands, s_star, obs.camera.T_world_camera[:2, 3], "F1")
        r, rows = C.score_scene(ppl, sc, th, "F1")
        for row in rows:
            people_rows.append({"domain": domain, "set": sset, "scene_id": scene_id} | row)
        per_scene.append({"domain": domain, "set": sset, "scene_id": scene_id, "n_det": r["n_det"],
                          "n_visible": r["n_visible"], "matched": r["matched"], "false": r["false"],
                          "exempt": r["exempt"], "misses": r["misses"],
                          "rejected": collections.Counter(c["reject"] for c in cands if c["reject"]),
                          "people": [{"xy": p["xy"], "score": p["score"], "box": p["box"], "yaws": p["yaws"]}
                                     for p in ppl]})

    # unit check (section 4): F1 error < 90 for > half of the frontal/back matched people with both shoulders used
    # and not edge-on, in each fit domain (None counted as failure; C11)
    unit = {}
    for dom in C.FIT_DOMAINS:
        sel = [p for p in people_rows if p["domain"] == dom and p["matched"] and
               (p["view_angle"] < 45 or p["view_angle"] > 135) and p["facing_diag"]["sh_used"] and
               p["facing_diag"]["edge_on"] is False]
        ok = sum(p["errs"]["F1"] is not None and p["errs"]["F1"] < 90 for p in sel)
        known = sum(p["errs"]["F1"] is not None for p in sel)
        unit[dom] = {"n": len(sel), "f1_lt90": ok, "share": ok / len(sel) if sel else None, "known": known,
                     "share_known": ok / known if known else None, "pass": bool(sel) and ok / len(sel) > 0.5}
    say(f"unit check (F1 sign on fit data): {unit}")
    assert all(u["pass"] for u in unit.values()), "unit check failed: a sign bug; fix and log in DEVIATIONS.md"

    # facing variant selection: within 45, unknown wrong, over the chosen setting's matched fit people
    fac = {}
    for v in P.VARIANTS:
        per = {}
        for dom in C.FIT_DOMAINS:
            m = [p for p in people_rows if p["domain"] == dom and p["matched"]]
            per[dom] = (sum(p["errs"][v] is not None and p["errs"][v] <= 45 for p in m), len(m))
        k_all = sum(per[d][0] for d in C.FIT_DOMAINS)
        n_all = sum(per[d][1] for d in C.FIT_DOMAINS)
        fac[v] = {"per_domain": per, "min": min(Fraction(*per[d]) for d in C.FIT_DOMAINS),
                  "pooled": Fraction(k_all, n_all)}
    variant = max(P.VARIANTS, key=lambda v: (fac[v]["min"], fac[v]["pooled"], -P.VARIANTS.index(v)))
    say("facing variants (within 45, unknown wrong, matched fit people of the chosen setting): " +
        "; ".join(f"{v}: " + ", ".join(f"{d} {fac[v]['per_domain'][d][0]}/{fac[v]['per_domain'][d][1]}"
                                        for d in C.FIT_DOMAINS) + f", min {float(fac[v]['min']):.4f}, pooled "
                  f"{float(fac[v]['pooled']):.4f}" for v in P.VARIANTS))
    say(f"chosen facing variant: {variant}")

    # fit check of section 10's bars (C14), chosen setting and variant, per fit domain
    bars = {}
    for dom in C.FIT_DOMAINS:
        acc = accs[chosen["i"]][dom]
        rec, fb, med = metrics(acc)
        line_rec = Fraction(*acc["line"]) if acc["line"][1] else None
        m = [p for p in people_rows if p["domain"] == dom and p["matched"]]
        nonprof = C.facing_summary([p["errs"][variant] for p in m if p["view"] != "profile"])
        prof = C.facing_summary([p["errs"][variant] for p in m if p["view"] == "profile"])
        b = {"recall>=0.95": rec >= Fraction(95, 100), "line recall>=0.95": None if line_rec is None else line_rec >= Fraction(95, 100),
             "false<=0.10": fb <= Fraction(1, 10), "median xy<=0.15": med <= 0.15,
             "non-profile median<=25": None if not nonprof["n"] else nonprof["median_u"] <= 25,
             "non-profile within45>=0.85": None if not nonprof["n"] else nonprof["w45_share"] >= 0.85,
             "profile front/back>=0.90": None if not prof["n"] else prof["fb_share"] >= 0.90}
        bars[dom] = {"values": {"recall": [acc["matched"], acc["n_visible"]], "line": acc["line"],
                                "staff": acc["staff"], "false": [acc["false"], acc["scenes"]], "median_xy": med,
                                "non_profile": nonprof, "profile": prof},
                     "bars": b, "all_met": all(x in (True, None) for x in b.values())}
        say(f"fit bars {dom}: {b}; recall {acc['matched']}/{acc['n_visible']}, line {acc['line']}, staff "
            f"{acc['staff']}, false {acc['false']}/{acc['scenes']}, median xy {med:.4f}; non-profile n {nonprof['n']} "
            f"median_u {nonprof['median_u']} w45 {nonprof['w45']}/{nonprof['n']}; profile n {prof['n']} fb "
            f"{prof['fb']}/{prof['n']}")
    fix_considered = not all(bars[d]["all_met"] for d in C.FIT_DOMAINS)
    say(f"one general fix considered (a fit-measurable success bar missed in a fit domain): {fix_considered}")

    # ---------------------------------------------------------------- write and hash
    def fr(x):
        return float(x) if isinstance(x, Fraction) else x

    with open(HERE / "fit_rows.jsonl", "w") as fh:
        for r in rows_out:
            fh.write(json.dumps({k: (fr(v) if k in ("min_recall", "max_fb") else v) for k, v in r.items()
                                 if k != "per_domain"} | {"per_domain": {d: {k: fr(v) for k, v in pd.items()}
                                                                         for d, pd in r["per_domain"].items()}},
                                default=fr) + "\n")
    with open(HERE / "fit_people_chosen.jsonl", "w") as fh:
        for p in people_rows:
            fh.write(json.dumps(p, default=str) + "\n")
    with open(HERE / "fit_scenes_chosen.jsonl", "w") as fh:
        for p in per_scene:
            fh.write(json.dumps(p, default=str) + "\n")
    G = [f"I09 A1 fit grid (FROZEN.md section 5), fit data only: F-real 21 scenes, F-ph 268 scenes. Written "
         f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')}.",
         "recall = matched/visible (I03 scorer, 0.5 m); fb = false people per scene (exempt not counted); med = median "
         "xy error (m) of matched pairs; feasible = fb <= 0.10 in both domains.",
         "Median detector-only latency (C1, 289 fit images after 5 warm-ups): " +
         ", ".join(f"{d} {lat[d] * 1000:.2f} ms" for d in C.DETECTORS),
         "i | setting | F-real recall | F-real fb | F-real med | F-ph recall | F-ph fb | F-ph med | min recall | max fb "
         "| max med | feasible"]
    for r in rows_out:
        pr, pp = r["per_domain"]["F-real"], r["per_domain"]["F-ph"]
        G.append(f"{r['i']:3d} | {r['key']} | {pr['matched']}/{pr['n_visible']} = {float(pr['recall']):.4f} | "
                 f"{pr['false']}/{pr['scenes']} = {float(pr['fb']):.4f} | {pr['med']:.4f} | "
                 f"{pp['matched']}/{pp['n_visible']} = {float(pp['recall']):.4f} | {pp['false']}/{pp['scenes']} = "
                 f"{float(pp['fb']):.4f} | {pp['med']:.4f} | {float(r['min_recall']):.4f} | {float(r['max_fb']):.4f} | "
                 f"{r['max_med']:.4f} | {'yes' if r['feasible'] else 'no'}")
    G.append("")
    G.append(f"feasible: {len(feas)} of {len(rows_out)}")
    G.append(f"CHOSEN #{chosen['i']}: {chosen['key']} ({why})")
    G += LOG[-12:]
    (HERE / "fit_grid.txt").write_text("\n".join(G) + "\n")
    chosen_json = {
        "setting": s_star, "key": chosen["key"], "index": chosen["i"], "chosen_by": why,
        "band_m": P.BANDS[s_star["band"]], "facing_variant": variant,
        "fit_metrics": {d: {k: fr(v) for k, v in chosen["per_domain"][d].items()} for d in C.FIT_DOMAINS},
        "detector_latency_median_s": lat, "n_feasible": len(feas),
        "facing_selection": {v: {"per_domain": fac[v]["per_domain"], "min": float(fac[v]["min"]),
                                 "pooled": float(fac[v]["pooled"])} for v in P.VARIANTS},
        "unit_check": unit, "fit_bars": bars, "one_general_fix_considered": fix_considered,
        "frozen_sha256": hashes, "fit_rows_sha256": C.sha(HERE / "fit_rows.jsonl"),
        "fit_people_chosen_sha256": C.sha(HERE / "fit_people_chosen.jsonl"),
        "written": time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    C.jdump(chosen_json, HERE / "chosen.json")
    lines = [f"{C.sha(HERE / n)}  {n}" for n in ("fit_grid.txt", "fit_rows.jsonl", "fit_people_chosen.jsonl",
                                                 "fit_scenes_chosen.jsonl", "chosen.json")]
    (HERE / "FIT.sha256").write_text("\n".join(lines) + "\n")
    (HERE / "CHOSEN.sha256").write_text("\n".join(lines) + "\n")
    say(f"wrote and hashed FIT.sha256 / CHOSEN.sha256 at {time.strftime('%Y-%m-%dT%H:%M:%S%z')}; "
        f"{time.monotonic() - t0:.0f} s")
    (HERE / "fit_grid.log").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
