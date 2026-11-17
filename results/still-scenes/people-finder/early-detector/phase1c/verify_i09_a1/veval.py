"""I09 A1 verifier: the evaluation (FROZEN.md sections 6-8 and 10), recomputed from the verifier's own sidecar raw
outputs (raw/rtdetr_v2_r50vd.jsonl) with the verifier's own pipeline (vpipe.py) and the frozen people scorer.

Frozen choice (recomputed in vfit.py): rtdetr_v2_r50vd, tau 0.25, band wide [1.00, 2.30], flat off, d_dup 0.50,
d_push 0.20; facing F5 (F4 post hoc). Evaluation scenes only through research_harness (distinct_scenes, scene).
Writes veval.txt, veval_scenes.jsonl, veval_people.jsonl, veval_summary.json.
"""
from __future__ import annotations

import collections
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402

I03 = paths.STILL / 'perception/depth-blob-detector'
A0_PAIRS = paths.STILL / 'facing/a0_pairs.jsonl'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(I03))
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H  # noqa: E402
import people_scorer as PS  # noqa: E402
import vpipe as VP  # noqa: E402
from sgi.geometry import bearing, wrap_angle  # noqa: E402

SET = dict(tau=0.25, band="wide", flat=False, d_dup=0.50, d_push=0.20)
REAL = ("real", "staffed", "plain")
VARS = ("F1", "F2", "F3", "F4", "F5")
NB = 2000


def set_of(scenes_dir):
    d = str(scenes_dir) + "/"
    for n in REAL:
        if f"realistic_final/{n}/" in d:
            return n
    if "dev_new/seed" in d or d.endswith("placeholder/eval/") or d.endswith("/placeholder_first14/"):
        return "placeholder"
    raise SystemExit(d)


def thresholds_for(refs):
    ths = [H.recorded_config(r)[1].thresholds for r in refs]
    for th in ths:
        PS.check_thresholds(th)
    for th in ths[1:]:
        for k in ("person_radius_m", "visibility_min_samples", "visibility_depth_tolerance_m", "grid_resolution_m"):
            assert getattr(th, k) == getattr(ths[0], k)
    return ths[0]


def err(a, b):
    return abs(math.degrees(wrap_angle(a - b)))


def vclass(a):
    return "frontal" if a < 45 else "oblique" if a < 75 else "profile" if a <= 105 else "back"


def layout(scene_id):
    for p in ("real_", "staffed_", "plain_"):
        if scene_id.startswith(p):
            return scene_id[len(p):]
    return scene_id


def wil(k, n):
    lo, hi = PS.wilson(k, n)
    return f"{k}/{n} = {k / n:.3f} [{lo:.3f}, {hi:.3f}]" if n else f"{k}/{n}"


def main():
    t0 = time.time()
    man = json.load(open(HERE / "manifest_verify.json"))
    # the stored manifest names each scene folder as it lay when it was written; key it by where it lies now
    here = lambda f: str((paths.REPO / H.relocated(f)).resolve())  # noqa: E731
    folder_rgb = {here(e["folder"]): e["rgb"] for e in man["images"] if e["part"] == "eval"}
    raw = {}
    for line in open(HERE / "raw" / "rtdetr_v2_r50vd.jsonl"):
        r = json.loads(line)
        raw[r["rgb"]] = r
    distinct = H.distinct_scenes()
    assert len(distinct) == 68
    scenes, people_rows = [], []
    for sref in distinct:
        sset = set_of(sref.scenes_dir)
        dom = "E-real" if sset in REAL else "E-ph"
        th = thresholds_for(sref.runs)
        sc = H.scene(sref.runs[0], sref.scene_id)
        obs = sc.observation
        cam_xy = obs.camera.T_world_camera[:2, 3]
        r = raw[folder_rgb[here(sref.folder)]]
        kps = list(r["kp"]) + [None] * (len(r["boxes"]) - len(r["kp"]))
        bases = [VP.box_base(obs, b, kp) for b, kp in zip(r["boxes"], kps) if b[4] >= SET["tau"]]
        ppl = VP.people(bases, cam_xy, **SET)
        res = PS.score_people([(p["x"], p["y"], p["yaw"]["F5"]) for p in ppl], sc, th,
                              exempt_xy=PS.exempt_points(sc.truth), facing="full")
        info = PS.truth_info(sc, th)
        vis = {p.id: p for p in info["visible"]}
        row = {"folder": str(sref.folder), "scene_id": sref.scene_id, "set": sset, "domain": dom,
               "layout": layout(sref.scene_id) if dom == "E-real" else sref.scene_id,
               **{k: res[k] for k in PS.POOL_KEYS}, "xy_err": res["xy_err"], "misses": res["misses"],
               "people": res["people"], "facing": res["facing"], "false_numbers": res["false_numbers"],
               "det": [{"x": p["x"], "y": p["y"], "score": p["score"], "yaw": p["yaw"]} for p in ppl],
               "matches": res["matches"]}
        # false detections: distance to nearest visible true person
        row["false_detail"] = [{"n": n, "score": ppl[n - 1]["score"],
                                "nearest_vis": min((math.dist((ppl[n - 1]["x"], ppl[n - 1]["y"]), (p.x, p.y))
                                                    for p in vis.values()), default=None)} for n in res["false_numbers"]]
        scenes.append(row)
        for num, tid in res["matches"].items():
            p = vis[tid]
            q = ppl[num - 1]
            va = abs(math.degrees(wrap_angle(bearing((p.x, p.y), cam_xy) - p.yaw)))
            people_rows.append({"scene_id": sref.scene_id, "set": sset, "domain": dom, "layout": row["layout"],
                                "pid": tid, "role": info["roles"][tid], "view_angle": va, "view": vclass(va),
                                "xy_err": math.dist((q["x"], q["y"]), (p.x, p.y)),
                                "err": {v: None if q["yaw"][v] is None else err(q["yaw"][v], p.yaw) for v in VARS}})
    with open(HERE / "veval_scenes.jsonl", "w") as fh:
        for r in scenes:
            fh.write(json.dumps(r, default=str) + "\n")
    with open(HERE / "veval_people.jsonl", "w") as fh:
        for r in people_rows:
            fh.write(json.dumps(r) + "\n")

    L = []
    S = {}

    def pooled(rows):
        P = PS.pool(rows)
        P["by_role"] = {c: [sum(1 for p in P["people"] if p["role_class"] == c and p["matched"]),
                            sum(1 for p in P["people"] if p["role_class"] == c)] for c in ("line", "group", "staff", "other")}
        return P
    L.append("== Detection ==")
    for name, sel in [(s, lambda r, s=s: r["set"] == s) for s in REAL + ("placeholder",)] + \
                     [(d, lambda r, d=d: r["domain"] == d) for d in ("E-real", "E-ph")] + [("pooled", lambda r: True)]:
        rows = [r for r in scenes if sel(r)]
        P = pooled(rows)
        S[name] = {"matched": P["matched"], "visible": P["n_visible"], "false": P["false"], "scenes": P["scenes"],
                   "exempt": P["exempt"], "near_hidden": P["near_hidden"], "merged_pairs": P["merged_pairs"],
                   "neighbour_pairs": P["neighbour_pairs"], "med_xy": P["xy_err_median"], "n_xy": len(P["xy_err"]),
                   "by_role": P["by_role"], "staff_behind": P["staff_behind"]}
        L.append(f"{name:12s} scenes {P['scenes']} recall {wil(P['matched'], P['n_visible'])} false {P['false']} "
                 f"({P['false'] / P['scenes']:.3f}/img) exempt {P['exempt']} near-hidden {P['near_hidden']} merges "
                 f"{P['merges']} merged pairs {P['merged_pairs']}/{P['neighbour_pairs']} med xy "
                 f"{P['xy_err_median']:.4f} (n {len(P['xy_err'])}) staff-behind misses {P['staff_behind']} by role {P['by_role']}")
    L.append("E-real misses and false detections:")
    for r in scenes:
        if r["domain"] == "E-real" and (r["misses"] or r["false_detail"]):
            L.append(f"  {r['scene_id']}: misses {[(m['id'], m['role_class'], m['lower_hidden']) for m in r['misses']]}; "
                     f"false {[(d['n'], round(d['score'], 3), round(d['nearest_vis'], 3)) for d in r['false_detail']]}")
    L.append("E-ph misses: " + str(collections.Counter((m['role_class']) for r in scenes if r['domain'] == 'E-ph' for m in r['misses'])))

    # facing by view class
    def fac_table(ppl, v, title):
        L.append(f"-- facing {title}, variant {v} --")
        out = {}
        for cls in ("frontal", "oblique", "profile", "back", "non-profile", "all"):
            ps = [p for p in ppl if (cls == "all" or (cls == "non-profile" and p["view"] != "profile") or p["view"] == cls)]
            e = [180.0 if p["err"][v] is None else p["err"][v] for p in ps]
            known = sum(p["err"][v] is not None for p in ps)
            w = sum(x <= 45 for x in e)
            fb = sum(x < 90 for x in e)
            med = statistics.median(e) if e else float("nan")
            out[cls] = {"n": len(ps), "known": known, "w45": w, "fb": fb, "median": med}
            L.append(f"  {cls:12s} n {len(ps):3d} known {known} w45 {w}/{len(ps)} front/back {fb}/{len(ps)} median {med:.2f}")
        return out
    for dom in ("E-real", "E-ph"):
        ps = [p for p in people_rows if p["domain"] == dom]
        S[f"facing_{dom}_F5"] = fac_table(ps, "F5", dom)
    S["facing_E-real_F4_posthoc"] = fac_table([p for p in people_rows if p["domain"] == "E-real"], "F4", "E-real POST HOC")
    pr = [p for p in people_rows if p["domain"] == "E-real" and p["view"] == "profile" and p["err"]["F5"] > 45]
    L.append("E-real profile people outside 45 (F5): " + str([(p["scene_id"], p["pid"], p["role"], round(p["err"]["F5"], 1)) for p in pr]))

    # bootstrap over the 14 E-real layouts
    er = [r for r in scenes if r["domain"] == "E-real"]
    lays = sorted({r["layout"] for r in er})
    assert len(lays) == 14
    rng = np.random.default_rng(0)
    bs = collections.defaultdict(list)
    ep = [p for p in people_rows if p["domain"] == "E-real"]
    for _ in range(NB):
        draw = [lays[i] for i in rng.integers(0, len(lays), len(lays))]
        rr = [r for l in draw for r in er if r["layout"] == l]
        pp = [p for l in draw for p in ep if p["layout"] == l]
        m = sum(r["matched"] for r in rr); v = sum(r["n_visible"] for r in rr)
        bs["recall"].append(m / v)
        bs["false"].append(sum(r["false"] for r in rr) / len(rr))
        npf = [180.0 if p["err"]["F5"] is None else p["err"]["F5"] for p in pp if p["view"] != "profile"]
        bs["np_w45"].append(sum(x <= 45 for x in npf) / len(npf))
        bs["np_med"].append(statistics.median(npf))
        bs["all_w45"].append(sum(p["err"]["F5"] is not None and p["err"]["F5"] <= 45 for p in pp) / len(pp))
    S["bootstrap"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in bs.items()}
    L.append(f"E-real layout bootstrap (14 layouts, {NB} draws, numpy default_rng(0), percentile): " +
             ", ".join(f"{k} [{a:.3f}, {b:.3f}]" for k, (a, b) in S["bootstrap"].items()))

    # A0-matched facing
    a0 = [json.loads(l) for l in open(A0_PAIRS)]
    det_by_scene = {}
    for r in er:
        det_by_scene[r["scene_id"]] = {tid: r["det"][int(n) - 1] for n, tid in r["matches"].items()}
    L.append("== A0-matched facing (section 8) ==")
    S["a0"] = {}
    for v in ("F5", "F4"):
        for model in sorted({p["row"] for p in a0}, key=lambda s: ["Qwen", "GLM", "Deep", "Haiku", "Sonnet", "Gemini"].index(next(k for k in ["Qwen", "GLM", "Deep", "Haiku", "Sonnet", "Gemini"] if s.startswith(k)))):
            ps = [p for p in a0 if p["row"] == model]
            mt, unm = [], 0
            for p in ps:
                d = det_by_scene[p["scene"]].get(p["pid"])
                if d is None:
                    unm += 1
                    continue
                y = d["yaw"][v]
                sc_ = H  # noqa
                mt.append((p, y))
            # detector error needs the true yaw: recompute from the matched person's err rows
            ok = 0
            known = 0
            inst = {}
            for p, y in mt:
                e = next(q for q in people_rows if q["scene_id"] == p["scene"] and q["pid"] == p["pid"])["err"][v]
                known += e is not None
                hit = e is not None and e <= 45
                ok += hit
                inst[(p["arm"], p["scene"], p["pid"])] = hit
            own = sum(p["err"] is not None and p["err"] <= 45 for p, _ in mt)
            own_all = sum(p["err"] is not None and p["err"] <= 45 for p in ps)
            # bootstrap over A0 layouts
            lay = sorted({p["layout"] for p in ps})
            rng = np.random.default_rng(0)
            hits = collections.defaultdict(list)
            for p, y in mt:
                e = next(q for q in people_rows if q["scene_id"] == p["scene"] and q["pid"] == p["pid"])["err"][v]
                hits[p["layout"]].append(e is not None and e <= 45)
            b = []
            for _ in range(NB):
                draw = [lay[i] for i in rng.integers(0, len(lay), len(lay))]
                hh = [h for l in draw for h in hits[l]]
                if hh:
                    b.append(sum(hh) / len(hh))
            ci = (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))
            S["a0"][f"{model}|{v}"] = {"pairs": len(ps), "matched": len(mt), "unmatched": unm, "w45": ok, "known": known,
                                      "inst": [sum(inst.values()), len(inst)], "own_same": own, "own_all": own_all, "ci": ci}
            L.append(f"  {v} {model:22s} pairs {len(ps)} det-matched {len(mt)} (unmatched {unm}) PRIMARY {ok}/{len(mt)} = "
                     f"{ok / len(mt):.3f} [boot {ci[0]:.3f}, {ci[1]:.3f}] among known {ok}/{known}; all pairs unmatched wrong "
                     f"{ok}/{len(ps)} = {ok / len(ps):.3f}; instances {sum(inst.values())}/{len(inst)}; model own same subset "
                     f"{own}/{len(mt)} = {own / len(mt):.3f}; own all {own_all}/{len(ps)} = {own_all / len(ps):.3f}")
    # which Gemini pairs are unmatched
    g = [p for p in a0 if p["row"].startswith("Gemini")]
    L.append("  Gemini unmatched pairs: " + str(collections.Counter((p["scene"], p["pid"]) for p in g if det_by_scene[p["scene"]].get(p["pid"]) is None)))

    # like for like
    vrows = [json.loads(l) for l in open(I03 / "stageA/vlm_rows.jsonl")]
    brows = {json.loads(l)["folder"]: json.loads(l) for l in open(I03 / "stageA/rows.jsonl")}
    drow = {here(r["folder"]): r for r in scenes}
    L.append("== Like for like (section 7) ==")
    S["lfl"] = {}
    for model in ["Qwen 3.8", "GLM-5.3", "DeepSeek V4.1", "Haiku 5.5", "Sonnet 5.5 low", "Gemini 3.8 low"]:
        for dom in ("E-real", "E-ph"):
            rows = [r for r in vrows if r["model"] == model and r["domain"] == dom]
            if not rows:
                continue
            lis = [r for r in rows if r["has_list"]]
            V = PS.pool(lis)
            D = PS.pool([drow[here(r["folder"])] for r in lis])
            B = PS.pool([brows[r["folder"]] for r in lis])
            Dall = PS.pool([drow[here(r["folder"])] for r in rows])
            Vall = PS.pool(rows)
            df, vf = D["facing"], V["facing"]
            S["lfl"][f"{model}|{dom}"] = {"keys": len(rows), "listed": len(lis), "vlm": [V["matched"], V["n_visible"], V["false"] / V["scenes"], V["merged_pairs"], vf["within45_full"], vf["known"]],
                                         "det": [D["matched"], D["n_visible"], D["false"] / D["scenes"], D["merged_pairs"], df["within45_full"], df["matched"], df["known"]],
                                         "blob": [B["matched"], B["n_visible"], B["false"] / B["scenes"], B["merged_pairs"], B["facing"]["within45_axis"], B["facing"]["known"]],
                                         "sens_det": [Dall["matched"], Dall["n_visible"], Dall["false"] / Dall["scenes"]],
                                         "sens_vlm": [Vall["matched"], Vall["n_visible"]]}
            L.append(f"  {model:15s} {dom} keys {len(rows)} ({len(rows) - len(lis)} no list) | VLM {V['matched']}/{V['n_visible']} = "
                     f"{V['matched'] / V['n_visible']:.3f}, false/ans {V['false'] / V['scenes']:.3f}, merged {V['merged_pairs']}/{V['neighbour_pairs']}, "
                     f"full w45 {vf['within45_full']}/{vf['known']} = {vf['within45_full'] / max(vf['known'], 1):.3f} | DET "
                     f"{D['matched']}/{D['n_visible']} = {D['matched'] / D['n_visible']:.3f}, false/key {D['false'] / D['scenes']:.3f}, "
                     f"merged {D['merged_pairs']}/{D['neighbour_pairs']}, w45 {df['within45_full']}/{df['matched']} = "
                     f"{df['within45_full'] / max(df['matched'], 1):.3f} | BLOBS {B['matched']}/{B['n_visible']} = "
                     f"{B['matched'] / B['n_visible']:.3f}, false/key {B['false'] / B['scenes']:.3f}, axis {B['facing']['within45_axis']}/{B['facing']['known']} | "
                     f"sens: DET all keys {Dall['matched']}/{Dall['n_visible']} = {Dall['matched'] / Dall['n_visible']:.3f} "
                     f"false {Dall['false'] / Dall['scenes']:.3f}; VLM empty lists {Vall['matched']}/{Vall['n_visible']} = "
                     f"{Vall['matched'] / Vall['n_visible']:.3f}")

    # criteria (E-real)
    E = S["E-real"]
    F = S["facing_E-real_F5"]
    gem = S["a0"]["Gemini 3.8 low (s23)|F5"]
    a0v = gem["w45"] / gem["matched"]
    succ = {"recall>=0.95": E["matched"] / E["visible"] >= 0.95,
            "line>=0.95": E["by_role"]["line"][0] / E["by_role"]["line"][1] >= 0.95,
            "false<=0.10": E["false"] / E["scenes"] <= 0.10, "medxy<=0.15": E["med_xy"] <= 0.15,
            "np median<=25": F["non-profile"]["median"] <= 25,
            "np w45>=0.85": F["non-profile"]["w45"] / F["non-profile"]["n"] >= 0.85,
            "profile fb>=0.90": F["profile"]["fb"] / F["profile"]["n"] >= 0.90}
    kill = {"recall<0.90": E["matched"] / E["visible"] < 0.90, "false>0.30": E["false"] / E["scenes"] > 0.30,
            "A0 Gemini<0.806": a0v < 0.806, "A0 Gemini<0.70": a0v < 0.70}
    L.append(f"== Criteria (E-real; runtime judged separately in vtime.py) == success bars {succ}; kill bars {kill}")
    Eph = S["E-ph"]
    L.append(f"E-ph: recall {Eph['matched']}/{Eph['visible']} = {Eph['matched'] / Eph['visible']:.3f} (kill line 0.90: "
             f"{'below' if Eph['matched'] / Eph['visible'] < 0.90 else 'not below'}), false {Eph['false']}/{Eph['scenes']}")
    S["criteria"] = {"success": succ, "kill": kill}
    L.append(f"{time.time() - t0:.0f} s")
    (HERE / "veval.txt").write_text("\n".join(L) + "\n")
    (HERE / "veval_summary.json").write_text(json.dumps(S, indent=1, default=str))
    print("\n".join(L))


if __name__ == "__main__":
    main()
