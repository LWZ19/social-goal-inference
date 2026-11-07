"""I09 A1 verifier: the fit grid (192 settings), the selection rule, the unit check, facing selection (F1-F4, then
F1-F5) and the fix trigger, recomputed from the cached fit raw outputs (a1/raw_fit, checked bit-identical with the
verifier's own sidecar pass on the 29 worked examples) with the verifier's own pipeline (vpipe.py).

Fit scenes are loaded with phase1b/i03/stageA/fitdata.py (fit_refs, load), as FROZEN.md section 1 says.
Writes vfit_rows.jsonl, vfit.txt, vfit_people_chosen.jsonl.
"""
from __future__ import annotations

import itertools
import json
import math
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, 'results/2026-10-17-beyond-vlm/phase1b/i03/stageA')
sys.path.insert(0, 'results/2026-10-17-beyond-vlm/phase1b/i03')
sys.path.insert(0, 'docs/model_notes/tools')
import fitdata as FD  # noqa: E402
import people_scorer as PS  # noqa: E402
import vpipe as VP  # noqa: E402
from sgi.geometry import bearing, wrap_angle  # noqa: E402

RAW = HERE.parent / "a1" / "raw_fit"
DETS = ("rtdetr_v2_r50vd", "dfine_large")
TAUS = (0.25, 0.40, 0.55, 0.70)
BANDS = ("wide", "narrow")
FLATS = (False, True)
DDUPS = (0.30, 0.50)
DPUSH = (0.00, 0.10, 0.20)
DOMS = ("F-real", "F-ph")
VARS = ("F1", "F2", "F3", "F4", "F5")


def settings():
    return [dict(detector=d, tau=t, band=b, flat=f, d_dup=u, d_push=p)
            for d, t, b, f, u, p in itertools.product(DETS, TAUS, BANDS, FLATS, DDUPS, DPUSH)]


def key(s):
    return (s["detector"], s["tau"], s["band"], s["flat"], s["d_dup"], s["d_push"])


def raw_index():
    idx = {}
    for d in DETS:
        for line in open(RAW / f"{d}.jsonl"):
            r = json.loads(line)
            k = (d, r["set"], r["scene_id"])
            assert k not in idx
            kp = [None] * len(r["boxes"])
            for p in r["pose"]:
                kp[p["box_index"]] = p["kp"]
            idx[k] = (r["boxes"], kp, r["t_det_s"])
    return idx


def err_deg(a, b):
    return abs(math.degrees(wrap_angle(a - b)))


def view_angle(p, cam_xy):
    return abs(math.degrees(wrap_angle(bearing((p.x, p.y), cam_xy) - p.yaw)))


def view_class(a):
    return "frontal" if a < 45 else "oblique" if a < 75 else "profile" if a <= 105 else "back"


def main():
    t0 = time.time()
    idx = raw_index()
    refs = FD.fit_refs()
    assert len(refs) == 289
    S = settings()
    assert len(S) == 192
    acc = {key(s): {d: {"m": 0, "v": 0, "f": 0, "n": 0, "xy": [], "line": [0, 0]} for d in DOMS} for s in S}
    bases_cache = {}
    lat = {d: [] for d in DETS}
    for k, (dom, sset, sid, loader) in enumerate(refs):
        sc, th = FD.load(loader)
        obs = sc.observation
        cam_xy = obs.camera.T_world_camera[:2, 3]
        ex = PS.exempt_points(sc.truth)
        for d in DETS:
            boxes, kps, tdet = idx[(d, sset, sid)]
            lat[d].append(tdet)
            bases = [VP.box_base(obs, b, kp) for b, kp in zip(boxes, kps) if b[4] >= 0.25]
            bases_cache[(d, sset, sid)] = bases
            for s in S:
                if s["detector"] != d:
                    continue
                ppl = VP.people(bases, cam_xy, tau=s["tau"], band=s["band"], flat=s["flat"], d_dup=s["d_dup"],
                                d_push=s["d_push"])
                r = PS.score_people([(p["x"], p["y"], None) for p in ppl], sc, th, exempt_xy=ex)
                a = acc[key(s)][dom]
                a["m"] += r["matched"]; a["v"] += r["n_visible"]; a["f"] += r["false"]; a["n"] += 1
                a["xy"] += r["xy_err"]
                for pp in r["people"]:
                    if pp["role_class"] == "line":
                        a["line"][0] += pp["matched"]; a["line"][1] += 1
        if k % 50 == 0:
            print(f"{k}/289 {time.time() - t0:.0f}s", flush=True)
    rows = []
    for s in S:
        row = dict(s)
        for d in DOMS:
            a = acc[key(s)][d]
            row[d] = {"matched": a["m"], "visible": a["v"], "recall": a["m"] / a["v"], "false": a["f"],
                      "scenes": a["n"], "fb": a["f"] / a["n"], "med_xy": statistics.median(a["xy"]) if a["xy"] else None,
                      "line": a["line"]}
        row["min_recall"] = min(row[d]["recall"] for d in DOMS)
        row["max_fb"] = max(row[d]["fb"] for d in DOMS)
        row["max_med"] = max(row[d]["med_xy"] for d in DOMS)
        rows.append(row)
    med_lat = {d: statistics.median(lat[d]) for d in DETS}

    def tiebreak(r):
        return (r["max_fb"], r["max_med"], med_lat[r["detector"]], r["tau"], 0 if r["band"] == "wide" else 1,
                int(r["flat"]), r["d_dup"], r["d_push"])
    L = []
    feas = [r for r in rows if all(r[d]["fb"] <= 0.10 for d in DOMS)]
    L.append(f"feasible (fb <= 0.10 in both domains): {len(feas)} of {len(rows)}")
    if feas:
        best = max(r["min_recall"] for r in feas)
        cands = sorted([r for r in feas if r["min_recall"] == best], key=tiebreak)
        how = "rule 1-3"
    else:
        a = [r for r in rows if r["min_recall"] >= 0.95]
        if a:
            lo = min(r["max_fb"] for r in a)
            cands = sorted([r for r in a if r["max_fb"] == lo], key=lambda r: (-r["min_recall"],) + tiebreak(r))
            how = "fallback 4a"
        else:
            b = [r for r in rows if r["max_fb"] <= 0.30]
            if b:
                hi = max(r["min_recall"] for r in b)
                cands = sorted([r for r in b if r["min_recall"] == hi], key=tiebreak)
                how = "fallback 4b"
            else:
                lo = min(r["max_fb"] for r in rows)
                cands = sorted([r for r in rows if r["max_fb"] == lo], key=lambda r: (-r["min_recall"],) + tiebreak(r))
                how = "fallback 4c"
    ch = cands[0]
    L.append(f"selection: {how}; candidates at the top value: {len(cands)}; detector-only median latency (s): {med_lat}")
    for r in cands[:6]:
        L.append(f"  cand {key(r)} min_recall {r['min_recall']:.4f} max_fb {r['max_fb']:.4f} max_med {r['max_med']:.4f}")
    L.append(f"CHOSEN {key(ch)}: " + "; ".join(
        f"{d} recall {ch[d]['matched']}/{ch[d]['visible']} = {ch[d]['recall']:.4f}, false {ch[d]['false']}/{ch[d]['scenes']}"
        f" = {ch[d]['fb']:.4f}, med xy {ch[d]['med_xy']:.4f}, line {ch[d]['line']}" for d in DOMS))
    # top 10 by min recall with max_fb <= 0.30 (context)
    b = sorted([r for r in rows if r["max_fb"] <= 0.30], key=lambda r: (-r["min_recall"],) + tiebreak(r))
    L.append("top 8 with max fb <= 0.30 (by min recall, then tie-break):")
    for r in b[:8]:
        L.append(f"  {key(r)} min_recall {r['min_recall']:.4f} max_fb {r['max_fb']:.4f} max_med {r['max_med']:.4f}")
    with open(HERE / "vfit_rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    # facing on the chosen setting's matched fit people
    per = []
    for dom, sset, sid, loader in refs:
        sc, th = FD.load(loader)
        obs = sc.observation
        cam_xy = obs.camera.T_world_camera[:2, 3]
        bases = bases_cache[(ch["detector"], sset, sid)]
        ppl = VP.people(bases, cam_xy, tau=ch["tau"], band=ch["band"], flat=ch["flat"], d_dup=ch["d_dup"],
                        d_push=ch["d_push"])
        r = PS.score_people([(p["x"], p["y"], None) for p in ppl], sc, th, exempt_xy=PS.exempt_points(sc.truth))
        vis = {p.id: p for p in PS.truth_info(sc, th)["visible"]}
        roles = PS.truth_info(sc, th)["roles"]
        for num, tid in r["matches"].items():
            p = vis[tid]
            q = ppl[num - 1]
            b0 = bases[q["i"]]
            va = view_angle(p, cam_xy)
            per.append({"domain": dom, "set": sset, "scene": sid, "pid": tid, "role": roles[tid], "view_angle": va,
                        "view": view_class(va), "true_yaw": p.yaw,
                        "yaw": q["yaw"], "sh_used": "sh" in b0["vecs"], "edge": b0["edge"],
                        "err": {v: (None if q["yaw"][v] is None else err_deg(q["yaw"][v], p.yaw)) for v in VARS}})
    with open(HERE / "vfit_people_chosen.jsonl", "w") as fh:
        for p in per:
            fh.write(json.dumps(p) + "\n")
    L.append(f"matched fit people (chosen setting): " + ", ".join(f"{d} {sum(p['domain'] == d for p in per)}" for d in DOMS))

    def w45(v, d):
        ps = [p for p in per if p["domain"] == d]
        return sum(p["err"][v] is not None and p["err"][v] <= 45 for p in ps), len(ps)
    for name, vs in (("F1-F4 (frozen)", VARS[:4]), ("F1-F5 (with the fix)", VARS)):
        sc_ = []
        for v in vs:
            k = [w45(v, d) for d in DOMS]
            mn = min(a / b for a, b in k)
            pooled = sum(a for a, _ in k) / sum(b for _, b in k)
            sc_.append((v, mn, pooled, k))
            L.append(f"  {name}: {v} " + ", ".join(f"{d} {a}/{b}" for d, (a, b) in zip(DOMS, k)) + f"; min {mn:.4f}; pooled {pooled:.4f}")
        best = sorted(sc_, key=lambda t: (-t[1], -t[2], vs.index(t[0])))[0]
        L.append(f"FACING CHOICE {name}: {best[0]}")
    # unit check (F1, error < 90, among frontal/back people with both shoulders used and not edge-on)
    for d in DOMS:
        ps = [p for p in per if p["domain"] == d and (p["view_angle"] < 45 or p["view_angle"] > 135)
              and p["sh_used"] and not p["edge"]]
        ok = sum(p["err"]["F1"] is not None and p["err"]["F1"] < 90 for p in ps)
        L.append(f"unit check {d}: F1 error < 90 in {ok}/{len(ps)}")
    # the section 10 bars on fit data, per variant F4 and F5
    for v in ("F4", "F5"):
        for d in DOMS:
            ps = [p for p in per if p["domain"] == d]
            npf = [p for p in ps if p["view"] != "profile"]
            pf = [p for p in ps if p["view"] == "profile"]
            e = [180.0 if p["err"][v] is None else p["err"][v] for p in npf]
            fb = sum(p["err"][v] is not None and p["err"][v] < 90 for p in pf)
            w = sum(x <= 45 for x in e)
            back = [p for p in ps if p["view"] == "back"]
            L.append(f"bars {v} {d}: non-profile median {statistics.median(e) if e else float('nan'):.1f} deg, within45 "
                     f"{w}/{len(e)}; profile front/back {fb}/{len(pf)}; back within45 "
                     f"{sum(p['err'][v] is not None and p['err'][v] <= 45 for p in back)}/{len(back)}")
    # face-rule firings: F3 differs from F2
    for d in DOMS:
        ps = [p for p in per if p["domain"] == d]
        fl = [p for p in ps if p["yaw"]["F2"] is not None and p["yaw"]["F3"] is not None
              and abs(wrap_angle(p["yaw"]["F2"] - p["yaw"]["F3"])) > 1e-9]
        from collections import Counter
        L.append(f"face-rule flips {d}: {len(fl)} by view {dict(Counter(p['view'] for p in fl))}; back people "
                 f"{sum(p['view'] == 'back' for p in ps)}")
    L.append(f"{time.time() - t0:.0f} s")
    (HERE / "vfit.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
