"""Verifier: fit-data facing variant at the chosen setting, then the Stage A evaluation on the 68 scenes with the
verifier's own detector and scorer; per-scene comparison with the compute agent's rows.jsonl (read as data).
Optional argv[1] = grid offset in metres (post hoc sensitivity; 0 for the main run)."""
import collections
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, 'docs/model_notes/tools')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_harness as H  # noqa: E402
from sgi.config import load_config  # noqa: E402
from sgi.geometry import lift_all  # noqa: E402
import v_core as C  # noqa: E402
import v_fit as F  # noqa: E402

HERE = Path(__file__).resolve().parent
I03 = HERE.parent
OFF = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
S = json.load(open(HERE / "v_select.json"))["orig_choice"]
assert json.load(open(HERE / "v_select.json"))["adopted"]["det"] == "orig"
Z_HI = S["z_hi"]


def detect(obs):
    pts = lift_all(obs.depth, obs.camera)
    if OFF:
        pts = pts.copy()
        pts[:, :2] += OFF                          # shifts the grid origin relative to the world by -OFF
    hm = C.HeightMap(pts, S["z_lo"], S["z_hi"])
    allb = hm.blobs(S["split_r"])
    kept = [b for b in allb if C.body(b, S, Z_HI)]
    for b in kept:
        b["ax"] = hm.axes(b)
    for b in allb:                                 # kept blobs are the same objects
        if OFF:
            b["x"] -= OFF
            b["y"] -= OFF
    cells_xy = {b["_b"]: (hm.cells[b["cells"]] + 0.5) * C.RES - OFF for b in allb}
    return kept, allb, cells_xy


def facing_fit():
    th = load_config().thresholds
    acc = {v: {d: [0, 0, 0] for d in ("F-real", "F-ph")} for v in ("all", "band")}
    for dom, st, ld in F.fit_scenes():
        obs, truth = F.load(ld)
        T = C.Truth(obs, truth, th)
        kept, _, _ = detect(obs)
        for v in ("all", "band"):
            r = C.score([(b["x"], b["y"], b["ax"][v]) for b in kept], T)
            a = acc[v][dom]
            a[0] += r["matched"]
            a[1] += r["w45_axis"]
            a[2] += r["unknown"]
    share = {v: min(acc[v][d][1] / acc[v][d][0] for d in acc[v]) for v in acc}
    variant = "band" if share["band"] > share["all"] else "all"
    print("fit facing (matched, within45 axis, unknown):", acc, "min share", share, "->", variant, flush=True)
    return variant


def set_of(d):
    d = str(d) + "/"
    for n in ("real", "staffed", "plain"):
        if f"realistic_final/{n}/" in d:
            return n
    if "dev_new/seed" in d or d.endswith("placeholder/eval/"):
        return "placeholder"
    raise SystemExit(d)


def miss_category(person, kept, allb, cells_xy, matches_by_blob):
    best, bestn = None, 0
    for b in allb:
        xy = cells_xy[b["_b"]]
        n = int((((xy[:, 0] - person.x) ** 2 + (xy[:, 1] - person.y) ** 2) <= 0.25 ** 2 + 1e-12).sum())
        if n > bestn:
            best, bestn = b, n
    if best is None:
        return "no cells within 0.25 m", None
    info = (round(best["top"], 3), round(best["foot"], 3))
    if any(best is k for k in kept):
        if best["_b"] in matches_by_blob:
            return "kept, matched to another person", info
        return "kept, unmatched, centre beyond 0.5 m", info
    if best["top"] > Z_HI - 0.05 + C.EPS:
        return "rejected: top above clip", info
    if best["foot"] > S["foot_hi"] + C.EPS:
        return "rejected: footprint above foot_hi", info
    if best["top"] < S["top_lo"]:
        return "rejected: top below top_lo", info
    return "rejected: footprint below foot_lo", info


def main():
    variant = facing_fit() if not OFF else "band"
    out = []
    for ref in H.distinct_scenes():
        st = set_of(ref.scenes_dir)
        dom = "E-ph" if st == "placeholder" else "E-real"
        ths = [H.recorded_config(r)[1].thresholds for r in ref.runs]
        for th in ths:
            assert th.person_radius_m == 0.25 and th.visibility_min_samples == 3 and th.visibility_depth_tolerance_m == 0.3
        sc = H.scene(ref.runs[0], ref.scene_id)
        T = C.Truth(sc.observation, sc.truth, ths[0])
        kept, allb, cells_xy = detect(sc.observation)
        dets = [(b["x"], b["y"], b["ax"][variant]) for b in kept]
        r = C.score(dets, T)
        by_blob = {kept[num - 1]["_b"]: pid for num, pid in r["matches"].items()}
        cats = []
        for m in r["misses"]:
            p = next(q for q in T.vis if q.id == m["id"])
            cats.append(miss_category(p, kept, allb, cells_xy, by_blob))
        false_blobs = [(round(b["x"], 3), round(b["y"], 3), round(b["top"], 3), round(b["foot"], 3),
                        round(min([math.dist((b["x"], b["y"]), (q.x, q.y)) for q in sc.truth.people] or [99]), 2))
                       for i, b in enumerate(kept) if (i + 1) not in r["matches"]]
        row = {k: r[k] for k in C.SUMK}
        row.update(set=st, domain=dom, scene=ref.scene_id, errs=r["errs"], misses=r["misses"], people=r["people"],
                   cats=cats, false_blobs=false_blobs, run=ref.runs[0].folder, folder=str(ref.folder))
        out.append(row)
        print(f"{st:12s} {ref.scene_id:38s} kept {len(kept)} matched {r['matched']}/{r['vis']} false {r['false']} "
              f"exempt {r['exempt']} {cats if cats else ''} {false_blobs if r['false'] else ''}", flush=True)
    with open(HERE / (f"eval_rows_off{OFF:.3f}.jsonl"), "w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")
    # tables
    groups = collections.OrderedDict()
    for st in ("real", "staffed", "plain", "placeholder"):
        groups[st] = [r for r in out if r["set"] == st]
    groups["E-real"] = [r for r in out if r["domain"] == "E-real"]
    groups["E-ph"] = [r for r in out if r["domain"] == "E-ph"]
    groups["pooled"] = out
    for g, rs in groups.items():
        P = C.pool(rs)
        lo, hi = C.wilson(P["matched"], P["vis"])
        sb = sum(m["staff_behind"] for m in P["misses"])
        os_ = sum(bool(m["role"]) and not m["staff_behind"] for m in P["misses"])
        ns = sum(not m["role"] for m in P["misses"])
        roles = collections.Counter((c, ok) for c, ok in P["people"])
        rolestr = " ".join(f"{c} {roles[(c, True)]}/{roles[(c, True)] + roles[(c, False)]}" for c in ("line", "group", "staff", "other"))
        cats = collections.Counter(c for r in rs for c, _ in r["cats"])
        print(f"{g:12s} scenes {P['scenes']} vis {P['vis']} recall {P['matched']}/{P['vis']} = {P['recall']:.3f} "
              f"[{lo:.3f}, {hi:.3f}] false {P['false']} ({P['fb']:.3f}) exempt {P['exempt']} nh {P['near_hidden']} "
              f"merges {P['merges']} mp {P['mpairs']}/{P['npairs']} med_err {P['med_err']:.3f} (n={len(P['errs'])}) "
              f"axis {P['w45_axis']}/{P['known']} unk {P['unknown']} | misses sb/os/ns {sb}/{os_}/{ns} | {rolestr} | {dict(cats)}")
    for dom in ("E-real", "E-ph"):
        P = C.pool(groups[dom])
        rec, fb, mp = P["recall"], P["fb"], P["mp"] if P["npairs"] else 0.0
        succ = rec >= 0.95 and fb <= 0.10 and (mp <= 0.10 if P["npairs"] else True)
        kill = rec < 0.85 or fb > 0.3 or (mp > 0.2 if P["npairs"] else False)
        print(f"VERDICT {dom}: recall {rec:.4f} fb {fb:.4f} mp {mp:.4f} success {succ} kill {kill}")
    # compare with the compute agent's rows.jsonl
    if not OFF:
        theirs = {r["folder"]: r for r in map(json.loads, open(I03 / "stageA" / "rows.jsonl"))}
        assert len(theirs) == 68
        nd = 0
        for r in out:
            t = theirs[r["folder"]]
            a = (r["vis"], r["matched"], r["false"], r["exempt"], r["near_hidden"], r["merges"], r["npairs"], r["mpairs"],
                 r["n_det"], r["w45_axis"], r["known"], sum(m["staff_behind"] for m in r["misses"]))
            b = (t["n_visible"], t["matched"], t["false"], t["exempt"], t["near_hidden"], t["merges"], t["neighbour_pairs"],
                 t["merged_pairs"], t["n_det"], t["facing"]["within45_axis"], t["facing"]["known"], t["staff_behind"])
            e1 = sorted(round(x, 4) for x in r["errs"])
            e2 = sorted(t["xy_err"])
            if a != b or any(abs(x - y) > 2e-4 for x, y in zip(e1, e2)) or len(e1) != len(e2):
                nd += 1
                print("DIFF", r["scene"], a, b)
        print(f"rows.jsonl comparison: {nd} of {len(out)} scenes differ")


if __name__ == "__main__":
    main()
