"""Verifier: check two post hoc claims of the fix step with my own code (vlib):
(a) the M3-audit 'lost' detections (C07 drops an A1 detection near hidden-by-rule staff): which image, why;
(b) facing of the people C07 re-lifts and matches (recomputed F5 yaw vs A1's original F5 yaw of the same box),
    over fit-train, fit-validate, development E-real and F-real. Writes verify/v_posthoc.txt."""
import sys

sys.dont_write_bytecode = True
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402

OUT = Path(__file__).resolve().parent
L = []


def say(s):
    L.append(s)
    print(s, flush=True)


def image(tag, raw, sc, th, s, variant, facing):
    obs = sc.observation
    cam = obs.camera
    cands = {c["index"]: c for c in V.P.candidates(raw, obs.depth, cam, s["tau"])}
    res = V.run_image(raw, obs.depth, cam, s, variant, ["C00", "C07"])
    hs = V.hidden_staff_marks(sc, th)
    for d in res["C00"]:
        for pid, x, y in hs:
            if math.dist(d["xy"], (x, y)) <= 0.6 and not any(math.dist(e["xy"], (x, y)) <= 0.6 for e in res["C07"]):
                c = cands[d["index"]]
                say(f"  (a) {tag}: A1 det idx {d['index']} score {d['score']:.3f} near hidden staff {pid} "
                    f"({math.dist(d['xy'], (x, y)):.2f} m) dropped by C07; flat {c['flat']} flat_share "
                    f"{c['flat_share']}, n_Q {c['n_Q']}")
    ppl = res["C07"]
    pts = [(p["xy"][0], p["xy"][1], p["yaw"]) for p in ppl]
    r = V.PS.score_people(pts, sc, th, exempt_xy=V.PS.exempt_points(sc.truth), facing="full")
    tp = {p.id: p for p in sc.truth.people}
    for num, tid in r["matches"].items():
        p = ppl[num - 1]
        if not p["m1"]:
            continue
        orig = cands[p["index"]]["yaw"][variant]
        e_new = None if p["yaw"] is None else abs(math.degrees(V.wrap_angle(p["yaw"] - tp[tid].yaw)))
        e_old = None if orig is None else abs(math.degrees(V.wrap_angle(orig - tp[tid].yaw)))
        facing.append((tag, tid, e_new, e_old))


def main():
    s, variant = V.frozen_setting()
    th = V.PS.default_thresholds()
    raw = {x["rgb"]: x for x in map(json.loads, (V.FIXDIR / "raw_fit/rtdetr_v2_r50vd.jsonl").read_text().splitlines())}
    facing = []
    for k in V.TRAIN + V.VALID:
        for row in V.ROWS:
            folder, sc = V.load_fit(k, row)
            image(f"seed{k}/{row}", raw[str(folder / "rgb.png")], sc, th, s, variant, facing)
    mane = json.loads((V.A1 / "manifest_eval.json").read_text())
    rawe = {x["rgb"]: x for x in map(json.loads, (V.A1 / "raw_eval/rtdetr_v2_r50vd.jsonl").read_text().splitlines())}
    byf = {e["folder"]: e for e in mane["images"]}
    for sref in V.H.distinct_scenes():
        e = byf.get(str(sref.folder))
        if e is None or e["domain"] != "E-real":
            continue
        sc = V.H.scene(sref.runs[0], sref.scene_id)
        image(f"dev/{sref.scene_id}", rawe[e["rgb"]], sc, V.H.recorded_config(sref.runs[0])[1].thresholds, s, variant,
              facing)
    sys.path.insert(0, str(V.I03 / "stageA"))
    import fitdata as FD
    rawf = {(x["set"], x["scene_id"]): x for x in map(json.loads, (V.A1 / "raw_fit/rtdetr_v2_r50vd.jsonl")
                                                       .read_text().splitlines())}
    for domain, sset, scene_id, loader in FD.fit_refs():
        if domain != "F-real":
            continue
        sc, thf = FD.load(loader)
        image(f"F-real/{sset}/{scene_id}", rawf[(sset, scene_id)], sc, thf, s, variant, facing)
    say(f"(b) re-lifted and matched under C07: {len(facing)}")
    w_new = sum(1 for f in facing if f[2] is not None and f[2] <= 45)
    u_new = sum(1 for f in facing if f[2] is None)
    w_old = sum(1 for f in facing if f[3] is not None and f[3] <= 45)
    u_old = sum(1 for f in facing if f[3] is None)
    say(f"    recomputed yaw: within 45 {w_new}, unknown {u_new}, other {len(facing) - w_new - u_new}")
    say(f"    A1 original yaw: within 45 {w_old}, unknown {u_old}, other {len(facing) - w_old - u_old} "
        f"(errors of the others: {[round(f[3]) for f in facing if f[3] is not None and f[3] > 45]})")
    for f in facing:
        say(f"    {f[0]} {f[1]} new {None if f[2] is None else round(f[2], 1)} old {None if f[3] is None else round(f[3], 1)}")
    (OUT / "v_posthoc.txt").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
