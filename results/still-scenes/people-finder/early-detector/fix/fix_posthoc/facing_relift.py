"""POST HOC (after the choice and the checks; changes nothing): facing of the chosen fix's re-lifted, matched people.
For every matched person whose detection the HR block re-lifted, the facing error (F5, deg) with the recomputed yaw
(the chosen fix, FIX_PROTOCOL.md section 3) and with A1's original yaw for the same box. Sets: fit-train, fit-validate,
A1 development E-real, A1 F-real. Project venv; writes facing_relift.txt next to this file."""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fix"))
import fixlib as X  # noqa: E402

F, P, PS, C, H = X.F, X.P, X.PS, X.C, X.H
sys.path.insert(0, str(F.RESEARCH / "phase1b/i03/stageA"))
import fitdata as FD  # noqa: E402

ch = json.loads((F.HERE / "chosen_fix2.json").read_text())
BEST = ch["chosen"]
assert BEST == "C07" and ch["checks"]["outcome"] == "confirmed"
L = []


def one(name, raw, sc, th, s, v):
    obs = sc.observation
    cam = obs.camera
    cands = P.candidates(raw, obs.depth, cam, s["tau"])
    a1yaw = {c["index"]: c.get("yaw") for c in cands}
    ppl = X.run_all(raw, obs.depth, cam, s, v, [BEST])[BEST]
    r, prow = C.score_scene(ppl, sc, th, v)
    by = {p.id: p for p in sc.truth.people}
    cam_xy = cam.T_world_camera[:2, 3]
    for num, tid in r["matches"].items():
        d = ppl[num - 1]
        if not d["m1"]:
            continue
        t = by[tid]
        e_new = C.err_deg(d["yaw"], t.yaw)
        e_old = C.err_deg(a1yaw[d["index"]][v], t.yaw)
        view = C.view_class(C.view_angle_deg(t, cam_xy))
        L.append(f"  {name:13s} {sc.inputs.scene_id if hasattr(sc, 'inputs') else ''} {tid} {view:8s} "
                 f"recomputed {('-' if e_new is None else f'{e_new:6.1f}')} | A1 yaw "
                 f"{('-' if e_old is None else f'{e_old:6.1f}')}")


def main():
    s, v = F.frozen_setting()
    raw = F.raw_lines(F.HERE / "raw_fit/rtdetr_v2_r50vd.jsonl")
    th = PS.default_thresholds()
    for e in F.fit_manifest()["images"]:
        sc = F.load_fit_scene(e["folder"])
        one(e["domain"], raw[e["rgb"]], sc, th, s, v)
    mane = json.loads((F.A1 / "manifest_eval.json").read_text())
    rawe = {x["rgb"]: x for x in map(json.loads, (F.A1 / "raw_eval/rtdetr_v2_r50vd.jsonl").read_text().splitlines())}
    by_folder = {e["folder"]: e for e in mane["images"]}
    for sref in H.distinct_scenes():
        e = by_folder[str(sref.folder)]
        if e["domain"] == "E-real":
            one("dev E-real", rawe[e["rgb"]], H.scene(sref.runs[0], sref.scene_id), C.thresholds_for(sref.runs), s, v)
    rawf = {(x["set"], x["scene_id"]): x for x in map(json.loads, (F.A1 / "raw_fit/rtdetr_v2_r50vd.jsonl")
                                                       .read_text().splitlines())}
    for domain, sset, scene_id, loader in FD.fit_refs():
        if domain == "F-real":
            sc, thf = FD.load(loader)
            one("F-real " + scene_id[-22:], rawf[(sset, scene_id)], sc, thf, s, v)
    head = ("POST HOC: re-lifted matched people under the chosen fix C07: F5 facing error (deg) with the recomputed "
            "yaw vs A1's original yaw of the same box")
    (HERE / "facing_relift.txt").write_text(head + "\n" + "\n".join(L) + "\n")
    print(head)
    print("\n".join(L))


if __name__ == "__main__":
    main()
