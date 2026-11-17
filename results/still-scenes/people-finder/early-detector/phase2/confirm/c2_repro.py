"""I09 confirmation, check C2 (FROZEN.md section 8): development reproduction, no test data.

The confirmation's project side and scoring (confirm_lib.score_image) run on A1's recorded raw evaluation pass
(a1/raw_eval/rtdetr_v2_r50vd.jsonl) for the 42 E-real scenes, loaded through the step-0 harness (H.distinct_scenes,
H.scene; A1's recorded thresholds through C.thresholds_for). They must reproduce a1/eval_people.json's people for
those scenes (same order; xy and yaw within 1e-9) and section 7's E-real (and real-arm) counts and medians exactly.
Writes c2_repro.txt. Project venv.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import confirm_lib as CL  # noqa: E402

C, H, PS = CL.C, CL.H, CL.PS
TOL = 1e-9


def dev_rows():
    """Score the 42 E-real development scenes with the confirmation's code: (rows, prows, people by folder, raw)."""
    for path, want in CL.DEV_RAW.items():
        assert CL.sha(path) == want, path
    s, variant = CL.frozen_setting()
    man = json.loads((CL.A1 / "manifest_eval.json").read_text())
    meta_raw = json.loads((CL.A1 / "raw_eval" / "meta.json").read_text())
    assert meta_raw["manifest_sha256"] == CL.DEV_RAW[CL.A1 / "manifest_eval.json"]
    raw = {x["rgb"]: x for x in map(json.loads, (CL.A1 / "raw_eval/rtdetr_v2_r50vd.jsonl").read_text().splitlines())}
    by_folder = {e["folder"]: e for e in man["images"]}
    distinct = H.distinct_scenes()
    assert len(distinct) == 68 and {str(d.folder) for d in distinct} == set(by_folder)
    rows, prows, people = [], [], {}
    for sref in distinct:
        e = by_folder[str(sref.folder)]
        if e["domain"] != "E-real":
            continue
        th = C.thresholds_for(sref.runs)
        sc = H.scene(sref.runs[0], sref.scene_id)
        meta = {"set": e["set"], "domain": e["domain"], "scene_id": sref.scene_id,
                "layout": C.layout_of(sref.scene_id), "folder": str(sref.folder)}
        row, pr, out = CL.score_image(raw[e["rgb"]], sc, th, s, variant, meta)
        rows.append(row)
        prows += pr
        people[str(sref.folder)] = out
    assert len(rows) == 42, len(rows)
    return rows, prows, people, raw


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= TOL


def main():
    t0 = time.monotonic()
    L = [f"I09 confirmation C2 (FROZEN.md section 8), {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
         f"Protocol and code hashes verified (C1): {json.dumps(CL.PROTOCOL_HASHES)}"]
    rows, prows, people, _ = dev_rows()
    ep = json.loads((CL.A1 / "eval_people.json").read_text())
    n_scenes = n_people = 0
    bad = []
    for folder, got in people.items():
        want = ep["scenes"][folder]["people"]
        n_scenes += 1
        if len(got) != len(want):
            bad.append((folder, "count", len(got), len(want)))
            continue
        for i, (g, w) in enumerate(zip(got, want)):
            n_people += 1
            if not (same(g["x"], w["x"]) and same(g["y"], w["y"]) and same(g["yaw"], w["yaw"])):
                bad.append((folder, i, (g["x"], g["y"], g["yaw"]), (w["x"], w["y"], w["yaw"])))
    L.append(f"People vs a1/eval_people.json: {n_scenes} E-real scenes, {n_people} people, same order; "
             f"mismatches (xy or yaw beyond {TOL}): {len(bad)}")
    for b in bad[:20]:
        L.append(f"  MISMATCH {b}")
    rec_rows, rec_prows = CL.load_dev_recorded()
    ok_all = not bad
    for name, sel in (("E-real (42 images)", lambda r: True), ("real arm (14 images)", lambda r: r["set"] == "real")):
        mine = CL.section7_numbers([r for r in rows if sel(r)], [p for p in prows if sel(p)])
        rec = CL.section7_numbers([r for r in rec_rows if sel(r)], [p for p in rec_prows if sel(p)])
        eq = mine == rec
        ok_all &= eq
        L.append(f"Section 7, {name}: confirmation code {json.dumps(mine)}")
        L.append(f"Section 7, {name}: A1 recorded rows   {json.dumps(rec)}")
        L.append(f"  exactly equal: {eq}")
    L.append(f"C2 {'PASS' if ok_all else 'FAIL'}; {time.monotonic() - t0:.0f} s")
    (CL.HERE / "c2_repro.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    assert ok_all, "C2 failed"


if __name__ == "__main__":
    main()
