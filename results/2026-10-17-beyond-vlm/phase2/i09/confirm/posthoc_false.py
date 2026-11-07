"""POST HOC (written after scoring and after reading tables.txt; not in FROZEN.md section 6; changes no verdict).

Where the 44 false people of the test set come from, from the scored rows and the scenes' truth only (no image is
viewed). For each false detection: the nearest visible true person (distance, role class, matched, lower body
hidden), the nearest true person who fails is_visible (distance, role), whether that hidden person's (x, y, 1.1 m)
projects into the picture, the detector score and the box (height in px, touching an image edge). Classes, first that
applies:
  A  displaced counter staff: within 0.8 m of a visible staff member whose lower body is hidden;
  B  a hidden true person: within 0.5 m of a true person who fails is_visible (the scorer's near_hidden);
  C  near a true person: within 1.5 m of any true person (visible or hidden);
  D  far from every true person (> 1.5 m).
Also: false people per image if class A, or A and B, were not counted (sensitivity, post hoc). Writes
posthoc_false.txt. Project venv.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import collections  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import confirm_lib as CL  # noqa: E402
import c3_truth  # noqa: E402
from sgi.geometry import in_image, project  # noqa: E402

PS = CL.PS


def main():
    CL.verify_sha_file(CL.HERE / "SCORE.sha256")
    man = c3_truth.load_manifest()
    rows = [json.loads(x) for x in (CL.HERE / "rows.jsonl").read_text().splitlines()]
    prows = [json.loads(x) for x in (CL.HERE / "people_rows.jsonl").read_text().splitlines()]
    by_key = {(e["seed"], e["scene_id"]): e for e in man["entries"]}
    lh = {(p["seed"], p["scene_id"], p["pid"]): p for p in prows}
    th = PS.default_thresholds()
    out, cls = [], collections.Counter()
    by_row = collections.defaultdict(collections.Counter)
    for r in rows:
        if not r["false_detail"]:
            continue
        sc = c3_truth.load_test_scene(by_key[(r["seed"], r["scene_id"])])
        info = PS.truth_info(sc, th)
        cam = sc.observation.camera
        W, Hh = cam.width, cam.height
        for fd in r["false_detail"]:
            if fd["exempt"]:
                continue
            xy = fd["xy"]
            nv = min(((math.dist(xy, (p.x, p.y)), p) for p in info["visible"]), key=lambda t: t[0], default=None)
            nh = min(((math.dist(xy, (p.x, p.y)), p) for p in info["hidden"]), key=lambda t: t[0], default=None)
            nv_row = lh[(r["seed"], r["scene_id"], nv[1].id)] if nv else None
            disp_staff = any(math.dist(xy, (p.x, p.y)) <= 0.8 and p.role and
                             lh[(r["seed"], r["scene_id"], p.id)]["lower_hidden"] for p in info["visible"])
            near_any = min([d for d in (nv[0] if nv else None, nh[0] if nh else None) if d is not None], default=None)
            c = ("A displaced counter staff" if disp_staff else
                 "B a hidden true person (within 0.5 m)" if nh and nh[0] <= PS.NEAR_M else
                 "C near a true person (within 1.5 m)" if near_any is not None and near_any <= 1.5 else
                 "D far from every true person")
            cls[c] += 1
            by_row[r["row_id"]][c[0]] += 1
            hid_in = None
            if nh:
                mk = project(cam, (nh[1].x, nh[1].y, 1.1))
                hid_in = bool(mk is not None and in_image(cam, mk[0], mk[1]))
            b = fd["box"]
            edge = [n for n, hit in (("left", b[0] <= 2), ("top", b[1] <= 2), ("right", b[2] >= W - 2),
                                     ("bottom", b[3] >= Hh - 2)) if hit]
            out.append({"seed": r["seed"], "scene": r["scene_id"], "number": fd["number"], "class": c,
                        "score": fd["score"],
                        "nearest_visible": None if nv is None else [round(nv[0], 2), nv[1].id, nv_row["role_class"],
                                                                     nv_row["matched"], nv_row["lower_hidden"]],
                        "nearest_hidden": None if nh is None else [round(nh[0], 2), nh[1].id, nh[1].role, hid_in],
                        "box_h_px": round(b[3] - b[1], 1), "edges": edge})
    n_img = len(rows)
    tot = sum(cls.values())
    a = sum(v for k, v in cls.items() if k.startswith("A"))
    ab = a + sum(v for k, v in cls.items() if k.startswith("B"))
    L = ["POST HOC (not pre-registered; written after scoring; changes no verdict): sources of the false people",
         f"false people {tot} in {n_img} images = {tot / n_img:.3f} per image; by class: " +
         "; ".join(f"{k}: {v}" for k, v in sorted(cls.items())),
         f"sensitivity (post hoc): without class A {tot - a}/{n_img} = {(tot - a) / n_img:.3f} per image; without "
         f"classes A and B {tot - ab}/{n_img} = {(tot - ab) / n_img:.3f} per image",
         "by row (classes A/B/C/D): " + "; ".join(f"{rid} {c['A']}/{c['B']}/{c['C']}/{c['D']}"
                                                  for rid, c in sorted(by_row.items())),
         "score distribution by class (min / median / max): " + "; ".join(
             f"{k[0]} " + "/".join(f"{x:.2f}" for x in (min(s), sorted(s)[len(s) // 2], max(s)))
             for k in sorted(cls) for s in [[o["score"] for o in out if o["class"] == k]]),
         "seed | scene | number | class | score | nearest visible [m, id, role class, matched, lower hidden] | "
         "nearest hidden [m, id, role, (x, y, 1.1 m) in picture] | box height px | box touches edge"]
    for o in out:
        L.append(f"{o['seed']} | {o['scene']} | {o['number']} | {o['class'][0]} | {o['score']:.3f} | "
                 f"{o['nearest_visible']} | {o['nearest_hidden']} | {o['box_h_px']} | {o['edges']}")
    (CL.HERE / "posthoc_false.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
