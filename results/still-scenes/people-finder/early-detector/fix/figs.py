"""I09 fix, diagnosis: annotated crops of fit-train images (seeds 110-115 only; refuses others). Project venv.

  python figs.py OUT_NAME SCENE_ID X0 Y0 X1 Y1 [--scale S]

Left: RGB with every candidate box >= 0.25 (green kept and matched, red kept and false, orange exempt, grey
duplicate/rejected, labelled by index and score), the torso hull of kept boxes (cyan), keypoints >= 0.30 (yellow),
and each true person's labelled vertical axis 0-1.7 m (magenta for staff, white for others; dots at 0.3 m steps).
Right: the depth image of the same crop (near = bright), with the same overlays.
"""
import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fitlib as F  # noqa: E402
from sgi.geometry import in_image, project  # noqa: E402

P = F.P


def main():
    out, sid = sys.argv[1], sys.argv[2]
    x0, y0, x1, y1 = map(int, sys.argv[3:7])
    scale = float(sys.argv[sys.argv.index("--scale") + 1]) if "--scale" in sys.argv else 2.0
    man = {e["scene_id"]: e for e in F.fit_manifest()["images"]}
    e = man[sid]
    assert e["domain"] == "fit-train" and e["seed"] in F.PG.FIT_TRAIN, "figures of fit-train only"
    F.PG.check(e["rgb"])
    sc = F.load_fit_scene(e["folder"])
    obs, cam = sc.observation, sc.observation.camera
    dets = [json.loads(x) for x in (F.HERE / "fit_train_dets.jsonl").read_text().splitlines()]
    dets = [d for d in dets if d["scene_id"] == sid]
    rgb = Image.open(e["rgb"]).convert("RGB")
    dep = obs.depth.astype(np.float64)
    crop_d = dep[y0:y1, x0:x1]
    fin = crop_d[np.isfinite(crop_d) & (crop_d > 0)]
    lo, hi = (np.percentile(fin, 2), np.percentile(fin, 98)) if fin.size else (0, 1)
    g = np.clip((hi - dep) / max(hi - lo, 1e-6), 0, 1)
    g[~np.isfinite(dep)] = 0
    dimg = Image.fromarray((g * 255).astype(np.uint8)).convert("RGB")
    panels = []
    for base in (rgb, dimg):
        im = base.crop((x0, y0, x1, y1)).resize((int((x1 - x0) * scale), int((y1 - y0) * scale)), Image.NEAREST)
        dr = ImageDraw.Draw(im)

        def T(u, v):
            return ((u - x0) * scale, (v - y0) * scale)
        for d in dets:
            b = d["box"]
            if d["fate"] == "kept":
                col = (255, 140, 0) if d.get("exempt") else (230, 30, 30) if d.get("false") else (40, 220, 40)
            else:
                col = (150, 150, 150)
            dr.rectangle([T(b[0], b[1]), T(b[2], b[3])], outline=col, width=2)
            dr.text(T(b[0] + 1, b[1] + 1), f"{d['index']}:{d['score']:.2f}", fill=col)
            kp = d.get("kp")
            if d["fate"] == "kept" and kp is not None:
                if d["region"] == "hull":
                    hull = P.convex_hull([(kp[j][0], kp[j][1]) for j in (5, 6, 12, 11)])
                    dr.polygon([T(*q) for q in hull], outline=(0, 230, 230))
                for j in range(17):
                    if kp[j][2] >= 0.30:
                        u, v = T(kp[j][0], kp[j][1])
                        dr.ellipse([u - 2, v - 2, u + 2, v + 2], fill=(255, 230, 0))
        for p in sc.truth.people:
            pts = [project(cam, (p.x, p.y, z)) for z in (0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.7)]
            pts = [q for q in pts if q is not None and in_image(cam, q[0], q[1])]
            col = (255, 0, 255) if p.role else (255, 255, 255)
            for q in pts:
                u, v = T(q[0], q[1])
                dr.ellipse([u - 2, v - 2, u + 2, v + 2], outline=col)
            if pts:
                dr.text(T(pts[-1][0] + 3, pts[-1][1]), p.id, fill=col)
        panels.append(im)
    W = sum(p.width for p in panels) + 6
    canvas = Image.new("RGB", (W, panels[0].height), (0, 0, 0))
    canvas.paste(panels[0], (0, 0))
    canvas.paste(panels[1], (panels[0].width + 6, 0))
    path = F.HERE / "figs" / out
    canvas.save(path)
    print(path)


if __name__ == "__main__":
    main()
