"""Verifier: score all 216 settings (original detector and the ADDENDUM_FIX drop-clip detector) on the fit data.

Fit data (FROZEN.md section 1): F-real = realistic worked examples via H.example (default config); F-ph = 8
placeholder worked examples via H.example + fresh seeds 100-109 via sgi.scene.load_scene. No evaluation scene.
Usage: v_fit.py SHARD NSHARDS  -> fit_rows_<SHARD>.jsonl
"""
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_harness as H  # noqa: E402
from sgi.config import load_config  # noqa: E402
from sgi.geometry import lift_all  # noqa: E402
from sgi.scene import load_scene  # noqa: E402
import v_core as C  # noqa: E402

HERE = Path(__file__).resolve().parent
FRESH = Path('results/still-scenes/perception/fresh_seeds')
GRID = {"z_lo": (0.10, 0.40), "z_hi": (1.90, 2.05, 2.20), "top_lo": (1.10, 1.30, 1.50), "split_r": (0.15, 0.25, 0.35),
        "foot_lo": (0.15, 0.30), "foot_hi": (0.70, 1.00)}
KEYS = tuple(GRID)
SETTINGS = [dict(zip(KEYS, c)) for c in itertools.product(*(GRID[k] for k in KEYS))]
CNT = ("vis", "matched", "false", "exempt", "near_hidden", "merges", "npairs", "mpairs", "n_det")


def fit_scenes():
    out = []
    for folder, sid in H.examples():
        s = str(folder)
        if '/realistic_final/' in s:
            out.append(("F-real", s.split('realistic_final/')[1].split('/')[0], ("ex", folder, sid)))
        elif s.endswith('data/placeholder/examples'):
            out.append(("F-ph", "ph_examples", ("ex", folder, sid)))
        else:
            raise SystemExit(f"unexpected examples folder {s}")
    for k in range(100, 110):
        assert k >= 100
        for split in ("eval", "examples"):
            for d in sorted((FRESH / f"seed{k}" / split).iterdir()):
                H.guard(d.name)
                out.append(("F-ph", f"seed{k}", ("fresh", d.parent, d.name)))
    return out


def load(ld):
    kind, folder, sid = ld
    if kind == "ex":
        sc = H.example(sid, folder)
        return sc.observation, sc.truth
    H.guard_scene(folder / sid)
    inputs, truth = load_scene(folder / sid)
    assert inputs.scene_id == sid
    return inputs.observation, truth


def run_scene(obs, truth, th):
    T = C.Truth(obs, truth, th)
    pts = lift_all(obs.depth, obs.camera)
    res = {}
    for det, drop in (("orig", False), ("fix", True)):
        rows = [None] * len(SETTINGS)
        for z_lo in GRID["z_lo"]:
            for z_hi in GRID["z_hi"]:
                hm = C.HeightMap(pts, z_lo, z_hi, drop_clip=drop)
                for split_r in GRID["split_r"]:
                    bl = hm.blobs(split_r)
                    for i, s in enumerate(SETTINGS):
                        if s["z_lo"] != z_lo or s["z_hi"] != z_hi or s["split_r"] != split_r:
                            continue
                        kept = [b for b in bl if C.body(b, s, z_hi)]
                        r = C.score([(b["x"], b["y"], None) for b in kept], T)
                        rows[i] = [r[k] for k in CNT] + [sum(m["staff_behind"] for m in r["misses"])]
        res[det] = rows
    return res, len(T.vis), len(truth.people)


def main():
    shard, nsh = int(sys.argv[1]), int(sys.argv[2])
    th = load_config().thresholds
    scenes = fit_scenes()
    print(f"fit scenes {len(scenes)}: F-real {sum(d == 'F-real' for d, _, _ in scenes)}, "
          f"F-ph {sum(d == 'F-ph' for d, _, _ in scenes)}", flush=True)
    t0 = time.time()
    with open(HERE / f"fit_rows_{shard}.jsonl", "w") as f:
        for n, (dom, st, ld) in enumerate(scenes):
            if n % nsh != shard:
                continue
            obs, truth = load(ld)
            res, nv, npeople = run_scene(obs, truth, th)
            f.write(json.dumps({"domain": dom, "set": st, "scene": ld[2], "folder": str(ld[1]), "vis": nv,
                                "people": npeople, "rows": res}) + "\n")
            f.flush()
            print(f"{n} {dom} {st} {ld[2]} vis {nv}/{npeople} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
