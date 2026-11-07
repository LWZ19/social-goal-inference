"""I09 confirmation, check C3 (FROZEN.md section 8): truth self-check on the test scenes (reads truth only).

The true visible people, given as detections with their true yaws, must score recall 1 (or nobody visible), 0 false
and every person within 45 deg on every test scene, under PS.default_thresholds() (configs/default.yaml). Each scene:
manifest hash check, section 3 path guard, H.guard_scene, then PS.FreshScene(load_scene(folder)). Writes c3_truth.txt.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import collections  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import confirm_lib as CL  # noqa: E402
import pathguard  # noqa: E402
from sgi.scene import load_scene  # noqa: E402

H, PS = CL.H, CL.PS


def load_manifest() -> dict:
    CL.verify_sha_file(CL.HERE / "MANIFEST.sha256")
    man = json.loads((CL.HERE / "manifest.json").read_text())
    assert man["required_seeds_200_209_complete"] is True
    return man


def load_test_scene(e: dict):
    folder = Path(e["folder"])
    pathguard.check(folder)
    pathguard.check(e["rgb"])
    H.guard_scene(folder)
    for f, key in (("rgb.png", "sha256_rgb_png"), ("depth.npy", "sha256_depth_npy"),
                   ("scene.json", "sha256_scene_json")):
        assert CL.sha(folder / f) == e[key], f"sha256 mismatch: {folder / f}"
    return PS.FreshScene(load_scene(folder))


def main():
    t0 = time.monotonic()
    man = load_manifest()
    th = PS.default_thresholds()
    PS.check_thresholds(th)
    n = ok_n = 0
    vis_total = 0
    roles = collections.Counter()
    fails = []
    for e in man["entries"]:
        sc = load_test_scene(e)
        info = PS.truth_info(sc, th)
        vis = info["visible"]
        r = PS.score_people([(p.x, p.y, p.yaw) for p in vis], sc, th, exempt_xy=PS.exempt_points(sc.truth),
                            facing="full")
        ok = (r["recall"] in (1.0, None) and r["false"] == 0 and r["exempt"] == 0 and r["merges"] == 0
              and r["facing"]["within45_full"] == r["matched"] == len(vis))
        n += 1
        ok_n += ok
        vis_total += len(vis)
        roles.update(info["roles"][p.id] for p in vis)
        if not ok:
            fails.append((e["seed"], e["scene_id"], r["recall"], r["false"], r["exempt"], r["merges"], r["facing"]))
    L = [f"I09 confirmation C3 (FROZEN.md section 8), {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
         f"manifest.json sha256 {CL.sha(CL.HERE / 'manifest.json')} (MANIFEST.sha256 verified)",
         f"thresholds: configs/default.yaml via PS.default_thresholds(); PS.check_thresholds passed",
         f"{ok_n} of {n} test scenes give recall 1 (or nobody visible), 0 false, 0 exempt, 0 merges, and every "
         f"matched person within 45 deg; visible people {vis_total}; by role class {dict(roles)}",
         f"failures: {fails}",
         f"C3 {'PASS' if ok_n == n else 'FAIL'}; {time.monotonic() - t0:.0f} s"]
    (CL.HERE / "c3_truth.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    assert ok_n == n, "C3 failed"


if __name__ == "__main__":
    main()
