"""Post-hoc 1, Q5 (added after Q1-Q4): goal and roster jitter per arm. Zero calls."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from posthoc1 import ARMS, MOVING, load  # noqa: E402


def main():
    sel, eps = load(MOVING / "runs_sweep" / (sys.argv[1] if len(sys.argv) > 1 else "dsv41"))
    print("| Arm | re-solve pairs | pose moved > 0.3 m | roster changed | 1-s samples | goal moved > 0.3 m | "
          "pose moves per minute (median over episodes) |")
    print("|---|---|---|---|---|---|---|")
    for arm in ARMS:
        n = mv = rc = ns = gm = 0
        per_min = []
        for lay in sel:
            e = eps.get((lay["name"], arm))
            if not e or not e.get("answers") or e["answers"][0].get("delivered_t") is None:
                continue
            rs = [r for r in e.get("resolves", []) if r.get("status") == "ok" and r.get("pose_world")]
            k = 0
            for a, b in zip(rs, rs[1:]):
                n += 1
                d = math.dist(a["pose_world"][:2], b["pose_world"][:2])
                mv += d > 0.3
                k += d > 0.3
                rc += set(a.get("roster") or []) != set(b.get("roster") or [])
            if len(rs) > 1:
                per_min.append(k / max((rs[-1]["sent_t"] - rs[0]["sent_t"]) / 60, 1e-6))
            t0 = e["answers"][0]["delivered_t"]
            s = [x for x in e["samples"] if x["t"] >= t0 and x.get("goal")]
            for a, b in zip(s, s[1:]):
                ns += 1
                gm += math.dist(a["goal"], b["goal"]) > 0.3
        per_min.sort()
        med = per_min[len(per_min) // 2] if per_min else float("nan")
        print(f"| {arm} | {n} | {mv / n:.3f} | {rc / n:.3f} | {ns} | {gm / ns:.3f} | {med:.1f} |")


if __name__ == "__main__":
    main()
