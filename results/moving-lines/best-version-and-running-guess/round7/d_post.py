"""Round 7 part D, POST HOC DESCRIPTION (written after reading c_gaps_ceiling_out.txt; picks nothing, claims nothing).

    .venv/bin/python round7/d_post.py SELECTION_JSON SCENARIO_DIR CEILING_ROOT \
        CONFIRM_ROOT

1. ROUND7 reading T asks for the layouts where OW scores below 0.5, with their reasons: per layout, OW's strict share,
   its share of window seconds at the front, its arrival loss, its top failing reason, and the selection's flags.
2. Each arm against CT3F (the best hand-built arm) and as a share of OW (T's "report methods also as a share of OW"),
   at band top 1.2: paired mean, sign test, lower bound.
3. CT4OT's goals when nobody is ahead: distance to slot 0 and heading error, to explain its front pass rate.
"""
import json
import math
import statistics
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from a_loss import MOVING, REPO, Line, episode as loss_episode, off, window  # noqa: E402
from c_gaps import gap, load  # noqa: E402


def main():
    sel_path, scen_dir, croot, froot = sys.argv[1:5]
    sel = json.loads((MOVING / sel_path).read_text())["layouts"]
    names = [lay["name"] for lay in sel]
    sc = {**load(sel, croot, ("OW", "CT4OT", "CT4O", "CT4PJT")), **load(sel, froot, ("CT4PJ", "CT3F"))}
    print(f"# Round 7 part D, POST HOC DESCRIPTION ({sel_path}; {croot}; {froot})\n")

    print("## 1. Layouts where OW scores below 0.5 (reading T)\n")
    print("| layout | stratum | lag | restless | members | OW strict | front share | arrival lost | top reason (s) |")
    print("|---|---|---|---|---|---|---|---|---|")
    low, high = [], []
    for lay in sel:
        hits = sorted((MOVING / croot).glob(f"**/{lay['name']}/seed{lay['seed']}/OW/episode.json"))
        if not hits:
            continue
        e = json.loads(hits[0].read_text())
        scen = yaml.safe_load((REPO / scen_dir / f"{lay['name']}.yaml").read_text())
        (q,) = scen["queues"].values()
        x = loss_episode(e, Line(q["slots"]))
        if x["strict"] is None:
            continue
        _, win = window(e)
        front = sum(1 for s in win if s["ahead"] == 0) / len(win)
        rec = (lay, x, front)
        (low if x["strict"] < 0.5 else high).append(rec)
        if x["strict"] < 0.5:
            top = max(x["fails"].items(), key=lambda kv: kv[1]) if x["fails"] else (("-", "-"), 0)
            print(f"| {lay['name']} | {lay.get('stratum')} | {lay.get('lag')} | {lay.get('restless')} | "
                  f"{lay.get('members')} | {x['strict']:.3f} | {front:.2f} | {x['arrival']:.2f} | "
                  f"{top[0][0]}, {top[0][1]} ({top[1]}) |")
    for name, grp in (("OW < 0.5", low), ("OW >= 0.5", high)):
        if not grp:
            continue
        lag = sum(bool(l.get("lag")) for l, _, _ in grp)
        rest = sum(bool(l.get("restless")) for l, _, _ in grp)
        strata = {}
        for l, _, _ in grp:
            strata[l.get("stratum")] = strata.get(l.get("stratum"), 0) + 1
        print(f"\n- {name}: {len(grp)} layouts; lag {lag}, restless {rest}; strata {strata}; front share mean "
              f"{statistics.mean(f for _, _, f in grp):.2f}; arrival lost mean "
              f"{statistics.mean(x['arrival'] for _, x, _ in grp):.2f}")

    print("\n## 2. Against CT3F, and as a share of OW (band top 1.2)\n")
    print("| arm − CT3F | n | mean | front part | behind part | wins/losses/ties | sign p | 95% lower bound |"
          " arm mean / OW mean |")
    print("|---|---|---|---|---|---|---|---|---|")
    ow = statistics.mean(sc[(n, "OW")][1.2]["all"] for n in names if sc.get((n, "OW")))
    for arm in ("CT4OT", "CT4O", "CT4PJT", "CT4PJ"):
        g = gap(sc, names, arm, "CT3F", 1.2)
        m = statistics.mean(sc[(n, arm)][1.2]["all"] for n in names if sc.get((n, arm)))
        print(f"| {arm} − CT3F | {g['n']} | {g['mean']:+.3f} | {g['front']:+.3f} | {g['behind']:+.3f} | "
              f"{g['w']}/{g['l']}/{g['t']} | {g['p']:.3f} | {g['lb']:+.3f} | {m / ow:.2f} |")
    m3 = statistics.mean(sc[(n, "CT3F")][1.2]["all"] for n in names if sc.get((n, "CT3F")))
    print(f"| CT3F | | | | | | | | {m3 / ow:.2f} |")

    print("\n## 3. CT4OT's goals when nobody is ahead\n")
    d0, hd = [], []
    for lay in sel:
        hits = sorted((MOVING / croot).glob(f"**/{lay['name']}/seed{lay['seed']}/CT4OT/episode.json"))
        if not hits:
            continue
        e = json.loads(hits[0].read_text())
        scen = yaml.safe_load((REPO / scen_dir / f"{lay['name']}.yaml").read_text())
        (q,) = scen["queues"].values()
        line = Line(q["slots"])
        start, _ = window(e)
        if start is None:
            continue
        for r in e["resolves"]:
            if (r.get("status") != "ok" or r.get("truth") is None or r.get("pose_world") is None
                    or r["sent_t"] < start - 1e-9 or r["truth"]["ahead"]):
                continue
            gx, gy, gyaw = r["pose_world"]
            d0.append(math.hypot(gx - line.slot0["x"], gy - line.slot0["y"]))
            hd.append(off(gyaw, math.atan2(line.face0[1] - gy, line.face0[0] - gx)))
    if d0:
        q = statistics.quantiles(d0, n=4)
        h = statistics.quantiles(hd, n=4)
        print(f"- {len(d0)} re-solves: goal to slot 0 median {statistics.median(d0):.2f} m, quartiles {q[0]:.2f} / "
              f"{q[2]:.2f}; within 0.6 m {sum(v <= 0.6 for v in d0) / len(d0):.3f}")
        print(f"- heading error to the counter point: median {statistics.median(hd):.1f} deg, quartiles {h[0]:.1f} / "
              f"{h[2]:.1f}; within 45 deg {sum(v <= 45 for v in hd) / len(hd):.3f}")


if __name__ == "__main__":
    main()
