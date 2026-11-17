"""CT4PJ's J1 failed (0.049, bar < 0.03). Is the port wrong? Zero calls; read-only on the runs.

1. J1's violations (m_ct4pj_score.behind_share's definition, unchanged), split by the arm's own logged `joined`.
2. The join test rebuilt offline from the logs: the previous re-solve's `in_turn` and `pose_world`, the robot's
   position at each re-solve's send time (linear between the sim's 1-s samples, so approximate near 0.35 m), and
   the 1.0 s dwell. Compared with the logged `joined` per re-solve and by first join.

    python3 results/moving-lines/best-version-and-running-guess/round3_check/ct4pj_join_check.py   (from the repo root, with the repo's .venv python)
"""
import collections
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "round4"))
sys.path.insert(0, str(HERE.parent / "round3"))
from m_ct4pj_score import load  # noqa: E402
from i_followups import line_of, pos  # noqa: E402

sel, eps = load("runs_pilot3/dsv41", ("CT4PJ", "CT3F"))
print("## 1. J1 violations by the arm's own joined flag\n")
for arm in ("CT4PJ", "CT3F"):
    by, lays, n = collections.Counter(), collections.Counter(), 0
    for lay in sel:
        e = eps[(lay["name"], arm)]
        line, S = line_of(lay["name"]), e["samples"]
        for r in e["resolves"]:
            if r.get("status") != "ok":
                continue
            s = min(S, key=lambda x: abs(x["t"] - r["sent_t"]))
            if not s["joined"] or abs(s["t"] - r["sent_t"]) > 0.6:
                continue
            n += 1
            last, p = r.get("last"), pos(r)
            if last is not None and last in p and line.project(*p[last])[0] > s["along"] + 0.25:
                by["arm joined" if r.get("joined") else "arm not joined"] += 1
                lays[lay["name"]] += 1
    print(f"- {arm}: {sum(by.values())} of {n} sim-joined re-solves; {dict(by)}; by layout {dict(lays.most_common())}")

print("\n## 2. The join test rebuilt from the logs (CT4PJ)\n")
agree, firsts = collections.Counter(), []
for lay in sel:
    e = eps[(lay["name"], "CT4PJ")]
    S, rs = e["samples"], [r for r in e["resolves"] if r.get("status") == "ok"]

    def robot_at(t):
        a = max((s for s in S if s["t"] <= t), key=lambda s: s["t"], default=S[0])
        b = min((s for s in S if s["t"] >= t), key=lambda s: s["t"], default=S[-1])
        if b["t"] == a["t"]:
            return a["robot"][:2]
        w = (t - a["t"]) / (b["t"] - a["t"])
        return [a["robot"][i] + w * (b["robot"][i] - a["robot"][i]) for i in (0, 1)]

    joined, since, prev, mine = False, None, None, None
    for r in rs:
        if not joined and prev is not None:
            near = prev["in_turn"] and math.dist(robot_at(r["sent_t"]), prev["pose_world"][:2]) <= 0.35
            since = (r["sent_t"] if since is None else since) if near else None
            joined = since is not None and r["sent_t"] - since >= 1.0 - 1e-9
        agree[joined == bool(r.get("joined"))] += 1
        mine = r["sent_t"] if joined and mine is None else mine
        prev = r
    logged = next((r["sent_t"] for r in rs if r.get("joined")), None)
    firsts.append((lay["name"], mine, logged))
print(f"- per re-solve: agree {agree[True]}, differ {agree[False]} ({agree[True] / sum(agree.values()):.3f})")
print(f"- first join the same in {sum(m == g for _, m, g in firsts)} of {len(firsts)} episodes; differing:")
for name, m, g in firsts:
    if m != g:
        print(f"  - {name}: rebuilt {m}, logged {g}")
