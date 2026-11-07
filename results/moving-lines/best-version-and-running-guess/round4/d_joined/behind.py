import sys
from collections import Counter
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "round3"))
from i_followups import load, line_of, pos, ARMS
sel, eps = load("runs_pilot3/dsv41")
print("| Arm | re-solves while joined (sim) | last is behind the robot | of which not waiting | leaves preceded within 3 s by a last behind the robot |")
print("|---|---|---|---|---|")
for arm in ARMS:
    n = beh = beh_nw = 0; leaves = pre = 0
    for lay in sel:
        e = eps.get((lay["name"], arm))
        if not e: continue
        line = line_of(lay["name"]); S = e["samples"]
        flagged = []
        for r in e["resolves"]:
            if r.get("status") != "ok": continue
            s = min(S, key=lambda x: abs(x["t"] - r["sent_t"]))
            if not s["joined"] or abs(s["t"] - r["sent_t"]) > 0.6: continue
            n += 1
            last, p = r.get("last"), pos(r)
            if last is None or last not in p: continue
            if line.project(*p[last])[0] > s["along"] + 0.25:
                beh += 1; flagged.append(r["sent_t"])
                if last not in r["truth"]["order"]: beh_nw += 1
        for ev in e.get("joins", []):
            if ev["event"] == "left":
                leaves += 1
                pre += any(ev["t"] - 3.0 <= t <= ev["t"] for t in flagged)
    print(f"| {arm} | {n} | {beh/max(n,1):.3f} | {beh_nw/max(beh,1):.2f} | {pre}/{leaves} |")
