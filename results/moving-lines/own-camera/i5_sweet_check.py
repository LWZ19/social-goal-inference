"""Item 5, POST HOC (the run's request after RESULTS_I5.md): does CT2's true-position tracking act on people
the robot's camera never sees, and is that why CTd beats CT2 in sweet_spot_line2_moving? Zero calls; read-only.
Per scene, over the 4 models' paired cells in runs_i5:
- people never matched by a live camera track (within 0.5 m) in a CTd episode, split by queue member or not;
- CT2's re-solves whose `last` (the person it comes after) is not a queue member at that moment (behind a bystander),
  and of those, how many are people the paired CTd episode's camera never saw."""
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mcompare as C  # noqa: E402

MODELS = {"sonnet55_low": range(10, 20), "dsv41": range(10, 15), "claude_haiku55": range(10, 15), "gemini_fast": range(10, 15)}
out = defaultdict(lambda: defaultdict(list))
for d, seeds in MODELS.items():
    eps = C.load(f"runs_i5/{d}", C.EVAL5, seeds, ("CTd", "CT2"))
    for sc in C.EVAL5:
        for s in seeds:
            ctd, ct2 = eps.get((sc, s, "CTd")), eps.get((sc, s, "CT2"))
            if not ctd or not ct2:
                continue
            seen, member = set(), {}
            for f in ctd.get("det", []):
                if "error" in f:
                    continue
                live = [(x, y) for _, x, y, stale in f["tracks"] if not stale]
                for i, tx, ty, m in f["truth"]:
                    member[i] = member.get(i, False) or m
                    if any(math.hypot(tx - x, ty - y) <= 0.5 for x, y in live):
                        seen.add(i)
            unseen = set(member) - seen
            out[sc]["unseen members"].append(sum(1 for i in unseen if member[i]))
            out[sc]["unseen non-members"].append(sum(1 for i in unseen if not member[i]))
            rs = [r for r in ct2.get("resolves", []) if r.get("status") == "ok" and r.get("truth") and r.get("last") is not None]
            byst = [r for r in rs if r["last"] not in r["truth"]["order"]]
            out[sc]["CT2 re-solves"].append(len(rs))
            out[sc]["CT2 behind a non-member"].append(len(byst))
            out[sc]["... a person the camera never saw"].append(sum(1 for r in byst if r["last"] in unseen))
print("| scene | cells | unseen members / episode | unseen non-members / episode | CT2 re-solves behind a non-member | of those, never seen by the camera |")
print("|---|---|---|---|---|---|")
for sc in C.EVAL5:
    v = out[sc]
    n = len(v["CT2 re-solves"])
    tot = sum(v["CT2 re-solves"])
    b = sum(v["CT2 behind a non-member"])
    print(f"| {sc} | {n} | {statistics.mean(v['unseen members']):.2f} | {statistics.mean(v['unseen non-members']):.2f} | "
          f"{b} of {tot} ({b / max(tot, 1):.2f}) | {sum(v['... a person the camera never saw'])} of {b} |")
