"""Record-level comparison of my verdicts and counts with the compute agent's rows (data files only; no code import)."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
B = HERE.parent / "stageB"
MAP = {"Badd": "Badd", "Bprune": "Bprune", "Breloc": "Breloc", "BrelocP": "BrelocP"}
def key(r): return (r["arm"], r["scene"], r["episode"], r["trial"])
tot = dis = 0
for w, ms in (("old", ["qwen38_low", "glm53", "dsv41"]), ("new", ["qwen38_low", "glm53", "dsv41", "haiku55", "sonnet55_low", "gemini38_low"]), ("ph", ["qwen38_low", "glm53", "dsv41"])):
    for m in ms:
        mine = {key(r): r for r in map(json.loads, (HERE / f"v_rows_{w}_{m}.jsonl").open())}
        theirs = {key(r): r for r in map(json.loads, (B / f"rows_{w}_{m}.jsonl").open())}
        assert set(mine) == set(theirs), (w, m)
        d = 0
        for k, a in mine.items():
            b = theirs[k]
            assert a["C_rec"] == b["C_rec"] and a["solver_step"] == b["solver_step"]
            for arm in MAP:
                va = a[arm]["ok"] if a["solver_step"] else a["C_rec"]
                if va != b[arm]:
                    d += 1; print("verdict differs", w, m, k, arm, va, b[arm])
                if a["solver_step"]:
                    bd = b[arm + "_d"]
                    for f1, f2 in (("added", "added"), ("pruned", "pruned"), ("exempt", "exempt"), ("moved", "moved"), ("added_missed", "added_matched_missed"), ("added_false", "added_false"), ("joined_added", "joined_added")):
                        if a[arm][f1] != bd[f2]:
                            d += 1; print("count differs", w, m, k, arm, f1, a[arm][f1], bd[f2])
                    if "cls" in a[arm] and int(bd["class"][0]) != a[arm]["cls"]:
                        d += 1; print("class differs", w, m, k, arm, a[arm]["cls"], bd["class"])
            if w == "ph" and a["solver_step"] and a["P_ph"] != b["P_i01"]:
                d += 1; print("P differs", w, m, k)
            tot += 1
        dis += d
        print(w, m, "records", len(mine), "disagreements", d)
print("total records", tot, "disagreements", dis)
