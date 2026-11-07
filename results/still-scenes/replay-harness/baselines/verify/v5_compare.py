"""(a) Per-record comparison of my rows with the first agent's records.jsonl. (b) Old vs new wording for GLM and
DeepSeek on the same (arm, scene, episode, trial): are people / target / SA the same (s23 C people answers are cache
copies of s19/s20)?"""
import json, collections
D = 'results/2026-10-17-beyond-vlm/rerun/i01_baselines/'
out = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)
mine = {}
for w in ("new", "old"):
    for l in open(D + f"verify/v1_rows_{w}.jsonl"):
        r = json.loads(l); mine[(w, r["stage"], r["model"], r["arm"], r["scene"], r["ep"], r["trial"])] = r
theirs = {}
for l in open(D + "records.jsonl"):
    r = json.loads(l); theirs[(r["wording"], r["stage"], r["model"], r["arm"], r["scene"], r["episode"], r["trial"])] = r
p("keys mine", len(mine), "theirs", len(theirs), "common", len(set(mine) & set(theirs)))
diff = collections.Counter()
for k in set(mine) & set(theirs):
    a, b = mine[k], theirs[k]
    def mv(x): return a["C"] if a.get(x) is None else a[x]
    for mk, tk in (("C", "C_rec"), ("SA", "SA"), ("SG", "SG"), ("CA", "CA")):
        if bool(mv(mk)) != bool(b[tk]):
            diff[mk] += 1
    if (a["SA"] is not None) != b["goal_step"]:
        diff["goal_step"] += 1
p("per-record disagreements:", dict(diff) or "none")
# (b) old vs new for glm53 / dsv41
for model in ("glm53", "dsv41"):
    same = collections.Counter()
    for k, r in mine.items():
        if k[0] != "new" or r["model"] != model:
            continue
        ko = [kk for kk in mine if kk[0] == "old" and kk[2] == model and kk[3:] == k[3:]]
        if len(ko) != 1:
            same["no pair"] += 1; continue
        o = mine[ko[0]]
        same["pairs"] += 1
        same[f"goal_step new={r['SA'] is not None} old={o['SA'] is not None}"] += 1
        if r["SA"] is not None and o["SA"] is not None:
            same["both goal step"] += 1
            same["same n people"] += r["people"] == o["people"]
            same["same SA"] += r["SA"] == o["SA"]
            same["same C"] += r["C"] == o["C"]
    p(model, dict(same))
open(D + "verify/v5_compare.txt", "w").write("\n".join(out) + "\n")
