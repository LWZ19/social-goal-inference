"""Compare the verifier's vtables.json with the compute agent's summary.json, number by number, and the verifier's
vrows with the compute agent's rows_<cell>.jsonl, signal by signal."""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONF = HERE.parent
mine = json.loads((HERE / "vtables.json").read_text())
theirs = json.loads((CONF / "summary.json").read_text())
TOL = 1e-12
diffs, n_cmp = [], 0


def cmp(label, a, b):
    global n_cmp
    n_cmp += 1
    if a is None or b is None:
        if a is not b and not (a is None and b is None):
            diffs.append((label, a, b))
        return
    if isinstance(a, list):
        for i, (x, y) in enumerate(zip(a, b)):
            cmp(f"{label}[{i}]", x, y)
        return
    a, b = float(a), float(b)
    if not (a == b or abs(a - b) <= TOL or (math.isnan(a) and math.isnan(b))):
        diffs.append((label, a, b))


GROUPS = {"primary": ("old_qwen", "old_glm", "old_deepseek"),
          "s23": ("new_glm", "new_deepseek", "new_haiku", "new_sonnet", "new_gemini"),
          "watch": ("watch_qwen", "watch_glm", "watch_deepseek")}


def their_auc_table(kind, grp):
    if grp == "primary":
        return {"T2": theirs["T2"], "T3": theirs["T3"], "T4": theirs["T4"]}[kind]
    sub = {"T2": "auc", "T3": "right_target", "T4": "wrong_target"}[kind]
    return theirs["T6"][grp][sub]


for kind in ("T2", "T3", "T4"):
    for grp in ("primary", "s23", "watch"):
        if grp not in mine[kind]:
            continue
        tt = their_auc_table(kind, grp)
        for s, ent in mine[kind][grp].items():
            if s.startswith("P1_") or s not in tt:
                continue
            t = tt[s]
            pooled_key = [k for k in t if k.startswith("pooled")]
            for c, e in ent.items():
                te = t["pooled" if "pooled" in t else pooled_key[0]] if c == "pooled" else t["cells"].get(c)
                if te is None:
                    diffs.append((f"{kind}/{grp}/{s}/{c}", "missing in theirs", e["auc"]))
                    continue
                cmp(f"{kind}/{grp}/{s}/{c}/auc", e["auc"], te.get("auc"))
                if "ci" in te:
                    cmp(f"{kind}/{grp}/{s}/{c}/ci", e["ci"], te["ci"])
                    cmp(f"{kind}/{grp}/{s}/{c}/dropped", e["dropped"], te["dropped"])
                if c == "pooled":
                    for k in ("stratified", "mean_cells"):
                        if k in te:
                            cmp(f"{kind}/{grp}/{s}/{c}/{k}", e[k], te[k])
# paired difference
cmp("paired/point", mine["paired_P1_minus_S5a"]["point"], theirs["T2"]["paired_P1_minus_S5a"].get("auc",
    theirs["T2"]["paired_P1_minus_S5a"].get("point")))
cmp("paired/ci", mine["paired_P1_minus_S5a"]["ci"], theirs["T2"]["paired_P1_minus_S5a"]["ci"])
# pooled 549
for s, v in mine["pooled549"].items():
    t = theirs["T2"]["pooled549"]
    tv = t.get(s, {}).get("auc") if isinstance(t.get(s), dict) else t.get(s)
    cmp(f"pooled549/{s}", v, tv)
# selective
for key, v in mine["T5"].items():
    grp, s, c = key.split("/")
    if grp == "primary":
        t = theirs["T5"].get(s, {}).get("pooled primary" if c == "pooled" else c)
    else:
        t = theirs["T6"][grp]["selective"].get(s, {}).get(c)
    if t is None:
        diffs.append((f"T5/{key}", "missing in theirs", v["acc80"]))
        continue
    for a, b in (("acc80", "acc80"), ("acc50", "acc50"), ("caught", "caught"), ("caught_wt", "caught_wrong_target"),
                 ("caught_rt", "caught_right_target"), ("base", "base")):
        cmp(f"T5/{key}/{a}", v[a], t[b])
# T8
for c, d in mine["T8"].items():
    t = theirs["T8"][c]
    for a, b in (("P1_zero", "P1_zero"), ("P1_pos", "P1_pos"), ("P1_inf", "P1_inf"),
                 ("P1_zero_heading_gt1", "P1_zero_heading"), ("P1b_inf", "P1b_inf"), ("S4_inf_missing", "S4_inf_missing"),
                 ("S4_inf_target", "S4_inf_target"), ("S3_zero", "S3_zero"), ("names_none", "names_none")):
        if a in d:
            cmp(f"T8/{c}/{a}", d[a], t.get(b))
# T7 full coefficients
for k, v in mine["s6"].items():
    t = theirs["T7"][k]["full"]
    tb = [t["b"]] + [w for w, act in zip(t["w"], t["active"]) if act]
    cmp(f"T7/{k}/beta", v["full"]["beta"], tb)
    cmp(f"T7/{k}/n_active", sum(v["full"]["kept"]), sum(t["active"]))

# row-level signal comparison
INF = float("inf")
rows_t = {}
for p in sorted(CONF.glob("rows_*.jsonl")):
    for line in open(p):
        r = json.loads(line)
        rows_t[r["key"]] = r
rows_m = {}
for g in ("oldwatch", "s23"):
    for line in open(HERE / f"vrows_{g}.jsonl"):
        r = json.loads(line)
        rows_m[r["key"]] = r
assert set(rows_t) == set(rows_m), (len(rows_t), len(rows_m))
rd = []
for k, m in rows_m.items():
    t = rows_t[k]
    if "P1_raw" in m:
        if round(m["P1_raw"], 6) != t["P1"]:
            rd.append((k, "P1", m["P1_raw"], t["P1"]))
        p1b = INF if m["tnum_C"] != m["tnum_D"] else round(m["P1_raw"], 6)
        if p1b != t["P1b"]:
            rd.append((k, "P1b", p1b, t["P1b"]))
    nv, nd, mm = m["S2_parts"]
    if (nv, nd, mm) != (t["n_v"], t["n_d"], t["n_match"]):
        rd.append((k, "S2parts", (nv, nd, mm), (t["n_v"], t["n_d"], t["n_match"])))
    if m["S3_parts"] != [len(t["unsupported"]), len(t["named"])]:
        rd.append((k, "S3parts", m["S3_parts"], [len(t["unsupported"]), len(t["named"])]))
    s4 = INF if m["S4_missing_place"] else (INF if m["S4_target_spread"] > 1.0 else round(m["S4_pose_spread"], 6))
    if s4 != t["S4"]:
        rd.append((k, "S4", s4, t["S4"]))
    if m["success"] != t["success"] or m["target_correct"] != t["target_correct"] or m["layout"] != t["layout"]:
        rd.append((k, "label/layout"))
print(f"table comparisons: {n_cmp}; differences beyond {TOL}: {len(diffs)}")
for d in diffs[:60]:
    print("  ", d)
print(f"row comparisons: {len(rows_m)} records; differences: {len(rd)}")
for d in rd[:30]:
    print("  ", d)
