"""Verifier extras: new-wording / placeholder line and fixable recovery, failure breakdown, prune breaks,
B-relocate-prune class shift. Reads my v_rows_* and I01 rows only. No model, no network."""
import collections, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
I01 = HERE.parents[2] / "phase1" / "i01_repairs"
ARMS = ("Badd", "Bprune", "Breloc", "BrelocP")
def key(r): return (r["arm"], r["scene"], r["episode"], r["trial"])
def v(r, a): return r["C_rec"] if not r["solver_step"] else r[a]["ok"]
def lf(r): return r["situation"] == "waiting_line" and r["solver_step"] and r["tc"] and not r["C_rec"] and (not r["in_region"] or not r["reachable"])
for w, ms in (("new", ["qwen38_low", "glm53", "dsv41", "haiku55", "sonnet55_low", "gemini38_low"]), ("ph", ["qwen38_low", "glm53", "dsv41"]), ("old", ["qwen38_low", "glm53", "dsv41"])):
    for m in ms:
        rs = [json.loads(l) for l in (HERE / f"v_rows_{w}_{m}.jsonl").open()]
        if w == "ph":
            P = {key(r): (r["P_ph"] if r["solver_step"] else r["C_rec"]) for r in rs}
        else:
            P = {key(r): o["P"] for o in map(json.loads, (I01 / f"rows_{w}_{m}.jsonl").open()) for r in [o]}
        fails = [r for r in rs if not r["C_rec"]]
        fx = [r for r in fails if P[key(r)]]
        lfs = [r for r in rs if lf(r)]
        parts = []
        for a in ARMS:
            parts.append(f"{a} lf {sum(v(r,a) for r in lfs)}/{len(lfs)} fixable {sum(v(r,a) for r in fx)}/{len(fx)}")
        ns = sum(not r["solver_step"] for r in fails); wt = sum(r["solver_step"] and not r["tc"] for r in fails)
        print(w, m, f"failures {len(fails)} not-solver {ns} wrong-target {wt}; " + "; ".join(parts))
# prune breaks in new wording
for m in ("dsv41", "haiku55"):
    for r in map(json.loads, (HERE / f"v_rows_new_{m}.jsonl").open()):
        if r["solver_step"] and r["C_rec"] and not r["Bprune"]["ok"]:
            print("prune break", m, r["scene"], r["episode"], r["trial"], "pruned", r["Bprune"].get("pruned_numbers"), "pruned_matched", r["Bprune"]["pruned_matched"], "cons numbers", r["cons_numbers"])
# BrelocP class shift (old GLM)
rs = [json.loads(l) for l in (HERE / "v_rows_old_glm53.jsonl").open()]
o = {key(x): x for x in map(json.loads, (I01 / "rows_old_glm53.jsonl").open())}
lfs = [r for r in rs if o[key(r)]["line_failure"]]
c1_add = {key(r) for r in lfs if not r["Badd"]["ok"] and r["Badd"]["cls"] == 1}
c1_rp = {key(r) for r in lfs if not r["BrelocP"]["ok"] and r["BrelocP"]["cls"] == 1}
print("GLM class1 Badd", len(c1_add), "BrelocP", len(c1_rp), "new in BrelocP", len(c1_rp - c1_add), "B-add class1 flipped/moved under BrelocP", len(c1_add - c1_rp))
c1_pr = {key(r) for r in lfs if not r["Bprune"]["ok"] and r["Bprune"]["cls"] == 1}
print("GLM class1 Bprune", len(c1_pr), "new vs Badd", len(c1_pr - c1_add))
# B-add class 1: how many state no come_after at all vs an empty/unusable one
for r in lfs:
    if not r["Badd"]["ok"] and r["Badd"]["cls"] == 1:
        pass
