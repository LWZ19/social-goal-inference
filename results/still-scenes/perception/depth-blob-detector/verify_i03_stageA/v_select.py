"""Verifier: apply FROZEN.md section 3's rule to the verifier's own fit rows; the fix re-fit and the union adoption;
compare with the compute agent's fit_selection.json / fit_selection_fix.json (read as data only)."""
import json
import sys
from fractions import Fraction as Fr
from pathlib import Path

HERE = Path(__file__).resolve().parent
I03 = HERE.parent
sys.path.insert(0, str(HERE))
from v_fit import SETTINGS, CNT  # noqa: E402

DOMS = ("F-real", "F-ph")
rows = [json.loads(l) for f in sorted(HERE.glob("fit_rows_*.jsonl")) for l in open(f)]
assert len(rows) == 289, len(rows)
print("scenes", {d: sum(r["domain"] == d for r in rows) for d in DOMS},
      "visible", {d: sum(r["vis"] for r in rows if r["domain"] == d) for d in DOMS},
      "people", {d: sum(r["people"] for r in rows if r["domain"] == d) for d in DOMS})


def table(det):
    out = []
    for i, s in enumerate(SETTINGS):
        c = {d: dict.fromkeys(list(CNT) + ["staff_behind", "scenes"], 0) for d in DOMS}
        for r in rows:
            v = r["rows"][det][i]
            cd = c[r["domain"]]
            cd["scenes"] += 1
            for k, x in zip(list(CNT) + ["staff_behind"], v):
                cd[k] += x
        m = {d: {"rec": Fr(c[d]["matched"], c[d]["vis"]) if c[d]["vis"] else Fr(1),
                 "fb": Fr(c[d]["false"], c[d]["scenes"]),
                 "mp": Fr(c[d]["mpairs"], c[d]["npairs"]) if c[d]["npairs"] else Fr(0)} for d in DOMS}
        out.append({"det": det, "i": i, "s": s, "c": c, "m": m})
    return out


def keyparts(r):
    s, m = r["s"], r["m"]
    minrec = min(m[d]["rec"] for d in DOMS)
    maxmp = max(m[d]["mp"] for d in DOMS)
    maxfb = max(m[d]["fb"] for d in DOMS)
    restr = (-s["split_r"], s["z_lo"], -s["z_hi"], s["top_lo"], s["foot_lo"], -s["foot_hi"])
    return minrec, maxmp, maxfb, restr


def choose(tab):
    feas = [r for r in tab if all(r["m"][d]["fb"] <= Fr(1, 10) for d in DOMS)]
    if feas:
        # first minimum in list order (min() keeps the first of equal keys)
        return min(feas, key=lambda r: (-keyparts(r)[0], keyparts(r)[1], keyparts(r)[2], keyparts(r)[3])), "feasible", len(feas)
    c1 = [r for r in tab if keyparts(r)[0] >= Fr(95, 100)]
    if c1:
        return min(c1, key=lambda r: (keyparts(r)[2], -keyparts(r)[0], keyparts(r)[1], keyparts(r)[3])), "4a", 0
    c2 = [r for r in tab if keyparts(r)[2] <= Fr(3, 10)]
    if c2:
        return min(c2, key=lambda r: (-keyparts(r)[0], keyparts(r)[1], keyparts(r)[2], keyparts(r)[3])), "4b", 0
    return min(tab, key=lambda r: (keyparts(r)[2], -keyparts(r)[0], keyparts(r)[1], keyparts(r)[3])), "4c", 0


def show(r):
    s = r["s"]
    c = r["c"]
    return (f"{r['det']} " + " ".join(f"{k}={v:.2f}" for k, v in s.items()) + " | " +
            " ; ".join(f"{d}: {c[d]['matched']}/{c[d]['vis']} false {c[d]['false']}/{c[d]['scenes']} exempt {c[d]['exempt']} "
                       f"mp {c[d]['mpairs']}/{c[d]['npairs']} merges {c[d]['merges']} staff_behind {c[d]['staff_behind']}"
                       for d in DOMS))


orig = table("orig")
fix = table("fix")
ch, how, nf = choose(orig)
print("ORIGINAL: feasible", nf, "chosen by", how)
print("  ", show(ch))
feas = [r for r in orig if all(r["m"][d]["fb"] <= Fr(1, 10) for d in DOMS)]
for r in feas:
    print("   feasible:", show(r))
chf, howf, nff = choose(fix)
print("FIX: feasible", nff, "chosen by", howf)
print("  ", show(chf))
fixfb = [max(r["m"][d]["fb"] for d in DOMS) for r in fix]
print("  fix: lowest max fb", float(min(fixfb)), "at", show(fix[fixfb.index(min(fixfb))]))
rec1 = [r for r in fix if all(r["m"][d]["rec"] == 1 for d in DOMS)]
if rec1:
    print("  fix: settings with recall 1 in both:", len(rec1), "max fb range",
          float(min(max(r['m'][d]['fb'] for d in DOMS) for r in rec1)), float(max(max(r['m'][d]['fb'] for d in DOMS) for r in rec1)))
union, howu, _ = choose(orig + fix)
print("ADOPTION on union (orig first):", howu, show(union))

# fit bars at chosen
c = ch["c"]
for d in DOMS:
    rec = c[d]["matched"] / c[d]["vis"]
    fb = c[d]["false"] / c[d]["scenes"]
    mp = c[d]["mpairs"] / c[d]["npairs"] if c[d]["npairs"] else 0
    print(f"fit bars {d}: recall {rec:.4f} fb {fb:.4f} mp {mp:.4f} success {rec >= 0.95 and fb <= 0.10 and mp <= 0.10} "
          f"kill {rec < 0.85 or fb > 0.3 or mp > 0.2}")

# compare with the compute agent's tables (as data)
for name, tab in (("fit_selection.json", orig), ("fit_selection_fix.json", fix)):
    other = json.load(open(I03 / name))["rows"]
    diffs = 0
    for r in tab:
        o = next(x for x in other if all(abs(x["setting"][k] - r["s"][k]) < 1e-9 for k in r["s"]))
        for d in DOMS:
            mine = (r["c"][d]["vis"], r["c"][d]["matched"], r["c"][d]["false"], r["c"][d]["exempt"], r["c"][d]["near_hidden"],
                    r["c"][d]["merges"], r["c"][d]["npairs"], r["c"][d]["mpairs"], r["c"][d]["n_det"], r["c"][d]["staff_behind"])
            oc = o["counts"][d]
            theirs = (oc["n_visible"], oc["matched"], oc["false"], oc["exempt"], oc["near_hidden"], oc["merges"],
                      oc["neighbour_pairs"], oc["merged_pairs"], oc["n_det"], oc["staff_behind"])
            if mine != theirs:
                diffs += 1
                if diffs <= 10:
                    print("  DIFF", name, d, r["s"], "mine", mine, "theirs", theirs)
    print(f"{name}: {diffs} setting-domain rows differ of {2 * len(tab)}")
json.dump({"orig_choice": ch["s"], "orig_how": how, "n_feasible": nf, "fix_choice": chf["s"], "fix_how": howf,
           "adopted": {"det": union["det"], "s": union["s"]}}, open(HERE / "v_select.json", "w"), indent=1)
