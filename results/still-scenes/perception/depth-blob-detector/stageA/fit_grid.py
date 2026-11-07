"""I03 Stage A step 3: fit the 216-setting grid on fit data only (FROZEN.md sections 2-3) and freeze the choice.

Loads only fit scenes (fitdata.fit_refs: F-real 21 worked examples, F-ph 8 placeholder worked examples + 260 fresh
seeds). Per setting and fit domain D: recall_D (matched / visible, pooled, at 0.5 m), fb_D (false bodies per scene;
exempt blobs not counted), mp_D (merged neighbour pairs / neighbour pairs, 0 when none).

Rule: feasible when fb_D <= 0.10 in both domains; among feasible, highest min_D recall_D; ties: lower max_D mp_D, lower
max_D fb_D, then least restrictive (larger split_r, lower z_lo, higher z_hi, lower top_lo, lower foot_lo, higher
foot_hi). If none feasible: lowest max_D fb_D among min recall >= 0.95; else highest min recall among max_D fb_D <=
0.30; else lowest max_D fb_D (each with the remaining keys of the same order as tie-breaks).

Then, at the chosen setting only: the facing-axis variant (all points vs 1.0-1.6 m band), by the higher of the two
domains' minimum axis accuracy (within 45 deg mod 180 over matched pairs; a pair with no axis counts as a miss), ties
to "all"; and whether the chosen setting meets the Stage A success bars on fit data (which decides whether the one
general fix is considered).

Writes ../fit_selection.json and ../FIT.sha256 (as the note names them), fit_grid.txt, chosen.json and
fit_rows_chosen.jsonl here. Run once, before any evaluation scene is loaded.
"""
from __future__ import annotations

import collections
import hashlib
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
I03 = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(I03))
import fitdata as FD  # noqa: E402
import people_scorer as PS  # noqa: E402
import blobs as B  # noqa: E402

DOMAINS = ("F-real", "F-ph")
COUNTS = ("scenes", "n_visible", "matched", "false", "exempt", "near_hidden", "merges", "neighbour_pairs",
          "merged_pairs", "n_det", "staff_behind")


def metrics(acc):
    out = {}
    for d in DOMAINS:
        c = acc[d]
        out[d] = {"recall": Fraction(c["matched"], c["n_visible"]) if c["n_visible"] else Fraction(1),
                  "fb": Fraction(c["false"], c["scenes"]),
                  "mp": Fraction(c["merged_pairs"], c["neighbour_pairs"]) if c["neighbour_pairs"] else Fraction(0)}
    return out


def keys(s, m):
    minrec = min(m[d]["recall"] for d in DOMAINS)
    maxmp = max(m[d]["mp"] for d in DOMAINS)
    maxfb = max(m[d]["fb"] for d in DOMAINS)
    restr = (-s["split_r"], s["z_lo"], -s["z_hi"], s["top_lo"], s["foot_lo"], -s["foot_hi"])
    return minrec, maxmp, maxfb, restr


def choose(rows):
    """(index, which step of the rule chose it)."""
    feas = [i for i, r in enumerate(rows) if all(r["m"][d]["fb"] <= Fraction(1, 10) for d in DOMAINS)]
    def k_main(i):
        minrec, maxmp, maxfb, restr = keys(rows[i]["s"], rows[i]["m"])
        return (-minrec, maxmp, maxfb, restr)
    if feas:
        return min(feas, key=k_main), "step 2-3: feasible (fb <= 0.10 in both domains), highest min recall"
    c1 = [i for i, r in enumerate(rows) if keys(r["s"], r["m"])[0] >= Fraction(95, 100)]
    if c1:
        def k1(i):
            minrec, maxmp, maxfb, restr = keys(rows[i]["s"], rows[i]["m"])
            return (maxfb, -minrec, maxmp, restr)
        return min(c1, key=k1), "step 4a: none feasible; lowest max fb among min recall >= 0.95"
    c2 = [i for i, r in enumerate(rows) if keys(r["s"], r["m"])[2] <= Fraction(3, 10)]
    if c2:
        return min(c2, key=k_main), "step 4b: none feasible; highest min recall among max fb <= 0.30"
    def k3(i):
        minrec, maxmp, maxfb, restr = keys(rows[i]["s"], rows[i]["m"])
        return (maxfb, -minrec, maxmp, restr)
    return min(range(len(rows)), key=k3), "step 4c: lowest max fb"


def bars(m, acc):
    """Stage A success and kill bars applied to fit data, per domain."""
    out = {}
    for d in DOMAINS:
        has_pairs = acc[d]["neighbour_pairs"] > 0
        succ = (m[d]["recall"] >= Fraction(95, 100) and m[d]["fb"] <= Fraction(1, 10)
                and (not has_pairs or m[d]["mp"] <= Fraction(1, 10)))
        kill = (m[d]["recall"] < Fraction(85, 100) or m[d]["fb"] > Fraction(3, 10)
                or (has_pairs and m[d]["mp"] > Fraction(2, 10)))
        out[d] = {"success": succ, "kill": kill}
    return out


def f(x, nd=4):
    return round(float(x), nd)


def main():
    t0 = time.monotonic()
    refs = FD.fit_refs()
    n_by = collections.Counter(r[0] for r in refs)
    assert n_by == {"F-real": 21, "F-ph": 268}, n_by
    grid = B.settings()
    assert len(grid) == 216
    acc = [{d: collections.Counter() for d in DOMAINS} for _ in grid]
    bands = sorted({(s["z_lo"], s["z_hi"]) for s in grid})
    for n, (domain, sset, scene_id, loader) in enumerate(refs, 1):
        sc, th = FD.load(loader)
        ex = PS.exempt_points(sc.truth)
        pts = B.lifted(sc.observation)
        per_band = {band: B.blobs_band(pts, band[0], band[1], B.GRID["split_r"], facing=False) for band in bands}
        for i, s in enumerate(grid):
            kb = B.kept(per_band[(s["z_lo"], s["z_hi"])][f"{s['split_r']:.2f}"], s)
            r = PS.score_people([(b["x"], b["y"], None) for b in kb], sc, th, exempt_xy=ex)
            c = acc[i][domain]
            c["scenes"] += 1
            for k in COUNTS[1:]:
                c[k] += r[k]
        if n % 25 == 0 or n == len(refs):
            print(f"{n}/{len(refs)} fit scenes, {time.monotonic() - t0:.0f} s", flush=True)
    rows = [{"i": i, "s": s, "acc": acc[i], "m": metrics(acc[i])} for i, s in enumerate(grid)]
    ci, how = choose(rows)
    chosen = rows[ci]
    fit_bars = bars(chosen["m"], chosen["acc"])
    fit_success = all(v["success"] for v in fit_bars.values())

    # facing variant and per-scene rows at the chosen setting (fit data only)
    s = chosen["s"]
    fac = {v: {d: collections.Counter() for d in DOMAINS} for v in ("all", "band")}
    scene_rows = []
    for domain, sset, scene_id, loader in refs:
        sc, th = FD.load(loader)
        ex = PS.exempt_points(sc.truth)
        kb, allb = B.detect(sc.observation, s)
        res = {}
        for v in ("all", "band"):
            res[v] = PS.score_people([(b["x"], b["y"], b[f"axis_{v}"]) for b in kb], sc, th, exempt_xy=ex,
                                     facing="axis")
            fac[v][domain]["matched"] += res[v]["facing"]["matched"]
            fac[v][domain]["w45"] += res[v]["facing"]["within45_axis"]
            fac[v][domain]["unknown"] += res[v]["facing"]["unknown"]
        r = res["all"]
        scene_rows.append({"domain": domain, "set": sset, "scene_id": scene_id, "n_visible": r["n_visible"],
                           "matched": r["matched"], "false": r["false"], "exempt": r["exempt"],
                           "near_hidden": r["near_hidden"], "merges": r["merges"],
                           "neighbour_pairs": r["neighbour_pairs"], "merged_pairs": r["merged_pairs"],
                           "misses": r["misses"], "false_xy": [(round(kb[j - 1]["x"], 3), round(kb[j - 1]["y"], 3),
                                                                round(kb[j - 1]["top"], 3),
                                                                round(kb[j - 1]["footprint"], 3))
                                                               for j in r["false_numbers"]],
                           "kept": len(kb), "axis_w45_all": r["facing"]["within45_axis"],
                           "axis_w45_band": res["band"]["facing"]["within45_axis"]})
    acc_v = {v: min(Fraction(fac[v][d]["w45"], fac[v][d]["matched"]) if fac[v][d]["matched"] else Fraction(0)
                    for d in DOMAINS) for v in ("all", "band")}
    facing_variant = "band" if acc_v["band"] > acc_v["all"] else "all"

    def row_json(r):
        return {"i": r["i"], "setting": r["s"], "counts": {d: dict(r["acc"][d]) for d in DOMAINS},
                "recall": {d: f(r["m"][d]["recall"]) for d in DOMAINS},
                "fb_per_scene": {d: f(r["m"][d]["fb"]) for d in DOMAINS},
                "merged_share": {d: f(r["m"][d]["mp"]) for d in DOMAINS},
                "min_recall": f(keys(r["s"], r["m"])[0]), "max_mp": f(keys(r["s"], r["m"])[1]),
                "max_fb": f(keys(r["s"], r["m"])[2]),
                "feasible": all(r["m"][d]["fb"] <= Fraction(1, 10) for d in DOMAINS)}

    sel = {"note": "I03 phase 1b Stage A fit selection (FROZEN.md section 3). Fit data only: F-real 21, F-ph 268.",
           "n_settings": len(rows), "n_feasible": sum(row_json(r)["feasible"] for r in rows),
           "chosen_index": ci, "chosen_setting": s, "chosen_by": how, "top_hi": round(s["z_hi"] - 0.05, 2),
           "chosen_fit_metrics": row_json(chosen), "fit_bars": fit_bars, "fit_success_all_bars": fit_success,
           "one_general_fix_needed": not fit_success,
           "facing": {"variant": facing_variant,
                      "accuracy_min_over_domains": {v: f(acc_v[v]) for v in acc_v},
                      "counts": {v: {d: dict(fac[v][d]) for d in DOMAINS} for v in fac},
                      "rule": "higher min-over-domains share of matched pairs within 45 deg mod 180 (unknown = miss); "
                              "tie -> all"},
           "rows": [row_json(r) for r in rows]}
    out = I03 / "fit_selection.json"
    out.write_text(json.dumps(sel, indent=1))
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    (I03 / "FIT.sha256").write_text(f"{digest}  fit_selection.json\n")
    chosen_json = {"setting": s, "top_hi": round(s["z_hi"] - 0.05, 2), "facing_variant": facing_variant,
                   "chosen_by": how, "fit_selection_sha256": digest, "fit_success_all_bars": fit_success,
                   "one_general_fix_needed": not fit_success}
    (HERE / "chosen.json").write_text(json.dumps(chosen_json, indent=1))
    with open(HERE / "fit_rows_chosen.jsonl", "w") as fh:
        for r in scene_rows:
            fh.write(json.dumps(r) + "\n")

    # the human table
    L = []
    L.append("I03 Stage A fit grid (FROZEN.md sections 2-3). Fit data only: F-real 21 scenes, F-ph 268 scenes "
             "(8 placeholder worked examples + 260 fresh-seed scenes, seeds 100-109).")
    L.append(f"visible people: F-real {chosen['acc']['F-real']['n_visible']}, F-ph {chosen['acc']['F-ph']['n_visible']}; "
             f"neighbour pairs: F-real {chosen['acc']['F-real']['neighbour_pairs']}, F-ph {chosen['acc']['F-ph']['neighbour_pairs']}")
    L.append("columns: setting | recall real k/n | recall ph k/n | fb/scene real (false) | fb/scene ph (false) | "
             "merged pairs real k/n | merged pairs ph k/n | min recall | max fb | feasible")
    order = sorted(range(len(rows)), key=lambda i: (-keys(rows[i]["s"], rows[i]["m"])[0],
                                                      keys(rows[i]["s"], rows[i]["m"])[1],
                                                      keys(rows[i]["s"], rows[i]["m"])[2],
                                                      keys(rows[i]["s"], rows[i]["m"])[3]))
    for i in order:
        r = rows[i]
        a, m = r["acc"], r["m"]
        feas = all(m[d]["fb"] <= Fraction(1, 10) for d in DOMAINS)
        L.append(f"{'*' if i == ci else ' '} {B.setting_key(r['s'])} | "
                 f"{a['F-real']['matched']}/{a['F-real']['n_visible']} | {a['F-ph']['matched']}/{a['F-ph']['n_visible']} | "
                 f"{float(m['F-real']['fb']):.3f} ({a['F-real']['false']}) | {float(m['F-ph']['fb']):.3f} ({a['F-ph']['false']}) | "
                 f"{a['F-real']['merged_pairs']}/{a['F-real']['neighbour_pairs']} | {a['F-ph']['merged_pairs']}/{a['F-ph']['neighbour_pairs']} | "
                 f"{float(keys(r['s'], m)[0]):.4f} | {float(keys(r['s'], m)[2]):.3f} | {'yes' if feas else 'no'}")
    L.append("")
    L.append(f"feasible settings: {sel['n_feasible']} of {len(rows)}")
    L.append(f"CHOSEN (row marked *): {B.setting_key(s)}, top_hi {s['z_hi'] - 0.05:.2f}; {how}")
    L.append(f"fit bars at the chosen setting: {fit_bars}; all success bars met on fit data: {fit_success}")
    L.append(f"facing variant: {facing_variant} (min-over-domains axis accuracy: all {float(acc_v['all']):.4f}, "
             f"band {float(acc_v['band']):.4f}); counts {sel['facing']['counts']}")
    L.append(f"fit_selection.json sha256 {digest}")
    L.append(f"{time.monotonic() - t0:.0f} s")
    (HERE / "fit_grid.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L[-7:]))


if __name__ == "__main__":
    main()
