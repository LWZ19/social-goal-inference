"""I03 Stage A: the one general fix (ADDENDUM_FIX.md, sha256 in ../ADDENDUM_FIX.sha256). Re-fit the same 216-setting
grid on the same fit data with drop_clip=True, then adopt by the same rule run on the union of the 432 rows, original
rows first (an exact tie keeps the original). Re-choose the facing variant at the adopted setting. Writes
../fit_selection_fix.json, ../FIT_FIX.sha256, fit_grid_fix.txt, fit_rows_chosen_fix.jsonl (if the fix is adopted),
and rewrites chosen.json (the original is kept in chosen_original.json). Fit data only."""
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
from fit_grid import COUNTS, DOMAINS, bars, choose, f, keys, metrics  # noqa: E402


def check_hashes():
    for name, sha in (("ADDENDUM_FIX.md", "ADDENDUM_FIX.sha256"), ("fit_selection.json", "FIT.sha256"),
                      ("FROZEN.md", "FROZEN.sha256")):
        got = hashlib.sha256((I03 / name).read_bytes()).hexdigest()
        assert got == (I03 / sha).read_text().split()[0], name
    orig = json.loads((HERE / "chosen_original.json").read_text())
    assert orig["fit_selection_sha256"] == (I03 / "FIT.sha256").read_text().split()[0]
    return orig


def row_json(r):
    return {"i": r["i"], "setting": r["s"], "counts": {d: dict(r["acc"][d]) for d in DOMAINS},
            "recall": {d: f(r["m"][d]["recall"]) for d in DOMAINS},
            "fb_per_scene": {d: f(r["m"][d]["fb"]) for d in DOMAINS},
            "merged_share": {d: f(r["m"][d]["mp"]) for d in DOMAINS},
            "min_recall": f(keys(r["s"], r["m"])[0]), "max_mp": f(keys(r["s"], r["m"])[1]),
            "max_fb": f(keys(r["s"], r["m"])[2]),
            "feasible": all(r["m"][d]["fb"] <= Fraction(1, 10) for d in DOMAINS)}


def facing_at(refs, s, drop_clip):
    fac = {v: {d: collections.Counter() for d in DOMAINS} for v in ("all", "band")}
    scene_rows = []
    for domain, sset, scene_id, loader in refs:
        sc, th = FD.load(loader)
        ex = PS.exempt_points(sc.truth)
        kb, _ = B.detect(sc.observation, s, drop_clip=drop_clip)
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
                           "merges": r["merges"], "neighbour_pairs": r["neighbour_pairs"],
                           "merged_pairs": r["merged_pairs"], "misses": r["misses"], "kept": len(kb)})
    acc_v = {v: min(Fraction(fac[v][d]["w45"], fac[v][d]["matched"]) if fac[v][d]["matched"] else Fraction(0)
                    for d in DOMAINS) for v in ("all", "band")}
    return ("band" if acc_v["band"] > acc_v["all"] else "all"), acc_v, fac, scene_rows


def main():
    t0 = time.monotonic()
    orig = check_hashes()
    sel0 = json.loads((I03 / "fit_selection.json").read_text())
    refs = FD.fit_refs()
    assert collections.Counter(r[0] for r in refs) == {"F-real": 21, "F-ph": 268}
    grid = B.settings()
    acc = [{d: collections.Counter() for d in DOMAINS} for _ in grid]
    bands = sorted({(s["z_lo"], s["z_hi"]) for s in grid})
    for n, (domain, sset, scene_id, loader) in enumerate(refs, 1):
        sc, th = FD.load(loader)
        ex = PS.exempt_points(sc.truth)
        pts = B.lifted(sc.observation)
        per_band = {band: B.blobs_band(pts, band[0], band[1], B.GRID["split_r"], facing=False, drop_clip=True)
                    for band in bands}
        for i, s in enumerate(grid):
            kb = B.kept(per_band[(s["z_lo"], s["z_hi"])][f"{s['split_r']:.2f}"], s)
            r = PS.score_people([(b["x"], b["y"], None) for b in kb], sc, th, exempt_xy=ex)
            c = acc[i][domain]
            c["scenes"] += 1
            for k in COUNTS[1:]:
                c[k] += r[k]
        if n % 50 == 0 or n == len(refs):
            print(f"{n}/{len(refs)} fit scenes, {time.monotonic() - t0:.0f} s", flush=True)
    fixed = [{"i": i, "s": s, "acc": acc[i], "m": metrics(acc[i])} for i, s in enumerate(grid)]
    fi, fhow = choose(fixed)
    # the original rows, rebuilt from their recorded counts
    original = []
    for r in sel0["rows"]:
        a = {d: collections.Counter(r["counts"][d]) for d in DOMAINS}
        original.append({"i": r["i"], "s": r["setting"], "acc": a, "m": metrics(a)})
    assert original[sel0["chosen_index"]]["s"] == orig["setting"]
    union = original + fixed
    ui, uhow = choose(union)
    adopted_fix = ui >= len(original)
    chosen_row = union[ui]
    s = chosen_row["s"]
    variant, acc_v, fac, scene_rows = facing_at(refs, s, adopted_fix)
    fit_bars = bars(chosen_row["m"], chosen_row["acc"])
    sel = {"note": "I03 Stage A re-fit with the one general fix (ADDENDUM_FIX.md: drop clip-reaching cells). "
                   "Fit data only: F-real 21, F-ph 268.",
           "addendum_sha256": (I03 / "ADDENDUM_FIX.sha256").read_text().split()[0],
           "n_settings": len(fixed), "n_feasible": sum(row_json(r)["feasible"] for r in fixed),
           "fix_chosen_index": fi, "fix_chosen_setting": fixed[fi]["s"], "fix_chosen_by": fhow,
           "fix_chosen_fit_metrics": row_json(fixed[fi]),
           "union_choice": {"index": ui, "detector": "fix" if adopted_fix else "original", "setting": s,
                            "chosen_by": uhow, "fit_metrics": row_json(chosen_row), "fit_bars": fit_bars,
                            "fit_success_all_bars": all(v["success"] for v in fit_bars.values())},
           "facing": {"variant": variant, "accuracy_min_over_domains": {v: f(acc_v[v]) for v in acc_v},
                      "counts": {v: {d: dict(fac[v][d]) for d in DOMAINS} for v in fac}},
           "rows": [row_json(r) for r in fixed]}
    out = I03 / "fit_selection_fix.json"
    out.write_text(json.dumps(sel, indent=1))
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    (I03 / "FIT_FIX.sha256").write_text(f"{digest}  fit_selection_fix.json\n")
    with open(HERE / ("fit_rows_chosen_fix.jsonl" if adopted_fix else "fit_rows_chosen_union.jsonl"), "w") as fh:
        for r in scene_rows:
            fh.write(json.dumps(r) + "\n")
    chosen = {"detector": "fix" if adopted_fix else "original", "drop_clip": adopted_fix, "setting": s,
              "top_hi": round(s["z_hi"] - 0.05, 2), "facing_variant": variant, "chosen_by": uhow,
              "fit_selection_sha256": orig["fit_selection_sha256"], "fit_selection_fix_sha256": digest,
              "addendum_fix_sha256": sel["addendum_sha256"],
              "fit_metrics": row_json(chosen_row), "fit_bars": fit_bars,
              "fit_success_all_bars": all(v["success"] for v in fit_bars.values()),
              "original_choice": orig}
    (HERE / "chosen.json").write_text(json.dumps(chosen, indent=1))
    L = ["I03 Stage A fit grid WITH THE ONE GENERAL FIX (ADDENDUM_FIX.md: drop clip-reaching cells). Fit data only.",
         "columns: setting | recall real k/n | recall ph k/n | fb/scene real (false) | fb/scene ph (false) | "
         "merged pairs real k/n | merged pairs ph k/n | min recall | max fb | feasible"]
    order = sorted(range(len(fixed)), key=lambda i: (-keys(fixed[i]["s"], fixed[i]["m"])[0],
                                                       keys(fixed[i]["s"], fixed[i]["m"])[1],
                                                       keys(fixed[i]["s"], fixed[i]["m"])[2],
                                                       keys(fixed[i]["s"], fixed[i]["m"])[3]))
    for i in order:
        r = fixed[i]
        a, m = r["acc"], r["m"]
        feas = all(m[d]["fb"] <= Fraction(1, 10) for d in DOMAINS)
        L.append(f"{'*' if i == fi else ' '} {B.setting_key(r['s'])} | "
                 f"{a['F-real']['matched']}/{a['F-real']['n_visible']} | {a['F-ph']['matched']}/{a['F-ph']['n_visible']} | "
                 f"{float(m['F-real']['fb']):.3f} ({a['F-real']['false']}) | {float(m['F-ph']['fb']):.3f} ({a['F-ph']['false']}) | "
                 f"{a['F-real']['merged_pairs']}/{a['F-real']['neighbour_pairs']} | {a['F-ph']['merged_pairs']}/{a['F-ph']['neighbour_pairs']} | "
                 f"{float(keys(r['s'], m)[0]):.4f} | {float(keys(r['s'], m)[2]):.3f} | {'yes' if feas else 'no'}")
    L += ["", f"feasible settings with the fix: {sel['n_feasible']} of {len(fixed)}",
          f"fix's own choice (row *): {B.setting_key(fixed[fi]['s'])}; {fhow}",
          f"ADOPTION (rule on the union of 432 rows, original first): {'FIX' if adopted_fix else 'ORIGINAL'} "
          f"{B.setting_key(s)}; {uhow}",
          f"fit bars at the adopted setting: {fit_bars}",
          f"facing variant at the adopted setting: {variant} (min-over-domains: all {float(acc_v['all']):.4f}, band "
          f"{float(acc_v['band']):.4f}); counts {sel['facing']['counts']}",
          f"fit_selection_fix.json sha256 {digest}", f"{time.monotonic() - t0:.0f} s"]
    (HERE / "fit_grid_fix.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L[-8:]))


if __name__ == "__main__":
    main()
