"""I09 A1 step 5 (FROZEN.md sections 5 and 12; ADDENDUM_FIX.md): the re-fit with the one general fix (F5), on FIT data
only. Verifies FROZEN, DEVIATIONS, FIT and ADDENDUM_FIX hashes; recomputes the 192 detection rows and asserts them
equal to fit_rows.jsonl (facing does not enter them); recomputes the chosen setting's matched fit people with F1-F5,
asserts F1-F4 equal chosen.json; runs the facing rule over the union F1-F5; re-checks section 10's fit bars with the
picked variant. Writes fit_fix.txt, fit_people_chosen_fix.jsonl, chosen_fix.json and FIT_FIX.sha256. Project venv."""
from __future__ import annotations

import json
import sys
import time
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import pipeline as P  # noqa: E402
import fitdata as FD  # noqa: E402
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("i09_fit_grid", HERE / "fit_grid.py")   # not phase1b's fit_grid.py
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)

LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def verify_fit():
    out = {}
    for line in (HERE / "FIT.sha256").read_text().splitlines():
        want, name = line.split()
        got = C.sha(HERE / name)
        assert got == want, name
        out[name] = got
    assert (HERE / "FIT.sha256").read_text() == (HERE / "CHOSEN.sha256").read_text()
    out["ADDENDUM_FIX.md"] = C.verify(HERE / "ADDENDUM_FIX.md", HERE / "ADDENDUM_FIX.sha256")
    return out


def main():
    t0 = time.monotonic()
    hashes = C.verify_frozen() | verify_fit()
    say(f"I09 A1 re-fit with the one general fix, {time.strftime('%Y-%m-%dT%H:%M:%S%z')}; hashes verify: {hashes}")
    chosen = json.loads((HERE / "chosen.json").read_text())
    s_star = chosen["setting"]
    man, meta, raw, lat = G.load_raw()
    frozen_rows = [json.loads(x) for x in (HERE / "fit_rows.jsonl").read_text().splitlines()]
    accs = [{dom: G.blank() for dom in C.FIT_DOMAINS} for _ in C.GRID]
    refs = FD.fit_refs()
    people_rows = []
    for domain, sset, scene_id, loader in refs:
        sc, th = FD.load(loader)
        obs = sc.observation
        cam_xy = obs.camera.T_world_camera[:2, 3]
        cands = {d: P.candidates(raw[d][(sset, scene_id)], obs.depth, obs.camera, min(C.TAUS)) for d in C.DETECTORS}
        for i, s in enumerate(C.GRID):
            ppl = P.people_for(cands[s["detector"]], s, cam_xy, "F1")
            r, rows = C.score_scene(ppl, sc, th, "F1")
            G.add(accs[i][domain], r, rows)
            if s == s_star:
                for row in rows:
                    people_rows.append({"domain": domain, "set": sset, "scene_id": scene_id} | row)
    same = 0
    for i, fr in enumerate(frozen_rows):
        for d in C.FIT_DOMAINS:
            for k in ("scenes", "n_visible", "matched", "false", "exempt", "near_hidden", "merges", "neighbour_pairs",
                      "merged_pairs", "n_det"):
                assert accs[i][d][k] == fr["per_domain"][d][k], (i, d, k)
            assert abs(G.metrics(accs[i][d])[2] - fr["per_domain"][d]["med"]) < 1e-12
        same += 1
    say(f"detection rows recomputed: {same} of 192 equal fit_rows.jsonl (counts and medians); chosen setting unchanged: "
        f"#{chosen['index']} {chosen['key']}")

    fac = {}
    for v in P.ALL_VARIANTS:
        per = {}
        for dom in C.FIT_DOMAINS:
            m = [p for p in people_rows if p["domain"] == dom and p["matched"]]
            per[dom] = (sum(p["errs"][v] is not None and p["errs"][v] <= 45 for p in m), len(m))
        fac[v] = {"per_domain": per, "min": min(Fraction(*per[d]) for d in C.FIT_DOMAINS),
                  "pooled": Fraction(sum(per[d][0] for d in C.FIT_DOMAINS), sum(per[d][1] for d in C.FIT_DOMAINS))}
    for v in P.VARIANTS:
        assert [list(x) for x in fac[v]["per_domain"].values()] == \
               [list(x) for x in chosen["facing_selection"][v]["per_domain"].values()], v
    say("F1-F4 recomputed: equal to chosen.json")
    variant = max(P.ALL_VARIANTS, key=lambda v: (fac[v]["min"], fac[v]["pooled"], -P.ALL_VARIANTS.index(v)))
    say("facing over the union (within 45, unknown wrong, matched fit people of the chosen setting): " +
        "; ".join(f"{v}: " + ", ".join(f"{d} {fac[v]['per_domain'][d][0]}/{fac[v]['per_domain'][d][1]}"
                                        for d in C.FIT_DOMAINS) + f", min {float(fac[v]['min']):.4f}, pooled "
                  f"{float(fac[v]['pooled']):.4f}" for v in P.ALL_VARIANTS))
    adopted = variant in P.FIX_VARIANTS
    say(f"union rule picks {variant}; the fix is {'ADOPTED' if adopted else 'NOT adopted'}")

    bars = {}
    for dom in C.FIT_DOMAINS:
        acc = accs[chosen["index"]][dom]
        rec, fb, med = G.metrics(acc)
        m = [p for p in people_rows if p["domain"] == dom and p["matched"]]
        nonprof = C.facing_summary([p["errs"][variant] for p in m if p["view"] != "profile"])
        prof = C.facing_summary([p["errs"][variant] for p in m if p["view"] == "profile"])
        b = {"recall>=0.95": rec >= Fraction(95, 100),
             "line recall>=0.95": Fraction(*acc["line"]) >= Fraction(95, 100) if acc["line"][1] else None,
             "false<=0.10": fb <= Fraction(1, 10), "median xy<=0.15": med <= 0.15,
             "non-profile median<=25": nonprof["median_u"] <= 25 if nonprof["n"] else None,
             "non-profile within45>=0.85": nonprof["w45_share"] >= 0.85 if nonprof["n"] else None,
             "profile front/back>=0.90": prof["fb_share"] >= 0.90 if prof["n"] else None}
        bars[dom] = {"bars": b, "non_profile": nonprof, "profile": prof,
                     "by_view": {vc: C.facing_summary([p["errs"][variant] for p in m if p["view"] == vc])
                                 for vc in C.VIEW_CLASSES}}
        say(f"fit bars {dom} with {variant}: {b}; non-profile n {nonprof['n']} median_u {nonprof['median_u']:.2f} "
            f"w45 {nonprof['w45']}/{nonprof['n']}; profile fb {prof['fb']}/{prof['n']}")

    with open(HERE / "fit_people_chosen_fix.jsonl", "w") as fh:
        for p in people_rows:
            fh.write(json.dumps(p, default=str) + "\n")
    final = {"setting": s_star, "key": chosen["key"], "index": chosen["index"], "band_m": P.BANDS[s_star["band"]],
             "facing_variant": variant, "fix_adopted": adopted, "chosen_by": chosen["chosen_by"],
             "facing_selection_union": {v: {"per_domain": fac[v]["per_domain"], "min": float(fac[v]["min"]),
                                            "pooled": float(fac[v]["pooled"])} for v in P.ALL_VARIANTS},
             "fit_bars_final": bars, "hashes_verified": hashes,
             "fit_people_chosen_fix_sha256": C.sha(HERE / "fit_people_chosen_fix.jsonl"),
             "written": time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    C.jdump(final, HERE / "chosen_fix.json")
    (HERE / "fit_fix.txt").write_text("\n".join(LOG) + "\n")
    lines = [f"{C.sha(HERE / n)}  {n}" for n in ("ADDENDUM_FIX.md", "fit_fix.txt", "fit_people_chosen_fix.jsonl",
                                                 "chosen_fix.json")]
    (HERE / "FIT_FIX.sha256").write_text("\n".join(lines) + "\n")
    say(f"wrote and hashed FIT_FIX.sha256 at {time.strftime('%Y-%m-%dT%H:%M:%S%z')}; {time.monotonic() - t0:.0f} s")
    (HERE / "fit_fix.txt").write_text("\n".join(LOG) + "\n")
    lines = [f"{C.sha(HERE / n)}  {n}" for n in ("ADDENDUM_FIX.md", "fit_fix.txt", "fit_people_chosen_fix.jsonl",
                                                 "chosen_fix.json")]
    (HERE / "FIT_FIX.sha256").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
