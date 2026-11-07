"""I03 Stage B tables from rows_<wording>_<model>.jsonl and step0_<wording>_<model>.jsonl (replay.py). Writes
tables.txt and summary.json beside this file. Definitions: FROZEN.md sections 7-8 and CLARIFICATIONS.md."""
from __future__ import annotations

import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = ("Badd", "Bprune", "Breloc", "BrelocP")
ARM_NAME = {"Badd": "B-add", "Bprune": "B-prune", "Breloc": "B-relocate", "BrelocP": "B-relocate-prune",
            "R3_i01": "I01 R3", "R4_i01": "I01 R4", "P_i01": "I01 P"}
NAME = {"qwen38_low": "Qwen", "glm53": "GLM", "dsv41": "DeepSeek", "haiku55": "Haiku", "sonnet55_low": "Sonnet",
        "gemini38_low": "Gemini"}
ORDER = {"old": ["qwen38_low", "glm53", "dsv41"],
         "new": ["qwen38_low", "glm53", "dsv41", "haiku55", "sonnet55_low", "gemini38_low"],
         "ph": ["qwen38_low", "glm53", "dsv41"]}
SITS = [("no_one", "nobody near"), ("waiting_line", "lines"), ("conversation_group", "groups"),
        ("audience", "audience")]
CACHE_COPY = {("new", "qwen38_low"): ("old", "qwen38_low"), ("new", "glm53"): ("old", "glm53"),
              ("new", "dsv41"): ("old", "dsv41")}


def load(prefix):
    out = {}
    for w, models in ORDER.items():
        for m in models:
            p = HERE / f"{prefix}_{w}_{m}.jsonl"
            if p.exists():
                out[(w, m)] = [json.loads(line) for line in p.open()]
    return out


def flips(rs, k, base="C_rec"):
    return [r for r in rs if r[k] and not r[base]]


def breaks(rs, k, base="C_rec"):
    return [r for r in rs if r[base] and not r[k]]


def cell(rs, k, base="C_rec"):
    return f"{sum(r[k] for r in rs)} (+{len(flips(rs, k, base))}/-{len(breaks(rs, k, base))})"


def keyset(rs):
    return {(r["arm"], r["scene"], r["episode"], r["trial"]) for r in rs}


def answers(rs):
    return len({r["answer"] for r in rs}), len({r["layout"] for r in rs})


def main():
    rows, step0 = load("rows"), load("step0")
    L, S = [], {}
    p = L.append
    p("I03 Stage B: solver replays with the frozen Stage A blobs (FROZEN.md sections 7-8; CLARIFICATIONS.md).")
    p("Cell format: successes (+flips / -breaks) against the recorded C verdict. Records that are not solver steps keep")
    p("their recorded verdict in every arm. Old wording = s19 Qwen/GLM, s20 DeepSeek, con, realistic arms (252 each).")
    p("")

    # ------------------------------------------------------------------ step 0
    p("TABLE 0. Step 0: replay with the recorded inputs (H.resolve). reproduced = place moved <= 1e-6 m; verdict = "
      "replayed success equals recorded success.")
    p(f"{'wording':8s}{'model':10s}{'records':>8s}{'solver':>8s}{'reprod':>8s}{'verdict':>8s}")
    tot = collections.Counter()
    for (w, m), rs in step0.items():
        st = [r for r in rs if r["solver_step"]]
        rep, ok = sum(r["C_reproduced"] for r in st), sum(r["C_verdict_ok"] for r in st)
        tot[(w == "ph", "st")] += len(st)
        tot[(w == "ph", "rep")] += rep
        tot[(w == "ph", "ok")] += ok
        p(f"{w:8s}{NAME[m]:10s}{len(rs):8d}{len(st):8d}{rep:8d}{ok:8d}")
        S[f"step0|{w}|{m}"] = [len(st), rep, ok]
    p(f"realistic (old + new): {tot[(False, 'rep')]} of {tot[(False, 'st')]} places reproduced, {tot[(False, 'ok')]} "
      f"of {tot[(False, 'st')]} verdicts; placeholder: {tot[(True, 'rep')]} of {tot[(True, 'st')]} places, "
      f"{tot[(True, 'ok')]} of {tot[(True, 'st')]} verdicts")
    S["step0_realistic"] = [tot[(False, "st")], tot[(False, "rep")], tot[(False, "ok")]]
    S["step0_ph"] = [tot[(True, "st")], tot[(True, "rep")], tot[(True, "ok")]]
    # the arms run repeated C; check it agrees with step 0
    for (w, m), rs in rows.items():
        st = [r for r in rs if r["solver_step"]]
        assert all(r["C_reproduced"] and r["C_verdict_ok"] for r in st) or w == "ph", (w, m)
    p("")

    # ------------------------------------------------------------------ cache copies
    p("TABLE 0b. Cache-copy check: people lists identical per (arm, scene, trial) key, new wording vs old wording.")
    for (w, m), (w0, m0) in CACHE_COPY.items():
        if (w, m) not in rows or (w0, m0) not in rows:
            continue
        a = {(r["arm"], r["scene"], r["trial"]): r["people_hash"] for r in rows[(w, m)] if r["has_people"]}
        b = {(r["arm"], r["scene"], r["trial"]): r["people_hash"] for r in rows[(w0, m0)] if r["has_people"]}
        shared = set(a) & set(b)
        same = sum(a[k] == b[k] for k in shared)
        p(f"  {NAME[m]:9s} new vs old: {same} of {len(shared)} shared keys identical (new keys with a list {len(a)}, "
          f"old {len(b)})")
        S[f"cache_copy|{m}"] = [same, len(shared), len(a), len(b)]
    p("")

    # ------------------------------------------------------------------ inputs
    p("TABLE 1. What the arms changed, over solver steps (sums; per step in brackets where useful).")
    p(f"{'wording':6s}{'model':9s}{'steps':>6s}{'det':>6s}{'unsup':>6s}{'(exempt)':>9s}{'unexpl':>7s}{'(non-tgt)':>10s}"
      f"{'tgtblob':>8s} | Badd: added (matched missed / reassigned / false) | Bprune: pruned (were matched) | "
      f"Breloc: pairs (to target blob; were matched; moved m median) added | BrelocP: pruned | close pairs (Badd, "
      f"Breloc) | recompute differs | added off image")
    import statistics
    for (w, m), rs in rows.items():
        st = [r for r in rs if r["solver_step"]]
        g = lambda k: sum(r[k] for r in st)  # noqa: E731
        ga = lambda a, k: sum(r[a + "_d"][k] for r in st)  # noqa: E731
        mv = [x for r in st for x in r["reloc_moved_m"]]
        p(f"{w:6s}{NAME[m]:9s}{len(st):6d}{g('n_det'):6d}{g('n_unsupported'):6d}{g('n_unsupported_exempt'):9d}"
          f"{g('n_unexplained'):7d}{g('n_unexplained_nontarget'):10d}{g('n_target_blobs'):8d} | "
          f"{ga('Badd', 'added')} ({ga('Badd', 'added_matched_missed')} / {ga('Badd', 'added_matched_reassigned')} / "
          f"{ga('Badd', 'added_false')}) | {ga('Bprune', 'pruned')} ({ga('Bprune', 'pruned_were_matched')}) | "
          f"{g('reloc_pairs')} ({g('reloc_pairs_to_target_blob')}; {g('reloc_moved_was_matched')}; "
          f"{statistics.median(mv) if mv else float('nan'):.2f}) {ga('Breloc', 'added')} | {ga('BrelocP', 'pruned')} | "
          f"{ga('Badd', 'close_pairs')}, {ga('Breloc', 'close_pairs')} | {g('reloc_recompute_differs')} | "
          f"{ga('Badd', 'added_off_image')}; no projection {ga('Badd', 'no_proj')}")
        S[f"inputs|{w}|{m}"] = {"steps": len(st), "det": g("n_det"), "unsupported": g("n_unsupported"),
                                 "unsupported_exempt": g("n_unsupported_exempt"), "unexplained": g("n_unexplained"),
                                 "unexplained_nontarget": g("n_unexplained_nontarget"),
                                 "badd_added": ga("Badd", "added"), "badd_added_missed": ga("Badd", "added_matched_missed"),
                                 "badd_added_false": ga("Badd", "added_false"), "bprune_pruned": ga("Bprune", "pruned"),
                                 "bprune_pruned_were_matched": ga("Bprune", "pruned_were_matched"),
                                 "reloc_pairs": g("reloc_pairs"), "reloc_recompute_differs": g("reloc_recompute_differs"),
                                 "steps_with_added": sum(r["Badd_d"]["added"] > 0 for r in st),
                                 "steps_joined_added": sum(r["Badd_d"]["joined_added"] > 0 for r in st)}
    p("  steps with >= 1 added blob under B-add / steps where an added blob joined the line (solver's joined_waiting):")
    for (w, m), rs in rows.items():
        st = [r for r in rs if r["solver_step"]]
        lst = [r for r in st if r["situation"] == "waiting_line"]
        p(f"  {w:6s}{NAME[m]:9s} all steps {sum(r['Badd_d']['added'] > 0 for r in st)}/{len(st)}; joined "
          f"{sum(r['Badd_d']['joined_added'] > 0 for r in st)}; line steps with added {sum(r['Badd_d']['added'] > 0 for r in lst)}"
          f"/{len(lst)}; line steps where an added blob joined {sum(r['Badd_d']['joined_added'] > 0 for r in lst)}")
    p("")

    # ------------------------------------------------------------------ main tables
    for w, title in (("old", "TABLE 2. Old wording (primary), per model x situation"),
                     ("new", "TABLE 2x. New wording (extra; Qwen, GLM and DeepSeek people answers are cache copies of "
                             "s19/s20)"),
                     ("ph", "TABLE 2p. Placeholder C runs (extra): s19 ph + dev_seed0-2 (Qwen, GLM), s20 ph + "
                            "dev_seed0-2 (DeepSeek)")):
        p(title + ". n records (solver steps).")
        cols = ARMS + ("R3_i01", "R4_i01", "P_i01")
        p(f"{'model':9s}{'situation':13s}{'n(st)':>9s}{'C':>5s}  " + "".join(f"{ARM_NAME[k]:>18s}" for k in cols))
        for m in ORDER[w]:
            rs = rows.get((w, m))
            if rs is None:
                continue
            for s, sname in SITS + [(None, "all")]:
                sub = [r for r in rs if s is None or r["situation"] == s]
                if not sub:
                    continue
                st = sum(r["solver_step"] for r in sub)
                p(f"{NAME[m]:9s}{sname:13s}{len(sub):5d}({st:3d}){sum(r['C_rec'] for r in sub):5d}  "
                  + "".join(f"{cell(sub, k):>18s}" for k in cols))
                for k in cols:
                    S[f"main|{w}|{m}|{sname}|{k}"] = [sum(r[k] for r in sub), len(flips(sub, k)), len(breaks(sub, k)),
                                                      len(sub), sum(r["C_rec"] for r in sub)]
        if w == "ph":
            nonrep = [(r["run"], r["scene"], r["episode"], r["trial"]) for m in ORDER[w] for r in rows.get((w, m), [])
                      if r["solver_step"] and not (r["C_reproduced"] and r["C_verdict_ok"])]
            p(f"placeholder solver steps not reproduced (place or verdict): {len(nonrep)} {nonrep[:10]}")
            if nonrep:
                p("against the replayed C instead:")
                for m in ORDER[w]:
                    rs = rows.get((w, m))
                    p(f"  {NAME[m]:9s} C replay {sum(r['C'] for r in rs)}; "
                      + "; ".join(f"{ARM_NAME[k]} {cell(rs, k, 'C')}" for k in ARMS))
        p("")

    # ------------------------------------------------------------------ criteria
    old = {m: rows[("old", m)] for m in ORDER["old"] if ("old", m) in rows}
    if len(old) == 3:
        q, g, d = old["qwen38_low"], old["glm53"], old["dsv41"]
        lines = lambda rs, k: sum(r[k] for r in rs if r["situation"] == "waiting_line")  # noqa: E731
        assert (lines(q, "C_rec"), lines(g, "C_rec"), lines(d, "C_rec")) == (86, 42, 68)
        assert all(sum(r["situation"] == "waiting_line" for r in rs) == 90 for rs in old.values())
        lf = {m: [r for r in rs if r["line_failure"]] for m, rs in old.items()}
        assert (len(lf["qwen38_low"]), len(lf["glm53"]), len(lf["dsv41"])) == (2, 29, 15)
        fixable = [r for m in ORDER["old"] for r in lf[m] if r["P_i01"]]
        assert len(fixable) == 25, len(fixable)
        p("TABLE 3. Stage B criteria (FROZEN.md section 8), old wording, realistic, 252 records per model. B-add is "
          "primary; the other arms are secondary, judged against the same bars.")
        allrs = q + g + d
        for k in ARMS + ("R3_i01", "R4_i01", "P_i01"):
            gl, dl, ql = lines(g, k), lines(d, k), lines(q, k)
            net = {m: sum(r[k] - r["C_rec"] for r in rs if r["situation"] != "waiting_line") for m, rs in old.items()}
            qt = sum(r[k] for r in q)
            rec = sum(r[k] for r in fixable)
            fl, br = len(flips(allrs, k)), len(breaks(allrs, k))
            bars = {"GLM lines >= 47": gl >= 47, "DeepSeek lines >= 71": dl >= 71,
                    "non-line net >= -1 (Qwen, GLM, DeepSeek)": all(v >= -1 for v in net.values()),
                    "Qwen total >= 234": qt >= 234, "fixable line failures recovered >= 13 of 25": rec >= 13}

            kill1 = (gl - 42 <= 2) and (dl - 68 <= 1)
            kill2 = br >= fl
            success = all(bars.values())
            verdict = "success" if success and not (kill1 or kill2) else "kill" if (kill1 or kill2) else "neither"
            if success and (kill1 or kill2):
                verdict = "success and kill both (contradiction)"
            p(f"  {ARM_NAME[k]:17s} GLM lines {gl}/90 ({gl - 42:+d}); DeepSeek lines {dl}/90 ({dl - 68:+d}); Qwen lines "
              f"{ql}/90; non-line net Qwen {net['qwen38_low']:+d}, GLM {net['glm53']:+d}, DeepSeek {net['dsv41']:+d}; "
              f"Qwen total {qt}/252; fixable line failures recovered {rec}/25; flips {fl}, breaks {br} over 756")
            p(f"  {'':17s} success bars {bars}; kill 1 (GLM gain <= 2 and DeepSeek gain <= 1) {kill1}; kill 2 (breaks >= "
              f"flips) {kill2}{' (both 0)' if fl == br == 0 else ''}; kill 3 cannot fire (no live check) -> {verdict}")
            S[f"criteria|{k}"] = {"glm_lines": gl, "ds_lines": dl, "qwen_lines": ql, "nonline_net": net, "qwen_total": qt,
                                  "fixable_recovered": rec, "flips": fl, "breaks": br, "bars": bars, "kill1": kill1,
                                  "kill2": kill2, "verdict": verdict}
        p("")

        # -------------------------------------------------------------- line failure sets
        p("TABLE 4. Pre-registered line-failure sets (phase1 rows_old_*: line_failure) flipped per arm, with the overlap "
          "with I01's P and R4 flips. Format: flipped k/n (of which also P / also R4 / not P).")
        for m in ORDER["old"]:
            rs = lf[m]
            parts = []
            for k in ARMS + ("R3_i01", "R4_i01", "P_i01"):
                f = [r for r in rs if r[k]]
                parts.append(f"{ARM_NAME[k]} {len(f)}/{len(rs)} ({sum(r['P_i01'] for r in f)}/{sum(r['R4_i01'] for r in f)}/"
                             f"{sum(not r['P_i01'] for r in f)})")
                S[f"lf|{m}|{k}"] = [len(f), len(rs), sum(r["P_i01"] for r in f), sum(r["R4_i01"] for r in f)]
            p(f"  {NAME[m]:9s} " + "; ".join(parts))
        p("  fixable line failures (P flips) recovered per arm, k/n:")
        for k in ARMS:
            per = {m: (sum(r[k] for r in lf[m] if r["P_i01"]), sum(r["P_i01"] for r in lf[m])) for m in ORDER["old"]}
            p(f"    {ARM_NAME[k]:17s} " + ", ".join(f"{NAME[m]} {a}/{b}" for m, (a, b) in per.items())
              + f"; pooled {sum(a for a, _ in per.values())}/{sum(b for _, b in per.values())}")
        p("")

        # -------------------------------------------------------------- all flips: overlap and fixable recovery
        p("TABLE 5. All records, old wording: flips per arm and their overlap with I01 P / R4 flips; fixable-offline "
          "recovery over all recorded failures (fixable = recorded failure that I01 P flips).")
        for m in ORDER["old"]:
            rs = old[m]
            fx = [r for r in rs if not r["C_rec"] and r["P_i01"]]
            fails = [r for r in rs if not r["C_rec"]]
            for k in ARMS:
                f, b = flips(rs, k), breaks(rs, k)
                p(f"  {NAME[m]:9s}{ARM_NAME[k]:17s} flips {len(f)} (also P {sum(r['P_i01'] for r in f)}, also R4 "
                  f"{sum(r['R4_i01'] for r in f)}, also R3 {sum(r['R3_i01'] for r in f)}, not P "
                  f"{sum(not r['P_i01'] for r in f)}; R3 flips also flipped {sum(r[k] for r in flips(rs, 'R3_i01'))}/"
                  f"{len(flips(rs, 'R3_i01'))}), breaks {len(b)} "
                  f"(R4 also breaks {sum(not r['R4_i01'] for r in b)}, P also breaks {sum(not r['P_i01'] for r in b)}); "
                  f"fixable recovered {sum(r[k] for r in fx)}/{len(fx)} of {len(fails)} failures; "
                  f"P flips also flipped {sum(r[k] for r in flips(rs, 'P_i01'))}/{len(flips(rs, 'P_i01'))}, R4 flips also "
                  f"flipped {sum(r[k] for r in flips(rs, 'R4_i01'))}/{len(flips(rs, 'R4_i01'))}")
                S[f"overlap|{m}|{k}"] = {"flips": len(f), "also_P": sum(r["P_i01"] for r in f),
                                         "also_R4": sum(r["R4_i01"] for r in f), "breaks": len(b),
                                         "fixable_recovered": [sum(r[k] for r in fx), len(fx)], "failures": len(fails)}
        p("")

        # -------------------------------------------------------------- distinct answers and layouts
        p("TABLE 6. Distinct people answers (run, scene, trial) and layouts behind the flips and breaks, old wording.")
        for m in ORDER["old"]:
            rs = old[m]
            for k in ARMS:
                f, b = flips(rs, k), breaks(rs, k)
                fa, fl_ = answers(f)
                ba, bl = answers(b)
                p(f"  {NAME[m]:9s}{ARM_NAME[k]:17s} flips {len(f)} from {fa} answers in {fl_} layouts "
                  f"{sorted(collections.Counter(r['layout'] for r in f).items())}; breaks {len(b)} from {ba} answers in "
                  f"{bl} layouts {sorted(collections.Counter(r['layout'] for r in b).items())}")
        p("")

        # -------------------------------------------------------------- why the remaining line failures fail
        p("TABLE 7. Why the pre-registered line failures that an arm leaves fail (CLARIFICATIONS.md point 18; classes "
          "fixed before solving; first class that applies).")
        for m in ORDER["old"]:
            rs = lf[m]
            c0 = collections.Counter(r["C_class"] for r in rs)
            p(f"  {NAME[m]:9s} recorded inputs (n {len(rs)}): {sorted(c0.items())}")
            for k in ARMS:
                left = [r for r in rs if not r[k]]
                c = collections.Counter(r[k + "_d"]["class"] for r in left)
                sub3 = [r[k + "_d"]["class_info"] for r in left if r[k + "_d"]["class"].startswith("3")]
                sub2 = [r[k + "_d"]["class_info"] for r in left if r[k + "_d"]["class"].startswith("2")]
                p(f"  {NAME[m]:9s} {ARM_NAME[k]:17s} left {len(left)}: {sorted(c.items())}; class 3: members unmatched "
                  f"{sum(i['members_unmatched'] for i in sub3)}, matched not in line "
                  f"{sum(i['members_matched_not_in_line'] for i in sub3)} (n {len(sub3)} steps); class 2: unsupported "
                  f"waiting numbers {sum(len(i['waiting_unsupported']) for i in sub2)} (n {len(sub2)} steps)")
                S[f"why|{m}|{k}"] = dict(c)
            left = [r for r in rs if not r["Badd"]]
            fx_left = [r for r in left if r["P_i01"]]
            p(f"  {NAME[m]:9s} B-add left {len(left)}: of them fixable offline (P) {len(fx_left)}, P-unfixable "
              f"{len(left) - len(fx_left)}; with no come_after stated {sum(not r['has_come_after'] for r in left)}; "
              f"empty people list {sum(r['n_det'] == 0 for r in left)}; R4 flips among them {sum(r['R4_i01'] for r in left)}")
        p("")
        p("TABLE 7b. All recorded failures, old wording, by what a people repair can touch (recorded C).")
        for m in ORDER["old"]:
            rs = old[m]
            fails = [r for r in rs if not r["C_rec"]]
            ns = sum(not r["solver_step"] for r in fails)
            wt = sum(r["solver_step"] and not r["target_correct"] for r in fails)
            rt = collections.Counter(r["situation"] for r in fails if r["solver_step"] and r["target_correct"])
            p(f"  {NAME[m]:9s} failures {len(fails)}: not a solver step {ns}; solver step, wrong target {wt}; solver step, "
              f"right target {sum(rt.values())} {sorted(rt.items())} (line failures {len(lf[m])})")
            for k in ARMS:
                left = [r for r in rs if not r[k]]
                lns = sum(not r["solver_step"] for r in left)
                lwt = sum(r["solver_step"] and not r[k + "_d"]["tc"] for r in left)
                lrt = collections.Counter(r["situation"] for r in left if r["solver_step"] and r[k + "_d"]["tc"])
                p(f"  {'':9s} {ARM_NAME[k]:17s} failures {len(left)}: not a solver step {lns}; wrong target {lwt}; right "
                  f"target {sum(lrt.values())} {sorted(lrt.items())}")
        p("")

    # ------------------------------------------------------------------ new wording and placeholder extras
    for w, title in (("new", "TABLE 8. New wording (extra): line failures flipped and fixable recovery"),
                     ("ph", "TABLE 8p. Placeholder (extra): line failures flipped and fixable recovery")):
        p(title)
        for m in ORDER[w]:
            rs = rows.get((w, m))
            if rs is None:
                continue
            lfs = [r for r in rs if r["line_failure"]]
            fx = [r for r in rs if not r["C_rec"] and r["P_i01"]]
            parts = [f"{ARM_NAME[k]} lf {sum(r[k] for r in lfs)}/{len(lfs)}, fixable {sum(r[k] for r in fx)}/{len(fx)}, "
                     f"flips {len(flips(rs, k))} (also P {sum(r['P_i01'] for r in flips(rs, k))}), breaks "
                     f"{len(breaks(rs, k))}" for k in ARMS]
            p(f"  {NAME[m]:9s} " + "; ".join(parts))
            for k in ARMS:
                S[f"extra|{w}|{m}|{k}"] = {"lf": [sum(r[k] for r in lfs), len(lfs)],
                                           "fixable": [sum(r[k] for r in fx), len(fx)],
                                           "flips": len(flips(rs, k)), "breaks": len(breaks(rs, k)),
                                           "succ": sum(r[k] for r in rs), "C": sum(r["C_rec"] for r in rs), "n": len(rs)}
        p("")
    if (HERE / "posthoc.txt").exists():
        p("=" * 100)
        p("POST HOC (posthoc.py, written after the tables above were read; it changes no arm and no criterion):")
        L.extend((HERE / "posthoc.txt").read_text().splitlines())
    (HERE / "tables.txt").write_text("\n".join(L) + "\n")
    (HERE / "summary.json").write_text(json.dumps(S, indent=1, default=str))
    print("\n".join(L))


if __name__ == "__main__":
    main()
