"""Independent verifier for I03 Stage B: tables and criteria from my own rows (v_rows_*.jsonl), FROZEN.md section 8.
Reads phase1/i01_repairs/rows_{old,new}_*.jsonl for I01's P / R3 / R4 and the pre-registered line-failure sets.
Writes v_tables.txt and v_summary.json. No model, no network.
"""
from __future__ import annotations

import collections
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
RD = HERE.parents[2]
I01 = RD / "phase1" / "i01_repairs"
ARMS = ("Badd", "Bprune", "Breloc", "BrelocP")
MODELS = {"old": ["qwen38_low", "glm53", "dsv41"],
          "new": ["qwen38_low", "glm53", "dsv41", "haiku55", "sonnet55_low", "gemini38_low"],
          "ph": ["qwen38_low", "glm53", "dsv41"]}
SHORT = {"qwen38_low": "Qwen", "glm53": "GLM", "dsv41": "DeepSeek", "haiku55": "Haiku", "sonnet55_low": "Sonnet",
         "gemini38_low": "Gemini"}
GROUP = {"no_one": "nobody", "waiting_line": "lines", "conversation_group": "groups", "audience": "groups"}
L = []


def out(*a):
    s = " ".join(str(x) for x in a)
    L.append(s)
    print(s)


def key(r):
    return (r["arm"], r["scene"], r["episode"], r["trial"])


def load(w, m):
    return [json.loads(l) for l in (HERE / f"v_rows_{w}_{m}.jsonl").open()]


def i01(w, m):
    p = I01 / f"rows_{w}_{m}.jsonl"
    return {key(r): r for r in map(json.loads, p.open())} if p.exists() else None


def verdict(r, arm):
    if arm == "C":
        return r["C_rec"]
    if not r["solver_step"]:
        return r["C_rec"]
    return r[arm]["ok"]


def line_failure(r):
    return (r["situation"] == "waiting_line" and r["solver_step"] and r["tc"] and not r["C_rec"]
            and (not r["in_region"] or not r["reachable"]))


def layout(scene):
    for p in ("real_", "staffed_", "plain_"):
        if scene.startswith(p):
            return scene[len(p):]
    return scene


def main():
    S = {}
    rows = {(w, m): load(w, m) for w, ms in MODELS.items() for m in ms}
    I = {(w, m): i01(w, m) for w, ms in MODELS.items() for m in ms if w != "ph"}
    # P / R3 / R4 per record
    for (w, m), rs in rows.items():
        for r in rs:
            if w == "ph":
                if r["solver_step"]:
                    r["P"], r["R3"], r["R4"] = r["P_ph"], r["R3_ph"], r["R4_ph"]
                else:
                    r["P"] = r["R3"] = r["R4"] = r["C_rec"]
                r["lf_i01"] = None
            else:
                o = I[(w, m)][key(r)]
                assert o["C_rec"] == r["C_rec"] and o["solver_step"] == r["solver_step"], key(r)
                r["P"], r["R3"], r["R4"], r["lf_i01"] = o["P"], o["R3"], o["R4"], o["line_failure"]
            r["lf"] = line_failure(r)

    out("STEP 0")
    tot = collections.Counter()
    for (w, m), rs in rows.items():
        st = [r for r in rs if r["solver_step"]]
        rp = sum(r["C_repro"] for r in st)
        vo = sum(r["C_now"] == r["C_rec"] for r in st)
        out(f"  {w:4s} {SHORT[m]:9s} records {len(rs)} solver {len(st)} reproduced {rp} verdict {vo}")
        g = "ph" if w == "ph" else "real"
        tot[g + "_n"] += len(st)
        tot[g + "_rp"] += rp
        tot[g + "_vo"] += vo
        S[f"step0_{w}_{m}"] = (len(st), rp, vo)
    out("  totals", dict(tot))
    S["step0_tot"] = dict(tot)

    out("\nLINE-FAILURE SETS (recomputed vs I01 rows)")
    for (w, m), rs in rows.items():
        if w == "ph":
            continue
        a = {key(r) for r in rs if r["lf"]}
        b = {key(r) for r in rs if r["lf_i01"]}
        out(f"  {w} {SHORT[m]:9s} recomputed {len(a)} i01 {len(b)} equal {a == b}")

    out("\nTOTALS (successes, +flips/-breaks vs recorded C)")
    for (w, m), rs in rows.items():
        line = [f"{w:4s} {SHORT[m]:9s} n={len(rs)}"]
        for arm in ("C",) + ARMS + ("Breloc_unpaired", "R3", "R4", "P"):
            def v(r):
                if arm in ("R3", "R4", "P"):
                    return r[arm]
                return verdict(r, arm)
            tot_ = sum(v(r) for r in rs)
            fl = sum(v(r) and not r["C_rec"] for r in rs)
            br = sum(r["C_rec"] and not v(r) for r in rs)
            grp = collections.Counter()
            for r in rs:
                grp[GROUP[r["situation"]]] += v(r)
            S[f"tot_{w}_{m}_{arm}"] = dict(total=tot_, flips=fl, breaks=br, **grp)
            line.append(f"{arm} {tot_} (+{fl}/-{br}) [N {grp['nobody']} L {grp['lines']} G {grp['groups']}]")
        out("  " + "; ".join(line))
    cnt = collections.Counter(r["situation"] for r in rows[("old", "glm53")])
    out("  situation n (old GLM):", dict(cnt))

    out("\nCRITERIA (old wording, B-add primary, others secondary)")
    old = {m: rows[("old", m)] for m in MODELS["old"]}
    rec_lines = {m: sum(r["C_rec"] for r in old[m] if r["situation"] == "waiting_line") for m in old}
    out("  recorded lines", rec_lines)
    fixable_lf = {m: [r for r in old[m] if r["lf"] and r["P"] and not r["C_rec"]] for m in old}
    out("  fixable line failures", {m: len(v) for m, v in fixable_lf.items()})
    for arm in ARMS + ("Breloc_unpaired",):
        lines = {m: sum(verdict(r, arm) for r in old[m] if r["situation"] == "waiting_line") for m in old}
        nonline = {m: sum(verdict(r, arm) - r["C_rec"] for r in old[m] if r["situation"] != "waiting_line")
                   for m in old}
        qtot = sum(verdict(r, arm) for r in old["qwen38_low"])
        fixrec = {m: sum(verdict(r, arm) for r in fixable_lf[m]) for m in old}
        flips = sum(verdict(r, arm) and not r["C_rec"] for m in old for r in old[m])
        breaks = sum(r["C_rec"] and not verdict(r, arm) for m in old for r in old[m])
        succ = {"GLM lines>=47": lines["glm53"] >= 47, "DS lines>=71": lines["dsv41"] >= 71,
                "nonline net>=-1": all(v >= -1 for v in nonline.values()), "Qwen total>=234": qtot >= 234,
                "fixable>=13": sum(fixrec.values()) >= 13}
        k1 = (lines["glm53"] - rec_lines["glm53"] <= 2) and (lines["dsv41"] - rec_lines["dsv41"] <= 1)
        k2 = breaks >= flips
        verdict_ = "success" if all(succ.values()) and not (k1 or k2) else ("kill" if (k1 or k2) else "neither")
        if all(succ.values()) and (k1 or k2):
            verdict_ = "success+kill?"
        S[f"crit_{arm}"] = dict(lines=lines, nonline=nonline, qtot=qtot, fixrec=fixrec, flips=flips, breaks=breaks,
                                succ=succ, kill1=k1, kill2=k2, verdict=verdict_)
        out(f"  {arm:16s} lines {lines} nonline {nonline} Qwen total {qtot} fixable {fixrec} "
            f"(pooled {sum(fixrec.values())}/25) flips {flips} breaks {breaks}")
        out(f"  {'':16s} success {succ} kill1 {k1} kill2 {k2} -> {verdict_}")
    # kill 3 reference: line failures not P-fixable
    lf_all = [r for m in old for r in old[m] if r["lf"]]
    out(f"  line failures {len(lf_all)}; not P-fixable {sum(not r['P'] for r in lf_all)}")

    out("\nPRE-REGISTERED LINE-FAILURE SETS FLIPPED (old)")
    for m in old:
        lfs = [r for r in old[m] if r["lf_i01"]]
        parts = []
        for arm in ARMS + ("R3", "R4", "P"):
            def v(r):
                return r[arm] if arm in ("R3", "R4", "P") else verdict(r, arm)
            f = [r for r in lfs if v(r)]
            parts.append(f"{arm} {len(f)}/{len(lfs)} (P {sum(r['P'] for r in f)}, R4 {sum(r['R4'] for r in f)}, "
                         f"R3 {sum(r['R3'] for r in f)})")
            S[f"lfset_{m}_{arm}"] = (len(f), len(lfs))
        out(f"  {SHORT[m]:9s} " + "; ".join(parts))

    out("\nFIXABLE RECOVERY (old): line failures and all failures")
    for arm in ARMS:
        a = []
        for m in old:
            fails = [r for r in old[m] if not r["C_rec"]]
            fx = [r for r in fails if r["P"]]
            fxl = [r for r in fx if r["lf"]]
            a.append(f"{SHORT[m]} lf {sum(verdict(r, arm) for r in fxl)}/{len(fxl)} all {sum(verdict(r, arm) for r in fx)}"
                     f"/{len(fx)} of {len(fails)} failures")
        out(f"  {arm:8s} " + "; ".join(a))

    out("\nOVERLAP OF FLIPS WITH I01 P / R4 / R3 FLIPS (old)")
    for arm in ARMS:
        for m in old:
            f = [r for r in old[m] if verdict(r, arm) and not r["C_rec"]]
            r3f = [r for r in old[m] if r["R3"] and not r["C_rec"]]
            out(f"  {arm:8s} {SHORT[m]:9s} flips {len(f)}: P {sum(r['P'] for r in f)} R4 {sum(r['R4'] for r in f)} "
                f"R3 {sum(r['R3'] for r in f)} notP {sum(not r['P'] for r in f)}; R3 flips {len(r3f)} of which "
                f"flipped here {sum(verdict(r, arm) for r in r3f)}; notP {[ (r['scene'], r['episode'], r['trial']) for r in f if not r['P']]}")

    out("\nANSWERS / LAYOUTS BEHIND FLIPS AND BREAKS")
    for w in ("old", "new", "ph"):
        for m in MODELS[w]:
            for arm in ARMS:
                rs = rows[(w, m)]
                f = [r for r in rs if verdict(r, arm) and not r["C_rec"]]
                b = [r for r in rs if r["C_rec"] and not verdict(r, arm)]
                fa = sorted({(r["run"], r["scene"], r["trial"]) for r in f})
                fl = collections.Counter(layout(r["scene"]) for r in f)
                ba = sorted({(r["run"][-30:], r["scene"], r["episode"], r["trial"]) for r in b})
                if w == "old" or f or b:
                    out(f"  {w} {SHORT[m]:9s} {arm:8s} flips {len(f)} from {len(fa)} answers in {len(fl)} layouts "
                        f"{dict(fl)}; breaks {len(b)} {ba}")
                if arm == "Badd" and f:
                    for r in f:
                        out(f"      flip {r['scene']} {r['episode']} t{r['trial']} added {r['Badd']['added']} "
                            f"joined {r['Badd']['joined']} joined_added {r['Badd']['joined_added']} P {r['P']} R3 {r['R3']}")

    out("\nB-ADD MECHANISM (solver steps)")
    for (w, m), rs in rows.items():
        st = [r for r in rs if r["solver_step"]]
        a = sum(r["Badd"]["added"] for r in st)
        am = sum(r["Badd"]["added_missed"] for r in st)
        ar = sum(r["Badd"]["added_reassigned"] for r in st)
        af = sum(r["Badd"]["added_false"] for r in st)
        steps_add = sum(r["Badd"]["added"] > 0 for r in st)
        joinedadd = sum(r["Badd"]["joined_added"] > 0 for r in st)
        out(f"  {w:4s} {SHORT[m]:9s} added {a} (missed {am}, reassigned {ar}, false {af}); steps with added "
            f"{steps_add}/{len(st)}; steps where an added joined {joinedadd}; off-image {sum(r['Badd']['off'] for r in st)}"
            f" noproj {sum(r['Badd']['noproj'] for r in st)}")
        false_by_layout = collections.Counter()
        for r in st:
            if r["Badd"]["added_false"]:
                false_by_layout[layout(r["scene"])] += r["Badd"]["added_false"]
        out(f"       false added by layout {dict(false_by_layout)}")
    for m in old:
        lfs = [r for r in old[m] if r["lf_i01"]]
        out(f"  old {SHORT[m]} line failures {len(lfs)}: added {sum(r['Badd']['added'] for r in lfs)} (missed "
            f"{sum(r['Badd']['added_missed'] for r in lfs)}), steps with added {sum(r['Badd']['added'] > 0 for r in lfs)},"
            f" steps where an added joined {sum(r['Badd']['joined_added'] > 0 for r in lfs)}, flipped "
            f"{sum(r['Badd']['ok'] for r in lfs)}")

    out("\nWHY LINE FAILURES LEFT BY AN ARM FAIL (old; classes 1 no used come_after, 2 unsupported waiting number,"
        " 3 member not in line, 4 complete)")
    for m in old:
        lfs = [r for r in old[m] if r["lf_i01"]]
        for arm in ("Badd", "BrelocP", "Bprune", "Breloc"):
            left = [r for r in lfs if not r[arm]["ok"]]
            c = collections.Counter(r[arm]["cls"] for r in left)
            cP = collections.Counter(r[arm]["cls"] for r in left if r["P"])
            nunsup = sum(len(r[arm]["cls_info"]["unsup"]) for r in left if r[arm]["cls"] == 2)
            nout = sum(r[arm]["cls_info"]["out"] for r in left if r[arm]["cls"] == 3)
            nocome = sum(1 for r in left if r[arm]["cls"] == 1 and "come_after" not in json.dumps(
                [k for k in (r.get("kinds") or [])]))
            out(f"  {SHORT[m]:9s} {arm:8s} left {len(left)}: {dict(sorted(c.items()))}; P-fixable by class "
                f"{dict(sorted(cP.items()))}; unsupported numbers (class 2) {nunsup}; members out (class 3) {nout}")

    out("\nWHAT PRUNE / RELOCATE ACT ON (solver steps)")
    for (w, m), rs in rows.items():
        st = [r for r in rs if r["solver_step"]]
        pr = sum(r["Bprune"]["pruned"] for r in st)
        pm = sum(r["Bprune"]["pruned_matched"] for r in st)
        ex = sum(r["Bprune"]["exempt"] for r in st)
        pairs = sum(r["reloc_pairs"] for r in st)
        mv = [x for r in st for x in r["reloc_moved_m"]]
        mvm = sum(r["Breloc"]["moved_was_matched"] for r in st)
        diff = sum(r["reloc_differs"] for r in st)
        close = sum(r[a]["close"] for r in st for a in ("Badd", "Breloc"))
        prp = sum(r["BrelocP"]["pruned"] for r in st)
        unp = sum(r["Breloc_unpaired"]["ok"] != r["Breloc"]["ok"] for r in st)
        out(f"  {w:4s} {SHORT[m]:9s} pruned {pr} (matched {pm}, unmatched {pr - pm}), exempt {ex}; reloc pairs {pairs}"
            f" (moved median {statistics.median(mv) if mv else float('nan'):.2f} m; were matched {mvm}); "
            f"BrelocP pruned {prp}; recompute differs {diff}; close pairs {close}; Breloc vs unpaired verdict differs {unp}")

    out("\nNUMBER COLLISIONS (post hoc check of the compute agent's claim)")
    for (w, m), rs in rows.items():
        st = [r for r in rs if r["solver_step"]]
        hits = []
        for r in st:
            dangling = [n for n in r["cons_numbers"] if n not in r["det_numbers"]]
            taken = [n for n in dangling if n >= r["add_start"] and n < r["add_start"] + r["Badd"]["added"]]
            if dangling and taken:
                hits.append((r["scene"], r["episode"], r["trial"], r["det_numbers"], dangling, taken, r["C_rec"],
                             r["Badd"]["ok"]))
        out(f"  {w} {SHORT[m]}: steps with dangling numbers taken by added people {len(hits)}")
        for h in hits:
            out(f"      {h}")

    out("\nCACHE-COPY CHECK (people lists per (arm, scene, trial), new vs old)")
    for m in ("qwen38_low", "glm53", "dsv41"):
        def lists(rs):
            d = collections.defaultdict(set)
            for r in rs:
                if r["people_json"] is not None:
                    d[(r["arm"], r["scene"], r["trial"])].add(r["people_json"])
            return d
        a, b = lists(rows[("new", m)]), lists(rows[("old", m)])
        shared = set(a) & set(b)
        out(f"  {SHORT[m]}: shared keys {len(shared)}, identical {sum(a[k] == b[k] for k in shared)}; new keys {len(a)},"
            f" old keys {len(b)}")

    (HERE / "v_tables.txt").write_text("\n".join(L) + "\n")
    (HERE / "v_summary.json").write_text(json.dumps(S, indent=1, default=str))


if __name__ == "__main__":
    main()
