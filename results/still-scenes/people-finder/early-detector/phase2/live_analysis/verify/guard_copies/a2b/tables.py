"""I09 A2 tables from a2/rows_<wording>_<model>.jsonl (and base_*), as FROZEN.md section 11 and CLARIFICATIONS.md.
Writes a2/tables.txt and a2/summary.json. Reads no run or scene."""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402

I01 = paths.STILL / "replay-harness" / "repairs"
MODELS = {"old": ["qwen38_low", "glm53", "dsv41"],
          "new": ["qwen38_low", "glm53", "dsv41", "haiku55", "sonnet55_low", "gemini38_low"],
          "ph": ["qwen38_low", "glm53", "dsv41"]}
SHORT = {"qwen38_low": "Qwen", "glm53": "GLM", "dsv41": "DeepSeek", "haiku55": "Haiku", "sonnet55_low": "Sonnet",
         "gemini38_low": "Gemini"}
ARMS = ("D", "Dpix", "Dkeep")
BASES = ("R3", "R4", "P")
SITS = ("waiting_line", "conversation_group", "no_one", "audience")
SIT_SHORT = {"waiting_line": "lines", "conversation_group": "groups", "no_one": "nobody near", "audience": "audience"}
OUT = []


def p(*a):
    OUT.append(" ".join(str(x) for x in a))


def load(w, m):
    path = HERE / f"rows_{w}_{m}.jsonl"
    return [json.loads(line) for line in path.open()] if path.exists() else None


def rid(r):
    return (r["arm"], r["scene"], r["episode"], r["trial"])


def flips(rows, arm):
    return [r for r in rows if r[arm] and not r["C_rec"]]


def breaks(rows, arm):
    return [r for r in rows if not r[arm] and r["C_rec"]]


def cell(rows, arm):
    s = sum(r[arm] for r in rows)
    if arm == "C_rec":
        return f"{s}"
    return f"{s} (+{len(flips(rows, arm))}/-{len(breaks(rows, arm))})"


def answers_layouts(rs):
    return len({r["answer"] for r in rs}), len({r["layout"] for r in rs})


def recs_str(rs):
    return ", ".join(f"{r['arm'][:2]}:{r['scene'].replace('real_', '').replace('staffed_', '').replace('plain_', '')}"
                     f" {r['episode']} t{r['trial']}" for r in rs)


def main():
    summary = {}
    data = {w: {m: load(w, m) for m in ms} for w, ms in MODELS.items()}
    missing = [(w, m) for w in data for m in data[w] if data[w][m] is None]
    p("I09 A2: solver replays with A1's frozen detector people (FROZEN.md section 11; a2/CLARIFICATIONS.md)")
    p("Arms: D (primary), Dpix and Dkeep (secondary). Flips/breaks are against the recorded C. Records that are not "
      "solver steps keep their recorded verdict.")
    if missing:
        p("MISSING row files:", missing)

    # ---------------------------------------------------------------- checks
    p("")
    p("== Table 0. Checks ==")
    for w in MODELS:
        for m in MODELS[w]:
            rows = data[w][m]
            if rows is None:
                continue
            st = [r for r in rows if r["solver_step"]]
            p(f"{w:3} {SHORT[m]:8} records {len(rows)}, solver steps {len(st)}, step-0 reproduced "
              f"{sum(r['C_reproduced'] for r in st)}/{len(st)}, verdict ok {sum(r['C_verdict_ok'] for r in st)}, "
              f"C again in arms pass ok {sum(r['C_again'] == r['C_rec'] for r in st)}; P/R3/R4/C_rec/line_failure/"
              f"solver_step vs reference rows: {sum(bool(r['ref_mismatch']) for r in rows)} records disagree; "
              f"empty lists {sum(r.get('empty_list', False) for r in st)}; D and Dpix lists = the detector's people "
              f"(count) in {sum(r['D_d']['n_people'] == r['n_dets'] and r['Dpix_d']['n_people'] == r['n_dets'] for r in st)}"
              f"/{len(st)}; D differs from Dpix in {sum(r['D'] != r['Dpix'] for r in rows)} records, from Dkeep in "
              f"{sum(r['D'] != r['Dkeep'] for r in rows)}")
    # cache copies (new wording Qwen, GLM, DeepSeek vs old)
    for m in ("qwen38_low", "glm53", "dsv41"):
        o, n = data["old"][m], data["new"][m]
        if o is None or n is None:
            continue
        def keyset(rows):
            k = collections.defaultdict(set)
            for r in rows:
                if r["has_people"]:
                    k[(r["arm"], r["scene"], r["trial"])].add(r["people_hash"])
            return k
        ko, kn = keyset(o), keyset(n)
        shared = set(ko) & set(kn)
        same = sum(ko[k] == kn[k] for k in shared)
        p(f"cache copy {SHORT[m]}: new-wording keys with a list {len(kn)}, shared with old {len(shared)}, identical "
          f"{same}")

    # ---------------------------------------------------------------- primary
    p("")
    p("== Table 1. Old wording (PRIMARY), successes of 252 (flips/breaks vs recorded C); lines of 90, groups of 54, "
      "nobody near of 108 ==")
    hdr = f"{'arm':10}" + "".join(f"{SHORT[m]:>18}" for m in MODELS["old"]) + "".join(
        f"{SHORT[m] + ' ' + s:>17}" for m in MODELS["old"] for s in ("L", "G", "N"))
    p(hdr)
    for arm in ("C_rec",) + ARMS + BASES:
        line = f"{('C (rec)' if arm == 'C_rec' else arm):10}"
        for m in MODELS["old"]:
            line += f"{cell(data['old'][m], arm):>18}"
        for m in MODELS["old"]:
            rows = data["old"][m]
            for s in ("waiting_line", "conversation_group", "no_one"):
                line += f"{sum(r[arm] for r in rows if r['situation'] == s):>17}"
        p(line)
    summary["old"] = {SHORT[m]: {arm: {"successes": sum(r[arm] for r in data["old"][m]),
                                       "flips": len(flips(data["old"][m], arm)) if arm != "C_rec" else 0,
                                       "breaks": len(breaks(data["old"][m], arm)) if arm != "C_rec" else 0,
                                       **{SIT_SHORT[s]: sum(r[arm] for r in data["old"][m] if r["situation"] == s)
                                          for s in SITS if any(r["situation"] == s for r in data["old"][m])}}
                                 for arm in ("C_rec",) + ARMS + BASES} for m in MODELS["old"]}

    # per model x situation, flips and breaks
    p("")
    p("== Table 2. Per model x situation: successes (flips/breaks vs recorded C), all wordings ==")
    for w in MODELS:
        for m in MODELS[w]:
            rows = data[w][m]
            if rows is None:
                continue
            for s in SITS:
                rs = [r for r in rows if r["situation"] == s]
                if not rs:
                    continue
                p(f"{w:3} {SHORT[m]:8} {SIT_SHORT[s]:12} n {len(rs):3}  C {sum(r['C_rec'] for r in rs):3}  " +
                  "  ".join(f"{a} {cell(rs, a)}" for a in ARMS + BASES))

    # ---------------------------------------------------------------- line failures, fixable
    p("")
    p("== Table 3. Pre-registered line failures flipped, fixable recovered (old wording) ==")
    pre = {}
    for m in MODELS["old"]:
        ref = [json.loads(line) for line in open(I01 / f"rows_old_{m}.jsonl")]
        pre[m] = {rid(r) for r in ref if r["line_failure"]}
        mine = {rid(r) for r in data["old"][m] if r["line_failure"]}
        assert pre[m] == mine, m
    assert [len(pre[m]) for m in MODELS["old"]] == [2, 29, 15]
    fixl = {m: [r for r in data["old"][m] if r["line_failure"] and r["P"]] for m in MODELS["old"]}
    assert sum(len(v) for v in fixl.values()) == 25 and [len(fixl[m]) for m in MODELS["old"]] == [2, 15, 8]
    p(f"{'arm':8} {'Qwen(2)':>8} {'GLM(29)':>8} {'DS(15)':>8}   fixable line failures recovered of 25 (Q/G/D)   "
      f"fixable over all failures recovered (Q/G/D)")
    summary["line_flips_old"] = {}
    for arm in ARMS + BASES:
        lf = [sum(r[arm] for r in data["old"][m] if r["line_failure"]) for m in MODELS["old"]]
        fx = [sum(r[arm] for r in fixl[m]) for m in MODELS["old"]]
        fa = [(sum(r[arm] for r in data["old"][m] if not r["C_rec"] and r["P"]),
               sum(1 for r in data["old"][m] if not r["C_rec"] and r["P"])) for m in MODELS["old"]]
        p(f"{arm:8} {lf[0]:>8} {lf[1]:>8} {lf[2]:>8}   {sum(fx):>3} ({fx[0]}/2, {fx[1]}/15, {fx[2]}/8)"
          f"{'':22}{' , '.join(f'{a}/{b}' for a, b in fa)}")
        summary["line_flips_old"][arm] = {"line_flipped": dict(zip(["Qwen", "GLM", "DeepSeek"], lf)),
                                          "fixable_line_recovered": dict(zip(["Qwen", "GLM", "DeepSeek"], fx)),
                                          "fixable_all_recovered": dict(zip(["Qwen", "GLM", "DeepSeek"], fa))}

    # ---------------------------------------------------------------- overlap
    p("")
    p("== Table 4. Overlap of each arm's flips and breaks with I01's P, R3 and R4 (record level, old wording) ==")
    for m in MODELS["old"]:
        rows = data["old"][m]
        for arm in ARMS:
            fa, ba = {rid(r) for r in flips(rows, arm)}, {rid(r) for r in breaks(rows, arm)}
            parts = []
            for b in BASES:
                fb, bb = {rid(r) for r in flips(rows, b)}, {rid(r) for r in breaks(rows, b)}
                parts.append(f"{b}: flips {len(fb)}, shared {len(fa & fb)}, {arm} only {len(fa - fb)}, {b} only "
                             f"{len(fb - fa)}; breaks {len(bb)}, shared {len(ba & bb)}")
            p(f"{SHORT[m]:8} {arm:5} flips {len(fa):2} breaks {len(ba):2} | " + " | ".join(parts))

    # ---------------------------------------------------------------- clustering
    p("")
    p("== Table 5. Clustering: people answers (run, scene, trial) and layouts behind the flips and breaks ==")
    summary["clusters"] = {}
    for w in ("old", "new", "ph"):
        for m in MODELS[w]:
            rows = data[w][m]
            if rows is None:
                continue
            for arm in ARMS:
                f_, b_ = flips(rows, arm), breaks(rows, arm)
                if not f_ and not b_:
                    continue
                af, lf = answers_layouts(f_)
                ab, lb = answers_layouts(b_)
                p(f"{w:3} {SHORT[m]:8} {arm:5} flips {len(f_)} from {af} answers in {lf} layouts "
                  f"{sorted({r['layout'] for r in f_})}; breaks {len(b_)} from {ab} answers in {lb} layouts "
                  f"{sorted({r['layout'] for r in b_})}")
                if w == "old" and arm == "D":
                    p(f"      flips: {recs_str(f_)}")
                    p(f"      breaks: {recs_str(b_)}")
                summary["clusters"][f"{w} {SHORT[m]} {arm}"] = {"flips": len(f_), "flip_answers": af,
                                                                "flip_layouts": lf, "breaks": len(b_),
                                                                "break_answers": ab, "break_layouts": lb}

    # ---------------------------------------------------------------- dropped, added, identity, close pairs
    p("")
    p("== Table 6. Fate of the VLM detections left unmatched (dropped in D and Dpix; kept in Dkeep), and the added "
      "detector people ==")
    p("Over solver steps; [over distinct people answers] in brackets. Roles by the recorded list's own matching to "
      "the visible true people.")
    summary["fate"] = {}
    for w in ("old", "new", "ph"):
        for m in MODELS[w]:
            rows = data[w][m]
            if rows is None:
                continue
            st = [r for r in rows if r["solver_step"]]
            for arm in ARMS:
                tot, tot_u = collections.Counter(), collections.Counter()
                seen = set()
                for r in st:
                    d = r[arm + "_d"]
                    c = collections.Counter(d["unmatched_vlm_fate"])
                    c["named"] = d["unmatched_vlm_named"]
                    c["named_come_after"] = d["unmatched_vlm_named_come_after"]
                    c["all"] = d["n_unmatched_vlm"]
                    for k in ("added", "added_matched_missed", "added_matched_found", "added_false", "close_pairs",
                              "close_pairs_added", "added_off_image", "added_box_centre_pixel", "added_yaw_none",
                              "matched_yaw_from_vlm"):
                        c[k] = d[k]
                    for k, v in d["identity"].items():
                        c["id_" + k] = v
                    for role in d["added_roles"]:
                        c["added_role_" + role] += 1
                    c["tnum_changed"] = int(d["tnum"] != r["tnum_C"])
                    tot.update(c)
                    key = (r["answer"], r["people_hash"])
                    if key not in seen:
                        seen.add(key)
                        tot_u.update(c)
                roles = ["true_staff", "true_line", "true_group", "true_other", "none"]
                p(f"{w:3} {SHORT[m]:8} {arm:5} unmatched VLM {tot['all']} [{tot_u['all']}]: " +
                  ", ".join(f"{k} {tot[k]} [{tot_u[k]}]" for k in roles) +
                  f"; named in a constraint {tot['named']} [{tot_u['named']}], in come_after "
                  f"{tot['named_come_after']} [{tot_u['named_come_after']}]")
                p(f"{'':18}added detector people {tot['added']} [{tot_u['added']}]: matched to a person the list "
                  f"missed {tot['added_matched_missed']} [{tot_u['added_matched_missed']}] (roles " +
                  ", ".join(f"{k[11:]} {v}" for k, v in sorted(tot.items()) if k.startswith("added_role_")) +
                  f"), matched to a person the list had {tot['added_matched_found']} [{tot_u['added_matched_found']}]"
                  f", false {tot['added_false']} [{tot_u['added_false']}]; pixel off image {tot['added_off_image']}, "
                  f"box-centre pixel {tot['added_box_centre_pixel']}, yaw None {tot['added_yaw_none']}")
                p(f"{'':18}relocated identity: same {tot['id_same']}, changed {tot['id_changed']}, none->someone "
                  f"{tot['id_none_to_someone']}, someone->none {tot['id_someone_to_none']}, none->none "
                  f"{tot['id_none_to_none']}; matched yaw from VLM (detector None) {tot['matched_yaw_from_vlm']}; "
                  f"close pairs (< 0.25 m) {tot['close_pairs']} (with an added person {tot['close_pairs_added']}); "
                  f"target number changed {tot['tnum_changed']}")
                summary["fate"][f"{w} {SHORT[m]} {arm}"] = {"steps": dict(tot), "answers": dict(tot_u)}

    # ---------------------------------------------------------------- remaining line failures
    p("")
    p("== Table 7. Line failures each arm leaves (old wording; descriptive classes, CLARIFICATIONS point 23) ==")
    for m in MODELS["old"]:
        rows = data["old"][m]
        for arm in ARMS:
            left = [r for r in rows if r["line_failure"] and not r[arm]]
            c = collections.Counter(r[arm + "_d"].get("class") for r in left)
            fixable_left = sum(r["P"] for r in left)
            p(f"{SHORT[m]:8} {arm:5} left {len(left)} (P-fixable {fixable_left}): " +
              ", ".join(f"{k}: {v}" for k, v in sorted(c.items(), key=lambda kv: str(kv[0]))))
        # also all waiting_line failures under the arm (not only pre-registered)

    # ---------------------------------------------------------------- POST HOC mechanism
    p("")
    p("== Table 7b. POST HOC (written after Tables 1-7 were read; changes no arm or verdict): mechanism of each D "
      "flip and break, and of the records where D and P disagree ==")
    p("Classes, first that applies: (a) a dropped detection was named in a recorded constraint; (b) something was "
      "dropped or added, but no named number was dropped; (c) neither: only positions and yaws changed (max move of a "
      "matched detection given).")

    def mech(r, arm="D"):
        d = r[arm + "_d"]
        named_drop = d["unmatched_vlm_named"] > 0 and arm != "Dkeep"
        mv = max(d["moved_m"].values(), default=0.0)
        if named_drop:
            return "a named number dropped", mv
        if d["dropped"] or d["added"] or (arm == "Dkeep" and d["n_unmatched_vlm"]):
            return "b other drop/add", mv
        return "c positions/yaws only", mv

    summary["posthoc_mechanism"] = {}
    for w in ("old", "new", "ph"):
        for m in MODELS[w]:
            rows = data[w][m]
            if rows is None:
                continue
            for kind, rs in (("flips", flips(rows, "D")), ("breaks", breaks(rows, "D"))):
                if not rs:
                    continue
                c = collections.Counter(mech(r)[0] for r in rs)
                mvs = sorted(round(mech(r)[1], 2) for r in rs if mech(r)[0].startswith("c"))
                p(f"{w:3} {SHORT[m]:8} D {kind:6} {len(rs):2}: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())) +
                  (f"; (c) max moves {mvs}" if mvs else ""))
                summary["posthoc_mechanism"][f"{w} {SHORT[m]} D {kind}"] = dict(c)
            dis = [r for r in rows if r["solver_step"] and r["D"] != r["P"]]
            if dis:
                c = collections.Counter(mech(r)[0] for r in dis)
                p(f"{w:3} {SHORT[m]:8} D != P in {len(dis)} records (D wins {sum(r['D'] for r in dis)}, P wins "
                  f"{sum(r['P'] for r in dis)}): " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))

    p("")
    p("== Table 7c. POST HOC sensitivity (old wording): net gain over C by layout, and D with its position/yaw-only "
      "flips removed ==")
    bars = {"qwen38_low": 232, "glm53": 199, "dsv41": 190}
    summary["posthoc_sensitivity"] = {}
    for m in MODELS["old"]:
        rows = data["old"][m]
        lay = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            lay[r["layout"]][0] += int(r["D"]) - int(r["C_rec"])
            lay[r["layout"]][1] += int(r["P"]) - int(r["C_rec"])
        nz = {k: v for k, v in sorted(lay.items()) if v != [0, 0]}
        d = sum(r["D"] for r in rows)
        c_only = sum(1 for r in flips(rows, "D") if mech(r)[0].startswith("c"))
        top = max(nz.items(), key=lambda kv: kv[1][0]) if nz else None
        p(f"{SHORT[m]:8} net D-C / P-C by layout: " + ", ".join(f"{k} {v[0]:+d}/{v[1]:+d}" for k, v in nz.items()))
        p(f"{'':9}D {d}; without its {c_only} position/yaw-only flips {d - c_only} (bar {bars[m]}); without the "
          f"layout with the largest D gain ({top[0] if top else '-'}, {top[1][0] if top else 0:+d}) "
          f"{d - (top[1][0] if top else 0)} over {len(rows) - sum(r['layout'] == (top[0] if top else '') for r in rows)}"
          f" records")
        summary["posthoc_sensitivity"][SHORT[m]] = {"by_layout_D_minus_C_P_minus_C": nz, "D": d,
                                                    "D_without_positions_yaw_only_flips": d - c_only}

    # ---------------------------------------------------------------- extras
    for w, n in (("new", 252), ("ph", 156)):
        p("")
        p(f"== Table 8{'a' if w == 'new' else 'b'}. Extras: {w} wording{'' if w == 'new' else ' (placeholder C runs)'},"
          f" successes of {n} (flips/breaks vs recorded C); lines ==")
        for m in MODELS[w]:
            rows = data[w][m]
            if rows is None:
                continue
            nl = sum(r["situation"] == "waiting_line" for r in rows)
            lf = [r for r in rows if r["line_failure"]]
            fx = [r for r in rows if r["line_failure"] and r["P"]]
            p(f"{SHORT[m]:8} n {len(rows)}  C {sum(r['C_rec'] for r in rows)}  " +
              "  ".join(f"{a} {cell(rows, a)}" for a in ARMS + BASES) +
              f" | lines of {nl}: C {sum(r['C_rec'] for r in rows if r['situation'] == 'waiting_line')} " +
              " ".join(f"{a} {sum(r[a] for r in rows if r['situation'] == 'waiting_line')}" for a in ARMS + BASES) +
              f" | line failures {len(lf)}, flipped " + " ".join(f"{a} {sum(r[a] for r in lf)}" for a in ARMS + BASES) +
              f" | fixable line {len(fx)}, recovered " + " ".join(f"{a} {sum(r[a] for r in fx)}" for a in ARMS))
            summary[f"{w} {SHORT[m]}"] = {a: {"successes": sum(r[a] for r in rows), "flips": len(flips(rows, a)),
                                              "breaks": len(breaks(rows, a))} for a in ARMS + BASES}
            summary[f"{w} {SHORT[m]}"]["C_rec"] = sum(r["C_rec"] for r in rows)

    # ---------------------------------------------------------------- criteria
    p("")
    p("== Table 9. A2 criteria (FROZEN.md section 11; judged on D, old wording) ==")
    crit = {}
    for arm in ARMS:
        s = {m: sum(r[arm] for r in data["old"][m]) for m in MODELS["old"]}
        q, g, d = s["qwen38_low"], s["glm53"], s["dsv41"]
        succ = {"GLM >= 199 and >= 186": g >= 199 and g >= 186, "DeepSeek >= 190 and >= 182": d >= 190 and d >= 182,
                "Qwen >= 232": q >= 232}
        kill = g < 186 and d < 182
        verdict = "success" if all(succ.values()) else "kill" if kill else "neither"
        crit[arm] = {"Qwen": q, "GLM": g, "DeepSeek": d, "success_clauses": succ, "kill": kill, "verdict": verdict}
        p(f"{arm:5} Qwen {q} (bar 232), GLM {g} (bar 199; C 186; P 202), DeepSeek {d} (bar 190; C 182; P 193): " +
          ", ".join(f"{k}: {'met' if v else 'NOT met'}" for k, v in succ.items()) +
          f"; kill (GLM < 186 and DeepSeek < 182): {'FIRES' if kill else 'not met'} -> {verdict}"
          f"{' (PRIMARY)' if arm == 'D' else ' (secondary)'}")
    for w in ("new", "ph"):
        for arm in ARMS:
            s = {m: sum(r[arm] for r in data[w][m]) for m in ("qwen38_low", "glm53", "dsv41") if data[w][m]}
            c = {m: sum(r["C_rec"] for r in data[w][m]) for m in s}
            pp = {m: sum(r["P"] for r in data[w][m]) for m in s}
            ok = {m: (s[m] >= c[m] - 2) if m == "qwen38_low" else (s[m] >= pp[m] - 3 and s[m] >= c[m]) for m in s}
            kill = s["glm53"] < c["glm53"] and s["dsv41"] < c["dsv41"]
            verdict = "success" if all(ok.values()) else "kill" if kill else "neither"
            crit[f"{w} {arm}"] = {SHORT[m]: s[m] for m in s} | {"verdict": verdict}
            p(f"secondary {w:3} {arm:5}: " + ", ".join(
                f"{SHORT[m]} {s[m]} (C {c[m]}, P {pp[m]}; bar "
                f"{'C-2 = ' + str(c[m] - 2) if m == 'qwen38_low' else 'max(P-3, C) = ' + str(max(pp[m] - 3, c[m]))}:"
                f" {'met' if ok[m] else 'NOT met'})" for m in s) +
              f"; kill (GLM < C and DeepSeek < C): {'FIRES' if kill else 'not met'} -> {verdict} (secondary, "
              f"same bars applied to these runs)")
    summary["criteria"] = crit
    p("")
    p("Prediction (FROZEN.md, inference): D between I01's R4 and P for GLM (200-202) and DeepSeek (188-193), Qwen "
      "234-236. Observed D: " + ", ".join(f"{k} {crit['D'][k]}" for k in ("Qwen", "GLM", "DeepSeek")))
    (HERE / "tables.txt").write_text("\n".join(OUT) + "\n")
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print("\n".join(OUT))


if __name__ == "__main__":
    main()
