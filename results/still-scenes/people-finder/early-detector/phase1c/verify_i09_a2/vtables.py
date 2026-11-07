"""Tables and criteria from my own vrows_*.jsonl, plus record-by-record comparison with I01's rows, phase1b stageB's
placeholder rows and the compute agent's a2/rows_* (read as raw data only)."""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
I09 = HERE.parent
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402

I01 = paths.STILL / "replay-harness" / "repairs"
SB = paths.STILL / "perception" / "depth-blob-detector" / "stageB"
A2 = I09 / "a2"
NAME = {"qwen38_low": "Qwen", "glm53": "GLM", "dsv41": "DeepSeek", "haiku55": "Haiku", "sonnet55_low": "Sonnet",
        "gemini38_low": "Gemini"}
MODELS = {"old": ["qwen38_low", "glm53", "dsv41"],
          "new": ["qwen38_low", "glm53", "dsv41", "haiku55", "sonnet55_low", "gemini38_low"],
          "ph": ["qwen38_low", "glm53", "dsv41"]}
SITS = {"waiting_line": "lines", "conversation_group": "groups", "no_one": "nobody", "audience": "audience"}
OUT = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


def load(path):
    return [json.loads(line) for line in open(path)] if Path(path).exists() else None


def key(r):
    return (r["arm"], r["scene"], r["episode"], r["trial"])


def layout(scene):
    return re.sub(r"^(real|staffed|plain)_", "", scene)


def answer(r):
    return (r["run"], r["scene"], r["trial"])


def flips(rows, arm, base="C_rec"):
    f = {key(r) for r in rows if r[arm] and not r[base]}
    b = {key(r) for r in rows if not r[arm] and r[base]}
    return f, b


summary = {}
for wording in ("old", "new", "ph"):
    for m in MODELS[wording]:
        rows = load(HERE / f"vrows_{wording}_{m}.jsonl")
        if rows is None:
            p(f"[{wording} {m}] missing")
            continue
        n = len(rows)
        ss = [r for r in rows if r["solver_step"]]
        rep = sum(r["step0_reproduced"] for r in ss)
        vok = sum(r["step0_verdict_ok"] for r in ss)
        maxmv = max(r["step0_moved_m"] for r in ss)
        tot = {a: sum(r[a] for r in rows) for a in ("C_rec", "D", "Dpix", "Dkeep", "P", "R3", "R4")}
        s = dict(n=n, solver_steps=len(ss), step0_reproduced=rep, step0_verdict_ok=vok, max_moved=maxmv, **tot)
        for a in ("D", "Dpix", "Dkeep", "P", "R3", "R4"):
            f, b = flips(rows, a)
            s[a + "_fb"] = (len(f), len(b))
        # situations
        bys = {}
        for sit in sorted({r["situation"] for r in rows}):
            rr = [r for r in rows if r["situation"] == sit]
            bys[sit] = dict(n=len(rr), **{a: sum(r[a] for r in rr) for a in ("C_rec", "D", "Dpix", "Dkeep", "P")})
            fD, bD = flips(rr, "D")
            bys[sit]["D_fb"] = (len(fD), len(bD))
        s["by_sit"] = bys
        s["empty_lists"] = sum(r.get("empty_list", False) for r in ss)
        s["empty_lists_changed"] = sum(r["D"] != r["C_rec"] for r in ss if r.get("empty_list"))
        s["close_pairs"] = {a: sum(r[a + "_d"]["close"] for r in ss) for a in ("D", "Dpix", "Dkeep")}
        s["yaw_from_vlm"] = {a: sum(r[a + "_d"]["yaw_from_vlm"] for r in ss) for a in ("D", "Dpix", "Dkeep")}
        # reference comparisons
        if wording in ("old", "new"):
            ref = {key(r): r for r in load(I01 / f"rows_{wording}_{m}.jsonl")}
            dis = [key(r) for r in rows if any(r[k] != ref[key(r)][k] for k in
                                                ("P", "R3", "R4", "C_rec", "line_failure", "solver_step"))]
            s["vs_I01_disagree"] = len(dis)
            assert len(ref) == n
        else:
            ref = {key(r): r for r in load(SB / f"rows_ph_{m}.jsonl")}
            dis = [key(r) for r in rows if (r["P"], r["R3"], r["R4"], r["C_rec"]) !=
                   (ref[key(r)]["P_i01"], ref[key(r)]["R3_i01"], ref[key(r)]["R4_i01"], ref[key(r)]["C_rec"])]
            s["vs_stageB_disagree"] = len(dis)
        ca = load(A2 / f"rows_{wording}_{m}.jsonl")
        if ca is not None:
            cak = {key(r): r for r in ca}
            assert len(cak) == n, (wording, m, len(cak), n)
            dd = collections.Counter()
            diff_keys = collections.defaultdict(list)
            for r in rows:
                c = cak[key(r)]
                for a in ("C_rec", "D", "Dpix", "Dkeep", "P", "R3", "R4", "line_failure", "solver_step"):
                    if r[a] != c[a]:
                        dd[a] += 1
                        diff_keys[a].append(key(r))
                if r["solver_step"]:
                    for a in ("D", "Dpix", "Dkeep"):
                        mine, theirs = r[a + "_d"], c[a + "_d"]
                        if a != "Dkeep" and sorted(mine["dropped"]) != sorted(theirs["dropped"]):
                            dd[a + "_dropped"] += 1
                        if len(mine["dropped"]) != theirs["n_unmatched_vlm"]:
                            dd[a + "_n_unmatched"] += 1
                        if len(mine["added"]) != theirs["added"]:
                            dd[a + "_added"] += 1
                        if mine["close"] != theirs["close_pairs"]:
                            dd[a + "_close"] += 1
                        cmap = {"1": "no_come_after", "2": "member_not_in_line", "3": "complete"}
                        tcls = theirs.get("class")
                        tcls = None if tcls is None else cmap[tcls[0]]
                        if mine["line_cls"] is not None and tcls is not None and mine["line_cls"] != tcls:
                            dd[a + "_linecls"] += 1
                        if (mine["line_cls"] is None) != (tcls is None):
                            dd[a + "_linecls_presence"] += 1
            s["vs_compute_agent"] = dict(dd)
            s["vs_compute_agent_keys"] = {k: v[:5] for k, v in diff_keys.items()}
        summary[(wording, m)] = (s, rows)

# ---------------------------------------------------------------- headline tables
for wording in ("old", "new", "ph"):
    p(f"\n=== {wording} wording ===")
    for m in MODELS[wording]:
        if (wording, m) not in summary:
            continue
        s, rows = summary[(wording, m)]
        p(f"{NAME[m]:9s} n {s['n']} solver {s['solver_steps']} step0 rep {s['step0_reproduced']} verdict_ok "
          f"{s['step0_verdict_ok']} maxmoved {s['max_moved']:.2e} | C {s['C_rec']} D {s['D']} {s['D_fb']} "
          f"Dpix {s['Dpix']} {s['Dpix_fb']} Dkeep {s['Dkeep']} {s['Dkeep_fb']} | P {s['P']} {s['P_fb']} R3 {s['R3']} "
          f"R4 {s['R4']} {s['R4_fb']}")
        p(f"          ref-disagree {s.get('vs_I01_disagree', s.get('vs_stageB_disagree'))}; vs compute agent "
          f"{s.get('vs_compute_agent')} {s.get('vs_compute_agent_keys', '')}; empty lists {s['empty_lists']} "
          f"(changed {s['empty_lists_changed']}); close pairs {s['close_pairs']}; yaw_from_vlm {s['yaw_from_vlm']}")
        for sit, b in s["by_sit"].items():
            p(f"          {SITS.get(sit, sit):8s} n {b['n']}: C {b['C_rec']} D {b['D']} {b['D_fb']} Dpix {b['Dpix']} "
              f"Dkeep {b['Dkeep']} P {b['P']}")

# ---------------------------------------------------------------- criteria (D, old)
p("\n=== criteria (FROZEN 11; D, old wording) ===")
crit = {}


def bars(wording):
    out = {}
    for m in ("qwen38_low", "glm53", "dsv41"):
        s, _ = summary[(wording, m)]
        if m == "qwen38_low":
            out[m] = s["C_rec"] - 2
        else:
            out[m] = max(s["P"] - 3, s["C_rec"])
    return out


for wording in ("old", "new", "ph"):
    if not all((wording, m) in summary for m in ("qwen38_low", "glm53", "dsv41")):
        continue
    b = bars(wording)
    for arm in ("D", "Dpix", "Dkeep"):
        vals = {m: summary[(wording, m)][0][arm] for m in b}
        succ = all(vals[m] >= b[m] for m in b)
        kill = (vals["glm53"] < summary[(wording, "glm53")][0]["C_rec"]
                and vals["dsv41"] < summary[(wording, "dsv41")][0]["C_rec"])
        verdict = "success" if succ else "kill" if kill else "neither"
        p(f"{wording:3s} {arm:5s} bars {b} values {vals} -> {verdict}")
        crit[(wording, arm)] = verdict
if ("old", "glm53") in summary:
    p("Note bars stated in FROZEN: GLM >= 199 (and >= 186), DeepSeek >= 190 (and >= 182), Qwen >= 232")

# ---------------------------------------------------------------- old-wording detail
if all(("old", m) in summary for m in MODELS["old"]):
    p("\n=== old wording detail (D) ===")
    for m in MODELS["old"]:
        s, rows = summary[("old", m)]
        ref = {key(r): r for r in load(I01 / f"rows_old_{m}.jsonl")}
        lf = {key(r) for r in ref.values() if r["line_failure"]}
        lf_mine = {key(r) for r in rows if r["line_failure"]}
        assert lf == lf_mine
        fix = {k for k in lf if ref[k]["P"]}
        fix_all = {key(r) for r in ref.values() if r["P"] and not r["C_rec"]}
        byk = {key(r): r for r in rows}
        line = {a: sum(byk[k][a] for k in lf) for a in ("D", "Dpix", "Dkeep", "P", "R3", "R4")}
        fixr = {a: sum(byk[k][a] for k in fix) for a in ("D", "Dpix", "Dkeep", "P", "R4")}
        fixall_r = {a: sum(byk[k][a] for k in fix_all) for a in ("D", "P")}
        p(f"{NAME[m]}: line failures {len(lf)}; flipped {line}; fixable line {len(fix)} recovered {fixr}; "
          f"fixable all {len(fix_all)} recovered {fixall_r}")
        # line classes left by D
        left = collections.Counter(byk[k]["D_d"]["line_cls"] for k in lf if not byk[k]["D"])
        leftP = sum(1 for k in lf if not byk[k]["D"] and ref[k]["P"])
        p(f"   line failures D leaves: {sum(left.values())} {dict(left)}; of them P-fixable {leftP}")
        # overlap
        fD, bD = flips(rows, "D")
        fP, bP = flips(rows, "P")
        fR3, _ = flips(rows, "R3")
        fR4, _ = flips(rows, "R4")
        p(f"   D flips {len(fD)} / P flips {len(fP)}: shared {len(fD & fP)}, D-only {len(fD - fP)}, "
          f"P-only {len(fP - fD)}; R4 flips {len(fR4)} shared with D {len(fD & fR4)}; R3 flips {len(fR3)} shared "
          f"{len(fD & fR3)}; D breaks {sorted(bD)}; P breaks {len(bP)}, shared with D breaks {len(bD & bP)}")
        # clusters
        fa = {answer(byk[k]) for k in fD}
        fl = collections.Counter(layout(k[1]) for k in fD)
        bl = collections.Counter(layout(k[1]) for k in bD)
        net = collections.Counter(fl)
        net.subtract(bl)
        fPl = collections.Counter(layout(k[1]) for k in fP)
        fPl.subtract(collections.Counter(layout(k[1]) for k in bP))
        p(f"   D flips from {len(fa)} answers in {len(fl)} layouts; net by layout {dict(net)}; "
          f"breaks from {len({answer(byk[k]) for k in bD})} answers; P net by layout {dict(fPl)}")
        # D vs P disagreements
        dp = [k for k in byk if byk[k]["D"] != byk[k]["P"]]
        dp_pos = [k for k in dp if not byk[k]["D_d"]["dropped"] and not byk[k]["D_d"]["added"]]
        p(f"   D != P in {len(dp)} records; with D dropping and adding no one: {len(dp_pos)}; D wins "
          f"{sum(byk[k]['D'] for k in dp)}")
        # mechanism (post hoc): flips with no drop and no add
        pos_only_f = [k for k in fD if not byk[k]["D_d"]["dropped"] and not byk[k]["D_d"]["added"]]
        pos_only_b = [k for k in bD if not byk[k]["D_d"]["dropped"] and not byk[k]["D_d"]["added"]]
        named_drop_f = [k for k in fD if any(f["named"] for f in byk[k]["D_d"]["fate"])]
        p(f"   POST HOC: D flips with no drop and no add {len(pos_only_f)} (max reloc "
          f"{[round(byk[k]['D_d']['moved_max'], 3) for k in pos_only_f]}); breaks of that kind {len(pos_only_b)}; "
          f"flips with a named drop {len(named_drop_f)}; D without pos-only flips = {s['D'] - len(pos_only_f)}")
        # fates
        for a in ("D", "Dpix"):
            fate_all = [(answer(byk[k]), f) for k in byk if byk[k]["solver_step"] for f in byk[k][a + "_d"]["fate"]]
            c_role = collections.Counter(f["role"] for _, f in fate_all)
            distinct = {(ans, f["n"]): f for ans, f in fate_all}
            d_role = collections.Counter(f["role"] for f in distinct.values())
            named = sum(f["named"] for _, f in fate_all)
            ca = sum(f["ca"] for _, f in fate_all)
            ca_d = sum(f["ca"] for f in distinct.values())
            idn = collections.Counter()
            for k in byk:
                if byk[k]["solver_step"]:
                    idn.update(byk[k][a + "_d"]["ident"])
            p(f"   {a} dropped {len(fate_all)} [{len(distinct)}] roles {dict(c_role)} [{dict(d_role)}]; named "
              f"{named}; in come_after {ca} [{ca_d}]; identity {dict(idn)}")
        added = collections.Counter()
        for k in byk:
            if byk[k]["solver_step"]:
                added.update(byk[k]["D_d"]["added_cls"])
        p(f"   D added: total {sum(added.values())} {dict(added)}")

# ---------------------------------------------------------------- new and ph: verdict breakdown
for wording in ("new", "ph"):
    for m in MODELS[wording]:
        if (wording, m) not in summary:
            continue
        s, rows = summary[(wording, m)]
        fD, bD = flips(rows, "D")
        p(f"{wording} {NAME[m]} D breaks layouts {collections.Counter(layout(k[1]) for k in bD)}; flips layouts "
          f"{collections.Counter(layout(k[1]) for k in fD)}")
        byk = {key(r): r for r in rows}
        p(f"   breaks with no drop and no add: {sum(1 for k in bD if not byk[k]['D_d']['dropped'] and not byk[k]['D_d']['added'])} of {len(bD)}")
        if wording == "ph":
            fate_all = [(answer(byk[k]), f) for k in byk if byk[k]["solver_step"] for f in byk[k]["D_d"]["fate"]]
            p(f"   ph {NAME[m]} D dropped {len(fate_all)} roles {dict(collections.Counter(f['role'] for _, f in fate_all))}"
              f" named {sum(f['named'] for _, f in fate_all)}")

(HERE / "vtables.txt").write_text("\n".join(OUT) + "\n")
