"""Independent verifier for the latency cluster (I21 replay, L1-L8, N1-N3). Reads runs only through research_harness.
Writes nothing but its stdout (tee'd to verify_out.txt by the caller)."""
import sys, collections, statistics as st
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H

REAL = ("real", "staffed", "plain")
mean = lambda v: st.mean(v) if v else float('nan')
med = lambda v: st.median(v) if v else float('nan')


def lat(rec, field="latency_s"):
    d = collections.defaultdict(float)
    for c in rec.result.get("calls") or []:
        d[c["stage"]] += c.get(field) or 0.0
    return d


def run1(stage, arm, model, method):
    rs = H.runs(stage=stage, arm=arm, model=model, method=method)
    assert len(rs) == 1, (stage, arm, model, method, rs)
    return rs[0]


def cp_rows(ref, field="latency_s"):
    """cp.py port: one row per record with people and constraints calls, in file order; seen reset per run."""
    rows, seen = [], set()
    for rec in H.records(ref):
        L = lat(rec, field)
        if "constraints" not in L or "people" not in L:
            continue
        pre = L["plan"] + L["locate"] + L["context"]
        s = pre + L["people"] + L["constraints"]
        c = max(pre, L["people"]) + L["constraints"]
        key = (rec.scene_id, rec.trial)
        first = key not in seen
        seen.add(key)
        rows.append(dict(s=s, c=c, ss=s if first else pre + L["constraints"], cs=c if first else pre + L["constraints"],
                         pre=pre, pe=L["people"], k=L["constraints"], first=first, rec=rec))
    return rows


def cp_line(label, rows):
    if not rows:
        return f"{label:44s} none"
    S, C, SS, CS = (mean([r[k] for r in rows]) for k in ("s", "c", "ss", "cs"))
    return (f"{label:44s} n={len(rows):3d} serial {S:5.1f} cp {C:5.1f} ({100*(1-C/S):4.1f}%) | shared serial {SS:5.1f} "
            f"cp {CS:5.1f} ({100*(1-CS/SS):4.1f}%) firsts={sum(r['first'] for r in rows)} "
            f"pre {mean([r['pre'] for r in rows]):.1f} pe {mean([r['pe'] for r in rows]):.1f} k {mean([r['k'] for r in rows]):.1f} "
            f"med serial {med([r['s'] for r in rows]):.1f} cp {med([r['c'] for r in rows]):.1f} pe>pre {sum(r['pe']>r['pre'] for r in rows)}")


print("=== L1/L2/L3/N2/N3: cp.py port (call latency_s) ===")
for stage, models in (("s19", ("qwen38_low", "glm53")), ("s21", ("qwen38_low",)),
                      ("s20", ("dsv41",)),
                      ("s23", ("glm53", "dsv41", "haiku55", "sonnet55_low", "gemini38_low"))):
    for model in models:
        pooled = []
        for arm in REAL:
            rows = cp_rows(run1(stage, arm, model, "con"))
            pooled += rows
            print(cp_line(f"{stage}_{arm}_{model}_con", rows))
        print(cp_line(f"{stage}_POOLED3_{model}_con", pooled))
# latency_total_s variant on s19 Qwen
for arm in REAL:
    print(cp_line(f"[latency_total_s] s19_{arm}_qwen38_low_con", cp_rows(run1("s19", arm, "qwen38_low", "con"), "latency_total_s")))

print("\n=== L4: cp2.py port (A1 - A2) ===")


def cp2_rows(ref):
    out = []
    for rec in H.records(ref):
        L = lat(rec)
        if "constraints" not in L:
            continue
        p, x, pe, k = L["plan"], L["context"], L["people"], L["constraints"]
        out.append(dict(serial=p + x + pe + k, A1=max(p + x, pe) + k, A2=max(p, pe) + max(x, k), k=k, nopeople="people" not in L))
    return out


for stage, models in (("s19", ("qwen38_low", "glm53")), ("s20", ("dsv41",)), ("s21", ("qwen38_low",)),
                      ("s23", ("glm53", "dsv41", "haiku55", "sonnet55_low", "gemini38_low"))):
    for model in models:
        pooled = []
        for arm in REAL:
            rows = cp2_rows(run1(stage, arm, model, "con"))
            pooled += rows
            a1, a2 = mean([r["A1"] for r in rows]), mean([r["A2"] for r in rows])
            print(f"{stage}_{arm}_{model}_con n={len(rows)} nopeople={sum(r['nopeople'] for r in rows)} serial {mean([r['serial'] for r in rows]):.2f} "
                  f"A1 {a1:.2f} A2 {a2:.2f} A1-A2 {a1-a2:.2f}")
        a1, a2 = mean([r["A1"] for r in pooled]), mean([r["A2"] for r in pooled])
        print(f"{stage}_POOLED3_{model}_con n={len(pooled)} A1 {a1:.2f} A2 {a2:.2f} A1-A2 {a1-a2:.2f}")
# s19 other arms too (ph, dev seeds), to see the max over all s19 runs
for ref in H.runs(stage="s19"):
    if ref.arm in REAL:
        continue
    rows = cp2_rows(ref)
    if rows:
        a1, a2 = mean([r["A1"] for r in rows]), mean([r["A2"] for r in rows])
        print(f"{ref.name} n={len(rows)} A1 {a1:.2f} A2 {a2:.2f} A1-A2 {a1-a2:.2f}")

print("\n=== L5: E = s16b qwen38_low fsc, serial plan+locate+context+goal; C concurrent / E ===")
for stage_e, stage_c in (("s16b", "s19"), ("s21", "s21")):
    Eall, Call, Csall, Eg = [], [], [], []
    Ekeys, Ckeys = {}, {}
    for arm in REAL:
        e = []
        for rec in H.records(run1(stage_e, arm, "qwen38_low", "fsc")):
            L = lat(rec)
            if "goal" not in L:
                continue
            v = L["plan"] + L["locate"] + L["context"] + L["goal"]
            e.append(v); Eg.append(L["goal"])
            Ekeys[(arm, rec.scene_id, rec.episode_id, rec.trial)] = v
        rows = cp_rows(run1(stage_c, arm, "qwen38_low", "con"))
        for r in rows:
            Ckeys[(arm, r["rec"].scene_id, r["rec"].episode_id, r["rec"].trial)] = r["c"]
        c = [r["c"] for r in rows]
        print(f"  {stage_e}/{stage_c} {arm}: E n={len(e)} mean {mean(e):.1f}; C cp n={len(c)} mean {mean(c):.1f}; ratio {mean(c)/mean(e):.2f}; "
              f"serial C/E {mean([r['s'] for r in rows])/mean(e):.2f}")
        Eall += e; Call += c; Csall += [r["s"] for r in rows]
    common = sorted(set(Ekeys) & set(Ckeys))
    print(f"  {stage_e}/{stage_c} POOLED: E n={len(Eall)} mean {mean(Eall):.1f} (goal call mean {mean(Eg):.1f} med {med(Eg):.1f}); "
          f"C cp n={len(Call)} mean {mean(Call):.1f}; ratio {mean(Call)/mean(Eall):.2f}; serial C {mean(Csall):.1f} C/E {mean(Csall)/mean(Eall):.2f}; "
          f"matched n={len(common)} ratio {mean([Ckeys[k] for k in common])/mean([Ekeys[k] for k in common]):.2f}")
# C saving averaged over all C records (standard-goal included)
allS, allC = [], []
for arm in REAL:
    for rec in H.records(run1("s19", arm, "qwen38_low", "con")):
        L = lat(rec)
        pre = L["plan"] + L["locate"] + L["context"]
        allS.append(pre + L["people"] + L["constraints"])
        allC.append((max(pre, L["people"]) if "people" in L else pre) + L["constraints"])
print(f"  s19 Qwen all C records n={len(allS)} serial {mean(allS):.1f} cp {mean(allC):.1f} ({100*(1-mean(allC)/mean(allS)):.1f}%)")

print("\n=== L6: measures.latency_s ===")
for stage, model in (("s19", "qwen38_low"), ("s19", "glm53"), ("s21", "qwen38_low")):
    g_rec, g_prov, g_tot, a_rec, n_eq, n_all, g_src = [], [], [], [], 0, 0, 0
    resid_all, resid_fresh = [], []
    for arm in REAL:
        sub_rec, sub_prov = [], []
        for rec in H.records(run1(stage, arm, model, "con")):
            n_all += 1
            m = rec.latency_s
            ts = sum(rec.timings_s.values())
            n_eq += abs(m - ts) < 1e-6
            a_rec.append(m)
            L = lat(rec); T = lat(rec, "latency_total_s")
            calls = rec.result.get("calls") or []
            resid_all.append(m - sum(c.get("latency_total_s") or 0 for c in calls))
            resid_fresh.append(m - sum(c.get("latency_total_s") or 0 for c in calls if not c.get("cached")))
            if "constraints" in L:
                g_src += rec.result.get("source") == "constraints"
                g_rec.append(m); g_prov.append(sum(L.values())); g_tot.append(sum(T.values()))
                sub_rec.append(m); sub_prov.append(sum(L.values()))
        print(f"  {stage}_{arm}_{model}_con goal-step n={len(sub_rec)} record {mean(sub_rec):.1f} ({med(sub_rec):.1f}) provider {mean(sub_prov):.1f}")
    print(f"  {stage} {model} POOLED goal-step n={len(g_rec)} (source=constraints {g_src}) record latency_s mean {mean(g_rec):.1f} median {med(g_rec):.1f}; "
          f"sum call latency_s {mean(g_prov):.1f} ({med(g_prov):.1f}); sum latency_total_s {mean(g_tot):.1f}; all records n={len(a_rec)} {mean(a_rec):.1f} ({med(a_rec):.1f}); "
          f"latency_s==sum(timings_s) {n_eq}/{n_all}; resid vs all calls med {med(resid_all):.2f}, vs fresh-only med {med(resid_fresh):.2f}")
# equality check over all listed runs
n_eq = n_all = 0
for ref in H.runs():
    for rec in H.records(ref):
        if rec.latency_s is None:
            continue
        n_all += 1
        n_eq += abs(rec.latency_s - sum(rec.timings_s.values())) < 1e-6
print(f"  ALL listed runs: measures.latency_s == sum(timings_s) in {n_eq}/{n_all}")

print("\n=== L7: cached calls carry the original call's latency ===")
cnt = collections.Counter()
for ref in H.runs(stage="s19"):
    for rec in H.records(ref):
        for c in rec.result.get("calls") or []:
            cnt[(c["stage"], bool(c.get("cached")), (c.get("latency_s") or 0) > 0)] += 1
print("  s19 (stage, cached, latency>0):", dict(sorted(cnt.items())))


def index(refs, stages, fresh_only):
    idx = {}
    for ref in refs:
        for rec in H.records(ref):
            for c in rec.result.get("calls") or []:
                if c["stage"] not in stages or (fresh_only and c.get("cached")):
                    continue
                key = (ref.arm, rec.scene_id, rec.trial, c["stage"], c.get("raw_text"))
                idx.setdefault(key, []).append((c.get("latency_s"), c.get("latency_total_s"), str(c.get("usage"))))
    return idx


def compare(label, src_refs, dst_refs, stages, src_fresh=True):
    idx = index(src_refs, stages, src_fresh)
    tally = collections.Counter()
    for ref in dst_refs:
        for rec in H.records(ref):
            for c in rec.result.get("calls") or []:
                if c["stage"] not in stages or not c.get("cached"):
                    continue
                key = (ref.arm, rec.scene_id, rec.trial, c["stage"], c.get("raw_text"))
                got = (c.get("latency_s"), c.get("latency_total_s"), str(c.get("usage")))
                if key not in idx:
                    tally[(c["stage"], "no source")] += 1
                elif got in idx[key]:
                    tally[(c["stage"], "identical")] += 1
                else:
                    tally[(c["stage"], "DIFFERENT")] += 1
    print(f"  {label}: {dict(sorted(tally.items()))}")


s21 = lambda m: [r for r in H.runs(stage="s21") if r.method == m]
compare("s21 fsc+con plan/context vs s21 cot fresh", s21("cot"), s21("fsc") + s21("con"), {"plan", "context"})
compare("s21 con people vs s19 Qwen realistic people", [r for r in H.runs(stage="s19", model="qwen38_low") if r.arm in REAL],
        s21("con"), {"people"}, src_fresh=False)
compare("s19 Qwen plan/context vs s16b Qwen cot (both cached)", [r for r in H.runs(stage="s16b", model="qwen38_low", method="cot") if r.arm in REAL],
        [r for r in H.runs(stage="s19", model="qwen38_low") if r.arm in REAL], {"plan", "context"}, src_fresh=False)
for model in ("glm53", "dsv41", "haiku55", "sonnet55_low", "gemini38_low"):
    s23m = lambda m: [r for r in H.runs(stage="s23", model=model) if r.method == m]
    compare(f"s23 {model} fsc+con plan/context vs s23 cot fresh", s23m("cot"), s23m("fsc") + s23m("con"), {"plan", "context"})
    compare(f"s23 {model} con people (cached) vs s23 con people fresh", s23m("con"), s23m("con"), {"people"})
compare("s23 glm53 con people vs s19 glm53 realistic", [r for r in H.runs(stage="s19", model="glm53") if r.arm in REAL],
        [r for r in H.runs(stage="s23", model="glm53", method="con")], {"people"}, src_fresh=False)
compare("s23 dsv41 con people vs s20 dsv41 realistic con", [r for r in H.runs(stage="s20", model="dsv41", method="con") if r.arm in REAL],
        [r for r in H.runs(stage="s23", model="dsv41", method="con")], {"people"}, src_fresh=False)
# s21 con: goal timing vs people+constraints totals
d = []
for ref in s21("con"):
    for rec in H.records(ref):
        T = lat(rec, "latency_total_s")
        if "constraints" in T:
            d.append(rec.timings_s.get("goal", 0) - T["people"] - T["constraints"])
print(f"  s21 con: timings_s.goal - (people+constraints latency_total_s) median {med(d):.2f} n={len(d)}")

print("\n=== L8/N1: fresh-call medians per stage (cached false) ===")


def fresh_medians(stage, model):
    by = collections.defaultdict(list)
    for ref in H.runs(stage=stage, model=model):
        if ref.arm not in REAL:
            continue
        for rec in H.records(ref):
            for c in rec.result.get("calls") or []:
                if c.get("cached"):
                    continue
                s = c["stage"]
                if s == "goal":
                    s = "goal_R" if ref.method == "cot" else "goal_E"
                by[(s, ref.method if s in ("plan", "context") else "")].append(c.get("latency_s") or 0.0)
    agg = collections.defaultdict(list)
    for (s, m), v in by.items():
        agg[s] += v
    parts = [f"{s} {med(agg[s]):.1f} (n {len(agg[s])}, mean {mean(agg[s]):.1f})" for s in
             ("plan", "context", "goal_R", "goal_E", "people", "constraints") if s in agg]
    where = {f"{s}@{m}": len(v) for (s, m), v in by.items() if s in ("plan", "context")}
    print(f"  {stage} {model}: " + "; ".join(parts) + f"   fresh plan/context by method: {where}")


fresh_medians("s21", "qwen38_low")
for model in ("glm53", "dsv41", "haiku55", "sonnet55_low", "gemini38_low"):
    fresh_medians("s23", model)
# for comparison, GLM at s19: carried latencies (all cached)
by = collections.defaultdict(list)
for ref in H.runs(stage="s19", model="glm53"):
    if ref.arm in REAL:
        for rec in H.records(ref):
            for c in rec.result.get("calls") or []:
                by[c["stage"]].append(c.get("latency_s") or 0)
print("  s19 glm53 (carried, all cached) medians:", {k: round(med(v), 1) for k, v in by.items()})
