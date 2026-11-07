"""Tables from v1 rows. Rows keep recorded verdict where SA/CA/SG could not be computed."""
import json, collections, sys
D = 'results/2026-10-17-beyond-vlm/rerun/i01_baselines/verify/'
SITS = ("no_one", "waiting_line", "conversation_group")
out = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)
for which in ("new", "old"):
    rows = [json.loads(l) for l in open(D + f"v1_rows_{which}.jsonl")]
    models = list(dict.fromkeys(r["model"] for r in rows))
    p(f"===== {which} wording: {len(rows)} records")
    def v(r, key):
        return r["C"] if r.get(key) is None else r[key]
    for key in ("C", "C_now", "SA", "SG", "CA"):
        p(f"-- {key} (all records; missing -> recorded C)")
        for m in models:
            rs = [r for r in rows if r["model"] == m]
            cells = []
            for s in SITS + ("pooled",):
                sub = rs if s == "pooled" else [r for r in rs if r["sit"] == s]
                cells.append(f"{s}={sum(bool(v(r, key)) for r in sub)}/{len(sub)}")
            p(f"  {m:14s} " + " ".join(cells))
    p("-- goal-step only (target and people recorded): C, SA, CA, n")
    for m in models:
        cells = []
        for s in SITS + ("pooled",):
            sub = [r for r in rows if r["model"] == m and (s == "pooled" or r["sit"] == s) and r["SA"] is not None]
            cells.append(f"{s}: C={sum(r['C'] for r in sub)} SA={sum(r['SA'] for r in sub)} CA={sum(r['CA'] for r in sub)} n={len(sub)}")
        p(f"  {m:14s} " + " | ".join(cells))
    p("-- lines: records stating come_after: n, C, CA")
    for m in models:
        sub = [r for r in rows if r["model"] == m and r["sit"] == "waiting_line" and r.get("n_ca")]
        p(f"  {m:14s} n={len(sub)} C={sum(r['C'] for r in sub)} CA={sum(r['CA'] for r in sub)}  lines goal-step={sum(1 for r in rows if r['model']==m and r['sit']=='waiting_line' and r['SA'] is not None)}")
    p("-- kept recorded (no goal step) per sit; no target; status/source breakdown")
    for m in models:
        rs = [r for r in rows if r["model"] == m]
        kept = collections.Counter(r["sit"] for r in rs if r["SA"] is None)
        notgt = collections.Counter(r["sit"] for r in rs if not r["has_target"])
        kept_succ = collections.Counter(r["sit"] for r in rs if r["SA"] is None and r["C"])
        ss = collections.Counter((r["status"], r["source"], r["people"] is None) for r in rs if r["SA"] is None)
        emp = collections.Counter(r["sit"] for r in rs if r["people"] == 0)
        mism = sum(1 for r in rs if (r["SA"] is not None) != r["solver_step"])
        p(f"  {m:14s} kept={sum(kept.values())} {dict(kept)} kept_succ={dict(kept_succ)} no_target={dict(notgt)} kinds={dict(ss)} empty_people={dict(emp)} goalstep!=solver_step:{mism}")
    p("-- C_now == C:", sum(r["C_now"] == r["C"] for r in rows), "/", len(rows))
    cr = [r for r in rows if "C_resolve" in r]
    p("-- full re-solve reproduces C success:", sum(r["C_resolve"] == r["C"] for r in cr), "/", len(cr),
      "; moved<=1e-6:", sum(1 for r in cr if r["C_moved"] is not None and r["C_moved"] <= 1e-6))
    sg = [r for r in rows if "SG_moved" in r]
    p("-- SG vs recorded standard pose: n", len(sg), "max moved", max((r["SG_moved"] for r in sg), default=None))
    p("-- SG at groups by scene layout (eval_10/11 vs eval_12)")
    for m in models:
        g = [r for r in rows if r["model"] == m and r["sit"] == "conversation_group"]
        def lay(r):
            import re
            return re.search(r"eval_\d+", r["scene"]).group(0)
        c = collections.Counter(); n = collections.Counter(); sa = collections.Counter(); cc = collections.Counter()
        for r in g:
            k = "eval_12" if lay(r) == "eval_12" else ("eval_10/11" if lay(r) in ("eval_10", "eval_11") else lay(r))
            n[k] += 1; c[k] += bool(v(r, "SG")); sa[k] += bool(v(r, "SA")); cc[k] += r["C"]
        p(f"  {m:14s} " + " ".join(f"{k}: SG={c[k]} SA={sa[k]} C={cc[k]} /{n[k]}" for k in sorted(n)))
    p("-- DeepSeek/any no_one goal-step records")
    for m in models:
        sub = [r for r in rows if r["model"] == m and r["sit"] == "no_one" and r["SA"] is not None]
        p(f"  {m:14s} n={len(sub)} C={sum(r['C'] for r in sub)} SA={sum(r['SA'] for r in sub)} SG={sum(bool(r['SG']) for r in sub)}")
open(D + "v3_tables.txt", "w").write("\n".join(out) + "\n")
