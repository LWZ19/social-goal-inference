"""L7 recheck with the episode id in the key for plan/context (people keyed per scene and trial)."""
import sys, collections
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
REAL = ("real", "staffed", "plain")
def key(ref, rec, c):
    ep = None if c["stage"] == "people" else rec.episode_id
    return (ref.arm, rec.scene_id, ep, rec.trial, c["stage"], c.get("raw_text"))
def index(refs, stages, fresh_only):
    idx = collections.defaultdict(list)
    for ref in refs:
        for rec in H.records(ref):
            for c in rec.result.get("calls") or []:
                if c["stage"] in stages and not (fresh_only and c.get("cached")):
                    idx[key(ref, rec, c)].append((c.get("latency_s"), c.get("latency_total_s"), str(c.get("usage"))))
    return idx
def compare(label, src, dst, stages, src_fresh=True):
    idx = index(src, stages, src_fresh); t = collections.Counter()
    for ref in dst:
        for rec in H.records(ref):
            for c in rec.result.get("calls") or []:
                if c["stage"] not in stages or not c.get("cached"): continue
                k = key(ref, rec, c); got = (c.get("latency_s"), c.get("latency_total_s"), str(c.get("usage")))
                t[(c["stage"], "no source" if k not in idx else ("identical" if got in idx[k] else "DIFFERENT"))] += 1
    print(f"{label}: {dict(sorted(t.items()))}")
compare("s21 fsc+con plan/context vs s21 cot fresh", [r for r in H.runs(stage="s21", method="cot")],
        [r for r in H.runs(stage="s21") if r.method != "cot"], {"plan", "context"})
for m in ("glm53", "dsv41", "haiku55", "sonnet55_low", "gemini38_low"):
    compare(f"s23 {m} fsc+con plan/context vs cot fresh", H.runs(stage="s23", model=m, method="cot"),
            H.runs(stage="s23", model=m, method="fsc") + H.runs(stage="s23", model=m, method="con"), {"plan", "context"})
    compare(f"s23 {m} con people cached vs same-run fresh people", H.runs(stage="s23", model=m, method="con"),
            H.runs(stage="s23", model=m, method="con"), {"people"})
