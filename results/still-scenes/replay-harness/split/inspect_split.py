"""Follow-ups to split.py: (1) is the target the same answer across R/E/C for one record key? (2) what are the
nobody-near (no_one) failures with the right target on C (Qwen, Gemini, GLM, DeepSeek)? (3) result.source per
situation and method. Reads only through research_harness; output saved to inspect_out.txt."""
import collections, sys, statistics as st
from pathlib import Path
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H

OUT = Path(__file__).resolve().parent / "inspect_out.txt"
out = []
def say(t=""):
    print(t); out.append(t)

ARMS = ("real", "staffed", "plain")
NEW = [("qwen38_low", "Qwen", "s21"), ("glm53", "GLM", "s23"), ("dsv41", "DeepSeek", "s23"),
       ("haiku55", "Haiku", "s23"), ("sonnet55_low", "Sonnet", "s23"), ("gemini38_low", "Gemini", "s23")]
OLD = [("qwen38_low", "Qwen", {"cot": "s16b", "fsc": "s16b", "con": "s19"}),
       ("glm53", "GLM", {"cot": "s16b", "fsc": "s16b", "con": "s19"}),
       ("dsv41", "DeepSeek", {"cot": "s20", "fsc": "s20", "con": "s20"})]

def load(stage, model, method):
    recs = {}
    for arm in ARMS:
        (ref,) = H.runs(stage=stage, arm=arm, model=model, method=method)
        for r in H.records(ref):
            recs[(arm, r.scene_id, r.episode_id, r.trial)] = r
    return recs

def tw(r):
    t = r.target
    return None if not t else tuple(round(c, 4) for c in (t.get("world") or ()))

say("(1) Target answer identical across methods for one key (arm, scene, episode, trial)")
for label_set, rows in (("new", [(m, l, {k: s for k in ("cot", "fsc", "con")}) for m, l, s in NEW]), ("old", OLD)):
    for model, label, stages in rows:
        d = {meth: load(stages[meth], model, meth) for meth in ("cot", "fsc", "con")}
        keys = set(d["cot"]) & set(d["fsc"]) & set(d["con"])
        same_t = sum(tw(d["cot"][k]) == tw(d["fsc"][k]) == tw(d["con"][k]) for k in keys)
        same_tc = sum(bool(d["cot"][k].score.get("target_correct")) == bool(d["fsc"][k].score.get("target_correct"))
                      == bool(d["con"][k].score.get("target_correct")) for k in keys)
        # is the locate call cached in E and C?
        cached = collections.Counter()
        for meth in ("cot", "fsc", "con"):
            for r in d[meth].values():
                for stage, t in r.timings.items():
                    cached[(meth, stage, "calls")] += t["calls"]
                    cached[(meth, stage, "cached")] += t["cached"]
        stg = sorted({s for (_, s, _) in cached})
        cs = "; ".join(f"{meth}:" + ",".join(f"{s} {cached[(meth, s, 'cached')]}/{cached[(meth, s, 'calls')]}"
                                              for s in stg if cached[(meth, s, 'calls')]) for meth in ("cot", "fsc", "con"))
        say(f"  {label_set} {label:9s} keys {len(keys)}  same target world in R,E,C: {same_t}  same target_correct: {same_tc}")
        say(f"      cached/calls per stage: {cs}")

say("")
say("(2) no_one records failing with the right target, C, new wording (all three methods shown side by side)")
for model, label, stage in NEW:
    d = {meth: load(stage, model, meth) for meth in ("cot", "fsc", "con")}
    rows = [(k, r) for k, r in d["con"].items() if r.situation == "no_one" and not r.score.get("success")]
    say(f"  {label}: {len(rows)} C failures at no_one")
    byscene = collections.Counter()
    for k, r in sorted(rows):
        s = r.score
        byscene[(k[1], k[2])] += 1
        alsoR = not d["cot"][k].score.get("success")
        alsoE = not d["fsc"][k].score.get("success")
        say(f"    {k[0]:7s} {k[1]:22s} {k[2]:10s} t{k[3]} src={r.result.get('source'):11s} tc={int(bool(s.get('target_correct')))} "
            f"inreg={int(bool(s.get('in_region')))} reach={int(bool(s.get('reachable')))} d_reg={s.get('distance_to_region_m')} "
            f"matched={s.get('matched_target_id')} target_world={tw(r)} R_fails={int(alsoR)} E_fails={int(alsoE)}")
    if rows:
        say(f"    by (scene, episode): {dict(byscene)}")

say("")
say("(3) result.source by situation and method, new wording (count of records)")
for model, label, stage in NEW:
    for meth in ("cot", "fsc", "con"):
        c = collections.Counter((r.situation, r.result.get("source")) for r in load(stage, model, meth).values())
        say(f"  {label:9s} {meth:3s} " + ", ".join(f"{s}/{src}:{n}" for (s, src), n in sorted(c.items(), key=str)))

say("")
say("(4) Right-target R failures at waiting_line: distance to region (m), median and quartiles, new wording")
for model, label, stage in NEW:
    v = [r.score.get("distance_to_region_m") for r in load(stage, model, "cot").values()
         if r.situation == "waiting_line" and r.score.get("target_correct") and not r.score.get("success")]
    v = [x for x in v if x is not None]
    if v:
        q = st.quantiles(v, n=4) if len(v) > 1 else [v[0]] * 3
        say(f"  {label:9s} n={len(v)} median {st.median(v):.2f}  q1 {q[0]:.2f}  q3 {q[2]:.2f}  <=0.15 m: {sum(x <= 0.15 for x in v)}")
OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
