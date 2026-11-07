"""Recorded successes per model x situation for R (cot), E (fsc), C (con), realistic arms, new and old wording."""
import sys, collections
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
ARMS = ("real", "staffed", "plain")
SITS = ("no_one", "waiting_line", "conversation_group")
out = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)
for label, sel in (("NEW", [("s21", "qwen38_low", {"cot": "s21", "fsc": "s21", "con": "s21"})] +
                   [("s23", m, {"cot": "s23", "fsc": "s23", "con": "s23"}) for m in ("glm53", "dsv41", "haiku55", "sonnet55_low", "gemini38_low")]),
                  ("OLD", [("s19", "qwen38_low", {"cot": "s16b", "fsc": "s16b", "con": "s19"}),
                           ("s19", "glm53", {"cot": "s16b", "fsc": "s16b", "con": "s19"}),
                           ("s20", "dsv41", {"cot": "s20", "fsc": "s20", "con": "s20"})])):
    p("==", label)
    for _, model, stages in sel:
        line = [model]
        for meth in ("cot", "fsc", "con"):
            cnt = collections.Counter(); tot = collections.Counter(); trials = collections.Counter()
            for arm in ARMS:
                rs = H.runs(stage=stages[meth], arm=arm, model=model, method=meth)
                assert len(rs) == 1, (stages[meth], arm, model, meth, rs)
                for rec in H.records(rs[0]):
                    tot[rec.situation] += 1; cnt[rec.situation] += bool(rec.score["success"]); trials[rec.trial] += 1
            line.append(f"{meth}: " + " ".join(f"{s}={cnt[s]}/{tot[s]}" for s in SITS) + f" pooled={sum(cnt.values())}/{sum(tot.values())} trials={dict(trials)} other_sits={set(tot)-set(SITS)}")
        p(" | ".join(line))
open('results/2026-10-17-beyond-vlm/rerun/i01_baselines/verify/v2_recorded.txt', "w").write("\n".join(out) + "\n")
