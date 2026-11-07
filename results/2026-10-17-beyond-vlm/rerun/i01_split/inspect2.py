"""Follow-ups 2: (5) wrong targets by scene (the target is shared by R/E/C, so cot records are used); (6) records in
region with the right target but unreachable: min_body_gap_m. New wording. Output saved to inspect2_out.txt."""
import collections, sys, statistics as st
from pathlib import Path
sys.path.insert(0, 'docs/model_notes/tools')
import research_harness as H
OUT = Path(__file__).resolve().parent / "inspect2_out.txt"
out = []
def say(t=""):
    print(t); out.append(t)
ARMS = ("real", "staffed", "plain")
NEW = [("qwen38_low", "Qwen", "s21"), ("glm53", "GLM", "s23"), ("dsv41", "DeepSeek", "s23"),
       ("haiku55", "Haiku", "s23"), ("sonnet55_low", "Sonnet", "s23"), ("gemini38_low", "Gemini", "s23")]
def load(stage, model, method):
    return [r for arm in ARMS for ref in H.runs(stage=stage, arm=arm, model=model, method=method) for r in H.records(ref)]
say("(5) wrong targets (score.target_correct False) by scene, of 3 trials x episodes; same for R, E, C")
for model, label, stage in NEW:
    recs = load(stage, model, "cot")
    n = collections.Counter(r.scene_id for r in recs)
    w = collections.Counter(r.scene_id for r in recs if not r.score.get("target_correct"))
    say(f"  {label:9s} total {sum(w.values())}: " + ", ".join(f"{s} {w[s]}/{n[s]}" for s in sorted(w)))
say("")
say("(6) right target, in region, unreachable: min_body_gap_m (goal to nearest person, minus both radii)")
for model, label, stage in NEW:
    for meth in ("cot", "fsc", "con"):
        v = [r for r in load(stage, model, meth) if r.score.get("target_correct") and r.score.get("in_region")
             and not r.score.get("reachable")]
        if not v:
            continue
        gaps = sorted(round(r.score.get("min_body_gap_m"), 3) for r in v if r.score.get("min_body_gap_m") is not None)
        say(f"  {label:9s} {meth:3s} n={len(v)} situations {dict(collections.Counter(r.situation for r in v))} gaps {gaps}")
OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
