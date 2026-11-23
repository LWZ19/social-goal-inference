"""Tables of the solver-alone and people-blind rows from records.jsonl and re_records.jsonl (written by
baselines.py). No scene or run is read here. Prints and writes tables.txt beside this file."""
from __future__ import annotations

import collections
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
SITS = ("no_one", "waiting_line", "conversation_group", "audience")
SHORT = {"no_one": "no_one", "waiting_line": "line", "conversation_group": "group", "audience": "audience"}
MODELS = ("qwen38_low", "glm53", "dsv41", "haiku55", "sonnet55_low", "gemini38_low")


def load(name):
    return [json.loads(line) for line in open(OUT / name)]


def main():
    rows, re_rows = load("records.jsonl"), load("re_records.jsonl")
    lines = []
    say = lambda s="": lines.append(s)
    for wording, label in (("new", "NEW wording (s21 Qwen; s23 others)"), ("old", "OLD wording (s19 Qwen/GLM; s20 DeepSeek; R/E from s16b and s20)")):
        W = [r for r in rows if r["wording"] == wording]
        RE = [r for r in re_rows if r["wording"] == wording]
        models = [m for m in MODELS if any(r["model"] == m for r in W)]
        sits = [s for s in SITS if any(r["situation"] == s for r in W)]
        say(f"===== {label}: realistic arms real+staffed+plain, all trials =====")
        say("Columns: n records; R, E (recorded, same stage); C rec (recorded); C now (recorded pose scored now);")
        say("SA = solver alone (recorded target+people, constraints=[]); SG = people-blind standard goal;")
        say("CA = solver with only come_after kept (no come_after -> SA). Non-goal-step records keep recorded verdict.")
        say(f"{'model':13s} {'situation':9s} {'n':>4s} {'R':>4s} {'E':>4s} {'Crec':>5s} {'Cnow':>5s} {'SA':>4s} {'SG':>4s} {'CA':>4s} {'kept_SA':>8s} {'kept_SG':>8s}")
        for m in models:
            for s in list(sits) + ["pooled"]:
                sub = [r for r in W if r["model"] == m and (s == "pooled" or r["situation"] == s)]
                re_sub = [r for r in RE if r["model"] == m and (s == "pooled" or r["situation"] == s)]
                R = sum(r["ok"] for r in re_sub if r["method"] == "R")
                nR = sum(1 for r in re_sub if r["method"] == "R")
                E = sum(r["ok"] for r in re_sub if r["method"] == "E")
                nE = sum(1 for r in re_sub if r["method"] == "E")
                n = len(sub)
                assert nR == n and nE == n, (m, s, n, nR, nE)
                say(f"{m:13s} {SHORT.get(s, s):9s} {n:4d} {R:4d} {E:4d} {sum(r['C_rec'] for r in sub):5d} "
                    f"{sum(r['C_now'] for r in sub):5d} {sum(r['SA'] for r in sub):4d} {sum(r['SG'] for r in sub):4d} "
                    f"{sum(r['CA'] for r in sub):4d} {sum(not r['goal_step'] for r in sub):8d} "
                    f"{sum(not r['has_target'] for r in sub):8d}")
            say()
        # compact rows
        for s in ("conversation_group", "waiting_line", "no_one", "pooled"):
            say(f"-- {SHORT.get(s, s)} (order {', '.join(models)}) --")
            for col in ("C_rec", "SA", "SG", "CA"):
                vals = []
                for m in models:
                    sub = [r for r in W if r["model"] == m and (s == "pooled" or r["situation"] == s)]
                    vals.append(f"{sum(r[col] for r in sub)}")
                n = len([r for r in W if r["model"] == models[0] and (s == "pooled" or r["situation"] == s)])
                say(f"  {col:6s}: {' / '.join(vals)} of {n}")
            for col in ("R", "E"):
                vals = [str(sum(r["ok"] for r in RE if r["model"] == m and r["method"] == col and (s == "pooled" or r["situation"] == s))) for m in models]
                say(f"  {col:6s}: {' / '.join(vals)}")
        say()
        # kept-recorded counts
        say("-- records keeping their recorded verdict --")
        for m in models:
            sub = [r for r in W if r["model"] == m]
            c = collections.Counter((r["situation"], r["source"], r["has_target"]) for r in sub if not r["goal_step"])
            notarget = sum(not r["has_target"] for r in sub)
            empty = collections.Counter(r["situation"] for r in sub if r["goal_step"] and r["n_people"] == 0)
            say(f"  {m}: no goal step {sum(c.values())} (of which no target {notarget}); by (situation, source): "
                f"{dict(c)}; goal steps with an empty people list (solver run on them): {dict(empty)}")
        say()
        # lines: come_after detail
        say("-- lines: come_after detail --")
        say(f"  {'model':13s} {'n':>3s} {'goal':>5s} {'w/ CA':>6s} {'Crec|CA':>8s} {'CAonly|CA':>10s} {'SA|CA':>6s} "
            f"{'Crec|goal':>10s} {'CAonly|goal':>12s} {'SA|goal':>8s} {'tc|goal':>8s} {'Crec|tc,CA':>11s} {'CAonly|tc,CA':>13s}")
        for m in models:
            sub = [r for r in W if r["model"] == m and r["situation"] == "waiting_line"]
            goal = [r for r in sub if r["goal_step"]]
            withca = [r for r in goal if r["has_come_after"]]
            tcca = [r for r in withca if r["target_correct"]]
            say(f"  {m:13s} {len(sub):3d} {len(goal):5d} {len(withca):6d} {sum(r['C_rec'] for r in withca):8d} "
                f"{sum(r['CA'] for r in withca):10d} {sum(r['SA'] for r in withca):6d} {sum(r['C_rec'] for r in goal):10d} "
                f"{sum(r['CA'] for r in goal):12d} {sum(r['SA'] for r in goal):8d} {sum(r['target_correct'] for r in goal):8d} "
                f"{sum(r['C_rec'] for r in tcca):6d}/{len(tcca):<4d} {sum(r['CA'] for r in tcca):8d}/{len(tcca):<4d}")
        say()
        say("-- goal-step records only (target and people recorded); non-goal-step records listed with (source, C rec) --")
        for s in ("conversation_group", "waiting_line"):
            for m in models:
                sub = [r for r in W if r["model"] == m and r["situation"] == s and r["goal_step"]]
                rest = [(r["source"], r["C_rec"]) for r in W if r["model"] == m and r["situation"] == s and not r["goal_step"]]
                say(f"  {SHORT[s]:6s} {m:13s} n {len(sub):3d}  C {sum(r['C_rec'] for r in sub):3d}  SA {sum(r['SA'] for r in sub):3d}  "
                    f"SG {sum(r['SG'] for r in sub):3d}  CA {sum(r['CA'] for r in sub):3d}  | not goal step: {rest}")
        say()
        # groups by scene and episode, all models of this wording pooled
        say("-- groups by arm, scene, episode (all models of this wording pooled): n, SG, SA, C rec --")
        by = collections.defaultdict(lambda: [0, 0, 0, 0])
        for r in W:
            if r["situation"] == "conversation_group":
                v = by[(r["arm"], r["scene"], r["episode"])]
                v[0] += 1; v[1] += r["SG"]; v[2] += r["SA"]; v[3] += r["C_rec"]
        for k, v in sorted(by.items()):
            say(f"  {k[0]:8s} {k[1]:40s} {k[2]:3s} n {v[0]:3d} SG {v[1]:3d} SA {v[2]:3d} C {v[3]:3d}")
        for tag in ("eval_10", "eval_11", "eval_12"):
            vs = [v for k, v in by.items() if tag in k[1]]
            say(f"  {tag}: n {sum(v[0] for v in vs)} SG {sum(v[1] for v in vs)} SA {sum(v[2] for v in vs)} C {sum(v[3] for v in vs)}")
        say("  SG at groups per model, eval_10+eval_11 / eval_12:")
        for m in models:
            a = [r for r in W if r["model"] == m and r["situation"] == "conversation_group"]
            e1011 = [r for r in a if "eval_10" in r["scene"] or "eval_11" in r["scene"]]
            e12 = [r for r in a if "eval_12" in r["scene"]]
            say(f"    {m:13s} {sum(r['SG'] for r in e1011)}/{len(e1011)}  {sum(r['SG'] for r in e12)}/{len(e12)}")
        say()
        # sanity
        mism = [r for r in W if r["C_now"] != r["C_rec"]]
        sg_std = [r.get("SG_same_as_recorded_m") for r in W if r.get("SG_same_as_recorded_m") is not None]
        say(f"-- sanity: C now != C rec in {len(mism)} of {len(W)} records; standard goal recomputed for "
            f"{len(sg_std)} source=standard records, max distance from the recorded pose "
            f"{max(sg_std) if sg_std else float('nan'):.2e} m, {sum(d > 1e-6 for d in sg_std)} moved > 1e-6 m")
        for r in mism:
            say(f"   mismatch: {r['model']} {r['arm']} {r['scene']} {r['episode']} t{r['trial']}")
        say()
    text = "\n".join(lines)
    print(text)
    (OUT / "tables.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
