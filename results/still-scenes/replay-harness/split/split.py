"""I01 deliverable (3): target finding vs placement given the right target.

Reads runs only through research_harness. No model call, nothing tuned. Prints every table and saves it to
split_out.txt beside this file.

Definitions (all from the recorded score of each record; a rescore check is printed at the end):
  found      = score.target_correct (a not-ok result counts as not found, as the scorer has it)
  succ|found = score.success among records with score.target_correct (success implies target_correct)
  failure split, hierarchical, of every failed record:
    wrong_tgt   target_correct False      (of which `notok`: result.status != "ok")
    out_region  right target, in_region False
    unreach     right target, in region, reachable False
    heading     right target, in region, reachable, heading_ok False
  overlap: right-target failures that are both out of region and unreachable are counted under out_region and also
  reported in the column `out&unr`.
Realistic arms pooled: real, staffed, plain (84 records each per run, 252 per model x method).
"""
import collections
import sys
from pathlib import Path

sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
import research_harness as H  # noqa: E402

OUT = Path(__file__).resolve().parent / "split_out.txt"
ARMS = ("real", "staffed", "plain")
SITS = ("no_one", "waiting_line", "conversation_group")
PEOPLE = ("waiting_line", "conversation_group")
METHOD = {"cot": "R", "fsc": "E", "fsc3": "E", "con": "C"}
lines_out = []


def say(text=""):
    print(text)
    lines_out.append(text)


def load(stage, model, method, trials=None):
    """Records of the realistic arms of one stage/model/method, optionally keeping only some trials."""
    recs, used = [], []
    for arm in ARMS:
        refs = H.runs(stage=stage, arm=arm, model=model, method=method)
        if len(refs) != 1:
            raise SystemExit(f"expected one run for {stage} {arm} {model} {method}, got {len(refs)}")
        used.append(refs[0].folder)
        for rec in H.records(refs[0]):
            if trials is None or rec.trial in trials:
                recs.append(rec)
    return recs, used


def cause(rec):
    s = rec.score
    if s.get("success"):
        return "ok"
    if not s.get("target_correct"):
        return "wrong_tgt"
    if not s.get("in_region"):
        return "out_region"
    if not s.get("reachable"):
        return "unreach"
    if not s.get("heading_ok"):
        return "heading"
    return "other"


def cell(recs):
    n = len(recs)
    found = sum(bool(r.score.get("target_correct")) for r in recs)
    succ = sum(bool(r.score.get("success")) for r in recs)
    c = collections.Counter(cause(r) for r in recs)
    notok = sum(r.result.get("status") != "ok" for r in recs)
    both = sum(1 for r in recs if cause(r) == "out_region" and not r.score.get("reachable"))
    return dict(n=n, found=found, succ=succ, wrong=c["wrong_tgt"], out=c["out_region"], unr=c["unreach"],
                head=c["heading"], other=c["other"], notok=notok, both=both)


def pct(a, b):
    return f"{100 * a / b:5.1f}%" if b else "   n/a"


HEAD = (f"{'model':13s} {'M':1s} {'situation':18s} {'n':>4s} {'succ':>5s} {'found':>5s} {'found%':>6s} "
        f"{'s|found':>8s} {'s|f%':>6s} | {'wrong':>5s} {'(notok)':>7s} {'out_reg':>7s} {'unreach':>7s} {'heading':>7s} "
        f"{'out&unr':>7s}")


def row(model, m, sit, k):
    say(f"{model:13s} {m:1s} {sit:18s} {k['n']:4d} {k['succ']:5d} {k['found']:5d} {pct(k['found'], k['n']):>6s} "
        f"{k['succ']:3d}/{k['found']:<4d} {pct(k['succ'], k['found']):>6s} | {k['wrong']:5d} {k['notok']:7d} "
        f"{k['out']:7d} {k['unr']:7d} {k['head']:7d} {k['both']:7d}" + (f"  OTHER {k['other']}" if k['other'] else ""))


def table(title, specs):
    """specs: list of (model label, method letter, records)."""
    say("")
    say("=" * len(HEAD))
    say(title)
    say("=" * len(HEAD))
    say(HEAD)
    store = {}
    for model, m, recs in specs:
        by = collections.defaultdict(list)
        for r in recs:
            by[r.situation].append(r)
        extra = sorted(set(by) - set(SITS))
        if extra:
            say(f"  NOTE unexpected situations {extra}")
        for sit in SITS:
            k = cell(by[sit])
            store[(model, m, sit)] = k
            row(model, m, sit, k)
        k = cell([r for s in PEOPLE for r in by[s]])
        store[(model, m, "people")] = k
        row(model, m, "people(line+group)", k)
        k = cell(recs)
        store[(model, m, "pooled")] = k
        row(model, m, "POOLED", k)
        say("-" * len(HEAD))
    return store


NEW = [("qwen38_low", "Qwen", "s21"), ("glm53", "GLM", "s23"), ("dsv41", "DeepSeek", "s23"),
       ("haiku55", "Haiku", "s23"), ("sonnet55_low", "Sonnet", "s23"), ("gemini38_low", "Gemini", "s23")]

say(__doc__)
all_recs = []
specs, runs_used = [], []
for model, label, stage in NEW:
    for method in ("cot", "fsc", "con"):
        recs, used = load(stage, model, method)
        runs_used += used
        all_recs += recs
        specs.append((label, METHOD[method], recs))
new = table("TABLE 1. New wording: Qwen s21, others s23; realistic arms pooled (real+staffed+plain), 3 trials", specs)

# Old wording: s16b (R, E) and s19 (C) for Qwen and GLM, s20 for DeepSeek.
specs_old = []
for model, label in (("qwen38_low", "Qwen"), ("glm53", "GLM")):
    for stage, method in (("s16b", "cot"), ("s16b", "fsc"), ("s19", "con")):
        recs, used = load(stage, model, method)
        runs_used += used
        all_recs += recs
        specs_old.append((label, METHOD[method], recs))
for method in ("cot", "fsc", "con"):
    recs, used = load("s20", "dsv41", method)
    runs_used += used
    all_recs += recs
    specs_old.append(("DeepSeek", METHOD[method], recs))
old = table("TABLE 2. Old wording: Qwen/GLM s16b (R,E) + s19 (C); DeepSeek s20; realistic arms pooled, 3 trials",
            specs_old)

# Qwen 9-draw pool: s16b + s22 trials 3-5 + s21 (R, E); C has only s19 + s21 (6 draws).
s22_dup = collections.Counter()
specs_pool = []
for method in ("cot", "fsc"):
    a, ua = load("s16b", "qwen38_low", method)
    b, ub = load("s22", "qwen38_low", method, trials={3, 4, 5})
    bdup, _ = load("s22", "qwen38_low", method, trials={0, 1, 2})
    c, uc = load("s21", "qwen38_low", method)
    runs_used += ub
    all_recs += b
    # check s22 trials 0-2 duplicate s16b record for record
    key = lambda r: (r.arm, r.scene_id, r.episode_id, r.trial)
    old_by = {key(r): r for r in a}
    for r in bdup:
        o = old_by.get(key(r))
        if o is None:
            s22_dup[(method, "missing in s16b")] += 1
        elif (o.score == r.score and o.pose_world == r.pose_world):
            s22_dup[(method, "same score+pose")] += 1
        else:
            s22_dup[(method, "DIFFERENT")] += 1
    s22_dup[(method, "s22 trials 3-5 records")] = len(b)
    specs_pool.append(("Qwen 9-draw", METHOD[method], a + b + c))
    specs_pool.append(("Qwen s22 t3-5", METHOD[method], b))
a, _ = load("s19", "qwen38_low", "con")
c, _ = load("s21", "qwen38_low", "con")
specs_pool.append(("Qwen 6-draw", "C", a + c))
pool = table("TABLE 3. Qwen pools: R,E = s16b + s22 trials 3-5 + s21 (9 draws, 756 records); C = s19 + s21 (6 draws,"
             " 504); s22 t3-5 alone shown", specs_pool)
say("s22 trials 0-2 against s16b (should all be identical cache copies): " + str(dict(s22_dup)))

# ---- Claim checks -------------------------------------------------------------------------------------------
say("")
say("=" * 100)
say("CLAIM CHECKS")
say("=" * 100)
say("Totals of 252 (R / E / C), new wording:")
for model, label, stage in NEW:
    say(f"  {label:9s} " + " / ".join(str(new[(label, m, 'pooled')]['succ']) for m in "REC")
        + "   (n = " + "/".join(str(new[(label, m, 'pooled')]['n']) for m in "REC") + ")")
say("R at waiting_line, success given the right target (new wording):")
for model, label, stage in sorted(NEW, key=lambda x: -new[(x[1], 'R', 'waiting_line')]['succ'] /
                                  max(1, new[(x[1], 'R', 'waiting_line')]['found'])):
    k = new[(label, "R", "waiting_line")]
    say(f"  {label:9s} {k['succ']}/{k['found']} = {pct(k['succ'], k['found'])}   (found {k['found']}/{k['n']})")
say("C at people scenes (waiting_line + conversation_group, of 144), new wording:")
for model, label, stage in NEW:
    k = new[(label, "C", "people")]
    no = new[(label, "C", "no_one")]
    tot = new[(label, "C", "pooled")]
    say(f"  {label:9s} people {k['succ']}/{k['n']}  no_one succ {no['succ']}/{no['n']} found {no['found']}/{no['n']}"
        f" succ|found {no['succ']}/{no['found']}  total {tot['succ']}/{tot['n']}")
say("Gap decomposition of C totals vs Sonnet (gap = people-scene gap + no_one gap; no_one gap split into"
    " records lost to wrong target vs lost with right target):")
ref = "Sonnet"
for model, label, stage in NEW:
    t = new[(label, "C", "pooled")]['succ'] - new[(ref, "C", "pooled")]['succ']
    p = new[(label, "C", "people")]['succ'] - new[(ref, "C", "people")]['succ']
    nl = new[(label, "C", "no_one")]
    nr = new[(ref, "C", "no_one")]
    wrong_gap = -(nl['wrong'] - nr['wrong'])
    place_gap = -((nl['out'] + nl['unr'] + nl['head'] + nl['other']) - (nr['out'] + nr['unr'] + nr['head'] + nr['other']))
    say(f"  {label:9s} total {t:+4d} = people {p:+4d} + no_one {t - p:+4d} (wrong-target {wrong_gap:+4d},"
        f" placement {place_gap:+4d})")
say("Same decomposition, all situations, wrong-target vs placement failures (C, vs Sonnet):")
for model, label, stage in NEW:
    rows = []
    for sit in SITS:
        kl, kr = new[(label, "C", sit)], new[(ref, "C", sit)]
        rows.append(f"{sit}: wrong {kl['wrong'] - kr['wrong']:+d} place "
                    f"{(kl['n'] - kl['succ'] - kl['wrong']) - (kr['n'] - kr['succ'] - kr['wrong']):+d}")
    say(f"  {label:9s} " + "; ".join(rows))

# ---- Failure split per model, pooled over situations, per method --------------------------------------------
say("")
say("Failure split per model x method (new wording, pooled over situations; share of failures):")
say(f"  {'model':9s} M  fails  wrong_tgt   out_region   unreach   heading   (notok, out&unr)")
for model, label, stage in NEW:
    for m in "REC":
        k = new[(label, m, "pooled")]
        f = k['n'] - k['succ']
        say(f"  {label:9s} {m}  {f:5d}  {k['wrong']:4d} {pct(k['wrong'], f)}  {k['out']:4d} {pct(k['out'], f)}  "
            f"{k['unr']:3d} {pct(k['unr'], f)}  {k['head']:3d} {pct(k['head'], f)}  ({k['notok']}, {k['both']})")

# ---- Rescore check: recorded success and target_correct against the package scorer now ----------------------
say("")
seen, mism, n = set(), collections.Counter(), 0
by_scene = collections.defaultdict(list)
for r in all_recs:
    k = (r.run, r.scene_id, r.episode_id, r.trial)
    if k in seen:
        continue
    seen.add(k)
    by_scene[(str(r.ref.scenes_dir), r.scene_id, r.run)].append(r)
for key in sorted(by_scene):
    group = by_scene[key]
    posed = [r for r in group if r.pose_world is not None and r.result.get("status") == "ok"]
    rest = [r for r in group if not (r.pose_world is not None and r.result.get("status") == "ok")]
    for r in posed:
        s = H.score_pose(r, r.pose_world)
        n += 1
        for f in ("success", "target_correct", "in_region", "reachable", "heading_ok"):
            if bool(getattr(s, f)) != bool(r.score.get(f)):
                mism[f] += 1
    for r in rest:
        s = H.rescore(r)
        n += 1
        for f in ("success", "target_correct"):
            if bool(getattr(s, f)) != bool(r.score.get(f)):
                mism[f] += 1
say(f"Rescore check over {n} distinct records (score_pose for ok records with a pose, rescore otherwise): "
    f"mismatches {dict(mism) or 'none'}")
say("Runs used: " + ", ".join(sorted(set(runs_used))))
OUT.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
