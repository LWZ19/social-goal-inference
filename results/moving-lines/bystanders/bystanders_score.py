"""BYSTANDERS_FROZEN.md (sha256 683b48f0...) scored: M1-M5 and P1-P8, zero calls. Written before any judged cell.

    python3 bystanders_score.py runs/dsv41                                   # judged: *_bystanders, seeds 10-14
    python3 bystanders_score.py runs/dsv41 --nh runs/dsv41 runs/sonnet55_low # plus the no-harm runs (M5, P6)
    python3 bystanders_score.py runs/dsv41 --dev                             # seeds 0-2: format check only
    (run folders are read relative to the working directory: from the repo root, results/data/moving/runs/dsv41)

Log format (DEVIATIONS 12): per resolve `last` (the solver's _turn last, a track id = scenario human index; None where
no place in turn is used), `turn_joined`, `exempt`, `non_followers` {id: t}; the first resolve also has
`answer_come_after` ([number, nearest answer-frame track, metres]) and `chain` (Gl). `log["bystanders"]` = indices.

Operational readings fixed here, before any judged data:
- Capture (M1): an arm that never re-solves (a standard-goal answer) has 0 capture. Otherwise, from the first resolve's delivery to the episode's last sample, each resolve's `last` holds until the
  next resolve is delivered; captured time = time with `last` in the bystanders. "Captured" = at least 1.0 s.
- A standard-goal answer names nobody. A named bystander (M2, P2) = an `answer_come_after` row whose nearest track is a bystander within 0.5 m.
- The model's answer is read from CT's first resolve (C1, CT2, CT3, CT3-j replay the same answer; checked).
- P8 "differ in capture" = captured time differs by at least 1.0 s.
- A false drop (M4) = a non-follower who is in the queue at that time: by the samples' `queue_ids` where logged; where
  not, every non-bystander non-follower counts (labelled as a proxy).
- Lone bystander = one whose `face` is not another bystander's place; the pair face each other.
- Short-line state = the members count of the nearest sample at or before the time.
"""
import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402
sys.path.insert(0, str(paths.DESIGN))
from summarise_fresh import measures, sign_p  # noqa: E402  (the frozen strict scoring, unchanged)

ROOT = paths.REPO
BYS = ["cafe_line2_bystanders", "room_coffee_cart_bystanders", "sweet_spot_line2_bystanders"]
ORIG = ["cafe_line2_moving", "room_coffee_cart_slots", "sweet_spot_line2_moving"]
ARMS = ["R1", "C1", "CT", "CT2", "Gl", "CT3", "CT3-j"]
CAPTURED_S, NAMED_M, DIFFER_S = 1.0, 0.5, 1.0


def load(roots, scenes, seeds):
    eps = []
    for root in roots:
        for f in Path(root).glob("*/seed*/*/episode.json"):
            d = json.loads(f.read_text())
            if d["scenario"] in scenes and d["seed"] in seeds:
                d["_path"] = str(f.parent)
                eps.append(d)
    return eps


def scenario_info(name):
    d = yaml.safe_load((ROOT / "sim/scenarios" / f"{name}.yaml").read_text())
    humans = d.get("humans", [])
    places = {i: tuple(h["stand"]) for i, h in enumerate(humans) if h.get("bystander")}
    lone = {i for i, h in enumerate(humans) if h.get("bystander") and tuple(h.get("face", ())) not in places.values()}
    mirror = {i for i, h in enumerate(humans) if h.get("queue") and h.get("turn_deg")}
    return dict(bystanders=set(places), lone=lone, mirror=mirror)


def members_at(samples, t, key="members"):
    prev = [s for s in samples if s["t"] <= t + 1e-9]
    s = prev[-1] if prev else samples[0]
    return s.get(key)


def capture(e):
    """(captured seconds, window seconds, Counter of captured seconds by (lone|pair, members), bystander ids)."""
    bys = set(e.get("bystanders") or [])
    rs = [r for r in e.get("resolves", []) if r.get("delivered_t") is not None]
    if not e.get("samples"):
        return None
    end = e["samples"][-1]["t"]
    if not rs:                                     # no re-solving (e.g. a standard-goal answer): nothing to capture
        first = next((a["delivered_t"] for a in e.get("answers", []) if a.get("delivered_t") is not None), end)
        return 0.0, max(end - first, 0.0), Counter()
    start = rs[0]["delivered_t"]
    info = scenario_info(e["scenario"]) if e["scenario"] in BYS else dict(lone=set())
    got, by = 0.0, Counter()
    for r, nxt in zip(rs, rs[1:] + [None]):
        t1 = end if nxt is None else nxt["delivered_t"]
        dt = max(0.0, min(t1, end) - r["delivered_t"])
        last = r.get("last")
        if last is not None and int(last) in bys:
            got += dt
            by[("lone" if int(last) in info["lone"] else "pair", members_at(e["samples"], r["delivered_t"]))] += dt
    return got, max(end - start, 0.0), by


def named_bystanders(e):
    first = (e.get("resolves") or [{}])[0]
    rows = first.get("answer_come_after")
    if rows is None:                               # a standard-goal answer names nobody
        answers = [a for a in e.get("answers", []) if a.get("status") == "ok"]
        return [] if answers and answers[0].get("source") == "standard" else None
    bys = set(e.get("bystanders") or [])
    return sorted({int(r[1]) for r in rows if r[1] is not None and r[2] is not None and r[2] <= NAMED_M and int(r[1]) in bys})


def chain_bystanders(e):
    first = (e.get("resolves") or [{}])[0]
    ch = first.get("chain")
    if ch is None:
        return None
    return sorted(int(i) for i in ch if int(i) in set(e.get("bystanders") or []))


def non_followers(e):
    """{track id: first t} over the episode, and whether queue_ids were logged."""
    out = {}
    for r in e.get("resolves", []):
        for k, t in (r.get("non_followers") or {}).items():
            out.setdefault(int(k), t)
    logged = any("queue_ids" in s for s in e.get("samples", []))
    return out, logged


def false_drops(e):
    nf, logged = non_followers(e)
    bys = set(e.get("bystanders") or [])
    bad = []
    for i, t in nf.items():
        if logged:
            q = members_at(e["samples"], t, "queue_ids") or []
            if i in q:
                bad.append(i)
        elif i not in bys:
            bad.append(i)
    return bad, logged


def cells(eps):
    by = defaultdict(dict)
    for e in eps:
        by[(e["label"], e["scenario"], e["seed"])][e["arm"]] = e
    return by


def paired(by, a, b, key="share_strict"):
    rows = [(k, measures(v[a])[key], measures(v[b])[key]) for k, v in by.items() if a in v and b in v]
    w = sum(x > y + 1e-9 for _, x, y in rows)
    l = sum(x < y - 1e-9 for _, x, y in rows)
    d = [x - y for _, x, y in rows]
    return rows, w, l, len(rows) - w - l, (statistics.mean(d) if d else float("nan")), \
        (statistics.mean(abs(v) for v in d) if d else float("nan"))


def verdict(ok):
    return "PASS" if ok else "FAIL"


def judged(eps, dev):
    by = cells(eps)
    n_cells = len(by)
    print(f"# Bystanders, {'DEV FORMAT CHECK (seeds 0-2): not a judgement' if dev else 'judged cells'}, seeds {sorted({k[2] for k in by})}: "
          f"{n_cells} cells, models {sorted({k[0] for k in by})}\n")
    missing = Counter(a for v in by.values() for a in ARMS if a not in v)
    if missing:
        print(f"Missing arm-cells: {dict(missing)}\n")

    # M3 table
    print("## M3: strict share and keep-up, by arm\n")
    print("| Arm | Cells | Strict share | Frozen share | Back in line within 10 s | Cut-in samples |")
    print("|---|---|---|---|---|---|")
    for arm in ARMS:
        m = [measures(v[arm]) for v in by.values() if arm in v]
        if m:
            print(f"| {arm} | {len(m)} | {statistics.mean(x['share_strict'] for x in m):.3f} | "
                  f"{statistics.mean(x['share_frozen'] for x in m):.3f} | {sum(x['back10'] for x in m)}/"
                  f"{sum(x['events'] for x in m)} | {sum(x['cut'] for x in m)} |")

    # M1 capture
    cap = {(k, arm): capture(v[arm]) for k, v in by.items() for arm in ("CT", "CT2", "Gl", "CT3", "CT3-j") if arm in v}
    print("\n## M1: capture (time the solver's last is a bystander, after the first resolve)\n")
    print("| Arm | Cells | Captured >= 1 s | Captured s (pooled) | Share of window | Lone / pair s | By members count |")
    print("|---|---|---|---|---|---|---|")
    pooled = {}
    for arm in ("CT", "CT2", "Gl", "CT3", "CT3-j"):
        xs = [c for (k, a), c in cap.items() if a == arm and c is not None]
        if not xs:
            continue
        got, win = sum(c[0] for c in xs), sum(c[1] for c in xs)
        lone = sum(s for c in xs for (kind, _), s in c[2].items() if kind == "lone")
        pair = sum(s for c in xs for (kind, _), s in c[2].items() if kind == "pair")
        mem = Counter()
        for c in xs:
            for (_, m), s in c[2].items():
                mem[m] += s
        pooled[arm] = got
        print(f"| {arm} | {len(xs)} | {sum(c[0] >= CAPTURED_S for c in xs)} | {got:.1f} | {got / max(win, 1e-9):.3f} | "
              f"{lone:.1f} / {pair:.1f} | {dict(sorted((k, round(v, 1)) for k, v in mem.items()))} |")
    print("\nPer cell captured seconds (CT2 / Gl / CT3 / CT3-j):")
    for k in sorted(by):
        row = []
        for arm in ("CT2", "Gl", "CT3", "CT3-j"):
            c = cap.get((k, arm))
            row.append("-" if c is None else f"{c[0]:.1f}")
        print(f"  {k[1]:30} s{k[2]:<3} " + " / ".join(row))

    # M2 rosters
    print("\n## M2: rosters at the answer\n")
    model_named, chain_named, mismatched = {}, {}, []
    for k, v in sorted(by.items()):
        src = next((a for a in ("CT", "CT2", "CT3", "CT3-j") if a in v and named_bystanders(v[a]) is not None), None)
        model_named[k] = None if src is None else named_bystanders(v[src])
        first = lambda x: (x.get("resolves") or [{}])[0].get("answer_come_after")
        ref = None if src is None else first(v[src])
        for a in ("CT", "CT2", "CT3", "CT3-j"):
            if a in v and v[a].get("resolves") and first(v[a]) != ref:
                mismatched.append((k, a))
        chain_named[k] = chain_bystanders(v["Gl"]) if "Gl" in v else None
        print(f"  {k[1]:30} s{k[2]:<3} model names bystanders {model_named[k]}; Gl chain bystanders {chain_named[k]}")
    if mismatched:
        print(f"  WARNING: answer_come_after differs across the C arms in {mismatched}")

    # M4 false drops
    print("\n## M4: non-followers (CT3)\n")
    proxy = False
    fd_cells, mirror_cells = 0, 0
    for k, v in sorted(by.items()):
        if "CT3" not in v:
            continue
        nf, logged = non_followers(v["CT3"])
        bad, logged = false_drops(v["CT3"])
        proxy = proxy or not logged
        info = scenario_info(k[1])
        mir = sorted(set(nf) & info["mirror"])
        fd_cells += bool(bad)
        mirror_cells += bool(mir)
        if nf:
            print(f"  {k[1]:30} s{k[2]:<3} non-followers {nf}; false drops {bad}; mirror {mir}"
                  f"{'' if logged else ' (proxy: queue_ids not logged)'}")
    print(f"  cells with a false drop: {fd_cells}; room cells with the mirror member dropped: {mirror_cells}")

    # Predictions
    print("\n## Predictions (BYSTANDERS_FROZEN.md)\n")
    keys = sorted(by)
    p1 = sum(bool(chain_named.get(k)) for k in keys if chain_named.get(k) is not None)
    print(f"- P1 G's roster: a bystander in Gl's chain in {p1} of {sum(chain_named.get(k) is not None for k in keys)} "
          f"cells (bar >= 6 of 15): {verdict(p1 >= 6)}")
    p2 = sum(bool(model_named.get(k)) for k in keys if model_named.get(k) is not None)
    p2_ok = p2 <= 3
    print(f"- P2 model's roster: bystanders named in {p2} of {sum(model_named.get(k) is not None for k in keys)} "
          f"answers (bar <= 3 of 15): {verdict(p2_ok)}")
    c2 = {k: cap.get((k, "CT2")) for k in keys}
    p3_cells = [k for k in keys if c2[k] is not None and c2[k][0] >= CAPTURED_S]
    scenes_hit = {k[1] for k in p3_cells}
    lone2 = sum(s for k in keys if c2[k] for (kind, _), s in c2[k][2].items() if kind == "lone")
    p3_ok = len(p3_cells) >= 7 and scenes_hit >= set(k[1] for k in keys)
    print(f"- P3 CT2 captured >= 1 s in {len(p3_cells)} of {len(keys)} cells, scenes {sorted(scenes_hit)} "
          f"(bar >= 7, every scene): {verdict(p3_ok)}; lone-man share of CT2 capture "
          f"{lone2 / max(pooled.get('CT2', 0), 1e-9):.2f} (reported, 'most' expected)")
    c3 = {k: cap.get((k, "CT3")) for k in keys}
    p4_cells = sum(1 for k in keys if c3[k] is not None and c3[k][0] >= CAPTURED_S)
    p4_ok = pooled.get("CT3", 0) <= pooled.get("CT2", 0) / 3 and p4_cells <= 4
    print(f"- P4 CT3 pooled capture {pooled.get('CT3', 0):.1f} s vs CT2 {pooled.get('CT2', 0):.1f} s (bar <= 1/3), "
          f"CT3 captured in {p4_cells} cells (bar <= 4): {verdict(p4_ok)}"
          f"{' (trivial: CT2 has no capture)' if not pooled.get('CT2') else ''}")
    rows, w, l, t, md, _ = paired(by, "CT3", "CT2")
    room = [(x, y) for k, x, y in rows if k[1].startswith("room")]
    room_ok = sum(x >= y - 1e-9 for x, y in room) >= 4
    sp = sign_p(w, l)
    p5_ok = w > l and md > 0 and room_ok
    beats = p5_ok and p4_ok and sp <= 0.10
    print(f"- P5 CT3 vs CT2 strict share: {w}/{l}/{t}, mean {md:+.3f}, sign p {sp:.3f}; room at or above CT2 "
          f"{sum(x >= y - 1e-9 for x, y in room)} of {len(room)} (bar >= 4): {verdict(p5_ok)}. "
          f"'CT3 beats CT2' {'CLAIMED' if beats else 'NOT SHOWN'} (needs P5, P4 and sign p <= 0.10)")
    p7_ok = mirror_cells <= 1
    print(f"- P7 mirror member dropped in {mirror_cells} room cells (bar <= 1): {verdict(p7_ok)}")
    diff = sum(1 for k in keys if cap.get((k, "CT3")) and cap.get((k, "CT3-j"))
               and abs(cap[(k, "CT3")][0] - cap[(k, "CT3-j")][0]) >= DIFFER_S)
    print(f"- P8 CT3 and CT3-j differ in capture (>= {DIFFER_S} s) in {diff} cells (bar <= 4): {verdict(diff <= 4)}")
    if proxy:
        print("  (M4/P7 used the proxy: queue_ids were not logged)")

    # Role knowledge reading
    print("\n## Role knowledge (the note's reading rule; G = Gl, DEVIATIONS 12)\n")
    either = [k for k in keys if (cap.get((k, "Gl")) or (0,))[0] >= CAPTURED_S or (cap.get((k, "CT3")) or (0,))[0] >= CAPTURED_S]
    g_more = sum(1 for k in either if cap.get((k, "Gl")) and cap.get((k, "CT3")) and cap[(k, "Gl")][0] > cap[(k, "CT3")][0])
    _, w3, l3, t3, md3, _ = paired(by, "CT3", "Gl")
    either2 = [k for k in keys if (cap.get((k, "Gl")) or (0,))[0] >= CAPTURED_S or (cap.get((k, "CT2")) or (0,))[0] >= CAPTURED_S]
    ct2_not_lower = sum(1 for k in either2 if cap.get((k, "Gl")) and cap.get((k, "CT2")) and cap[(k, "CT2")][0] >= cap[(k, "Gl")][0])
    matters = p2_ok and g_more >= 8 and w3 > l3
    discards = p2_ok and p3_ok and ct2_not_lower > len(either2) / 2
    print(f"- Gl capture > CT3 capture in {g_more} of {len(either)} cells where either is captured (bar >= 8); "
          f"CT3 vs Gl strict share {w3}/{l3}/{t3}, mean {md3:+.3f}")
    print(f"- CT2 capture not lower than Gl's in {ct2_not_lower} of {len(either2)} cells where either is captured")
    m_text = "the model's roles matter" if matters else ''
    print(f"- Reading: {m_text}"
          f"{' | the model knows, CT2 discards it' if discards else ''}"
          f"{' | roles from the model do not matter here (P2 failed)' if not p2_ok else ''}"
          f"{' | none of the three readings holds' if not (matters or discards or not p2_ok) else ''}")
    print("\nPaired strict share, other comparisons (reported):")
    for a, b in (("CT2", "Gl"), ("CT3", "CT"), ("CT2", "CT"), ("CT3-j", "CT2"), ("CT2", "R1"), ("CT3", "R1"),
                 ("CT2", "C1")):
        _, w_, l_, t_, md_, _ = paired(by, a, b)
        if w_ + l_ + t_:
            print(f"  {a} vs {b}: {w_}/{l_}/{t_}, mean {md_:+.3f}, sign p {sign_p(w_, l_):.3f}")


def no_harm(eps):
    by = cells(eps)
    print(f"\n# M5 / P6: no-harm on the originals ({len(by)} cells)\n")
    for label in sorted({k[0] for k in by}):
        sub = {k: v for k, v in by.items() if k[0] == label}
        for arm in ("CT3", "CT3-j"):
            rows, w, l, t, md, mad = paired(sub, arm, "CT2")
            if not rows:
                continue
            fd = sum(bool(false_drops(v[arm])[0]) for v in sub.values() if arm in v)
            proxy = not all(false_drops(v[arm])[1] for v in sub.values() if arm in v)
            bar_fd = 2 if len(rows) <= 15 else 3
            ok = abs(md) <= 0.02 and mad <= 0.06 and fd <= bar_fd
            judged_arm = arm == "CT3"
            print(f"- {label} {arm} vs CT2: {len(rows)} cells, {w}/{l}/{t}, mean shift {md:+.3f} (bar within 0.02), "
                  f"mean |d| {mad:.3f} (bar <= 0.06), cells with a false drop {fd} (bar <= {bar_fd})"
                  f"{' [proxy]' if proxy else ''}: {verdict(ok) if judged_arm else 'reported'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--nh", nargs="*", default=[], help="roots holding the no-harm cells on the originals")
    ap.add_argument("--dev", action="store_true", help="seeds 0-2, format check only")
    ap.add_argument("--seeds", help="first-last, e.g. 15-34 (BYSTANDERS_CONFIRM_FROZEN.md); default 10-14")
    ap.add_argument("--scenes", help="comma-separated subset of the bystander scenes; default all three")
    a = ap.parse_args()
    if a.seeds:
        lo, hi = (int(x) for x in a.seeds.split("-"))
        seeds = set(range(lo, hi + 1))
    else:
        seeds = set(range(0, 3)) if a.dev else set(range(10, 15))
    scenes = set(a.scenes.split(",")) if a.scenes else set(BYS)
    eps = load(a.roots, scenes, seeds)
    sources = Counter((e["scenario"], (e.get("answers") or [{}])[0].get("source")) for e in eps if e["arm"] == "CT")
    print(f"CT first-answer sources: {dict(sorted(sources.items()))}\n")
    if not eps:
        print("no bystander episodes found")
    else:
        judged(eps, a.dev)
    if a.nh:
        nh = [e for e in load(a.nh, set(ORIG), set(range(10, 20))) if e["arm"] in ("CT2", "CT3", "CT3-j")]
        if a.dev:
            nh = []
        if nh:
            no_harm(nh)


if __name__ == "__main__":
    main()
