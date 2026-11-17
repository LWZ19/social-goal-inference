"""BYSTANDERS_SONNET_ADDENDUM.md scored (zero calls): Sonnet 5.5 on the three bystander scenes, seeds 10-19.
Room's 10 cells are BYSTANDERS_FROZEN.md's optional arm (capture bar C2, share descriptive); cafe and sweet_spot's 20
are the addendum. Functions are bystanders_score.py's, unchanged.

    python3 bystanders_sonnet_score.py results/data/moving/runs/sonnet55_low   (from the repo root)
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402
sys.path.insert(0, str(paths.DESIGN))
from bystanders_score import (BYS, CAPTURED_S, capture, cells, false_drops, load, measures, named_bystanders,  # noqa: E402
                              paired)
from summarise_fresh import sign_p  # noqa: E402

ROOM = "room_coffee_cart_bystanders"


def c2(by, cap, label):
    pooled = {a: sum(cap[(k, a)][0] for k in by if cap.get((k, a))) for a in ("CT2", "CT3")}
    n3 = sum(1 for k in by if cap.get((k, "CT3")) and cap[(k, "CT3")][0] >= CAPTURED_S)
    n2 = sum(1 for k in by if cap.get((k, "CT2")) and cap[(k, "CT2")][0] >= CAPTURED_S)
    ok = pooled["CT3"] <= pooled["CT2"] / 3 and n3 <= 0.2 * len(by)
    trivial = " (trivial: CT2 has no capture)" if not pooled["CT2"] else ""
    print(f"- C2 {label}: CT3 {pooled['CT3']:.1f} s vs CT2 {pooled['CT2']:.1f} s (bar <= 1/3); CT3 captured in {n3}, "
          f"CT2 in {n2}, of {len(by)} cells (bar CT3 <= 20%): {'PASS' if ok else 'FAIL'}{trivial}")
    return ok


def main():
    eps = [e for e in load(sys.argv[1:], set(BYS), set(range(10, 20))) if e["label"].startswith("claude_sonnet")]
    by = cells(eps)
    print(f"# Sonnet bystanders: {len(by)} cells (planned 30: room 10, cafe 10, sweet_spot 10)")
    src = Counter((k[1], (v.get("CT2") or {}).get("answers", [{}])[0].get("source")) for k, v in by.items())
    print(f"CT2 first-answer sources: {dict(sorted(src.items()))}\n")
    cap = {(k, a): capture(v[a]) for k, v in by.items() for a in ("CT2", "CT3", "Gl") if a in v}

    room = {k: v for k, v in by.items() if k[1] == ROOM}
    rest = {k: v for k, v in by.items() if k[1] != ROOM}
    print("## Room (BYSTANDERS_FROZEN.md optional arm)")
    if room:
        c2(room, cap, "room")
    print("\n## Addendum: cafe + sweet_spot")
    ok_c2 = c2(rest, cap, "cafe+sweet") if rest else False

    named = {k: named_bystanders(v.get("CT2") or v.get("CT3")) for k, v in by.items()}
    n_named = sum(bool(x) for x in named.values() if x is not None)
    ok_s2 = n_named <= 6
    print(f"- S2 roster (all {len(by)} cells): bystanders named in {n_named} answers (bar <= 6 of 30): "
          f"{'PASS' if ok_s2 else 'FAIL'}")

    keep = {k: v for k, v in by.items() if (v.get("CT2") or {}).get("answers", [{}])[0].get("source") == "constraints"}
    either = [k for k in keep if (cap.get((k, "Gl")) or (0,))[0] >= CAPTURED_S or (cap.get((k, "CT3")) or (0,))[0] >= CAPTURED_S]
    g_more = sum(1 for k in either if cap.get((k, "Gl")) and cap.get((k, "CT3")) and cap[(k, "Gl")][0] > cap[(k, "CT3")][0])
    _, w, l, t, md, _ = paired(keep, "CT3", "Gl")
    p = sign_p(w, l)
    ok_r = len(either) > 0 and g_more >= 0.8 * len(either) and w > l and p <= 0.05
    print(f"- SR roles (constraint-answer cells, {len(keep)} of all 30): Gl capture > CT3 in {g_more} of {len(either)} "
          f"(bar >= 80%); CT3 vs Gl strict share {w}/{l}/{t}, mean {md:+.3f}, sign p {p:.3f} (bar <= 0.05): "
          f"{'PASS' if ok_r else 'FAIL'}")

    print("\n## Reported only (no claim unless a sign p <= 0.05 is stated as such)")
    for label, sub in (("room", room), ("cafe+sweet", rest), ("all", by)):
        for a, b in (("CT3", "CT2"), ("CT2", "R1"), ("CT3", "R1"), ("CT2", "Gl")):
            _, w_, l_, t_, md_, _ = paired(sub, a, b)
            if w_ + l_ + t_:
                print(f"  {label}: {a} vs {b}: {w_}/{l_}/{t_}, mean {md_:+.3f}, sign p {sign_p(w_, l_):.3f}")
    for arm in ("R1", "CT2", "CT3", "Gl"):
        m = [measures(v[arm]) for v in by.values() if arm in v]
        if m:
            print(f"  strict share {arm}: {sum(x['share_strict'] for x in m) / len(m):.3f} over {len(m)} cells; "
                  f"cut-in samples {sum(x['cut'] for x in m)}")
    fd = sum(bool(false_drops(v["CT3"])[0]) for v in by.values() if "CT3" in v)
    print(f"  cells with a CT3 false drop: {fd}")


if __name__ == "__main__":
    main()
