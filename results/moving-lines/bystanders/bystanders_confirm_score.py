"""BYSTANDERS_CONFIRM_FROZEN.md scored (zero calls): bars C1, C2 and R on DeepSeek seeds 15-44, cafe and sweet_spot
bystander scenes. Uses bystanders_score.py's functions unchanged (capture, rosters, the frozen strict share).

    python3 bystanders_confirm_score.py runs/dsv41
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bystanders_score import capture, cells, load, paired, false_drops, CAPTURED_S  # noqa: E402
from summarise_fresh import sign_p  # noqa: E402

SCENES = {"cafe_line2_bystanders", "sweet_spot_line2_bystanders"}
SEEDS = set(range(15, 45))


def main():
    by = cells(load(sys.argv[1:], SCENES, SEEDS))
    print(f"# Bystanders confirmation: {len(by)} cells (planned 60)")
    src = Counter((v.get("CT2") or {}).get("answers", [{}])[0].get("source") for v in by.values())
    print(f"CT2 first-answer sources: {dict(src)}")
    cap = {(k, a): capture(v[a]) for k, v in by.items() for a in ("CT2", "CT3", "CT3b", "Gl") if a in v}
    pooled = {a: sum(c[0] for (k, b), c in cap.items() if b == a and c) for a in ("CT2", "CT3", "CT3b", "Gl")}
    n_cap = {a: sum(1 for (k, b), c in cap.items() if b == a and c and c[0] >= CAPTURED_S) for a in pooled}
    print(f"capture pooled s: {', '.join(f'{a} {pooled[a]:.1f}' for a in pooled)}; "
          f"cells captured >= 1 s: {', '.join(f'{a} {n_cap[a]}' for a in n_cap)}")

    _, w, l, t, md, _ = paired(by, "CT3", "CT2")
    p = sign_p(w, l)
    c1 = w > l and p <= 0.05
    print(f"C1 CT3 vs CT2 strict share: {w}/{l}/{t}, mean {md:+.3f}, sign p {p:.3f} (bar p <= 0.05, wins > losses): "
          f"{'PASS' if c1 else 'FAIL'}")
    c2 = pooled["CT3"] <= pooled["CT2"] / 3 and n_cap["CT3"] <= 0.2 * len(by)
    print(f"C2 capture: CT3 {pooled['CT3']:.1f} s vs CT2 {pooled['CT2']:.1f} s (bar <= 1/3), CT3 captured in "
          f"{n_cap['CT3']} of {len(by)} cells (bar <= 20%): {'PASS' if c2 else 'FAIL'}")
    print(f"'CT3 beats CT2 on bystander scenes': {'CLAIMED' if c1 and c2 else 'NOT SHOWN'} (needs C1 and C2)")

    keep = {k: v for k, v in by.items() if (v.get("CT2") or {}).get("answers", [{}])[0].get("source") == "constraints"}
    either = [k for k in keep if (cap.get((k, "Gl")) or (0,))[0] >= CAPTURED_S or (cap.get((k, "CT3")) or (0,))[0] >= CAPTURED_S]
    g_more = sum(1 for k in either if cap.get((k, "Gl")) and cap.get((k, "CT3")) and cap[(k, "Gl")][0] > cap[(k, "CT3")][0])
    _, w2, l2, t2, md2, _ = paired(keep, "CT3", "Gl")
    p2 = sign_p(w2, l2)
    r = len(either) > 0 and g_more >= 0.8 * len(either) and w2 > l2 and p2 <= 0.05
    print(f"R (constraint-answer cells, {len(keep)}): Gl capture > CT3 in {g_more} of {len(either)} (bar >= 80%); "
          f"CT3 vs Gl strict share {w2}/{l2}/{t2}, mean {md2:+.3f}, sign p {p2:.3f} (bar <= 0.05): "
          f"{'PASS' if r else 'FAIL'}")

    fd = sum(bool(false_drops(v["CT3"])[0]) for v in by.values() if "CT3" in v)
    fdb = sum(bool(false_drops(v["CT3b"])[0]) for v in by.values() if "CT3b" in v)
    print(f"Reported: cells with a false drop CT3 {fd}, CT3b {fdb}")
    for a, b in (("CT3b", "CT2"), ("CT3b", "CT3"), ("CT2", "Gl")):
        _, w_, l_, t_, md_, _ = paired(by, a, b)
        if w_ + l_ + t_:
            print(f"Reported (post hoc arm or secondary): {a} vs {b}: {w_}/{l_}/{t_}, mean {md_:+.3f}, "
                  f"sign p {sign_p(w_, l_):.3f}")


if __name__ == "__main__":
    main()
