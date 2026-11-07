"""Item 4 (I4_FROZEN.md): RTw (tracking R1's marked point, no roles) against CT2 on the fresh headline cells. Zero
calls; read-only.

    .venv/bin/python fill/i4_rtw/i4_score.py           (the frozen reading)
    .venv/bin/python fill/i4_rtw/i4_score.py --test    (old data: seeds 0-2, CT)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mcompare as C  # noqa: E402

# Amendment 1 (I4_FROZEN.md): RTw and CT2 are run together in one invocation into runs_i4/<model> (same machine
# load); the earlier RTw and CT2 in runs/<model> are read only for repeat checks.
POOLED = {"Sonnet 5.5": ("runs_i4/sonnet55_low", range(10, 20)), "DeepSeek V4.1": ("runs_i4/dsv41", range(10, 15)),
          "Haiku 5.5": ("runs_i4/claude_haiku55", range(10, 15)), "Gemini 3.8": ("runs_i4/gemini_fast", range(10, 15))}
REPORTED = {"x-test-3": ("runs_i4/x3", range(10, 13)), "Qwen 3.8": ("runs_i4/soclaas_qwen38_low", range(10, 15)),
            "GLM-5.3": ("runs_i4/soclaas_glm53", range(10, 15))}
OLD = ("cafe", "room_coffee_cart", "cafe_line2_moving", "cafe_line6_moving", "sweet_spot_line2_moving")


def anchor(eps) -> str:
    rt = [e for (_, _, a), e in eps.items() if a == "RTw" and e is not None]
    none = sum(1 for e in rt if any(t.get("anchor", 0) is None for t in e.get("tracked", [])))
    left = sum(1 for e in rt if any(t.get("anchor_in_queue") is False for t in e.get("tracked", [])))
    return f"{len(rt)} RTw episodes: point farther than 3 m from anyone (no anchor) {none}; anchor left the queue {left}"


def main():
    test = "--test" in sys.argv
    ref = "CT" if test else "CT2"
    scen = OLD if test else C.EVAL5
    groups = ({"DeepSeek (seeds 0-2)": ("runs/dsv41", range(0, 3)), "x-test-3 (seeds 0-2)": ("runs/x3", range(0, 3)),
               "Sonnet (seeds 0-2)": ("runs/sonnet55_low", range(0, 3))} if test else {**POOLED, **REPORTED})
    print(f"# Item 4: RTw against {ref}{' (TEST on old data)' if test else ''}\n")
    pooled_ab, pooled_rr, rows, rows_r = [], [], [], []
    for name, (root, seeds) in groups.items():
        eps = C.load(root, scen, seeds, ("RTw", ref, "R1"))
        n, s, f = C.means(eps, "RTw")
        if n == 0:
            print(f"- {name}: no RTw episodes yet")
            continue
        for arm in ("RTw", ref, "R1"):
            k, s, f = C.means(eps, arm)
            print(f"- {name} {arm}: {k} episodes, strict {s:.3f}, primary {f:.3f}, vlm_error episodes "
                  f"{sum(1 for (_, _, a), e in eps.items() if a == arm and C.errors(e))}")
        print(f"  {anchor(eps)}")
        if not test:
            old = C.load(root.replace("runs_i4/", "runs/"), scen, seeds, ("RTw", ref))
            for arm in ("RTw", ref):
                both = {(*k[:2], "new"): e for k, e in eps.items() if k[2] == arm}
                both.update({(*k[:2], "old"): e for k, e in old.items() if k[2] == arm})
                rp = C.pairs(both, "new", "old")
                if rp:
                    dd = [x["share_strict"] - y["share_strict"] for _, x, y in rp]
                    print(f"  repeat check, {arm} this run vs the earlier run, {len(rp)} cells: mean d "
                          f"{sum(dd) / len(dd):+.3f}, mean |d| {sum(abs(v) for v in dd) / len(dd):.3f}")
        ab, rr = C.pairs(eps, ref, "RTw"), C.pairs(eps, "RTw", "R1")
        rows.append(C.row(f"{name}: {ref} vs RTw", ab))
        rows_r.append(C.row(f"{name}: RTw vs R1", rr))
        if test or name in POOLED:
            pooled_ab += ab
            pooled_rr += rr
    rows.append(C.row(f"pooled ({'test groups' if test else 'Sonnet, DeepSeek, Haiku, Gemini'}): {ref} vs RTw", pooled_ab))
    rows_r.append(C.row("pooled: RTw vs R1", pooled_rr))
    print("\n## Primary comparison\n\n" + C.table(rows))
    print("\n## Secondary: does tracking the point help over asking once?\n\n" + C.table(rows_r))
    p = rows[-1]
    expected = 125 if not test else p["n"]
    print(f"\nBAR H4 (pooled, strict: {ref} wins > losses and sign p <= 0.05): {C.better(p, 'strict')}"
          f"{'' if p['n'] == expected else f'  [pooled pairs {p[chr(110)]}, expected {expected}: check mechanics]'}")
    print(f"Same under the primary scoring (reported): {C.better(p, 'frozen')}")
    for r in rows[:-1]:
        print(f"- per model, reported: {r['name']}: strict {C.better(r, 'strict')}, primary {C.better(r, 'frozen')}")


if __name__ == "__main__":
    main()
