"""Releases CT4PJ.md's J2-J5 after J1 failed on its premise (CT4PJ_CHECK.md). It uses m_ct4pj_score's own
functions, unchanged, and the same computations as its main() after the J1 gate. Zero calls; read-only.

    .venv/bin/python round4/n_ct4pj_release.py runs_pilot3/dsv41
    (from best-version-and-running-guess/; RUN_ROOT relative to results/data/moving/)
"""
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m_ct4pj_score import episode_scores, joined_share, load, sign_p, stays, wlt  # noqa: E402


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "runs_pilot3/dsv41"
    new, base = "CT4PJ", "CT4P"
    sel, eps = load(root, ("CT3F", base, new))
    print(f"# CT4PJ.md J2-J5, released after J1's premise failure (CT4PJ_CHECK.md): {root}\n")
    print("- episodes: " + ", ".join(f"{a} {sum(1 for k in eps if k[1] == a)}" for a in ("CT3F", base, new)))
    sn = [x for (nm, a), e in eps.items() if a == new for x in stays(e)]
    sb = [x for (nm, a), e in eps.items() if a == base for x in stays(e)]
    mn = statistics.median(sn) if sn else float("nan")
    mb = statistics.median(sb) if sb else float("nan")
    print(f"- J2, median time joined before leaving: {new} {mn:.1f} s (n {len(sn)}) vs {base} {mb:.1f} s (n {len(sb)}) "
          f"-> {'held' if mn > mb else 'NOT held'}")

    def per_layout(f, a):
        e = eps.get(a)
        return None if e is None else f(e)

    pairs = [(per_layout(joined_share, (x["name"], new)), per_layout(joined_share, (x["name"], base))) for x in sel]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    w, l, t = wlt(pairs)
    print(f"- J3, joined share after the first service: {new} above {base} on {w}, below {l}, tied {t}; sign p "
          f"{sign_p(w, l):.3f} -> {'held' if w > l else 'NOT held'}")

    def strict(nm, a):
        e = eps.get((nm, a))
        return None if e is None else episode_scores(e)["after_service_strict"]

    pairs = [(strict(x["name"], new), strict(x["name"], base)) for x in sel]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    w, l, t = wlt(pairs)
    print(f"- J4, strict in line from the first service: {new} above {base} on {w}, below {l}, tied {t}; sign p "
          f"{sign_p(w, l):.3f} -> {'held' if w > l else 'NOT held'}")
    pairs = [(strict(x["name"], "CT3F"), strict(x["name"], new)) for x in sel]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    w, l, t = wlt(pairs)
    print(f"- J5, CT3F ahead of {new}: CT3F better {w}, worse {l}, tied {t}; sign p {sign_p(w, l):.3f} -> "
          f"{'held' if w > l else 'NOT held'}")
    means = {a: statistics.mean([v for v in (strict(x['name'], a) for x in sel) if v is not None])
             for a in ("CT3F", base, new)}
    print("\nMeans over layouts (strict from the first service): " +
          ", ".join(f"{a} {v:.3f}" for a, v in means.items()))


if __name__ == "__main__":
    main()
