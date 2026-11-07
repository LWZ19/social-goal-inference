"""Scores CT4PJ.md's J1-J5. Written before any CT4-PJ run and before the research read CT4P's outcome.
Zero calls; read-only. Stops before J2-J5 if J1 fails.

    .venv/bin/python round4/m_ct4pj_score.py runs_pilot3/dsv41 [NEW BASE]
    (from ideas2_1b/; NEW defaults to CT4PJ and BASE to CT4P. Other arms are a dry run of the code, not a result.)
"""
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "round3"))
from i_followups import MOVING, first_service_t, in_line, line_of, pos, sign_p  # noqa: E402
from score_sweep import episode_scores  # noqa: E402


def load(root, arms):
    sel = json.loads((MOVING / "sweep" / "pilot_selection.json").read_text())["layouts"]
    eps = {}
    for lay in sel:
        for arm in arms:
            hits = sorted((MOVING / root).glob(f"**/{lay['name']}/seed{lay['seed']}/{arm}/episode.json"))
            if hits:
                eps[(lay["name"], arm)] = json.loads(hits[0].read_text())
    return sel, eps


def behind_share(sel, eps, arm):
    n = b = 0
    for lay in sel:
        e = eps.get((lay["name"], arm))
        if not e:
            continue
        line, S = line_of(lay["name"]), e["samples"]
        for r in e["resolves"]:
            if r.get("status") != "ok":
                continue
            s = min(S, key=lambda x: abs(x["t"] - r["sent_t"]))
            if not s["joined"] or abs(s["t"] - r["sent_t"]) > 0.6:
                continue
            n += 1
            last, p = r.get("last"), pos(r)
            if last is not None and last in p and line.project(*p[last])[0] > s["along"] + 0.25:
                b += 1
    return b / max(n, 1), n


def stays(e):
    out, t_join = [], None
    for ev in e.get("joins", []):
        if ev["event"] == "joined":
            t_join = ev["t"]
        elif ev["event"] == "left" and t_join is not None:
            out.append(ev["t"] - t_join)
            t_join = None
    return out


def joined_share(e):
    t0 = first_service_t(e)
    if t0 is None:
        return None
    s = [x for x in e["samples"] if x["t"] >= t0 - 1e-9]
    return sum(x["joined"] for x in s) / len(s) if s else None


def wlt(pairs):
    w = sum(1 for a, b in pairs if a - b >= 0.01)
    l = sum(1 for a, b in pairs if b - a >= 0.01)
    return w, l, len(pairs) - w - l


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "runs_pilot3/dsv41"
    new = sys.argv[2] if len(sys.argv) > 2 else "CT4PJ"
    base = sys.argv[3] if len(sys.argv) > 3 else "CT4P"
    dry = (new, base) != ("CT4PJ", "CT4P")
    sel, eps = load(root, ("CT3F", base, new))
    print(f"# CT4PJ.md scored: {root}, new {new}, base {base}" + (" (DRY RUN: not a result)" if dry else "") + "\n")
    print("- episodes: " + ", ".join(f"{a} {sum(1 for k in eps if k[1] == a)}" for a in ("CT3F", base, new)))
    share, n = behind_share(sel, eps, new)
    ok = share < 0.03
    print(f"- J1, last behind the robot while joined: {new} {share:.3f} (n {n}; bar < 0.03) -> "
          f"{'held' if ok else 'NOT held: the port is wrong; J2-J5 withheld'}")
    if not ok:
        return
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
