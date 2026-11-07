"""Scores CT4P.md's predictions Q1-Q4 and its direction check. Written before any CT4-P run. Zero calls; read-only.

    .venv/bin/python round3/l_ct4p_score.py runs_pilot3/dsv41 [ARM]
    (from ideas2_1b/; ARM defaults to CT4P. Any other ARM is a dry run of the code, not a result.)
"""
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from i_followups import MOVING, first_service_t, line_of, pos, sign_p  # noqa: E402
from j_decompose import goal_check  # noqa: E402
from score_sweep import episode_scores  # noqa: E402  (path set by i_followups)

HIGH = ["cafe_1002", "cafe_1000", "cafe_1001", "cafe_1160", "cafe_1031", "sweet_1112", "sweet_1130", "cafe_1081",
        "sweet_1011", "cafe_1183", "cafe_1090", "sweet_1129"]
INFORMATIVE = ["cafe_1002", "cafe_1000", "cafe_1160", "cafe_1031", "sweet_1130"]


def load(root, arms):
    sel = json.loads((MOVING / "sweep" / "pilot_selection.json").read_text())["layouts"]
    eps = {}
    for lay in sel:
        for arm in arms:
            hits = sorted((MOVING / root).glob(f"**/{lay['name']}/seed{lay['seed']}/{arm}/episode.json"))
            if hits:
                eps[(lay["name"], arm)] = json.loads(hits[0].read_text())
    return sel, eps


def strict(eps, name, arm):
    e = eps.get((name, arm))
    return None if e is None else episode_scores(e)["after_service_strict"]


def wlt(pairs):
    w = sum(1 for a, b in pairs if a - b >= 0.01)
    l = sum(1 for a, b in pairs if b - a >= 0.01)
    return w, l, len(pairs) - w - l


def goal_ok_right_clean(sel, eps, arm):
    ok = n = 0
    for lay in sel:
        e = eps.get((lay["name"], arm))
        t0 = first_service_t(e) if e else None
        if t0 is None:
            continue
        line = line_of(lay["name"])
        for r in e["resolves"]:
            if r.get("status") != "ok" or not r.get("pose_world") or r["sent_t"] < t0:
                continue
            tr = r["truth"]
            if not tr["ahead"] or r.get("last") != tr["last"]:
                continue
            if any(i not in tr["order"] for i in (r.get("roster") or [])):
                continue
            n += 1
            ok += goal_check(r, line) == "ok"
    return ok / max(n, 1), n


def angle(u, v):
    c = (u[0] * v[0] + u[1] * v[1]) / (math.hypot(*u) * math.hypot(*v))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def direction_check(sel, eps, arm):
    angles, fallback, n = [], 0, 0
    for lay in sel:
        e = eps.get((lay["name"], arm))
        if not e:
            continue
        line = line_of(lay["name"])
        for r in e["resolves"]:
            if r.get("status") != "ok":
                continue
            n += 1
            src = r.get("direction_source")
            fallback += src is not None and src != "path"
            d, tr, p = r.get("solve_direction") or r.get("line_direction"), r["truth"], pos(r)
            if lay["name"] in HIGH and d and tr["last"] in p:
                al, _ = line.project(*p[tr["last"]])
                k = max(0, min(len(line.length) - 1, sum(al >= s for s in line.start) - 1))
                angles.append(angle(d, line.unit[k]))
    return (statistics.median(angles) if angles else float("nan")), fallback, n


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "runs_pilot3/dsv41"
    arm = sys.argv[2] if len(sys.argv) > 2 else "CT4P"
    sel, eps = load(root, ("CT3F", "CT4", arm))
    print(f"# CT4P.md scored: {root}, arm {arm}" + ("" if arm == "CT4P" else " (DRY RUN: not a result)") + "\n")
    print(f"- episodes: " + ", ".join(f"{a} {sum(1 for k in eps if k[1] == a)}" for a in ("CT3F", "CT4", arm)))
    med, fb, n = direction_check(sel, eps, arm)
    print(f"- direction check: median angle to the true axis at the true last, high-error layouts: {med:.1f}° "
          f"(bar < 15) -> {'PASS' if med < 15 else 'FAIL: stop, the port is wrong'}; fallback on {fb} of {n} "
          f"re-solves\n")
    print("| Layout | group | CT3F | CT4 | " + arm + " |")
    print("|---|---|---|---|---|")
    for lay in sel:
        nm = lay["name"]
        g = "high*" if nm in INFORMATIVE else "high" if nm in HIGH else "low"
        vals = [strict(eps, nm, a) for a in ("CT3F", "CT4", arm)]
        print(f"| {nm} | {g} | " + " | ".join("-" if v is None else f"{v:.3f}" for v in vals) + " |")
    print("\n## Predictions\n")
    pairs = [(strict(eps, nm, arm), strict(eps, nm, "CT4")) for nm in INFORMATIVE]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    w, l, t = wlt(pairs)
    print(f"- Q1, {arm} > CT4 on the 5 informative high-error layouts: {w}/{l}/{t} (better / worse / tied) -> "
          f"{'held' if w >= 4 else 'NOT held'} (needs 4)")
    low = [x["name"] for x in sel if x["name"] not in HIGH]
    ds = [strict(eps, nm, arm) - strict(eps, nm, "CT4") for nm in low
          if strict(eps, nm, arm) is not None and strict(eps, nm, "CT4") is not None]
    md = statistics.mean(ds) if ds else float("nan")
    print(f"- Q2, |mean({arm} - CT4)| on the 12 low-error layouts: {md:+.3f} (n {len(ds)}) -> "
          f"{'held' if abs(md) < 0.03 else 'NOT held'} (bar 0.03)")
    g, ng = goal_ok_right_clean(sel, eps, arm)
    print(f"- Q3, goals in line with the right last and a clean roster: {g:.2f} (n {ng}) -> "
          f"{'held' if g >= 0.55 else 'NOT held'} (bar 0.55; CT4 0.40, CT3F 0.64 in the pilot)")
    pairs = [(strict(eps, x["name"], "CT3F"), strict(eps, x["name"], arm)) for x in sel]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    w, l, t = wlt(pairs)
    print(f"- Q4, CT3F ahead of {arm} over all layouts: CT3F better {w}, worse {l}, tied {t}; sign p "
          f"{sign_p(w, l):.3f} -> {'held' if w > l else 'NOT held'}")
    allm = {a: statistics.mean([v for v in (strict(eps, x['name'], a) for x in sel) if v is not None])
            for a in ("CT3F", "CT4", arm)}
    print("\nMeans over layouts: " + ", ".join(f"{a} {v:.3f}" for a, v in allm.items()))


if __name__ == "__main__":
    main()
