"""Round 3 pilot: score PILOT_PREDICTIONS.md (sha256 cb29fc67) from the closed-loop logs. Written before stage 2
existed. Zero calls; read-only on the runs.

Per episode (sweep/score_sweep.episode_scores unchanged, plus POSTHOC_1 Q5's jitter measures):
- goal jumps per minute: consecutive ok re-solves whose pose_world moved more than 0.3 m, per minute of re-solves;
- roster changes per re-solve;
- cut-in samples (summarise_fresh.measures);
- strict share in line from the first service after delivery (episode_scores' after_service_strict);
- right last (episode_scores' right_last).
Per prediction: layouts where the pair goes the stated way / the other way / tied (|d| < 0.01 for shares, d = 0
for counts and rates), the pooled means, and held / not held ("held" = more layouts the stated way).

    .venv/bin/python round3/h_pilot_score.py runs_pilot3/dsv41      (from ideas2_1b/)
"""
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MOVING = HERE.parent.parent
sys.path.insert(0, str(MOVING))
sys.path.insert(0, str(MOVING / "sweep"))
from score_sweep import episode_scores  # noqa: E402
from summarise_fresh import sign_p  # noqa: E402

ARMS = ("CT2", "CT3F", "CT4", "W0", "CT5")


def jitter(e):
    rs = [r for r in e.get("resolves", []) if r.get("status") == "ok" and r.get("pose_world")]
    if len(rs) < 2:
        return None, None
    jumps = sum(1 for a, b in zip(rs, rs[1:]) if math.dist(a["pose_world"][:2], b["pose_world"][:2]) > 0.3)
    changes = sum(1 for a, b in zip(rs, rs[1:]) if set(a.get("roster") or []) != set(b.get("roster") or []))
    minutes = max((rs[-1]["sent_t"] - rs[0]["sent_t"]) / 60, 1e-9)
    return jumps / minutes, changes / (len(rs) - 1)


def load(root):
    sel = json.loads((MOVING / "sweep" / "pilot_selection.json").read_text())["layouts"]
    out = {}
    for lay in sel:
        for arm in ARMS:
            hits = sorted((MOVING / root).glob(f"**/{lay['name']}/seed{lay['seed']}/{arm}/episode.json"))
            if hits:
                e = json.loads(hits[0].read_text())
                s = episode_scores(e)
                s["jumps"], s["roster_changes"] = jitter(e)
                out[(lay["name"], arm)] = s
    return sel, out


def compare(sel, sc, a, b, key, better, share=True):
    """Layouts where a is better than b by `better` ("lower" or "higher")."""
    w = l = t = 0
    va, vb = [], []
    for lay in sel:
        x, y = sc.get((lay["name"], a), {}).get(key), sc.get((lay["name"], b), {}).get(key)
        if x is None or y is None:
            continue
        va.append(x)
        vb.append(y)
        d = (x - y) if better == "higher" else (y - x)
        if abs(d) < (0.01 if share else 1e-9):
            t += 1
        elif d > 0:
            w += 1
        else:
            l += 1
    ma = statistics.mean(va) if va else float("nan")
    mb = statistics.mean(vb) if vb else float("nan")
    return w, l, t, ma, mb


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "runs_pilot3/dsv41"
    sel, sc = load(root)
    print(f"# Round 3 pilot scored against PILOT_PREDICTIONS.md: {root}\n")
    print(f"- episodes: {len(sc)} ({', '.join(f'{a} {sum(1 for k in sc if k[1] == a)}' for a in ARMS)})\n")
    print("| Arm | goal jumps / min | roster changes / re-solve | cut-in samples | strict in line from 1st service | "
          "right last |")
    print("|---|---|---|---|---|---|")
    for arm in ARMS:
        def m(k, arm=arm):
            v = [s[k] for (n, a), s in sc.items() if a == arm and s.get(k) is not None]
            return f"{statistics.mean(v):.3f} (n {len(v)})" if v else "-"
        print(f"| {arm} | {m('jumps')} | {m('roster_changes')} | {m('cut')} | {m('after_service_strict')} | "
              f"{m('right_last')} |")
    print("\n## Predictions\n")
    rows = [
        ("P1 goal jumps: CT5 < CT4", "CT5", "CT4", "jumps", "lower", False, lambda w, l: w > l and w >= 17),
        ("P1 goal jumps: W0 < CT4", "W0", "CT4", "jumps", "lower", False, lambda w, l: w > l),
        ("P3 cut-ins: CT5 < CT4", "CT5", "CT4", "cut", "lower", False, lambda w, l: w > l),
        ("P3 cut-ins: W0 not below CT4", "W0", "CT4", "cut", "lower", False, lambda w, l: not w > l),
        ("P4 strict in line: CT5 > CT4", "CT5", "CT4", "after_service_strict", "higher", True, lambda w, l: w > l),
        ("P4 strict in line: W0 > CT4", "W0", "CT4", "after_service_strict", "higher", True, lambda w, l: w > l),
        ("P5 right last: CT5 < CT4", "CT4", "CT5", "right_last", "higher", True, lambda w, l: w > l),
    ]
    for label, a, b, key, better, share, ok in rows:
        w, l, t, ma, mb = compare(sel, sc, a, b, key, better, share)
        print(f"- {label}: {w}/{l}/{t} (stated way / other way / tied), means {a} {ma:.3f} vs {b} {mb:.3f}, "
              f"sign p {sign_p(w, l):.3f} -> {'held' if ok(w, l) else 'NOT held'}")
    for arm in ("W0", "CT5"):
        def pooled(a):
            v = [s["roster_changes"] for (n, x), s in sc.items() if x == a and s.get("roster_changes") is not None]
            return statistics.mean(v) if v else float("nan")
        r = pooled(arm) / pooled("CT4")
        print(f"- P2 roster changes: {arm} / CT4 = {r:.2f} (stated < 0.6) -> {'held' if r < 0.6 else 'NOT held'}")
    for arm in ("CT5", "W0"):
        print(f"\n## {arm} against CT3F (no prediction; reported; decides whether a confirmatory test is worth "
              f"running)\n")
        for key, better, share in (("after_service_strict", "higher", True), ("cut", "lower", False),
                                   ("jumps", "lower", False), ("right_last", "higher", True)):
            w, l, t, ma, mb = compare(sel, sc, arm, "CT3F", key, better, share)
            print(f"- {key} ({better} is better): {arm} better on {w}, worse on {l}, tied {t}; means {arm} {ma:.3f} "
                  f"vs CT3F {mb:.3f}; sign p {sign_p(w, l):.3f}")


if __name__ == "__main__":
    main()
