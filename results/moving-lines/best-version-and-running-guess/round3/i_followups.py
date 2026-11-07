"""Round 3 pilot follow-ups (FOLLOWUPS.md): X1, X2 (fixed in ROUND3.md section 11 before any output), and the
post-hoc PH-A and PH-B. Zero calls; read-only on the runs.

    .venv/bin/python round3/i_followups.py runs_pilot3/dsv41   (from ideas2_1b/)
"""
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
MOVING = HERE.parent.parent
REPO = Path(".")
sys.path.insert(0, str(MOVING))
sys.path.insert(0, str(MOVING / "sweep"))
from score_sweep import episode_scores  # noqa: E402
from summarise_fresh import in_line, sign_p  # noqa: E402

ARMS = ("CT3F", "CT4", "W0", "CT5")
BAND_M, HEADING_DEG, CLEAR_M = (0.5, 1.2), 45.0, 0.55   # sim/sgi_habitat/experiment.py


class Line:
    """Copy of sim/sgi_habitat/experiment.py's Line.project: (along, side) of a point on the slot polyline."""

    def __init__(self, slots):
        self.p = np.array([[s["x"], s["y"]] for s in slots], dtype=float)
        seg = np.diff(self.p, axis=0)
        self.length = np.hypot(seg[:, 0], seg[:, 1])
        self.unit = seg / self.length[:, None]
        self.start = np.concatenate([[0.0], np.cumsum(self.length)[:-1]])

    def project(self, x, y):
        best, n = None, len(self.length)
        for k in range(n):
            d = np.array([x, y]) - self.p[k]
            s = float(d @ self.unit[k])
            lo = -math.inf if k == 0 else 0.0
            hi = math.inf if k == n - 1 else float(self.length[k])
            s = min(max(s, lo), hi)
            foot = self.p[k] + s * self.unit[k]
            side = float(np.hypot(*(np.array([x, y]) - foot)))
            if best is None or side < best[1] - 1e-9:
                best = (float(self.start[k]) + s, side)
        return best


def load(root):
    sel = json.loads((MOVING / "sweep" / "pilot_selection.json").read_text())["layouts"]
    eps = {}
    for lay in sel:
        for arm in ARMS:
            hits = sorted((MOVING / root).glob(f"**/{lay['name']}/seed{lay['seed']}/{arm}/episode.json"))
            if hits:
                eps[(lay["name"], arm)] = json.loads(hits[0].read_text())
    return sel, eps


def line_of(name):
    sc = yaml.safe_load((REPO / "sim" / "scenarios" / "pilot3" / f"{name}.yaml").read_text())
    (q,) = sc["queues"].values()
    return Line(q["slots"])


def first_service_t(e):
    rs = e["resolves"]
    for a, b in zip(rs, rs[1:]):
        if a["truth"]["order"] and a["truth"]["order"][0] not in b["truth"]["order"]:
            return b["sent_t"]
    return None


def x1(sel, eps):
    print("## X1: CT5 minus CT3F, strict share from the first service, by restless (\"ours\" layouts)\n")
    for arm in ("CT5", "W0", "CT4"):
        for flag in (True, False):
            lays = [x for x in sel if x["stratum"] == "ours" and bool(x["restless"]) == flag]
            w = l = t = 0
            ds = []
            for lay in lays:
                a, b = eps.get((lay["name"], arm)), eps.get((lay["name"], "CT3F"))
                if not a or not b:
                    continue
                sa, sb = episode_scores(a)["after_service_strict"], episode_scores(b)["after_service_strict"]
                if sa is None or sb is None:
                    continue
                d = sa - sb
                ds.append(d)
                t += abs(d) < 0.01
                w += d >= 0.01
                l += d <= -0.01
            print(f"- {arm} - CT3F, {'restless' if flag else 'not restless'} (n {len(ds)}): mean d "
                  f"{statistics.mean(ds):+.3f}; {arm} better {w}, worse {l}, tied {t}; sign p {sign_p(w, l):.3f}")
    print()


def pos(r):
    return {int(i): (x, y) for i, x, y in r["tracks"]}


def x2(sel, eps):
    print("## X2: samples from the first service on that are not strictly in line, by the goal's last\n")
    print("| Arm | samples | strict in line | (a) cut | (b) not waiting | (c) waiter off axis | (d) other |")
    print("|---|---|---|---|---|---|---|")
    cross, dsplit = {}, {}
    for arm in ARMS:
        k, xt, ds = Counter(), Counter(), Counter()
        for lay in sel:
            e = eps.get((lay["name"], arm))
            if not e:
                continue
            t0 = first_service_t(e)
            if t0 is None:
                continue
            line = line_of(lay["name"])
            rs = [r for r in e["resolves"] if r.get("status") == "ok"]
            for s in (x for x in e["samples"] if x["t"] >= t0 - 1e-9):
                k["n"] += 1
                if in_line(s, "strict"):
                    k["in"] += 1
                    continue
                r = max((r for r in rs if r["sent_t"] <= s["t"] + 1e-9), key=lambda r: r["sent_t"], default=None)
                if r is None:
                    k["d"] += 1
                    ds["no re-solve yet"] += 1
                    continue
                tr, last, order = r["truth"], r.get("last"), r["truth"]["order"]
                p = pos(r)
                if last is not None and last in order:
                    off = last in p and line.project(*p[last])[1] > 0.6
                    if tr["last"] is not None and last in tr["ahead"] and \
                            order.index(last) < order.index(tr["last"]):
                        kind = "nearer front"
                    elif last == tr["last"]:
                        kind = "right"
                    else:
                        kind = "behind robot"
                    xt[(kind, "off" if off else "on")] += 1
                    if kind == "nearer front":
                        k["a"] += 1
                        continue
                    if off:
                        k["c"] += 1
                        continue
                elif last is not None:
                    k["b"] += 1
                    continue
                k["d"] += 1
                if s["ahead"] == 0:
                    ds["nobody ahead"] += 1
                elif not s["in_line"]:
                    why = ("behind the last" if s["behind_m"] < 0 else
                           "gap" if not BAND_M[0] <= s["gap_m"] <= BAND_M[1] else
                           "heading" if s["heading_off_deg"] > HEADING_DEG else
                           "nearest" if s["nearest_m"] < CLEAR_M else "?")
                    ds[("no last: " if last is None else "") + why] += 1
                else:
                    ds[("no last: " if last is None else "") + "side > 0.6"] += 1
        bad = max(k["n"] - k["in"], 1)
        print(f"| {arm} | {k['n']} | {k['in'] / max(k['n'], 1):.3f} | " +
              " | ".join(f"{k[c] / bad:.3f}" for c in "abcd") + " |")
        cross[arm], dsplit[arm] = xt, ds
    print("\n(a)-(d) are shares of the samples not strictly in line.\n")
    print("Cross-tab, waiting lasts on samples not strictly in line (counts): kind × on / off the axis\n")
    for arm in ARMS:
        xt = cross[arm]
        print(f"- {arm}: " + "; ".join(f"{kind} {xt[(kind, 'on')]} on / {xt[(kind, 'off')]} off"
                                      for kind in ("right", "nearer front", "behind robot")))
    print("\n(d) split by the first failing sim condition (counts)\n")
    for arm in ARMS:
        print(f"- {arm}: " + ", ".join(f"{a} {b}" for a, b in dsplit[arm].most_common()))
    print()


def pha(sel, eps):
    print("## PH-A (post hoc): why the goal jumps, over re-solve pairs whose pose moved more than 0.3 m\n")
    print("| Arm | jumps | (1) last changed | (2) same last, it moved > 0.15 m | (3) same last, roster or exempt "
          "changed | (4) other people moved |")
    print("|---|---|---|---|---|---|")
    for arm in ARMS:
        k = Counter()
        for lay in sel:
            e = eps.get((lay["name"], arm))
            if not e:
                continue
            rs = [r for r in e["resolves"] if r.get("status") == "ok" and r.get("pose_world")]
            for a, b in zip(rs, rs[1:]):
                if math.dist(a["pose_world"][:2], b["pose_world"][:2]) <= 0.3:
                    continue
                k["n"] += 1
                pa, pb = pos(a), pos(b)
                la, lb = a.get("last"), b.get("last")
                if la != lb:
                    k[1] += 1
                elif la is not None and la in pa and la in pb and math.dist(pa[la], pb[la]) > 0.15:
                    k[2] += 1
                elif set(a.get("roster") or []) != set(b.get("roster") or []) or \
                        set(a.get("exempt") or []) != set(b.get("exempt") or []):
                    k[3] += 1
                else:
                    k[4] += 1
        n = max(k["n"], 1)
        print(f"| {arm} | {k['n']} | " + " | ".join(f"{k[c] / n:.3f}" for c in (1, 2, 3, 4)) + " |")
    print()


def phb(sel, eps):
    print("## PH-B (post hoc): ticks where the solver's last is a bystander or walker (not in the sim's queue)\n")
    print("| Arm | ticks with someone ahead | last not waiting | its side from the axis (median m) | its along "
          "minus the true last's (median m) | run behind the same one (median s) | bystander / walker |")
    print("|---|---|---|---|---|---|---|")
    for arm in ARMS:
        n = m = 0
        sides, alongs, runs = [], [], []
        kinds = Counter()
        for lay in sel:
            e = eps.get((lay["name"], arm))
            if not e:
                continue
            line = line_of(lay["name"])
            bys = set(e.get("bystanders") or [])
            cur, start, prev_t = None, None, None
            for r in (r for r in e["resolves"] if r.get("status") == "ok"):
                tr = r["truth"]
                if not tr["ahead"]:
                    continue
                n += 1
                last, p = r.get("last"), pos(r)
                bad = last is not None and last not in tr["order"]
                if bad:
                    m += 1
                    kinds["bystander" if last in bys else "walker"] += 1
                    if last in p:
                        al, sd = line.project(*p[last])
                        sides.append(sd)
                        if tr["last"] in p:
                            alongs.append(al - line.project(*p[tr["last"]])[0])
                if bad and last == cur:
                    prev_t = r["sent_t"]
                    continue
                if cur is not None:
                    runs.append(prev_t - start + 0.5)
                cur, start, prev_t = (last, r["sent_t"], r["sent_t"]) if bad else (None, None, None)
            if cur is not None:
                runs.append(prev_t - start + 0.5)
        med = (lambda v: f"{statistics.median(v):.2f}" if v else "-")
        print(f"| {arm} | {n} | {m / max(n, 1):.3f} | {med(sides)} | {med(alongs)} | {med(runs)} | "
              f"{kinds['bystander']} / {kinds['walker']} |")
    print()


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "runs_pilot3/dsv41"
    sel, eps = load(root)
    print(f"# Round 3 pilot follow-ups: {root}, {len(eps)} episodes\n")
    x1(sel, eps)
    x2(sel, eps)
    pha(sel, eps)
    phb(sel, eps)


if __name__ == "__main__":
    main()
