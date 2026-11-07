"""Was the line served while the model was thinking? (post hoc, zero calls; for the second ideas round)

For every bystander-scene cell: the first answer's latency window [asked_t, delivered_t], the services (an id leaves the
sim's queue_ids) inside it and in the 20 s before delivery, and how CT3's leftover capture (time the solver's last is a
bystander) splits around the first service after delivery. If most leftover capture sits in cells whose latency window
already held a service, then checking the answer against what happened while the model thought would remove it.

    python3 latency_window.py results/data/moving/runs/dsv41 results/data/moving/runs/sonnet55_low   (from the repo root)
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bystanders_score import BYS, capture, load, named_bystanders  # noqa: E402

FOLLOW_S = 5.0          # CT3's co_motion_s: a named person has this long to follow an advance


def services(e):
    out, prev = [], None
    for s in e.get("samples", []):
        q = s.get("queue_ids")
        if q is None:
            continue
        if prev is not None and set(prev) - set(q):
            out.append(s["t"])
        prev = q
    return out


def captured_spans(e):
    bys = set(e.get("bystanders") or [])
    rs = [r for r in e.get("resolves", []) if r.get("delivered_t") is not None]
    end = e["samples"][-1]["t"]
    spans = []
    for r, nxt in zip(rs, rs[1:] + [None]):
        t1 = end if nxt is None else nxt["delivered_t"]
        if r.get("last") is not None and int(r["last"]) in bys:
            spans.append((r["delivered_t"], min(t1, end)))
    return spans


def main():
    eps = load(sys.argv[1:], set(BYS), set(range(10, 45)))
    rows = []
    for e in eps:
        if e["arm"] not in ("CT2", "CT3"):
            continue
        a = next((x for x in e.get("answers", []) if x.get("delivered_t") is not None), None)
        if a is None or not e.get("samples"):
            continue
        model = Path(e["_path"]).parts[-4]
        sv = services(e)
        lat = [t for t in sv if a["asked_t"] <= t <= a["delivered_t"]]
        hist = [t for t in sv if a["delivered_t"] - 20.0 <= t <= a["delivered_t"]]
        after = [t for t in sv if t > a["delivered_t"]]
        cut = (after[0] + FOLLOW_S) if after else float("inf")
        spans = captured_spans(e)
        before = sum(max(0.0, min(t1, cut) - t0) for t0, t1 in spans)
        total = sum(t1 - t0 for t0, t1 in spans)
        rows.append(dict(model=model, arm=e["arm"], scene=e["scenario"], seed=e["seed"], latency=a["latency_s"],
                         asked=a["asked_t"], delivered=a["delivered_t"], lat_sv=lat, hist_sv=hist,
                         first_after=after[0] if after else None, cap=total, cap_before=before,
                         named=named_bystanders(e)))
    for model in sorted({r["model"] for r in rows}):
        for arm in ("CT2", "CT3"):
            rs = [r for r in rows if r["model"] == model and r["arm"] == arm]
            if not rs:
                continue
            n_lat = sum(bool(r["lat_sv"]) for r in rs)
            n_hist = sum(bool(r["hist_sv"]) for r in rs)
            cap = sum(r["cap"] for r in rs)
            cap_lat = sum(r["cap"] for r in rs if r["lat_sv"])
            cap_hist = sum(r["cap"] for r in rs if r["hist_sv"])
            capb = sum(r["cap_before"] for r in rs)
            capb_lat = sum(r["cap_before"] for r in rs if r["lat_sv"])
            ncap = sum(r["cap"] >= 1.0 for r in rs)
            ncap_lat = sum(r["cap"] >= 1.0 and bool(r["lat_sv"]) for r in rs)
            lats = sorted(r["latency"] for r in rs)
            print(f"{model:13} {arm}: {len(rs)} cells, median latency {lats[len(lats) // 2]:.1f} s; "
                  f"a service inside the latency window in {n_lat}, within 20 s before delivery in {n_hist}")
            print(f"    capture {cap:.1f} s in {ncap} cells; in cells with a latency-window service {cap_lat:.1f} s "
                  f"({ncap_lat} cells), with a 20 s-history service {cap_hist:.1f} s")
            print(f"    capture before the first post-delivery service + {FOLLOW_S:.0f} s: {capb:.1f} s "
                  f"(of it in latency-window cells {capb_lat:.1f} s); after it {cap - capb:.1f} s")
    print("\nCT3 cells with capture >= 1 s:")
    for r in sorted((r for r in rows if r["arm"] == "CT3" and r["cap"] >= 1.0),
                    key=lambda r: (r["model"], r["scene"], r["seed"])):
        print(f"  {r['model']:13} {r['scene']:28} s{r['seed']:<3} asked {r['asked']:.1f} delivered {r['delivered']:.1f} "
              f"latency-window services {[round(t, 1) for t in r['lat_sv']]} first after "
              f"{None if r['first_after'] is None else round(r['first_after'], 1)}; capture {r['cap']:.1f} s "
              f"(before first advance+5 s {r['cap_before']:.1f}); named {r['named']}")


if __name__ == "__main__":
    main()


def evidence_split(paths):
    """Second check: CT3's captured seconds by how many services had already happened since the request (each one a
    chance to see the named bystander not take part), whether the follow check ever named that bystander, and how far
    the captured bystander moved over the logged tracks."""
    eps = [e for e in load(paths, set(BYS), set(range(10, 45))) if e["arm"] == "CT3" and e.get("samples")]
    tot = defaultdict(float)
    fired = defaultdict(lambda: [0, 0])
    moved = []
    for e in eps:
        model = Path(e["_path"]).parts[-4]
        sv = services(e)
        bys = set(e.get("bystanders") or [])
        spans = captured_spans(e)
        if not spans:
            continue
        for t0, t1 in spans:
            t = t0
            while t < t1 - 1e-9:
                step = min(0.1, t1 - t)
                k = sum(1 for s in sv if s <= t)
                tot[(model, min(k, 3))] += step
                t += step
        caught = {int(r["last"]) for r in e["resolves"] if r.get("last") is not None and int(r["last"]) in bys}
        nf = {int(k) for r in e["resolves"] for k in (r.get("non_followers") or {})}
        fired[model][0] += 1
        fired[model][1] += bool(caught & nf)
        for b in caught:
            pos = [(t[1], t[2]) for r in e["resolves"] for t in r.get("tracks", []) if int(t[0]) == b]
            if pos:
                x0, y0 = pos[0]
                moved.append(max(((x - x0) ** 2 + (y - y0) ** 2) ** 0.5 for x, y in pos))
    print("\nCT3 captured seconds by services already seen since the request (0, 1, 2, 3+):")
    for model in sorted({m for m, _ in tot}):
        row = [tot.get((model, k), 0.0) for k in range(4)]
        print(f"  {model:13} " + ", ".join(f"{k if k < 3 else '3+'}: {v:.1f} s" for k, v in enumerate(row))
              + f"; after 2 or more services {sum(row[2:]):.1f} of {sum(row):.1f} s")
        print(f"    captured cells where the follow check ever named the captured bystander: "
              f"{fired[model][1]} of {fired[model][0]}")
    if moved:
        moved.sort()
        print(f"  captured bystanders' largest displacement over the logged tracks: median {moved[len(moved) // 2]:.2f} m, "
              f"max {moved[-1]:.2f} m (n = {len(moved)})")


if __name__ == "__main__" and len(sys.argv) > 1:
    evidence_split(sys.argv[1:])
