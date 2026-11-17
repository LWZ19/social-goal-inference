"""Evidence that accumulates, replayed on CT3's logged tracks (post hoc, zero calls; for the second ideas round).

CT3 drops a named person only on one chain of events: the waiting person directly ahead of them, among those still
taken as waiting, advances 0.4 m and they do not follow within 5 s. This replay asks what a rule that counts every
such chance would have done, on the same tracks and the same answer:

- an event for named person i: any tracked person j who was within 2 m of i and nearer the answer's target point comes
  0.4 m nearer it within 3 s (once per j per 5 s);
- a non-follow: i comes less than 0.25 m nearer the target over [t - 3 s, t + 5 s];
- i is dropped once it has k non-follows (k = 1, 2, 3).

Reported: when each named bystander would be dropped; CT3's captured seconds after that time (an upper bound on what
the drop removes: it assumes a dropped bystander is not taken in again, which CT3's join gate ensures for anyone who
has not walked 1 m or stepped 0.55 m nearer; captured bystanders never moved more than 0.35 m); and drops of named
people who were in the sim's queue at that time (false drops), over bystander and original scenes.
Tracks are logged only from the answer's delivery, so evidence from the latency window is not used here.

    python3 evidence_replay.py results/data/moving/runs/dsv41 results/data/moving/runs/sonnet55_low   (from the repo root)
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bystanders_score import load, members_at  # noqa: E402

SCENES = {"cafe_line2_bystanders", "room_coffee_cart_bystanders", "sweet_spot_line2_bystanders",
          "cafe_line2_moving", "room_coffee_cart_slots", "sweet_spot_line2_moving"}
NEAR_M, ADV_M, ADV_S, EVERY_S, FOLLOW_M, AFTER_S, NAMED_M = 2.0, 0.4, 3.0, 5.0, 0.25, 5.0, 0.5


def series(e):
    out = defaultdict(list)
    for r in e.get("resolves", []):
        if r.get("delivered_t") is None:
            continue
        for tid, x, y in r.get("tracks", []):
            out[int(tid)].append((r["delivered_t"], (x, y)))
    return out


def at(s, t):
    """Position at the last record at or before t (None before the first)."""
    prev = None
    for ts, xy in s:
        if ts > t + 1e-9:
            break
        prev = xy
    return prev


def replay(e):
    f = Path(e["_path"]) / "001" / "answer.json"
    if not f.exists():                             # no recorded answer here (e.g. a standard-goal or failed answer)
        return None
    tw = json.loads(f.read_text()).get("target_world")
    first = (e.get("resolves") or [{}])[0]
    rows = first.get("answer_come_after")
    if not tw or rows is None:
        return None
    tgt = (tw[0], tw[1])
    named = sorted({int(r[1]) for r in rows if r[1] is not None and r[2] is not None and r[2] <= NAMED_M})
    S = series(e)
    gap = lambda tid, t: (lambda p: None if p is None else math.dist(p, tgt))(at(S[tid], t))
    times = sorted({ts for s in S.values() for ts, _ in s})
    events = []                                    # (t, j)
    last_adv = defaultdict(lambda: -1e9)
    for t in times:
        for j in S:
            g0, g1 = gap(j, t - ADV_S), gap(j, t)
            if g0 is not None and g1 is not None and g0 - g1 >= ADV_M and t - last_adv[j] > EVERY_S:
                last_adv[j] = t
                events.append((t, j))
    drops = {}
    for i in named:
        if i not in S:
            continue
        n, when = 0, {}
        for t, j in events:
            if j == i:
                continue
            pi0, pj0 = at(S[i], t - ADV_S), at(S[j], t - ADV_S)
            gi, gj = gap(i, t), gap(j, t)
            if None in (pi0, pj0, gi, gj) or math.dist(pi0, pj0) > NEAR_M or gj >= gi:
                continue
            gi0, gi1 = gap(i, t - ADV_S), gap(i, t + AFTER_S)
            if gi1 is None or gi0 - gi1 >= FOLLOW_M:
                continue
            n += 1
            when.setdefault(n, t + AFTER_S)
        drops[i] = when
    return named, drops


def main():
    eps = [e for e in load(sys.argv[1:], SCENES, set(range(10, 45))) if e["arm"] == "CT3" and e.get("samples")]
    agg = defaultdict(lambda: defaultdict(float))
    false = defaultdict(lambda: defaultdict(int))
    named_true = defaultdict(int)
    drop_times = defaultdict(list)
    for e in eps:
        model = Path(e["_path"]).parts[-4]
        out = replay(e)
        if out is None:
            continue
        named, drops = out
        bys = set(e.get("bystanders") or [])
        kind = "bystander scenes" if bys else "original scenes"
        rs = [r for r in e["resolves"] if r.get("delivered_t") is not None]
        end = e["samples"][-1]["t"]
        for r, nxt in zip(rs, rs[1:] + [None]):
            b = r.get("last")
            if b is None or int(b) not in bys:
                continue
            t0, t1 = r["delivered_t"], min(end, end if nxt is None else nxt["delivered_t"])
            agg[(model, kind)]["capture"] += max(0.0, t1 - t0)
            for k in (1, 2, 3):
                d = drops.get(int(b), {}).get(k)
                if d is not None:
                    agg[(model, kind)][f"removed k={k}"] += max(0.0, t1 - max(t0, d))
        for i in named:
            q = None
            if i in bys:
                for k in (1, 2, 3):
                    if k in drops.get(i, {}):
                        drop_times[(model, k)].append(drops[i][k] - rs[0]["delivered_t"])
                continue
            named_true[(model, kind)] += 1
            for k in (1, 2, 3):
                d = drops.get(i, {}).get(k)
                if d is None:
                    continue
                q = members_at(e["samples"], d, "queue_ids") or []
                if i in q:
                    false[(model, kind)][k] += 1
    for key in sorted(agg.keys() | named_true.keys()):
        a = agg.get(key, {})
        cap = a.get("capture", 0.0)
        print(f"{key[0]:13} {key[1]:17} CT3 capture {cap:6.1f} s; removed if dropped after k non-follows: "
              + ", ".join(f"k={k} {a.get(f'removed k={k}', 0.0):.1f} s" for k in (1, 2, 3)))
        print(f"{'':31} named people in the queue: {named_true.get(key, 0)}; dropped while in the queue (false drops): "
              + ", ".join(f"k={k} {false.get(key, {}).get(k, 0)}" for k in (1, 2, 3)))
    for key in sorted(drop_times):
        v = sorted(drop_times[key])
        print(f"{key[0]:13} k={key[1]}: named bystanders dropped {len(v)}, seconds after delivery median "
              f"{v[len(v) // 2]:.1f}, max {v[-1]:.1f}")


if __name__ == "__main__":
    main()
