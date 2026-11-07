"""Zero calls, scenario files only: for every candidate layout (make_sweep.py), which model-free rules take the lone
non-member into the line at the photo frame, at the nominal places and under the sim's fidget (each standing
non-member uniformly within 0.35 m of their place; 500 draws). The rules:

  chain  : start from the person nearest the target point (slot 0's face point); add anyone within 1.0 m of anyone
           taken; repeat (the Gl arm's geometry, IDEAS_interface F2).
  absorb : the solver's place-in-turn rule (`solver._turn`, as in bystanders_geometry.py): with the true members as
           the waiting people, a person within 1.0 m of the place 1.0 m behind the last and within 35 deg of
           straight behind becomes the new last; iterated.
  static : cognitive_science idea 1's leave-one-out check: would the solver's place-in-turn put a waiting person
           where they stand, i.e. at most 0.3 m off the axis and 0.3-1.6 m behind the last member.

Strata (nominal places; agreed with the research, 3 Nov): ours = absorb and static take the lone non-member
(the two rules the pipeline itself uses) and the chain doesn't (all three together leave no place in line, so that
stratum is empty); chain = otherwise, the chain takes the lone non-member or a pair member; mixed = otherwise, absorb
or static takes him, or absorb takes a pair member; easy = no rule takes anyone. Also reported: whether the
place 1.0 m behind the last member (the solver's standoff) is at least 0.6 m from everyone.

    python3 sweep_report.py candidates/   -> candidates/report.json and a printed summary
"""
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import yaml

STANDOFF, HALF, CHAIN, FIDGET, N = 1.0, math.radians(35.0), 1.0, 0.35, 500


def chain_takes(people, target, who):
    start = min(people, key=lambda p: math.dist(p[1], target))
    taken, grew = {start[0]}, True
    while grew:
        grew = False
        for n, xy in people:
            if n not in taken and any(math.dist(xy, q) < CHAIN for m, q in people if m in taken):
                taken.add(n)
                grew = True
    return who in taken


def absorb_takes(waiting, others, along, who):
    waiting, taken = list(waiting), []
    while True:
        last = waiting[-1][1]
        place = (last[0] + STANDOFF * along[0], last[1] + STANDOFF * along[1])
        cands = []
        for n, xy in others:
            if n in taken:
                continue
            back = (xy[0] - last[0], xy[1] - last[1])
            nb = math.hypot(*back)
            behind = back[0] * along[0] + back[1] * along[1] >= math.cos(HALF) * nb - 1e-9
            gap = math.dist(xy, place)
            if gap < STANDOFF and behind:
                cands.append((gap, n, xy))
        if not cands:
            return who in taken
        _, n, xy = min(cands)
        taken.append(n)
        waiting.append((n, xy))


def static_takes(last, along, side, xy):
    rel = (xy[0] - last[0], xy[1] - last[1])
    a = rel[0] * along[0] + rel[1] * along[1]
    s = rel[0] * side[0] + rel[1] * side[1]
    return abs(s) <= 0.3 and 0.3 <= a <= 1.6


def report(path: Path, rng: random.Random) -> dict:
    sc = yaml.safe_load(path.read_text())
    (_, q), = sc["queues"].items()
    slots = [(s["x"], s["y"]) for s in q["slots"]]
    target = tuple(q["slots"][0]["face"])
    along = (slots[1][0] - slots[0][0], slots[1][1] - slots[0][1])
    L = math.hypot(*along)
    along = (along[0] / L, along[1] / L)
    side = (-along[1], along[0])
    n = sum(1 for h in sc["humans"] if h.get("queue"))
    members = [(f"m{i}", slots[i]) for i in range(n)]
    standing = [(h["avatar"], tuple(h["stand"])) for h in sc["humans"] if h.get("stand") is not None]
    bys = [h["avatar"] for h in sc["humans"] if h.get("bystander")]
    lone, pair = bys[0], bys[1:]
    fixed = [(a, xy) for a, xy in standing if a not in bys]           # e.g. the barista: does not fidget
    last = slots[n - 1]

    def rules(stand):
        pos = dict(stand)
        everyone = members + stand
        return {"chain": chain_takes(everyone, target, lone),
                "absorb": absorb_takes(members, stand, along, lone),
                "static": static_takes(last, along, side, pos[lone]),
                "pair_chain": any(chain_takes(everyone, target, p) for p in pair),
                "pair_absorb": any(absorb_takes(members, stand, along, p) for p in pair)}

    nominal = rules(standing)
    counts = Counter()
    for _ in range(N):
        jit = []
        for a, xy in standing:
            if a in bys:
                r, t = FIDGET * math.sqrt(rng.random()), rng.random() * 2 * math.pi
                xy = (xy[0] + r * math.cos(t), xy[1] + r * math.sin(t))
            jit.append((a, xy))
        for k, v in rules(jit).items():
            counts[k] += v
    k = [nominal[r] for r in ("chain", "absorb", "static")]
    stratum = ("ours" if (nominal["absorb"] and nominal["static"] and not nominal["chain"]) else
               "chain" if (nominal["chain"] or nominal["pair_chain"]) else
               "mixed" if (any(k) or nominal["pair_absorb"]) else "easy")
    place = (last[0] + STANDOFF * along[0], last[1] + STANDOFF * along[1])
    clear = min(math.dist(place, xy) for _, xy in members[:-1] + standing)
    return {"name": path.stem, "members": n, "stratum": stratum, "nominal": nominal,
            "fidget_share": {r: round(c / N, 3) for r, c in counts.items()},
            "standoff_clear_m": round(clear, 3), "standoff_valid": clear >= 0.6}


def main(folder: str) -> int:
    folder = Path(folder)
    meta = {c["name"]: c for c in json.loads((folder / "candidates.json").read_text())}
    rng = random.Random(0)
    rows = []
    for path in sorted(folder.glob("*_[0-9]*.yaml")):
        row = report(path, rng)
        row.update({k: meta[path.stem][k] for k in ("base", "offset_m", "past_last_m", "facing", "walkers")})
        rows.append(row)
    (folder / "report.json").write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} candidates")
    for base in ("cafe", "sweet"):
        sub = [r for r in rows if r["base"] == base]
        print(f"  {base}: strata {dict(Counter(r['stratum'] for r in sub))}; "
              f"chain {sum(r['nominal']['chain'] for r in sub)}, absorb {sum(r['nominal']['absorb'] for r in sub)}, "
              f"static {sum(r['nominal']['static'] for r in sub)}; standoff valid {sum(r['standoff_valid'] for r in sub)}; "
              f"pair taken by chain {sum(r['nominal']['pair_chain'] for r in sub)}, by absorb "
              f"{sum(r['nominal']['pair_absorb'] for r in sub)}")
    for st in ("ours", "chain", "mixed", "easy"):
        sub = [r for r in rows if r["stratum"] == st]
        if sub:
            off = sorted(r["offset_m"] for r in sub)
            past = sorted(r["past_last_m"] for r in sub)
            print(f"  {st:5s} n={len(sub):3d}: offset {off[0]:.2f}-{off[-1]:.2f} m, past {past[0]:.2f}-{past[-1]:.2f} m, "
                  f"members {dict(Counter(r['members'] for r in sub))}, facing {dict(Counter(r['facing'] for r in sub))}, "
                  f"standoff valid {sum(r['standoff_valid'] for r in sub)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
