"""Item 11 layouts (results/moving-lines/best-version-new-layouts/I11_FROZEN.md; RANGES.md: candidates 3000-3449 per base, the
run only): select_pilot.py's logic unchanged, with its own folders and quotas. Zero calls; deterministic.

Quotas: ours 48, chain 12, mixed 6, easy 6; half from each base.
Within a stratum and base, candidates are sorted by their distance past the last member (`past_last_m`) and cut into
as many equal bins as layouts are wanted (a Latin-hypercube spread over that distance); from each bin the candidate is
taken that best evens out facing and line length so far (ties: the lower candidate index).
Stress flags, crossed 2 x 2 and balanced within each stratum and base, assigned in turn along the distance order:
  lag      : the queue's members wait U(0, 4) s before stepping up, and 20% of them wait for two free places;
  restless : the lone non-member makes one excursion towards the counter's face point, at U(10, 60) s, of
             U(0.75, 1.15) m (about 0.6-1.0 m realised), holds U(3, 6) s and goes back.
Writes sim/scenarios/i11/<name>.yaml (the candidate plus its flags) and i11_selection.json in results/moving-lines/test-layouts/.

    python3 select_i11.py
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "_lib" / "paths.py").exists()) / "_lib"))
import paths  # noqa: E402

HERE = paths.MOVING / "test-layouts"            # the layout candidates and selections
ROOT = paths.REPO
OUT = ROOT / "sim" / "scenarios" / "i11"
CAND = HERE / "i11_candidates"
QUOTA = {"ours": 48, "chain": 12, "mixed": 6, "easy": 6}
FLAGS = [(False, False), (True, True), (False, True), (True, False)]       # (lag, restless), in turn


def pick(cands, want):
    cands = sorted(cands, key=lambda r: (r["past_last_m"], r["name"]))
    bins = [cands[round(i * len(cands) / want):round((i + 1) * len(cands) / want)] for i in range(want)]
    facing, members, chosen = Counter(), Counter(), []
    for b in bins:
        best = min(b, key=lambda r: (facing[r["facing"]] + members[r["members"]], r["name"]))
        facing[best["facing"]] += 1
        members[best["members"]] += 1
        chosen.append(best)
    return chosen


def main() -> int:
    rows = json.loads((CAND / "report.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    selected = []
    for stratum, total in QUOTA.items():
        turn = 0                                   # flags rotate across both bases of a stratum
        for base in ("cafe", "sweet"):
            pool = [r for r in rows if r["stratum"] == stratum and r["base"] == base]
            want = total // 2
            if len(pool) < want:
                raise SystemExit(f"{stratum}/{base}: {len(pool)} candidates for {want}")
            for r in pick(pool, want):
                lag, restless = FLAGS[turn % 4]
                turn += 1
                selected.append({**r, "lag": lag, "restless": restless})
    for r in selected:
        src = CAND / f"{r['name']}.yaml"
        text = src.read_text()
        header = "".join(line + "\n" for line in text.splitlines() if line.startswith("#"))
        scene = yaml.safe_load(text)
        (qname, q), = scene["queues"].items()
        if r["lag"]:
            q["step_lag"], q["lazy_share"] = [0.0, 4.0], 0.2
        if r["restless"]:
            lone = next(h for h in scene["humans"] if h.get("bystander"))
            lone["excursion"] = {"at": [10.0, 60.0], "dist": [0.75, 1.15], "hold": [3.0, 6.0],
                                 "toward": list(q["slots"][0]["face"])}
        header += (f"# Selected by select_i11.py (item 11, candidates 3000-3449): stratum {r['stratum']}, lag {r['lag']}, restless {r['restless']}.\n")
        (OUT / f"{r['name']}.yaml").write_text(header + yaml.safe_dump(scene, sort_keys=False))
    for r in selected:                             # a seed per layout: with one seed for all, every layout drew the
        r["seed"] = int(hashlib.sha256(r["name"].encode()).hexdigest()[:8], 16) % 100000   # same first service time
    doc = {"quota": QUOTA, "n": len(selected),
           "layouts": [{k: r[k] for k in ("name", "seed", "base", "stratum", "members", "offset_m", "past_last_m",
                                          "facing", "walkers", "lag", "restless", "standoff_valid")} for r in selected]}
    (HERE / "i11_selection.json").write_text(json.dumps(doc, indent=1))
    digest = hashlib.sha256(b"".join((OUT / f"{r['name']}.yaml").read_bytes() for r in selected)).hexdigest()[:8]
    print(f"{len(selected)} layouts -> {OUT}; scenario files' sha256 {digest}")
    for stratum in QUOTA:
        sub = [r for r in selected if r["stratum"] == stratum]
        print(f"  {stratum:5s} {len(sub):2d}: bases {dict(Counter(r['base'] for r in sub))}, facing "
              f"{dict(Counter(r['facing'] for r in sub))}, members {dict(Counter(r['members'] for r in sub))}, "
              f"flags {dict(Counter((r['lag'], r['restless']) for r in sub))}, past {min(r['past_last_m'] for r in sub):.2f}-"
              f"{max(r['past_last_m'] for r in sub):.2f} m, walkers {dict(Counter(r['walkers'] for r in sub))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
