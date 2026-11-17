"""Cache and oracle checks of the ceiling run (no scores are read).

1. Every arm replays CT2's first answer from the cache: same pose and status as CT2 in the confirm run.
2. Photo-time arms (CT4O, CT4OT, CT4PJT) have `history_t`.
3. CT4O and CT4OT: the roster at each re-solve is the sim's people ahead (those the tracker sees).
4. OW: every pseudo-resolve has roster = truth ahead and last = truth last.
5. The target oracle changes the goal: CT4OT's first re-solve pose differs from CT4O's.

Usage: python3 check_ceiling.py CEILING_ROOT CONFIRM_ROOT   (run folders relative to the working directory, e.g. results/data/moving/...)
"""
import collections
import json
import math
import pathlib
import sys

root, confirm = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
ARMS = ["OW", "CT4OT", "CT4O", "CT4PJT"]
PHOTO = {"CT4O", "CT4OT", "CT4PJT"}
ORACLE_ROSTER = {"CT4O", "CT4OT"}


def track_id(t):
    return t.get("id") if isinstance(t, dict) else (t[0] if isinstance(t, list) else t)


groups = sorted({p.parent.parent for p in root.glob("**/seed*/*/episode.json")})
print(f"groups {len(groups)}")
faults = collections.defaultdict(list)
roster = collections.Counter()
target_moved = collections.Counter()
for g in groups:
    rel = g.relative_to(root)
    name = f"{g.parent.name}/{g.name}"
    ref_f = confirm / rel / "CT2" / "episode.json"
    if not ref_f.exists():
        faults["no CT2 in the confirm run"].append(name)
        continue
    ref = json.loads(ref_f.read_text())["answers"][0]
    eps = {a: json.loads((g / a / "episode.json").read_text()) for a in ARMS if (g / a / "episode.json").exists()}
    for arm in ARMS:
        if arm not in eps:
            faults["missing episode"].append(f"{name} {arm}")
    for arm, e in eps.items():
        a = e["answers"][0]
        if not a.get("cached"):
            faults["first answer not from cache"].append(f"{name} {arm} calls={a.get('calls')}")
        if a.get("pose_world") != ref.get("pose_world") or a.get("status") != ref.get("status"):
            faults["first answer differs from CT2"].append(f"{name} {arm}")
        if arm in PHOTO and a.get("status") == "ok" and a.get("session") and not e.get("history_t"):
            faults["photo-time arm without history_t"].append(f"{name} {arm}")
        for r in e.get("resolves") or []:
            if r.get("status") != "ok":
                continue
            ahead = set((r.get("truth") or {}).get("ahead") or [])
            got = set(r.get("roster") or [])
            if arm == "OW":
                ok = got == ahead and r.get("last") == r["truth"].get("last")
                roster[(arm, "roster and last = truth" if ok else "differs")] += 1
            elif arm in ORACLE_ROSTER:
                seen = {track_id(t) for t in r.get("tracks") or []}
                if got == (ahead & seen):
                    roster[(arm, "roster = truth ahead (tracked)")] += 1
                elif got <= ahead:
                    roster[(arm, "roster inside truth ahead")] += 1
                else:
                    roster[(arm, "roster has someone not ahead")] += 1
                    faults["oracle roster has someone not ahead"].append(f"{name} {arm} t={r.get('sent_t')}")
    if "CT4O" in eps and "CT4OT" in eps:
        r0 = (eps["CT4O"].get("resolves") or [None])[0]
        r1 = (eps["CT4OT"].get("resolves") or [None])[0]
        if r0 and r1 and r0.get("pose_world") and r1.get("pose_world"):
            d = math.dist(r0["pose_world"][:2], r1["pose_world"][:2])
            target_moved["first goal differs > 0.05 m" if d > 0.05 else "first goal same"] += 1

print("\nroster checks (ok re-solves):")
for (arm, k), n in sorted(roster.items()):
    print(f"  {arm:6} {k}: {n}")
print("\nCT4OT against CT4O, first re-solve goal:", dict(target_moved))
print("\nfaults:" if faults else "\nfaults: none")
for k, v in faults.items():
    print(f"  {k}: {len(v)}")
    for x in v[:8]:
        print(f"    {x}")
