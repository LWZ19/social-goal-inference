"""Cache and oracle checks of CT3FT (no scores are read).

1. CT3FT replays CT2's cached first answer (runs_confirm4): same pose and status.
2. history_t present; CT3F's co-motion fields logged on every ok re-solve.
3. The target oracle changes the goal: CT3FT's first re-solve pose against CT3F's (runs_confirm4).

Usage: python3 check_ct3ft.py CEILING_ROOT CONFIRM_ROOT
"""
import collections, json, math, pathlib, sys

root, confirm = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
faults = collections.defaultdict(list)
moved = collections.Counter()
groups = sorted({p.parent.parent for p in root.glob("**/seed*/CT3FT/episode.json")})
print(f"groups {len(groups)}")
for g in groups:
    rel, name = g.relative_to(root), f"{g.parent.name}/{g.name}"
    e = json.loads((g / "CT3FT" / "episode.json").read_text())
    ref = json.loads((confirm / rel / "CT2" / "episode.json").read_text())["answers"][0]
    f3 = json.loads((confirm / rel / "CT3F" / "episode.json").read_text())
    a = e["answers"][0]
    if not a.get("cached"):
        faults["first answer not from cache"].append(name)
    if a.get("pose_world") != ref.get("pose_world") or a.get("status") != ref.get("status"):
        faults["first answer differs from CT2"].append(name)
    if a.get("status") == "ok" and a.get("session") and not e.get("history_t"):
        faults["no history_t"].append(name)
    for r in e.get("resolves") or []:
        if r.get("status") == "ok" and r.get("co_motion_events") is None:
            faults["ok re-solve without co-motion fields"].append(f"{name} t={r.get('sent_t')}")
    r0, r1 = (f3.get("resolves") or [None])[0], (e.get("resolves") or [None])[0]
    if r0 and r1 and r0.get("pose_world") and r1.get("pose_world"):
        d = math.dist(r0["pose_world"][:2], r1["pose_world"][:2])
        moved["first goal differs > 0.05 m" if d > 0.05 else "first goal same"] += 1
print("CT3FT against CT3F, first re-solve goal:", dict(moved))
print("\nfaults:" if faults else "\nfaults: none")
for k, v in faults.items():
    print(f"  {k}: {len(v)}")
    for x in v[:8]:
        print(f"    {x}")
