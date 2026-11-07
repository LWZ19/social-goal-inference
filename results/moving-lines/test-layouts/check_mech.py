"""Mechanical checks of a sweep run, as RUN_NOTE.md defines them (no scores are read).

1. re-solve errors; 2. a missing `history_t` or `truth`; 3. crashed or empty episodes;
4. arms that do not replay CT2's first answer from the cache.

Usage: python3 check_mech.py RUN_ROOT [ARMS...]
"""
import collections
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
arms = sys.argv[2:] or ["CT2", "CT3", "Gl", "CT3F", "CT4", "R1"]
REPLAY = {"CT3", "Gl", "CT3F", "CT4"}
PHOTO = {"CT3F", "CT4", "CT3p"}

groups = sorted({p.parent.parent for p in root.glob("**/seed*/*/episode.json")})
print(f"groups {len(groups)}, arms {arms}")
faults = collections.defaultdict(list)
status = collections.Counter()
answer_status = collections.Counter()
n_resolves = collections.defaultdict(list)
for g in groups:
    name = f"{g.parent.name}/{g.name}"
    eps = {}
    for arm in arms:
        f = g / arm / "episode.json"
        if not f.exists():
            faults["missing episode"].append(f"{name} {arm}")
            continue
        try:
            eps[arm] = json.loads(f.read_text())
        except Exception as e:                         # a crashed write
            faults["unreadable episode"].append(f"{name} {arm}: {e}")
    for arm, e in eps.items():
        if not e.get("samples"):
            faults["empty episode"].append(f"{name} {arm}")
        if not e.get("answers"):
            faults["no answer"].append(f"{name} {arm}")
            continue
        a = e["answers"][0]
        answer_status[(arm, a.get("status"))] += 1
        rs = e.get("resolves") or []
        n_resolves[arm].append(len(rs))
        for r in rs:
            status[(arm, r.get("status"))] += 1
            if "truth" not in r or r["truth"] is None:
                faults["resolve without truth"].append(f"{name} {arm} t={r.get('sent_t')}")
        if arm in PHOTO and a.get("status") == "ok" and a.get("session") and not e.get("history_t"):
            faults["photo-time arm without history_t"].append(f"{name} {arm}")
        if arm != "R1" and a.get("status") == "ok" and a.get("session") and not rs:
            faults["answer with a session but no re-solves"].append(f"{name} {arm}")
    if "CT2" in eps and eps["CT2"].get("answers"):
        ref = eps["CT2"]["answers"][0]
        for arm in REPLAY & eps.keys():
            a = (eps[arm].get("answers") or [{}])[0]
            if not a.get("cached"):
                faults["replay arm not from cache"].append(f"{name} {arm} calls={a.get('calls')}")
            if a.get("pose_world") != ref.get("pose_world") or a.get("status") != ref.get("status"):
                faults["replay arm answer differs from CT2"].append(f"{name} {arm}")

print("\nfirst-answer status by arm:")
for (arm, s), n in sorted(answer_status.items()):
    print(f"  {arm:5} {s}: {n}")
print("\nre-solve status by arm:")
for (arm, s), n in sorted(status.items()):
    print(f"  {arm:5} {s}: {n}")
print("\nre-solves per episode (min / median / max):")
for arm, ns in sorted(n_resolves.items()):
    ns = sorted(ns)
    print(f"  {arm:5} {ns[0]} / {ns[len(ns) // 2]} / {ns[-1]}")
print("\nfaults:" if faults else "\nfaults: none")
for k, v in faults.items():
    print(f"  {k}: {len(v)}")
    for x in v[:8]:
        print(f"    {x}")
