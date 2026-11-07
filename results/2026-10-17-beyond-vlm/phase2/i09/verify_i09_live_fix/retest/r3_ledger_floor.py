"""R3 (unit level): the second-round Ledger's floor (LedgerLost), on the driver module of build 41940c3c.

    r3_ledger_floor.py OUT_DIR

The driver module is imported from LIVE (its main() is not run; PYTHONDONTWRITEBYTECODE keeps LIVE untouched).
L1 delete / L2 truncate in place / L3 one sent row dropped in place / L4 a non-sent row appended / L5 two instances,
then delete / L6 a corrupt line / L7 note() after a delete / L8 a new instance on a file that lost rows before it
started / L9 forked workers sharing one file while the parent deletes it mid-run.
"""
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

V = Path(__file__).resolve().parent
sys.path.insert(0, str(V))
import audit  # noqa: E402

AUD = audit.install("r3_floor")
LIVE = Path("results/2026-10-17-beyond-vlm/phase2/i09/live")
sys.path.insert(0, str(LIVE))
import driver as D  # noqa: E402
assert Path(D.__file__).resolve() == LIVE / "driver.py"
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
INFO = {"arm": "x", "scene": "s", "episode": "e", "trial": 0, "stage": "constraints"}


def fresh(name):
    p = OUT / f"{name}.jsonl"
    if p.exists():
        p.unlink()
    return p


def outcome(fn):
    try:
        fn()
        return "ok"
    except BaseException as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {str(exc)[:160]}"


res = {}
# L1 delete
p = fresh("l1"); L = D.Ledger(p, "glm53", 120, 60)
for _ in range(3): L.reserve(INFO)
os.remove(p)
res["L1_delete"] = {"reserve_after": outcome(lambda: L.reserve(INFO)), "file_recreated": p.exists(),
                    "recreated_size": p.stat().st_size if p.exists() else None, "ledger_total_attr_after": L.total}
# L2 truncate in place
p = fresh("l2"); L = D.Ledger(p, "glm53", 120, 60)
for _ in range(3): L.reserve(INFO)
open(p, "r+").truncate(0)
res["L2_truncate"] = {"reserve_after": outcome(lambda: L.reserve(INFO))}
# L3 drop one sent row in place
p = fresh("l3"); L = D.Ledger(p, "glm53", 120, 60)
for _ in range(3): L.reserve(INFO)
lines = p.read_text().splitlines(keepends=True)
with open(p, "r+") as f:
    f.write("".join(lines[1:])); f.truncate()
res["L3_drop_one"] = {"reserve_after": outcome(lambda: L.reserve(INFO))}
# L4 a non-sent row appended (answered / refused): must go on
p = fresh("l4"); L = D.Ledger(p, "glm53", 120, 60)
for _ in range(3): L.reserve(INFO)
with open(p, "a") as f:
    f.write(json.dumps({"event": "answered", "model": "glm53"}) + "\n")
res["L4_append_nonsent"] = {"reserve_after": outcome(lambda: L.reserve(INFO)), "total": L.total}
# L5 two instances share the file, then delete
p = fresh("l5"); A = D.Ledger(p, "glm53", 120, 60); B = D.Ledger(p, "dsv41", 120, 60)
A.reserve(INFO); A.reserve(INFO); B.reserve(INFO)
os.remove(p)
res["L5_two_instances"] = {"B_after": outcome(lambda: B.reserve(INFO)), "A_after": outcome(lambda: A.reserve(INFO))}
# L6 a corrupt line
p = fresh("l6"); L = D.Ledger(p, "glm53", 120, 60)
L.reserve(INFO)
with open(p, "a") as f:
    f.write('{"event": "sent", "model": "glm5\n')
res["L6_corrupt_line"] = {"reserve_after": outcome(lambda: L.reserve(INFO)),
                          "is_a_STOP": False}
try:
    L.reserve(INFO)
except D.STOPS:
    res["L6_corrupt_line"]["is_a_STOP"] = True
except Exception:
    pass
# L7 note() after a delete (as GuardedBackend's "answered" note after the send)
p = fresh("l7"); L = D.Ledger(p, "glm53", 120, 60)
L.reserve(INFO)
os.remove(p)
res["L7_note_after_delete"] = {"note_after": outcome(lambda: L.note("answered", INFO))}
# L8 the floor is per instance: a new instance on a file that lost rows before it started starts from what it finds
p = fresh("l8"); L = D.Ledger(p, "glm53", 120, 60)
for _ in range(5): L.reserve(INFO)
os.remove(p)
L2 = D.Ledger(p, "glm53", 120, 60)
res["L8_new_instance_after_loss"] = {"at_start": L2.at_start, "reserve": outcome(lambda: L2.reserve(INFO))}


# L9 forked workers, parent deletes the file mid-run
def worker(path, wid, q, barrier):
    ok, why = 0, None
    try:
        led = D.Ledger(Path(path), "glm53", 120, 60)
        barrier.wait()
        for i in range(200):
            led.reserve({**INFO, "scene": f"w{wid}", "episode": f"e{i}"})
            ok += 1
            time.sleep(0.01)
            led.note("answered", {**INFO, "scene": f"w{wid}", "episode": f"e{i}"})
    except BaseException as exc:  # noqa: BLE001
        why = type(exc).__name__
    q.put({"wid": wid, "ok": ok, "why": why})


p = fresh("l9"); ctx = mp.get_context("fork"); q = ctx.Queue(); bar = ctx.Barrier(5)
procs = [ctx.Process(target=worker, args=(str(p), w, q, bar)) for w in range(4)]
for pr in procs: pr.start()
bar.wait()
while True:
    n = sum('"sent"' in ln for ln in p.read_text().splitlines()) if p.exists() else 0
    if n >= 20:
        break
    time.sleep(0.002)
os.rename(p, OUT / "l9_before_delete.jsonl")      # keep what was there, then the path is gone
got = [q.get(timeout=120) for _ in procs]
for pr in procs: pr.join()
before = [json.loads(x) for x in (OUT / "l9_before_delete.jsonl").read_text().splitlines()]
after = [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
res["L9_concurrent_delete"] = {
    "workers": sorted(got, key=lambda r: r["wid"]),
    "sent_before_delete_file": sum(r["event"] == "sent" for r in before),
    "sent_in_recreated_file": sum(r["event"] == "sent" for r in after),
    "reserved_by_workers": sum(r["ok"] for r in got),
}
checks = {
    "L1": res["L1_delete"]["reserve_after"].startswith("LedgerLost"),
    "L2": res["L2_truncate"]["reserve_after"].startswith("LedgerLost"),
    "L3": res["L3_drop_one"]["reserve_after"].startswith("LedgerLost"),
    "L4": res["L4_append_nonsent"]["reserve_after"] == "ok",
    "L5": res["L5_two_instances"]["A_after"].startswith("LedgerLost") and res["L5_two_instances"]["B_after"].startswith("LedgerLost"),
    "L7": res["L7_note_after_delete"]["note_after"].startswith("LedgerLost"),
    "L9": all(w["why"] == "LedgerLost" for w in got) and res["L9_concurrent_delete"]["sent_in_recreated_file"] == 0
          and res["L9_concurrent_delete"]["reserved_by_workers"] == res["L9_concurrent_delete"]["sent_before_delete_file"],
}
res["checks"] = checks
res["audit"] = audit.summary()
(OUT / "r3_floor_result.json").write_text(json.dumps(res, indent=1, default=str))
print(json.dumps(res, indent=1, default=str))
print("R3 floor", "PASS" if all(checks.values()) else "FAIL", checks)
