"""Unit-level tests of driver.py's Ledger and Lock (84bab2e4), zero calls, all files in review/ledger_tests/.

L1  8 processes of one model race reservations on one ledger (caps 120/60): exactly 60 sent rows?
L2  4 + 4 processes of two models (caps 50/30): total 50, neither model above 30?
L0  control: the first build's Ledger (source of before_fix_e6e73181/driver.py, run in its own namespace) under L1's
    race, to show the test can see a broken ledger.
L3  ledger.jsonl deleted between reservations: does the count survive?
L4  Lock: A takes it; the file is removed by hand; B takes it; A exits. Is B's lock still there?
L5  a torn last line in ledger.jsonl: what does a new Ledger do?
"""
import ast
import collections
import fcntl  # noqa: F401  (the old namespace needs the same imports)
import json
import multiprocessing as mp
import os
import random
import shutil
import sys
import time
from pathlib import Path

REVIEW = Path(__file__).resolve().parent
sys.path.insert(0, str(REVIEW))
import audit  # noqa: E402

LIVE = REVIEW.parents[1] / "live"
sys.path.insert(0, str(LIVE))
import driver as D  # noqa: E402

W = REVIEW / "ledger_tests"
if W.exists():
    shutil.rmtree(W)
W.mkdir()
RESULTS = {}


def old_ledger_class():
    """The first build's Ledger, compiled from its own source, with the names it uses."""
    src = (LIVE / "before_fix_e6e73181" / "driver.py").read_text()
    tree = ast.parse(src)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Ledger")
    ns = {"Path": Path, "json": json, "os": os, "collections": collections, "CAP_TOTAL": 120, "CAP_MODEL": 60,
          "CapReached": D.CapReached, "_now": D._now}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "old_driver_Ledger", "exec"), ns)
    return ns["Ledger"]


def worker(cls_name, path, model, caps, start_at, attempts, q):
    cls = D.Ledger if cls_name == "new" else old_ledger_class()
    ledger = cls(Path(path), model, *caps)                  # each "invocation" builds its ledger before the race
    while time.time() < start_at:
        time.sleep(0.001)
    sent = refused = 0
    for i in range(attempts):
        try:
            ledger.reserve({"arm": "x", "scene": f"s{os.getpid()}", "episode": str(i), "trial": 0, "stage": "constraints"})
            sent += 1
        except D.CapReached:
            refused += 1
            break
        time.sleep(random.random() * 0.002)
    q.put((os.getpid(), model, sent, refused))


def race(tag, cls_name, models, caps, attempts):
    path = W / f"{tag}.jsonl"
    q = mp.Queue()
    start_at = time.time() + 3.0
    procs = [mp.Process(target=worker, args=(cls_name, str(path), m, caps, start_at, attempts, q)) for m in models]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    got = [q.get() for _ in procs]
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    per = collections.Counter(r["model"] for r in rows if r["event"] == "sent")
    totals_seq = [r["total"] for r in rows if r["event"] == "sent"]
    RESULTS[tag] = {"class": cls_name, "caps": caps, "processes": len(models), "sent_rows": sum(per.values()),
                    "sent_by_model": dict(per), "refused_rows": sum(r["event"] == "refused" for r in rows),
                    "worker_sent": sum(g[2] for g in got),
                    "running_totals_strictly_increasing": totals_seq == list(range(1, len(totals_seq) + 1)),
                    "bad_lines": 0}
    print(tag, json.dumps(RESULTS[tag]))


if __name__ == "__main__":
    mp.set_start_method("fork")
    race("L1_one_model", "new", ["glm53"] * 8, (120, 60), 40)
    race("L2_two_models", "new", ["glm53"] * 4 + ["dsv41"] * 4, (50, 30), 40)
    race("L0_control_old", "old", ["glm53"] * 8, (120, 60), 40)

    # L3: delete the ledger between reservations.
    p3 = W / "L3.jsonl"
    led = D.Ledger(p3, "glm53", 120, 5)
    for i in range(5):
        led.reserve({"i": i})
    try:
        led.reserve({"i": 5})
        sixth = "sent"
    except D.CapReached:
        sixth = "refused"
    p3.unlink()
    try:
        led.reserve({"i": 6})
        after = "sent"
    except D.CapReached:
        after = "refused"
    RESULTS["L3_delete"] = {"cap_model": 5, "sixth_before_delete": sixth, "after_delete": after,
                            "count_after": led.per_model["glm53"]}
    print("L3", RESULTS["L3_delete"])

    lp = W / "L4.lock"
    a = D.Lock(lp).__enter__()
    lp.unlink()                                     # "remove it only if no driver is running" -- done wrongly
    b = D.Lock(lp).__enter__()
    b_token = lp.read_text()
    a.__exit__(None, None, None)
    present_after_a = lp.exists()
    try:
        c = D.Lock(lp).__enter__()
        third = "taken while B still holds it"
        c.__exit__(None, None, None)
    except SystemExit:
        third = "refused"
    RESULTS["L4_lock_by_path"] = {"b_token": b_token.strip(), "b_lock_present_after_a_exits": present_after_a,
                                  "third_invocation": third, "a_held": a.held, "b_held": b.held}
    print("L4", RESULTS["L4_lock_by_path"])

    # L5: a torn last line.
    p5 = W / "L5.jsonl"
    led5 = D.Ledger(p5, "glm53", 120, 60)
    led5.reserve({"i": 0})
    with open(p5, "a") as f:
        f.write('{"time": "x", "event": "sent", "mod')   # as if a writer died mid-line
    try:
        D.Ledger(p5, "glm53", 120, 60)
        l5 = "constructed"
    except Exception as exc:
        l5 = f"{type(exc).__name__}: {str(exc)[:80]}"
    RESULTS["L5_torn_line"] = {"new_ledger": l5}
    print("L5", RESULTS["L5_torn_line"])

    RESULTS["audit"] = {"env_hits": len(audit.HITS["env"]), "main_cache_hits": len(audit.HITS["main_cache"])}
    (REVIEW / "ledger_tests.json").write_text(json.dumps(RESULTS, indent=1))
