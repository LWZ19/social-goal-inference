"""T2a: the new driver's Ledger under real concurrency, without the lock file.

    t2a_ledger_stress.py OUT_DIR

The NEW driver module is imported (its main() is not run). Worker processes (fork) start together at a barrier, each
with its own Ledger instance on ONE ledger file, and each loops reserve() + note("answered") as GuardedBackend does,
until CapReached or 150 attempts. Variants:
  v1: caps 120/60, 4 x glm53 + 2 x dsv41       -> expect glm 60, dsv 60, total 120
  v2: caps 50/40,  3 x glm53 + 1 x dsv41       -> expect each model <= 40, total == 50 (total cap binds)
  v3: caps 120/60, 6 x glm53 + 0 x dsv41, with a pre-filled ledger of 55 glm53 'sent' rows -> glm 60 (only 5 more)
Checks per variant: every line valid JSON; per-model sent <= cap and total <= cap (== cap, since demand exceeds);
the 'total' field of the sent rows is exactly 1..N (no two reservations saw the same count); per model 'model_total'
is exactly 1..n; sum of successful reserves reported by workers == sent rows.
"""
import collections
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

V = Path(__file__).resolve().parent
sys.path.insert(0, str(V))
import audit  # noqa: E402

AUD = audit.install("t2a")
LIVE = Path("results/2026-10-17-beyond-vlm/phase2/i09/live")
if len(sys.argv) > 2 and sys.argv[2] == "old":     # negative control: the FIRST build's Ledger (count at start only)
    import types
    D = types.ModuleType("i09_first_build")
    D.__file__ = str(LIVE / "driver.py")
    sys.modules["i09_first_build"] = D
    _old = LIVE / "before_fix_e6e73181" / "driver.py"
    exec(compile(_old.read_text(encoding="utf-8"), str(_old), "exec"), D.__dict__)
else:
    sys.path.insert(0, str(LIVE))
    import driver as D  # noqa: E402
    assert Path(D.__file__).resolve() == LIVE / "driver.py"
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)


def worker(path, model, max_total, max_model, barrier, q, wid):
    ok = refused = 0
    err = None
    try:
        led = D.Ledger(Path(path), model, max_total, max_model)
        barrier.wait()
        for i in range(150):
            info = {"arm": "x", "scene": f"w{wid}", "episode": f"e{i}", "trial": 0, "stage": "constraints"}
            try:
                led.reserve(info)
            except D.CapReached:
                refused += 1
                break
            ok += 1
            led.note("answered", {**info, "finish": "ok"})
    except BaseException as exc:                       # noqa: BLE001
        err = repr(exc)
    q.put({"wid": wid, "model": model, "ok": ok, "refused": refused, "err": err,
           "env_hits": AUD["env_hits"], "main_cache_hits": AUD["main_cache_hits"]})


def run(name, max_total, max_model, n_glm, n_dsv, prefill=0):
    path = OUT / f"{name}_ledger.jsonl"
    if path.exists():
        path.unlink()
    if prefill:
        with open(path, "w") as f:
            for i in range(prefill):
                f.write(json.dumps({"time": "prefill", "event": "sent", "model": "glm53", "total": i + 1,
                                    "model_total": i + 1}) + "\n")
    ctx = mp.get_context("fork")
    models = ["glm53"] * n_glm + ["dsv41"] * n_dsv
    barrier, q = ctx.Barrier(len(models)), ctx.Queue()
    procs = [ctx.Process(target=worker, args=(str(path), m, max_total, max_model, barrier, q, i))
             for i, m in enumerate(models)]
    t0 = time.time()
    for p in procs:
        p.start()
    res = [q.get(timeout=300) for _ in procs]
    for p in procs:
        p.join()
    lines = path.read_text().splitlines()
    bad = []
    rows = []
    for ln in lines:
        try:
            rows.append(json.loads(ln))
        except ValueError:
            bad.append(ln[:120])
    sent = [r for r in rows if r.get("event") == "sent" and r.get("time") != "prefill"]
    all_sent = [r for r in rows if r.get("event") == "sent"]
    by_model = collections.Counter(r["model"] for r in all_sent)
    totals = sorted(r["total"] for r in sent)
    exp_totals = list(range(prefill + 1, prefill + len(sent) + 1))
    mt_ok = all(sorted(r["model_total"] for r in all_sent if r["model"] == m) == list(range(1, by_model[m] + 1))
                for m in by_model)
    reserved_by_workers = sum(r["ok"] for r in res)
    out = {"variant": name, "caps": [max_total, max_model], "workers": collections.Counter(models),
           "prefill": prefill, "seconds": round(time.time() - t0, 2), "lines": len(lines), "invalid_json_lines": bad,
           "sent_by_model": dict(by_model), "sent_total": len(all_sent),
           "events": dict(collections.Counter(r.get("event") for r in rows)),
           "totals_are_1_to_N": totals == exp_totals, "model_totals_are_1_to_n": mt_ok,
           "reserved_by_workers": reserved_by_workers, "workers_result": sorted(res, key=lambda r: r["wid"]),
           "exit_codes": [p.exitcode for p in procs]}
    return out


results = {
    "v1": run("v1", 120, 60, 4, 2),
    "v2": run("v2", 50, 40, 3, 1),
    "v3": run("v3", 120, 60, 6, 0, prefill=55),
}
checks = {
    "v1": results["v1"]["sent_by_model"] == {"glm53": 60, "dsv41": 60} and results["v1"]["sent_total"] == 120,
    "v2": results["v2"]["sent_total"] == 50 and all(n <= 40 for n in results["v2"]["sent_by_model"].values()),
    "v3": results["v3"]["sent_by_model"] == {"glm53": 60} and results["v3"]["reserved_by_workers"] == 5,
}
for k, r in results.items():
    checks[k] = checks[k] and not r["invalid_json_lines"] and r["totals_are_1_to_N"] and r["model_totals_are_1_to_n"] \
        and r["reserved_by_workers"] == r["sent_total"] - r["prefill"] and all(w["err"] is None for w in r["workers_result"])
results["checks"] = checks
results["audit"] = audit.summary()
results["worker_env_hits"] = sum(w["env_hits"] for r in ("v1", "v2", "v3") for w in results[r]["workers_result"])
(OUT / "t2a_result.json").write_text(json.dumps(results, indent=1, default=str))
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "workers_result"} if isinstance(v, dict) else v
                  for k, v in results.items()}, indent=1, default=str))
print("T2a", "PASS" if all(checks.values()) else "FAIL", checks)
